"""
ILU Triton attention kernels:
  - Paged prefill GQA (Flash Attention v2, online softmax, direct paged KV access)
  - Paged decode GQA (KV gather + causal attention)
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from .utils import LOG2E, libentry, smart_triton_autotune


@triton.jit
def causal_mask_fn(mask_ptr, mask_size, mask_stride_m, mask_stride_n, q_start, kv_start, Q_BLOCK, KV_BLOCK):
    offset_causal = min(max(kv_start - q_start, -mask_size), mask_size)
    offsets_mask_causal = (
        (tl.arange(0, Q_BLOCK)[:, None]) * mask_stride_m
        + (mask_size + offset_causal + tl.arange(0, KV_BLOCK)[None, :]) * mask_stride_n
    )
    mask_causal = tl.load(mask_ptr + offsets_mask_causal).to(tl.int1)
    return mask_causal


@smart_triton_autotune(
    configs=[
        triton.Config({"BLOCK_M": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128}, num_warps=8, num_stages=2),
    ],
    selected_idx=0,
    key=["NUM_Q_HEADS", "HEAD_DIM", "PAGE_SIZE"],
)
@triton.jit
def _paged_prefill_fav2_kernel(
    Q,
    K_cache,
    V_cache,
    Out,
    aux_mask_ptr,
    cu_q_lens_ptr,
    seqlens_kv_ptr,
    block_tables_ptr,
    stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_bt_batch, stride_bt_block,
    stride_mask_m, stride_mask_n,
    sm_scale,
    AUX_MASK_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_AUX_MASK: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_D, "BLOCK_D must be >= HEAD_DIM")

    local_q_block_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    b_id = tl.program_id(2)

    q_start = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
    q_seq_len = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32) - q_start
    kv_seq_len = tl.load(seqlens_kv_ptr + b_id).to(tl.int32)

    q_blk_start = local_q_block_id * BLOCK_M
    if q_blk_start >= q_seq_len:
        return

    if GQA_INTERLEAVE:
        kv_head_id = q_head_id % NUM_KV_HEADS
    else:
        kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

    q_blk_end = tl.minimum(q_blk_start + BLOCK_M, q_seq_len)
    q_blk_len = q_blk_end - q_blk_start

    Q_blk_ptr = tl.make_block_ptr(
        base=Q + (q_start + q_blk_start) * stride_qt + q_head_id * stride_qh,
        shape=(q_blk_len, HEAD_DIM),
        strides=(stride_qt, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    q = tl.load(Q_blk_ptr, boundary_check=(0, 1), padding_option="zero")

    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    kv_cache_len = kv_seq_len - q_seq_len

    if IS_CAUSAL:
        kv_loop_end = tl.minimum(kv_seq_len, q_blk_end + kv_cache_len)
    else:
        kv_loop_end = kv_seq_len

    num_kv_pages = tl.cdiv(kv_loop_end, PAGE_SIZE)
    num_full_pages = kv_seq_len // PAGE_SIZE
    num_full_pages = tl.minimum(num_full_pages, num_kv_pages)

    qk_scale = sm_scale * LOG2E

    for page_idx in tl.range(0, num_full_pages):
        physical_block = tl.load(
            block_tables_ptr + b_id * stride_bt_batch + page_idx * stride_bt_block
        )

        K_T_blk_ptr = tl.make_block_ptr(
            base=K_cache + physical_block * stride_kb + kv_head_id * stride_kh,
            shape=(HEAD_DIM, PAGE_SIZE),
            strides=(stride_kd, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_D, BLOCK_N),
            order=(0, 1),
        )
        V_blk_ptr = tl.make_block_ptr(
            base=V_cache + physical_block * stride_vb + kv_head_id * stride_vh,
            shape=(PAGE_SIZE, HEAD_DIM),
            strides=(stride_vn, stride_vd),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

        for kv_inner_idx in tl.range(0, PAGE_SIZE // BLOCK_N):
            kv_blk_start = page_idx * PAGE_SIZE + kv_inner_idx * BLOCK_N

            if HEAD_DIM == BLOCK_D:
                k_T = tl.load(K_T_blk_ptr)
                v = tl.load(V_blk_ptr)
            else:
                k_T = tl.load(K_T_blk_ptr, boundary_check=(0,), padding_option="zero")
                v = tl.load(V_blk_ptr, boundary_check=(1,), padding_option="zero")

            qk = tl.dot(q, k_T)
            qk = qk * qk_scale

            if IS_CAUSAL:
                if USE_AUX_MASK:
                    mask = causal_mask_fn(
                        aux_mask_ptr,
                        AUX_MASK_SIZE,
                        stride_mask_m,
                        stride_mask_n,
                        kv_cache_len + q_blk_start,
                        kv_blk_start,
                        BLOCK_M,
                        BLOCK_N,
                    )
                    qk = tl.where(mask, qk, float("-inf"))
                else:
                    offs_m = q_blk_start + tl.arange(0, BLOCK_M)
                    offs_n = kv_blk_start + tl.arange(0, BLOCK_N)
                    causal_mask = (offs_m[:, None] + kv_cache_len) >= offs_n[None, :]
                    qk = tl.where(causal_mask, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(qk - m_ij[:, None])

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(K_cache.dtype.element_ty), v, acc=acc)

            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_ij

            K_T_blk_ptr = tl.advance(K_T_blk_ptr, (0, BLOCK_N))
            V_blk_ptr = tl.advance(V_blk_ptr, (BLOCK_N, 0))

    for page_idx in tl.range(num_full_pages, num_kv_pages):
        kv_page_start = page_idx * PAGE_SIZE
        kv_blk_end = tl.minimum(kv_page_start + PAGE_SIZE, kv_seq_len)
        kv_blk_len = kv_blk_end - kv_page_start

        physical_block = tl.load(
            block_tables_ptr + b_id * stride_bt_batch + page_idx * stride_bt_block
        )

        K_T_blk_ptr = tl.make_block_ptr(
            base=K_cache + physical_block * stride_kb + kv_head_id * stride_kh,
            shape=(HEAD_DIM, kv_blk_len),
            strides=(stride_kd, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_D, BLOCK_N),
            order=(0, 1),
        )
        V_blk_ptr = tl.make_block_ptr(
            base=V_cache + physical_block * stride_vb + kv_head_id * stride_vh,
            shape=(kv_blk_len, HEAD_DIM),
            strides=(stride_vn, stride_vd),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

        num_inner_blocks = tl.cdiv(kv_blk_len, BLOCK_N)
        for kv_inner_idx in tl.range(0, num_inner_blocks):
            kv_blk_start = kv_page_start + kv_inner_idx * BLOCK_N

            k_T = tl.load(K_T_blk_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_blk_ptr, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k_T)
            qk = qk * qk_scale

            if IS_CAUSAL:
                if USE_AUX_MASK:
                    mask = causal_mask_fn(
                        aux_mask_ptr,
                        AUX_MASK_SIZE,
                        stride_mask_m,
                        stride_mask_n,
                        kv_cache_len + q_blk_start,
                        kv_blk_start,
                        BLOCK_M,
                        BLOCK_N,
                    )
                    qk = tl.where(mask, qk, float("-inf"))
                else:
                    offs_m = q_blk_start + tl.arange(0, BLOCK_M)
                    offs_n = kv_blk_start + tl.arange(0, BLOCK_N)
                    causal_mask = (offs_m[:, None] + kv_cache_len) >= offs_n[None, :]
                    qk = tl.where(causal_mask, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(qk - m_ij[:, None])

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(K_cache.dtype.element_ty), v, acc=acc)

            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_ij

            K_T_blk_ptr = tl.advance(K_T_blk_ptr, (0, BLOCK_N))
            V_blk_ptr = tl.advance(V_blk_ptr, (BLOCK_N, 0))

    l_i = tl.maximum(l_i, 1e-6)
    acc = acc / l_i[:, None]

    O_blk_ptr = tl.make_block_ptr(
        base=Out + (q_start + q_blk_start) * stride_ot + q_head_id * stride_oh,
        shape=(q_blk_len, HEAD_DIM),
        strides=(stride_ot, stride_od),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(O_blk_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))


@libentry()
@triton.jit
def _paged_prefill_causal_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_j,
    stride_k_h,
    stride_k_d,
    stride_v_j,
    stride_v_h,
    stride_v_d,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    Tq: tl.constexpr,
    Tk: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    sm_scale,
    diag_off: tl.constexpr,
    OUT_T: tl.constexpr,
):
    """
    For each (query row t_i, head h): softmax over key j with causal mask
    j <= t_i + diag_off, diag_off = kv_seq_len - q_seq_len.

    D_PAD is next power of 2 >= D (ILU Triton needs power-of-2 vector tiles for arange/zeros).
    """
    pid = tl.program_id(0)
    pnum = tl.num_programs(0)
    total = Tq * H

    offs_d = tl.arange(0, D_PAD)
    mask_d = offs_d < D

    for flat in tl.range(pid, total, pnum):
        t_i = flat // H
        h = flat % H

        q_base = t_i * stride_q_t + h * stride_q_h
        q_vec = tl.load(q_ptr + q_base + offs_d * stride_q_d, mask=mask_d, other=0.0).to(tl.float32)

        m_max = tl.full((), -float("inf"), tl.float32)
        for j in range(Tk):
            allowed = j <= t_i + diag_off
            k_base = j * stride_k_j + h * stride_k_h
            k_vec = tl.load(k_ptr + k_base + offs_d * stride_k_d, mask=mask_d, other=0.0).to(tl.float32)
            s = tl.sum(q_vec * k_vec) * sm_scale
            s = tl.where(allowed, s, float("-inf"))
            m_max = tl.maximum(m_max, s)

        denom = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((D_PAD,), dtype=tl.float32)
        for j in range(Tk):
            allowed = j <= t_i + diag_off
            k_base = j * stride_k_j + h * stride_k_h
            v_base = j * stride_v_j + h * stride_v_h
            k_vec = tl.load(k_ptr + k_base + offs_d * stride_k_d, mask=mask_d, other=0.0).to(tl.float32)
            v_vec = tl.load(v_ptr + v_base + offs_d * stride_v_d, mask=mask_d, other=0.0).to(tl.float32)
            s = tl.sum(q_vec * k_vec) * sm_scale
            s = tl.where(allowed, s, float("-inf"))
            p = tl.exp(s - m_max)
            denom = denom + p
            acc = acc + p * v_vec

        if Tk > 0:
            out_vec = acc / denom
        else:
            out_vec = acc
        o_base = t_i * stride_o_t + h * stride_o_h
        tl.store(out_ptr + o_base + offs_d * stride_o_d, out_vec.to(OUT_T), mask=mask_d)


def _launch_causal_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
    q_seq_len: int,
    kv_seq_len: int,
) -> None:
    tq, h, d = q.shape
    tk = k.shape[0]
    assert k.shape == (tk, h, d) and v.shape == (tk, h, d)
    diag_off = kv_seq_len - q_seq_len

    if q.dtype == torch.float16:
        out_t = tl.float16
    elif q.dtype == torch.bfloat16:
        out_t = tl.bfloat16
    else:
        out_t = tl.float32

    total_tasks = tq * h
    block = 256
    grid = (triton.cdiv(total_tasks, block),)

    d_pad = triton.next_power_of_2(d)

    _paged_prefill_causal_attn_kernel[grid](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        Tq=tq,
        Tk=tk,
        H=h,
        D=d,
        D_PAD=d_pad,
        sm_scale=float(sm_scale),
        diag_off=int(diag_off),
        OUT_T=out_t,
    )


@smart_triton_autotune(
    configs=[
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    selected_idx=0,
    key=["HQ", "D", "PAGE_SIZE"],
)
@libentry()
@triton.jit
def _paged_decode_gqa_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    seqlens_ptr,
    block_tables_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_ob,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    HQ,
    HKV,
    D,
    GROUP,
    SOFTMAX_SCALE,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GQA_LAYOUT: tl.constexpr,
    ZERO_EMPTY_SEQLEN: tl.constexpr,
    OUT_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_h >= HQ:
        return

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    out_ptrs = o_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od

    seq_len = tl.load(seqlens_ptr + pid_b).to(tl.int32)
    if seq_len <= 0:
        if ZERO_EMPTY_SEQLEN:
            tl.store(out_ptrs, tl.zeros((BLOCK_D,), dtype=tl.float32).to(OUT_T), mask=d_mask)
        return
    kv_h = tl.where(GQA_LAYOUT == 0, pid_h % HKV, pid_h // GROUP)

    q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0)

    neg_inf = -1.0e30
    m = tl.full((1,), neg_inf, tl.float32)
    l = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    num_blocks = tl.cdiv(seq_len, BLOCK_N)
    for block_idx in tl.range(0, num_blocks):
        start_n = block_idx * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        s_mask = offs_n < seq_len

        logical_block_idx = offs_n // PAGE_SIZE
        offset_in_block = offs_n % PAGE_SIZE
        physical_block_id = tl.load(
            block_tables_ptr + pid_b * stride_bt_batch + logical_block_idx * stride_bt_block,
            mask=s_mask,
            other=0,
        )

        k_ptrs = (
            k_cache_ptr
            + physical_block_id[:, None] * stride_k_block
            + kv_h * stride_k_head
            + offset_in_block[:, None] * stride_k_blksz
            + offs_d[None, :] * stride_k_dim
        )
        v_ptrs = (
            v_cache_ptr
            + physical_block_id[:, None] * stride_v_block
            + kv_h * stride_v_head
            + offset_in_block[:, None] * stride_v_blksz
            + offs_d[None, :] * stride_v_dim
        )
        k = tl.load(k_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0)
        scores = tl.sum(k * q[None, :], axis=1) * SOFTMAX_SCALE
        scores = tl.where(s_mask, scores, neg_inf)
        m_new = tl.maximum(m, tl.max(scores, axis=0))
        alpha = tl.math.exp(m - m_new)
        p = tl.where(s_mask, tl.math.exp(scores - m_new), 0.0)
        l = l * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m = m_new

    l = tl.maximum(l, 1e-6)
    out = acc / l
    tl.store(out_ptrs, out.to(OUT_T), mask=d_mask)


def paged_attention_prefill_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    seqlens_kv: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    softmax_scale: Optional[float] = None,
    aux_mask: Optional[torch.Tensor] = None,
    max_q_lens: Optional[int] = None,
    max_total_seq_lens: Optional[int] = None,
) -> torch.Tensor:
    total_q_tokens, num_q_heads, head_dim = q.shape
    _, num_kv_heads, block_size, _ = key_cache.shape
    batch_size = cu_q_lens.shape[0] - 1

    sm_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)

    if seqlens_kv is None:
        seqlens_kv = (cu_q_lens[1:] - cu_q_lens[:-1]).to(torch.int32)
    else:
        seqlens_kv = seqlens_kv.to(torch.int32)

    use_aux_mask = aux_mask is not None

    out = torch.empty_like(q)
    block_tables_i32 = block_tables.to(torch.int32)

    BLOCK_D = triton.next_power_of_2(head_dim)

    if use_aux_mask:
        mask_ptr = aux_mask
        mask_stride_m = aux_mask.stride(0)
        mask_stride_n = aux_mask.stride(1)
        mask_size = aux_mask.shape[0]
    else:
        mask_ptr = q
        mask_stride_m = 0
        mask_stride_n = 0
        mask_size = 0

    def grid(META):
        bm = META["BLOCK_M"]
        if max_q_lens is not None:
            max_q_blocks = (max_q_lens + bm - 1) // bm
        else:
            q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
            max_q_blocks = ((q_lens + bm - 1) // bm).max().item()
        return (max_q_blocks, num_q_heads, batch_size)

    _paged_prefill_fav2_kernel[grid](
        q,
        key_cache,
        value_cache,
        out,
        mask_ptr,
        cu_q_lens,
        seqlens_kv,
        block_tables_i32,
        q.stride(0), q.stride(1), q.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        block_tables_i32.stride(0), block_tables_i32.stride(1),
        mask_stride_m, mask_stride_n,
        float(sm_scale),
        AUX_MASK_SIZE=mask_size,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        GQA_INTERLEAVE=gqa_interleave,
        HEAD_DIM=head_dim,
        BLOCK_D=BLOCK_D,
        PAGE_SIZE=block_size,
        BLOCK_N=min(block_size, 128),
        IS_CAUSAL=True,
        USE_AUX_MASK=use_aux_mask,
    )

    return out


def _paged_decode_gather_and_causal(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    softmax_scale: float,
) -> torch.Tensor:
    """ILU paged decode: direct paged KV Triton, one Q token per batch row."""
    batch_size, num_q_heads, head_dim = q.shape
    _, num_kv_heads, page_size, head_dim_cache = key_cache.shape

    assert head_dim == head_dim_cache
    group = num_q_heads // num_kv_heads
    o = torch.empty_like(q)
    block_d = triton.next_power_of_2(head_dim)
    layout_id = 0 if gqa_interleave else 1

    # MAX_S_LEN = 0
    if block_tables.shape[1] == 0:
        return torch.zeros_like(q)

    if q.dtype == torch.float16:
        out_t = tl.float16
    elif q.dtype == torch.bfloat16:
        out_t = tl.bfloat16
    else:
        out_t = tl.float32

    seqlens_i32 = seqlens.to(torch.int32)
    block_tables_i32 = block_tables.to(torch.int32)
    zero_empty_seqlen = not torch.cuda.is_current_stream_capturing()

    grid = (batch_size, num_q_heads)
    _paged_decode_gqa_kernel[grid](
        q,
        key_cache,
        value_cache,
        o,
        seqlens_i32,
        block_tables_i32,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        block_tables_i32.stride(0),
        block_tables_i32.stride(1),
        num_q_heads,
        num_kv_heads,
        head_dim,
        group,
        float(softmax_scale),
        BLOCK_D=block_d,
        PAGE_SIZE=page_size,
        GQA_LAYOUT=layout_id,
        ZERO_EMPTY_SEQLEN=zero_empty_seqlen,
        OUT_T=out_t,
    )
    return o


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["HQ", "D", "PAGE_SIZE", "COMPUTE_INT8"],
)
@libentry()
@triton.jit
def _paged_decode_quant_gqa_kernel(
    q_ptr,
    k_cache_ptr,
    key_scale_ptr,
    v_cache_ptr,
    value_scale_ptr,
    o_ptr,
    seqlens_ptr,
    block_tables_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_ks_head,
    stride_ks_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_vs_head,
    stride_vs_dim,
    stride_ob,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    HQ,
    HKV,
    D,
    GROUP,
    SOFTMAX_SCALE,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    GQA_LAYOUT: tl.constexpr,
    COMPUTE_INT8: tl.constexpr,
    OUT_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_h >= HQ:
        return

    seq_len = tl.load(seqlens_ptr + pid_b).to(tl.int32)
    kv_h = tl.where(GQA_LAYOUT == 0, pid_h % HKV, pid_h // GROUP)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    k_scale = tl.load(
        key_scale_ptr + kv_h * stride_ks_head + offs_d * stride_ks_dim,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    v_scale = tl.load(
        value_scale_ptr + kv_h * stride_vs_head + offs_d * stride_vs_dim,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    if COMPUTE_INT8:
        q_scaled = q * k_scale
        q_amax = tl.max(tl.abs(q_scaled), axis=0)
        q_quant_scale = q_amax / 127.0
        q_quant_scale = tl.where(q_quant_scale < 1.0e-6, 1.0, q_quant_scale)
        q_quant_f = q_scaled / q_quant_scale
        q_quant_pos = (q_quant_f + 0.5).to(tl.int32)
        q_quant_neg = (q_quant_f - 0.5).to(tl.int32)
        q_quant_i32 = tl.where(q_quant_f >= 0.0, q_quant_pos, q_quant_neg)
        q_quant_i32 = tl.minimum(tl.maximum(q_quant_i32, -128), 127)
        q_for_score = q_quant_i32.to(tl.float32)
    else:
        q_for_score = q
        q_quant_scale = 1.0

    neg_inf = -1.0e30
    m = tl.full((1,), neg_inf, tl.float32)
    l = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    num_blocks = tl.cdiv(seq_len, BLOCK_N)

    if COMPUTE_INT8:
        # Pass 1: online softmax to compute m and l
        for block_idx in tl.range(0, num_blocks):
            start_n = block_idx * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)
            s_mask = offs_n < seq_len

            logical_block_idx = offs_n // PAGE_SIZE
            offset_in_block = offs_n % PAGE_SIZE
            physical_block_id = tl.load(
                block_tables_ptr + pid_b * stride_bt_batch + logical_block_idx * stride_bt_block,
                mask=s_mask,
                other=0,
            )

            k_ptrs = (
                k_cache_ptr
                + physical_block_id[:, None] * stride_k_block
                + kv_h * stride_k_head
                + offset_in_block[:, None] * stride_k_blksz
                + offs_d[None, :] * stride_k_dim
            )
            k = tl.load(k_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            scores = tl.sum(k * q_for_score[None, :], axis=1) * q_quant_scale * SOFTMAX_SCALE
            scores = tl.where(s_mask, scores, neg_inf)
            m_new = tl.maximum(m, tl.max(scores, axis=0))
            alpha = tl.math.exp(m - m_new)
            p = tl.where(s_mask, tl.math.exp(scores - m_new), 0.0)
            l = l * alpha + tl.sum(p, axis=0)
            m = m_new

        # p_amax is always 1.0 since max(exp(scores - m)) = exp(0) = 1.0
        safe_l = tl.where(l > 0, l, 1.0)
        p_scale = (1.0 / safe_l) / 127.0
        p_scale = tl.where(p_scale < 1.0e-6, 1.0, p_scale)

        # Pass 2: quantized P @ V
        for block_idx in tl.range(0, num_blocks):
            start_n = block_idx * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)
            s_mask = offs_n < seq_len

            logical_block_idx = offs_n // PAGE_SIZE
            offset_in_block = offs_n % PAGE_SIZE
            physical_block_id = tl.load(
                block_tables_ptr + pid_b * stride_bt_batch + logical_block_idx * stride_bt_block,
                mask=s_mask,
                other=0,
            )

            k_ptrs = (
                k_cache_ptr
                + physical_block_id[:, None] * stride_k_block
                + kv_h * stride_k_head
                + offset_in_block[:, None] * stride_k_blksz
                + offs_d[None, :] * stride_k_dim
            )
            v_ptrs = (
                v_cache_ptr
                + physical_block_id[:, None] * stride_v_block
                + kv_h * stride_v_head
                + offset_in_block[:, None] * stride_v_blksz
                + offs_d[None, :] * stride_v_dim
            )
            k = tl.load(k_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v = tl.load(v_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            scores = tl.sum(k * q_for_score[None, :], axis=1) * q_quant_scale * SOFTMAX_SCALE
            scores = tl.where(s_mask, scores, neg_inf)
            p = tl.where(s_mask, tl.math.exp(scores - m) / safe_l, 0.0)
            p_quant_f = p / p_scale
            p_quant_i32 = (p_quant_f + 0.5).to(tl.int32)
            p_quant_i32 = tl.minimum(tl.maximum(p_quant_i32, -128), 127)
            acc += tl.sum(p_quant_i32.to(tl.float32)[:, None] * v, axis=0)

        out = acc * p_scale * v_scale
        out = tl.where(l > 0, out, 0.0)
    else:
        # Single pass: online softmax
        for block_idx in tl.range(0, num_blocks):
            start_n = block_idx * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)
            s_mask = offs_n < seq_len

            logical_block_idx = offs_n // PAGE_SIZE
            offset_in_block = offs_n % PAGE_SIZE
            physical_block_id = tl.load(
                block_tables_ptr + pid_b * stride_bt_batch + logical_block_idx * stride_bt_block,
                mask=s_mask,
                other=0,
            )

            k_ptrs = (
                k_cache_ptr
                + physical_block_id[:, None] * stride_k_block
                + kv_h * stride_k_head
                + offset_in_block[:, None] * stride_k_blksz
                + offs_d[None, :] * stride_k_dim
            )
            v_ptrs = (
                v_cache_ptr
                + physical_block_id[:, None] * stride_v_block
                + kv_h * stride_v_head
                + offset_in_block[:, None] * stride_v_blksz
                + offs_d[None, :] * stride_v_dim
            )
            k = tl.load(k_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v = tl.load(v_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            scores = tl.sum(k * k_scale[None, :] * q_for_score[None, :], axis=1) * SOFTMAX_SCALE
            scores = tl.where(s_mask, scores, neg_inf)
            m_new = tl.maximum(m, tl.max(scores, axis=0))
            alpha = tl.math.exp(m - m_new)
            p = tl.where(s_mask, tl.math.exp(scores - m_new), 0.0)
            l = l * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v * v_scale[None, :], axis=0)
            m = m_new

        safe_l = tl.where(l > 0, l, 1.0)
        out = acc / safe_l
        out = tl.where(l > 0, out, 0.0)

    out_ptrs = o_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, out.to(OUT_T), mask=d_mask)


def paged_attention_decode_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Paged KV decode attention (one query step per batch row)."""
    _, _, head_dim = q.shape
    _, _, _, head_dim_cache = key_cache.shape

    assert head_dim == head_dim_cache
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    return _paged_decode_gather_and_causal(
        q,
        key_cache,
        value_cache,
        seqlens,
        block_tables,
        gqa_interleave,
        float(softmax_scale),
    )


def paged_attention_decode_with_kv_dequant_impl(
    q: torch.Tensor,
    query_scale: Optional[torch.Tensor],
    key_cache: torch.Tensor,
    key_scale: torch.Tensor,
    value_cache: torch.Tensor,
    value_scale: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    compute_dtype: torch.dtype,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Paged KV decode attention with int8 KV cache and per-channel scales."""
    if query_scale is not None:
        raise NotImplementedError("ILU quant paged decode does not support quantized query.")
    if key_cache.dtype != torch.int8 or value_cache.dtype != torch.int8:
        raise TypeError("paged_attention_decode_with_kv_dequant_impl expects int8 key/value cache.")

    batch_size, num_q_heads, head_dim = q.shape
    _, num_kv_heads, page_size, head_dim_cache = key_cache.shape

    assert head_dim == head_dim_cache
    assert key_scale.shape == (num_kv_heads, head_dim)
    assert value_scale.shape == (num_kv_heads, head_dim)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    group = num_q_heads // num_kv_heads
    out = torch.empty_like(q)
    block_d = triton.next_power_of_2(head_dim)
    layout_id = 0 if gqa_interleave else 1
    compute_int8 = compute_dtype == torch.int8

    if q.dtype == torch.float16:
        out_t = tl.float16
    elif q.dtype == torch.bfloat16:
        out_t = tl.bfloat16
    else:
        out_t = tl.float32

    seqlens_i32 = seqlens.to(torch.int32)
    block_tables_i32 = block_tables.to(torch.int32)

    grid = (batch_size, num_q_heads)
    _paged_decode_quant_gqa_kernel[grid](
        q,
        key_cache,
        key_scale,
        value_cache,
        value_scale,
        out,
        seqlens_i32,
        block_tables_i32,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        key_scale.stride(0),
        key_scale.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        value_scale.stride(0),
        value_scale.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        block_tables_i32.stride(0),
        block_tables_i32.stride(1),
        num_q_heads,
        num_kv_heads,
        head_dim,
        group,
        float(softmax_scale),
        BLOCK_D=block_d,
        PAGE_SIZE=page_size,
        GQA_LAYOUT=layout_id,
        COMPUTE_INT8=compute_int8,
        OUT_T=out_t,
    )
    return out
