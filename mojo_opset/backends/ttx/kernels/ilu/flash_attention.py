"""
ILU: paged prefill GQA — KV gather + GQA expand in PyTorch; causal attention in Triton.
Aligned with MojoPagedPrefillGQA (mask j <= i + (kv_seq_len - q_seq_len)).

Paged decode (``paged_attention_decode_impl`` / ``paged_decode_kernel``)
------------------------------------------------------------------------
Host and kernel impose the following; callers must satisfy them or the launch will fail:

* **KV block size upper bound**: ``block_size`` (the sequence-length slot count per physical
  KV cache block, i.e. ``key_cache.shape[2]``) is asserted **≤ 128**. This matches the
  current Triton row-tile limit for the decode path (temporary restriction; message may
  say ``temp:``).

* **Page size equals Triton N tile**: ``paged_decode_kernel`` passes ``PAGE_SIZE`` and
  ``BLOCK_SIZE_N`` both from the same ``block_size`` and enforces
  ``tl.static_assert(PAGE_SIZE == BLOCK_SIZE_N)``. So the **paged KV page size** used in
  the kernel must be exactly the **block_ptr row tile** along the sequence axis — one
  logical page per ``make_block_ptr`` tile. Do not mix a different nominal page size with
  a different ``BLOCK_SIZE_N``.

"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from .utils import libentry


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

        out_vec = acc / denom
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


def paged_attention_prefill_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqlens_kv: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    softmax_scale: Optional[float] = None,
    aux_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del aux_mask

    total_q_tokens, num_q_heads, head_dim = q.shape
    _, num_kv_heads, block_size, _ = key_cache.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    outputs = torch.zeros(total_q_tokens, num_q_heads, head_dim, dtype=q.dtype, device=q.device)

    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    batch_size = len(q_lens)

    for i in range(batch_size):
        q_seq_len = q_lens[i].item()
        start_loc = cu_seqlens_q[i].item()
        end_loc = cu_seqlens_q[i + 1].item()
        q_batch = q[start_loc:end_loc].contiguous()
        if seqlens_kv is None:
            kv_seq_len = q_seq_len
        else:
            kv_seq_len = seqlens_kv[i].item()

        num_blocks_for_seq = (kv_seq_len + block_size - 1) // block_size
        k_unpadded = torch.zeros(kv_seq_len, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)
        v_unpadded = torch.zeros(kv_seq_len, num_kv_heads, head_dim, dtype=q.dtype, device=q.device)

        for j in range(num_blocks_for_seq):
            physical_block_id = block_tables[i, j].item()

            start_pos_in_seq = j * block_size
            end_pos_in_seq = min(start_pos_in_seq + block_size, kv_seq_len)
            tokens_in_block = end_pos_in_seq - start_pos_in_seq

            k_slice = key_cache[physical_block_id, :, :tokens_in_block, :]
            k_unpadded[start_pos_in_seq:end_pos_in_seq, :, :] = k_slice.permute(1, 0, 2)

            v_slice = value_cache[physical_block_id, :, :tokens_in_block, :]
            v_unpadded[start_pos_in_seq:end_pos_in_seq, :, :] = v_slice.permute(1, 0, 2)

        if num_q_heads != num_kv_heads:
            g = num_q_heads // num_kv_heads
            # repeat: head dim becomes [kv_0..kv_{K-1}] tiled g times → aligns with decode
            #   kv_head_id = q_head_id % num_kv_heads when gqa_interleave / GQA_INTERLEAVE.
            # repeat_interleave: each kv head repeated g times in order → aligns with
            #   kv_head_id = q_head_id // g when not interleaved. See module docstring.
            if gqa_interleave:
                k_expanded = k_unpadded.repeat((1, g, 1))
                v_expanded = v_unpadded.repeat((1, g, 1))
            else:
                k_expanded = k_unpadded.repeat_interleave(g, dim=1)
                v_expanded = v_unpadded.repeat_interleave(g, dim=1)
        else:
            k_expanded = k_unpadded
            v_expanded = v_unpadded

        k_expanded = k_expanded.contiguous()
        v_expanded = v_expanded.contiguous()
        out_slice = torch.empty_like(q_batch)
        _launch_causal_attn_triton(q_batch, k_expanded, v_expanded, out_slice, softmax_scale, q_seq_len, kv_seq_len)
        outputs[start_loc:end_loc] = out_slice

    return outputs


@libentry()
@triton.jit
def paged_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    seqlens_ptr,
    block_tables_ptr,
    BATCH_SIZE,
    NUM_TOTAL_BLOCKS,
    MAX_NUM_BLOCKS_PER_SEQ,
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
    sm_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    OUT_T: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_SIZE_D, "HEAD_DIM should be less than BLOCK_SIZE_D")
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    num_tasks = BATCH_SIZE * NUM_Q_HEADS

    for q_task_id in tl.range(pid, num_tasks, n_progs):
        q_head_id = q_task_id % NUM_Q_HEADS
        b_id = q_task_id // NUM_Q_HEADS
        # Matches paged_attention_prefill_impl: interleave → % ; grouped → //
        if GQA_INTERLEAVE:
            kv_head_id = q_head_id % NUM_KV_HEADS
        else:
            kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

        kv_seq_len = tl.load(seqlens_ptr + b_id)

        offs_d = tl.arange(0, BLOCK_SIZE_D)
        q_ptrs = q_ptr + b_id * stride_qb + q_head_id * stride_qh + offs_d * stride_qd
        q = tl.load(q_ptrs, mask=offs_d < HEAD_DIM, other=0.0)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

        num_logical_blocks = tl.cdiv(kv_seq_len, PAGE_SIZE)

        # PAGE_SIZE and BLOCK_SIZE_N are both host `block_size`; see module docstring.
        tl.static_assert(PAGE_SIZE == BLOCK_SIZE_N, "PAGE_SIZE should be equal to BLOCK_SIZE_N")

        for logical_block_idx in tl.range(0, num_logical_blocks):
            physical_block_id = tl.load(block_tables_ptr + b_id * stride_bt_batch + logical_block_idx * stride_bt_block)

            kv_block_start_in_seq = logical_block_idx * PAGE_SIZE
            kv_block_end_in_seq = tl.minimum(kv_block_start_in_seq + PAGE_SIZE, kv_seq_len)
            kv_block_len = kv_block_end_in_seq - kv_block_start_in_seq
            k_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr + physical_block_id * stride_k_block + kv_head_id * stride_k_head,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_k_blksz, stride_k_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr + physical_block_id * stride_v_block + kv_head_id * stride_v_head,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len

            k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")

            qk = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
            qk *= sm_scale
            qk = tl.where(mask, qk, float("-inf"))

            m_j = tl.max(qk, axis=0)
            m_ij = tl.maximum(m_i, m_j)
            qk = qk - m_ij

            p = tl.math.exp(qk)

            p_cast = p.to(k.dtype)

            v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

            l_ij = tl.sum(p, axis=0)

            alpha = tl.math.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij

            acc = acc * alpha

            acc += tl.sum((p_cast[:, None] * v).to(tl.float32), axis=0)

            m_i = m_ij

        m_i = m_i + tl.math.log(l_i)
        acc = acc / l_i

        o_ptrs = o_ptr + b_id * stride_ob + q_head_id * stride_oh + offs_d * stride_od
        tl.store(o_ptrs, acc.to(OUT_T), mask=offs_d < HEAD_DIM)


def paged_attention_decode_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Paged KV decode attention (one query step per batch row).

    Requirements (see module docstring):

    * ``key_cache`` / ``value_cache`` block length ``block_size`` (dim 2) must be **≤ 128**.
    * That same ``block_size`` is ``PAGE_SIZE`` and ``BLOCK_SIZE_N`` in ``paged_decode_kernel``;
      the kernel statically asserts ``PAGE_SIZE == BLOCK_SIZE_N``.
    """
    batch_size, num_q_heads, head_dim = q.shape
    num_total_blocks, num_kv_heads, block_size, head_dim_cache = key_cache.shape

    assert block_size <= 128, f"temp: only support block_size <= 128, but got {block_size}"
    max_num_blocks_per_seq = block_tables.shape[1]

    assert head_dim == head_dim_cache
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    o = torch.empty_like(q)

    block = 256
    grid = (triton.cdiv(batch_size * num_q_heads, block),)
    block_size_d = triton.next_power_of_2(head_dim)

    if q.dtype == torch.float16:
        out_t = tl.float16
    elif q.dtype == torch.bfloat16:
        out_t = tl.bfloat16
    else:
        out_t = tl.float32

    paged_decode_kernel[grid](
        q,
        key_cache,
        value_cache,
        o,
        seqlens,
        block_tables.to(torch.int32),
        batch_size,
        num_total_blocks,
        max_num_blocks_per_seq,
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
        block_tables.stride(0),
        block_tables.stride(1),
        softmax_scale,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        block_size,
        BLOCK_SIZE_D=block_size_d,
        BLOCK_SIZE_N=block_size,
        OUT_T=out_t,
    )
    return o
