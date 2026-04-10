"""
ILU Triton SWA operators for contiguous KV infer and paged prefill.

This implementation keeps the control flow on host for varlen / paged handling
and uses a generic masked attention Triton kernel for the per-sequence compute.
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from .utils import ilu_grid_dim_from_row_tasks
from .utils import libentry


def _generate_window_mask(
    q_seq_len: int,
    kv_seq_len: int,
    device: torch.device,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
) -> torch.Tensor:
    kv_computed_len = kv_seq_len - q_seq_len
    q_pos = torch.arange(kv_computed_len, kv_computed_len + q_seq_len, device=device)[:, None]
    kv_pos = torch.arange(0, kv_seq_len, device=device)[None, :]
    causal_mask = q_pos >= kv_pos
    if local_window_size is None and global_window_size is None:
        return causal_mask

    local_window_mask = q_pos <= (kv_pos + local_window_size) if local_window_size is not None else False
    global_window_mask = kv_pos < global_window_size if global_window_size is not None else False
    return causal_mask & (local_window_mask | global_window_mask)


def _expand_kv_heads(x: torch.Tensor, num_q_heads: int, num_kv_heads: int, gqa_interleave: bool) -> torch.Tensor:
    if num_q_heads == num_kv_heads:
        return x
    repeat = num_q_heads // num_kv_heads
    if gqa_interleave:
        return x.repeat((1, repeat, 1))
    return x.repeat_interleave(repeat, dim=1)


@libentry()
@triton.jit
def _swa_masked_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    out_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_m0,
    stride_m1,
    TQ: tl.constexpr,
    TK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    sm_scale,
    OUT_T: tl.constexpr,
):
    pid = tl.program_id(0)
    pnum = tl.num_programs(0)
    total = TQ * H

    offs_d = tl.arange(0, D_PAD)
    mask_d = offs_d < D

    for flat in tl.range(pid, total, pnum):
        qi = flat // H
        h = flat % H

        q_base = qi * stride_q_t + h * stride_q_h
        q_vec = tl.load(q_ptr + q_base + offs_d * stride_q_d, mask=mask_d, other=0.0).to(tl.float32)

        m_max = tl.full((), -float("inf"), tl.float32)
        for j in range(TK):
            allowed = tl.load(mask_ptr + qi * stride_m0 + j * stride_m1)
            k_base = j * stride_k_t + h * stride_k_h
            k_vec = tl.load(k_ptr + k_base + offs_d * stride_k_d, mask=mask_d, other=0.0).to(tl.float32)
            s = tl.sum(q_vec * k_vec) * sm_scale
            s = tl.where(allowed, s, float("-inf"))
            m_max = tl.maximum(m_max, s)

        denom = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((D_PAD,), dtype=tl.float32)
        for j in range(TK):
            allowed = tl.load(mask_ptr + qi * stride_m0 + j * stride_m1)
            k_base = j * stride_k_t + h * stride_k_h
            v_base = j * stride_v_t + h * stride_v_h
            k_vec = tl.load(k_ptr + k_base + offs_d * stride_k_d, mask=mask_d, other=0.0).to(tl.float32)
            v_vec = tl.load(v_ptr + v_base + offs_d * stride_v_d, mask=mask_d, other=0.0).to(tl.float32)
            s = tl.sum(q_vec * k_vec) * sm_scale
            s = tl.where(allowed, s, float("-inf"))
            p = tl.exp(s - m_max)
            denom = denom + p
            acc = acc + p * v_vec

        out_vec = acc / denom
        o_base = qi * stride_o_t + h * stride_o_h
        tl.store(out_ptr + o_base + offs_d * stride_o_d, out_vec.to(OUT_T), mask=mask_d)


def _launch_swa_masked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
) -> None:
    tq, h, d = q.shape
    tk = k.shape[0]
    assert q.shape == (tq, h, d)
    assert k.shape == (tk, h, d)
    assert v.shape == (tk, h, d)
    assert mask.shape == (tq, tk) and mask.dtype == torch.bool

    if q.dtype == torch.float16:
        out_t = tl.float16
    elif q.dtype == torch.bfloat16:
        out_t = tl.bfloat16
    else:
        out_t = tl.float32

    d_pad = triton.next_power_of_2(d)
    grid = (ilu_grid_dim_from_row_tasks(tq * h),)

    _swa_masked_fwd_kernel[grid](
        q,
        k,
        v,
        mask,
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
        mask.stride(0),
        mask.stride(1),
        TQ=tq,
        TK=tk,
        H=h,
        D=d,
        D_PAD=d_pad,
        sm_scale=float(sm_scale),
        OUT_T=out_t,
    )


@libentry()
@triton.jit
def _swa_acc_fwd_mxn(
    acc_ptr,
    l_i,
    m_i,
    q,
    k_block_ptr,
    v_block_ptr,
    mask,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    if mask is False:
        return acc_ptr, l_i, m_i

    k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
    qk = tl.dot(q, tl.trans(k))
    qk = qk * qk_scale
    if mask is not None and mask is not True:
        qk = tl.where(mask, qk, float("-inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp(qk)
    p_cast = p.to(k.dtype)

    v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc_ptr = acc_ptr * alpha[:, None]
    acc_ptr = tl.dot(p_cast, v, acc_ptr)
    m_i = m_ij
    return acc_ptr, l_i, m_i


@libentry()
@triton.jit
def _swa_infer_kernel(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    bsz,
    cu_seqlens_q_ptr,
    cu_seqlens_kv_ptr,
    scale,
    stride_ot,
    stride_oh,
    stride_od,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    IS_CAUSAL: tl.constexpr,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    cu_q_chunks = 0
    q_offsets = tl.arange(0, BLOCK_M)
    kv_offsets = tl.arange(0, BLOCK_N)

    for b_id in range(bsz):
        q_start = tl.load(cu_seqlens_q_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_seqlens_q_ptr + b_id + 1).to(tl.int32)
        kv_start = tl.load(cu_seqlens_kv_ptr + b_id).to(tl.int32)
        kv_end = tl.load(cu_seqlens_kv_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end - q_start
        kv_seq_len = kv_end - kv_start
        kv_computed_len = kv_seq_len - q_seq_len

        num_q_chunks = tl.cdiv(q_seq_len, BLOCK_M)
        prev_q_tasks = cu_q_chunks * NUM_Q_HEADS
        cu_q_chunks += num_q_chunks
        new_q_tasks = num_q_chunks * NUM_Q_HEADS

        for q_task_id in range((prev_q_tasks + pid) % n_programs, new_q_tasks, n_programs):
            q_block_id = q_task_id // NUM_Q_HEADS
            q_head_id = q_task_id % NUM_Q_HEADS
            if GQA_INTERLEAVE:
                kv_head_id = q_head_id % NUM_KV_HEADS
            else:
                kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

            q_block_start = q_block_id * BLOCK_M
            q_block_end = min(q_block_start + BLOCK_M, q_seq_len)
            q_block_len = q_block_end - q_block_start
            q_valid = (q_block_start + q_offsets) < q_seq_len
            q_abs = q_block_start + q_offsets + kv_computed_len

            q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_qt, stride_qd),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            q_block = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

            num_global_window_blocks, non_global_window_start_block, num_total_blocks = _swa_split_blocks(
                q_block_start + kv_computed_len,
                q_block_len,
                kv_seq_len,
                BLOCK_N,
                IS_CAUSAL,
                GLOBAL_WINDOW,
                LOCAL_WINDOW,
            )

            for kv_block_id in range(num_global_window_blocks):
                kv_block_start = kv_block_id * BLOCK_N
                kv_abs = kv_block_start + kv_offsets
                kv_valid = kv_abs < kv_seq_len
                mask = q_valid[:, None] & kv_valid[None, :]
                if IS_CAUSAL:
                    causal_mask = q_abs[:, None] >= kv_abs[None, :]
                    global_mask = kv_abs[None, :] < GLOBAL_WINDOW
                    if LOCAL_WINDOW is not None:
                        local_mask = q_abs[:, None] <= (kv_abs + LOCAL_WINDOW)[None, :]
                        global_mask = global_mask | local_mask
                    mask = mask & causal_mask & global_mask

                k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + kv_start * stride_kt + kv_head_id * stride_kh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + kv_start * stride_vt + kv_head_id * stride_vh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                acc, l_i, m_i = _swa_acc_fwd_mxn(
                    acc,
                    l_i,
                    m_i,
                    q_block,
                    k_block_ptr,
                    v_block_ptr,
                    mask,
                    scale,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_D,
                )

            for kv_block_id in range(non_global_window_start_block, num_total_blocks):
                kv_block_start = kv_block_id * BLOCK_N
                kv_abs = kv_block_start + kv_offsets
                kv_valid = kv_abs < kv_seq_len
                mask = q_valid[:, None] & kv_valid[None, :]
                if IS_CAUSAL:
                    causal_mask = q_abs[:, None] >= kv_abs[None, :]
                    if LOCAL_WINDOW is not None:
                        local_mask = q_abs[:, None] <= (kv_abs + LOCAL_WINDOW)[None, :]
                        causal_mask = causal_mask & local_mask
                    mask = mask & causal_mask

                k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + kv_start * stride_kt + kv_head_id * stride_kh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + kv_start * stride_vt + kv_head_id * stride_vh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                acc, l_i, m_i = _swa_acc_fwd_mxn(
                    acc,
                    l_i,
                    m_i,
                    q_block,
                    k_block_ptr,
                    v_block_ptr,
                    mask,
                    scale,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_D,
                )

            l_safe = tl.where(q_valid, l_i, 1.0)
            out_block = acc / tl.where(l_safe[:, None] > 0, l_safe[:, None], 1.0)
            out_block = tl.where(l_i[:, None] > 0, out_block, 0.0)
            out_block = tl.where(q_valid[:, None], out_block, 0.0)
            o_block_ptr = tl.make_block_ptr(
                base=o_ptr + q_start * stride_ot + q_head_id * stride_oh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_ot, stride_od),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            tl.store(o_block_ptr, out_block.to(o_ptr.type.element_ty), boundary_check=(0, 1))


@libentry()
@triton.jit
def _swa_paged_prefill_kernel(
    o_ptr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    bsz,
    cu_seqlens_q_ptr,
    kv_lens_ptr,
    block_table_ptr,
    scale,
    stride_ot,
    stride_oh,
    stride_od,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vp,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_block_table_b,
    stride_block_table_p,
    IS_CAUSAL: tl.constexpr,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    tl.static_assert(PAGE_SIZE % BLOCK_N == 0, "BLOCK_N must divide PAGE_SIZE for paged KV tiling")
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    cu_q_chunks = 0
    q_offsets = tl.arange(0, BLOCK_M)
    kv_offsets = tl.arange(0, BLOCK_N)

    for b_id in range(bsz):
        q_start = tl.load(cu_seqlens_q_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_seqlens_q_ptr + b_id + 1).to(tl.int32)
        kv_seq_len = tl.load(kv_lens_ptr + b_id).to(tl.int32)
        q_seq_len = q_end - q_start
        kv_computed_len = kv_seq_len - q_seq_len

        num_q_chunks = tl.cdiv(q_seq_len, BLOCK_M)
        prev_q_tasks = cu_q_chunks * NUM_Q_HEADS
        cu_q_chunks += num_q_chunks
        new_q_tasks = num_q_chunks * NUM_Q_HEADS

        for q_task_id in range((prev_q_tasks + pid) % n_programs, new_q_tasks, n_programs):
            q_block_id = q_task_id // NUM_Q_HEADS
            q_head_id = q_task_id % NUM_Q_HEADS
            if GQA_INTERLEAVE:
                kv_head_id = q_head_id % NUM_KV_HEADS
            else:
                kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

            q_block_start = q_block_id * BLOCK_M
            q_block_end = min(q_block_start + BLOCK_M, q_seq_len)
            q_block_len = q_block_end - q_block_start
            q_valid = (q_block_start + q_offsets) < q_seq_len
            q_abs = q_block_start + q_offsets + kv_computed_len

            q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_qt, stride_qd),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            q_block = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

            num_global_window_blocks, non_global_window_start_block, num_total_blocks = _swa_split_blocks(
                q_block_start + kv_computed_len,
                q_block_len,
                kv_seq_len,
                BLOCK_N,
                IS_CAUSAL,
                GLOBAL_WINDOW,
                LOCAL_WINDOW,
            )

            for kv_block_id in range(num_global_window_blocks):
                kv_block_start = kv_block_id * BLOCK_N
                kv_block_end = min(kv_block_start + BLOCK_N, kv_seq_len)
                kv_block_len = kv_block_end - kv_block_start
                logical_page_id = kv_block_start // PAGE_SIZE
                kv_block_start_in_page = kv_block_start % PAGE_SIZE
                physical_page_id = tl.load(
                    block_table_ptr + b_id * stride_block_table_b + logical_page_id * stride_block_table_p
                )
                kv_abs = kv_block_start + kv_offsets
                kv_valid = kv_abs < kv_seq_len
                mask = q_valid[:, None] & kv_valid[None, :]
                if IS_CAUSAL:
                    causal_mask = q_abs[:, None] >= kv_abs[None, :]
                    global_mask = kv_abs[None, :] < GLOBAL_WINDOW
                    if LOCAL_WINDOW is not None:
                        local_mask = q_abs[:, None] <= (kv_abs + LOCAL_WINDOW)[None, :]
                        global_mask = global_mask | local_mask
                    mask = mask & causal_mask & global_mask

                k_block_ptr = tl.make_block_ptr(
                    base=k_cache_ptr
                    + physical_page_id * stride_kp
                    + kv_head_id * stride_kh
                    + kv_block_start_in_page * stride_kt,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                v_block_ptr = tl.make_block_ptr(
                    base=v_cache_ptr
                    + physical_page_id * stride_vp
                    + kv_head_id * stride_vh
                    + kv_block_start_in_page * stride_vt,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                acc, l_i, m_i = _swa_acc_fwd_mxn(
                    acc,
                    l_i,
                    m_i,
                    q_block,
                    k_block_ptr,
                    v_block_ptr,
                    mask,
                    scale,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_D,
                )

            for kv_block_id in range(non_global_window_start_block, num_total_blocks):
                kv_block_start = kv_block_id * BLOCK_N
                kv_block_end = min(kv_block_start + BLOCK_N, kv_seq_len)
                kv_block_len = kv_block_end - kv_block_start
                logical_page_id = kv_block_start // PAGE_SIZE
                kv_block_start_in_page = kv_block_start % PAGE_SIZE
                physical_page_id = tl.load(
                    block_table_ptr + b_id * stride_block_table_b + logical_page_id * stride_block_table_p
                )
                kv_abs = kv_block_start + kv_offsets
                kv_valid = kv_abs < kv_seq_len
                mask = q_valid[:, None] & kv_valid[None, :]
                if IS_CAUSAL:
                    causal_mask = q_abs[:, None] >= kv_abs[None, :]
                    if LOCAL_WINDOW is not None:
                        local_mask = q_abs[:, None] <= (kv_abs + LOCAL_WINDOW)[None, :]
                        causal_mask = causal_mask & local_mask
                    mask = mask & causal_mask

                k_block_ptr = tl.make_block_ptr(
                    base=k_cache_ptr
                    + physical_page_id * stride_kp
                    + kv_head_id * stride_kh
                    + kv_block_start_in_page * stride_kt,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                v_block_ptr = tl.make_block_ptr(
                    base=v_cache_ptr
                    + physical_page_id * stride_vp
                    + kv_head_id * stride_vh
                    + kv_block_start_in_page * stride_vt,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                acc, l_i, m_i = _swa_acc_fwd_mxn(
                    acc,
                    l_i,
                    m_i,
                    q_block,
                    k_block_ptr,
                    v_block_ptr,
                    mask,
                    scale,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_D,
                )

            l_safe = tl.where(q_valid, l_i, 1.0)
            out_block = acc / l_safe[:, None]
            out_block = tl.where(q_valid[:, None], out_block, 0.0)
            o_block_ptr = tl.make_block_ptr(
                base=o_ptr + q_start * stride_ot + q_head_id * stride_oh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_ot, stride_od),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            tl.store(o_block_ptr, out_block.to(o_ptr.type.element_ty), boundary_check=(0, 1))


def swa_infer_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> torch.Tensor:
    total_q_tokens, num_q_heads, head_dim = q.shape
    _, num_kv_heads, _ = k.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    outputs = torch.empty_like(q)
    batch_size = cu_seqlens_q.shape[0] - 1
    block_d = triton.next_power_of_2(head_dim)
    block_m = 64 if (q.dtype == torch.float32 or block_d >= 128) else 128
    block_n = 64
    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    total_q_chunks = int(torch.div(q_lens + block_m - 1, block_m, rounding_mode="floor").sum().item())
    grid = (ilu_grid_dim_from_row_tasks(total_q_chunks * num_q_heads),)

    _swa_infer_kernel[grid](
        outputs,
        q,
        k,
        v,
        batch_size,
        cu_seqlens_q,
        cu_seqlens_kv,
        softmax_scale,
        outputs.stride(0),
        outputs.stride(1),
        outputs.stride(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        is_causal,
        global_window_size,
        local_window_size,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_stages=1,
    )
    return outputs


def swa_paged_prefill_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqlens_kv: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> torch.Tensor:
    total_q_tokens, num_q_heads, head_dim = q.shape
    _, num_kv_heads, block_size, _ = key_cache.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    outputs = torch.empty_like(q)
    batch_size = cu_seqlens_q.shape[0] - 1
    if seqlens_kv is None:
        seqlens_kv = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    block_n = min(128, triton.next_power_of_2(block_size))
    if block_size % block_n != 0:
        raise ValueError(
            f"KV block_size ({block_size}) must be divisible by Triton tile size ({block_n}); "
            "use a compatible page size (e.g. power of two, multiple of 128 for large pages)."
        )
    block_d = triton.next_power_of_2(head_dim)
    block_m = 64 if (q.dtype == torch.float32 or block_d >= 128 or block_n >= 128) else 128
    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    total_q_chunks = int(torch.div(q_lens + block_m - 1, block_m, rounding_mode="floor").sum().item())
    grid = (ilu_grid_dim_from_row_tasks(total_q_chunks * num_q_heads),)

    _swa_paged_prefill_kernel[grid](
        outputs,
        q,
        key_cache,
        value_cache,
        batch_size,
        cu_seqlens_q,
        seqlens_kv,
        block_tables,
        softmax_scale,
        outputs.stride(0),
        outputs.stride(1),
        outputs.stride(2),
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
        block_tables.stride(0),
        block_tables.stride(1),
        is_causal,
        global_window_size,
        local_window_size,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        PAGE_SIZE=block_size,
        num_stages=1,
    )
    return outputs

@triton.jit
def _swa_split_blocks(
    q_block_start_id,
    q_block_len,
    kv_seq_len,
    BLOCK_SIZE_N,
    IS_CAUSAL,
    GLOBAL_WINDOW_SIZE,
    LOCAL_WINDOW_SIZE,
):
    if not IS_CAUSAL:
        return 0, 0, tl.cdiv(kv_seq_len, BLOCK_SIZE_N)

    num_total_blocks = tl.cdiv(q_block_start_id + q_block_len, BLOCK_SIZE_N)
    if GLOBAL_WINDOW_SIZE is None and LOCAL_WINDOW_SIZE is None:
        return 0, 0, num_total_blocks

    if GLOBAL_WINDOW_SIZE is not None:
        num_global_window_blocks = tl.minimum(
            tl.cdiv(GLOBAL_WINDOW_SIZE, BLOCK_SIZE_N), num_total_blocks
        )
    else:
        num_global_window_blocks = 0

    if LOCAL_WINDOW_SIZE is not None:
        local_window_start_id = tl.maximum(q_block_start_id - LOCAL_WINDOW_SIZE, 0)
        local_window_start_block = local_window_start_id // BLOCK_SIZE_N
    else:
        local_window_start_block = num_total_blocks

    non_global_window_start_block = tl.maximum(num_global_window_blocks, local_window_start_block)

    return num_global_window_blocks, non_global_window_start_block, num_total_blocks

@triton.jit
def _sdpa_acc_fwd_1xN(
    acc_ptr,
    l_i,
    m_i,
    q,  # Accumulator, local l, local m, query vector
    K_block_ptr,
    V_block_ptr,  # Key and value block pointers for current stage
    mask,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    fp8_v: tl.constexpr,
):
    if mask is False:
        return acc_ptr, l_i, m_i
    # Decode is 1 x N attention; tl.dot TC path needs M,N,K >= 16 on typical Triton builds, so use fused mul-add.
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    qk = tl.sum((q[None, :] * k).to(tl.float32), axis=1)

    qk = qk * qk_scale
    if mask is not None and mask is not True:
        qk = tl.where(mask, qk, float("-inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 0))
    qk = qk - m_ij

    p = tl.math.exp(qk)

    p_cast = p.to(k.dtype)

    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    l_ij = tl.sum(p, axis=0)
    alpha = tl.math.exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc_ptr = acc_ptr * alpha
    acc_ptr += tl.sum((p_cast[:, None] * v).to(tl.float32), axis=0)

    m_i = m_ij
    return acc_ptr, l_i, m_i


@libentry()
@triton.jit
def _swa_infer_token_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    bsz,
    cu_seqlens_q_ptr,
    cu_seqlens_kv_ptr,
    softmax_scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    IS_CAUSAL: tl.constexpr,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_SIZE_D, "HEAD_DIM should be <= BLOCK_SIZE_D")
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    cu_q_tasks = 0
    for b_id in range(bsz):
        q_start = tl.load(cu_seqlens_q_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_seqlens_q_ptr + b_id + 1).to(tl.int32)
        kv_start = tl.load(cu_seqlens_kv_ptr + b_id).to(tl.int32)
        kv_end = tl.load(cu_seqlens_kv_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end - q_start
        kv_seq_len = kv_end - kv_start
        kv_computed_len = kv_seq_len - q_seq_len

        num_tasks = q_seq_len * NUM_Q_HEADS
        for q_task_id in range(pid, num_tasks, n_progs):
            q_head_id = q_task_id % NUM_Q_HEADS
            q_token_id = q_task_id // NUM_Q_HEADS
            if GQA_INTERLEAVE:
                kv_head_id = q_head_id % NUM_KV_HEADS
            else:
                kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

            q_abs = q_token_id + kv_computed_len
            offs_d = tl.arange(0, BLOCK_SIZE_D)
            q_ptrs = q_ptr + (q_start + q_token_id) * stride_qt + q_head_id * stride_qh + offs_d * stride_qd
            q = tl.load(q_ptrs, mask=offs_d < HEAD_DIM, other=0.0)

            m_i = -float("inf")
            l_i = 0.0
            acc = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

            num_global_window_blocks, non_global_window_start_block, num_total_blocks = _swa_split_blocks(
                q_abs,
                1,
                kv_seq_len,
                BLOCK_SIZE_N,
                IS_CAUSAL,
                GLOBAL_WINDOW,
                LOCAL_WINDOW,
            )

            for kv_block_id in range(num_global_window_blocks):
                kv_block_start = kv_block_id * BLOCK_SIZE_N
                kv_block_end = min(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
                kv_block_len = kv_block_end - kv_block_start
                kv_pos = kv_block_start + tl.arange(0, BLOCK_SIZE_N)
                kv_mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len

                mask = kv_mask
                if IS_CAUSAL:
                    global_mask = kv_pos < GLOBAL_WINDOW
                    if LOCAL_WINDOW is not None:
                        local_mask = (kv_pos + LOCAL_WINDOW) >= q_abs
                        global_mask = global_mask | local_mask
                    causal_mask = kv_pos <= q_abs
                    mask = kv_mask & global_mask & causal_mask

                k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + kv_start * stride_kt + kv_head_id * stride_kh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )
                v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + kv_start * stride_vt + kv_head_id * stride_vh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )

                acc, l_i, m_i = _sdpa_acc_fwd_1xN(
                    acc,
                    l_i,
                    m_i,
                    q,
                    k_block_ptr,
                    v_block_ptr,
                    mask,
                    softmax_scale,
                    HEAD_DIM,
                    BLOCK_SIZE_D,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_D,
                    v_ptr.dtype.element_ty == tl.float8e5,
                )

            for kv_block_id in range(non_global_window_start_block, num_total_blocks):
                kv_block_start = kv_block_id * BLOCK_SIZE_N
                kv_block_end = min(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
                kv_block_len = kv_block_end - kv_block_start
                kv_pos = kv_block_start + tl.arange(0, BLOCK_SIZE_N)
                kv_mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len

                mask = kv_mask
                if IS_CAUSAL:
                    causal_mask = kv_pos <= q_abs
                    if LOCAL_WINDOW is not None:
                        local_mask = (kv_pos + LOCAL_WINDOW) >= q_abs
                        causal_mask = causal_mask & local_mask
                    mask = kv_mask & causal_mask

                k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + kv_start * stride_kt + kv_head_id * stride_kh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )
                v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + kv_start * stride_vt + kv_head_id * stride_vh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )

                acc, l_i, m_i = _sdpa_acc_fwd_1xN(
                    acc,
                    l_i,
                    m_i,
                    q,
                    k_block_ptr,
                    v_block_ptr,
                    mask,
                    softmax_scale,
                    HEAD_DIM,
                    BLOCK_SIZE_D,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_D,
                    v_ptr.dtype.element_ty == tl.float8e5,
                )

            out = acc / l_i
            out_ptrs = o_ptr + (q_start + q_token_id) * stride_ot + q_head_id * stride_oh + offs_d * stride_od
            tl.store(out_ptrs, out.to(o_ptr.dtype.element_ty), mask=offs_d < HEAD_DIM)

        pid = (pid - num_tasks % n_progs + n_progs) % n_progs


@libentry()
@triton.jit
def _swa_paged_prefill_token_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    bsz,
    cu_seqlens_q_ptr,
    kv_lens_ptr,
    block_tables_ptr,
    softmax_scale,
    stride_qt,
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
    stride_ot,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    IS_CAUSAL: tl.constexpr,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_SIZE_D, "HEAD_DIM should be <= BLOCK_SIZE_D")
    tl.static_assert(
        PAGE_SIZE % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must divide PAGE_SIZE for paged KV tiling"
    )
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    for b_id in range(bsz):
        q_start = tl.load(cu_seqlens_q_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_seqlens_q_ptr + b_id + 1).to(tl.int32)
        kv_seq_len = tl.load(kv_lens_ptr + b_id).to(tl.int32)
        q_seq_len = q_end - q_start
        kv_computed_len = kv_seq_len - q_seq_len

        num_tasks = q_seq_len * NUM_Q_HEADS
        for q_task_id in range(pid, num_tasks, n_progs):
            q_head_id = q_task_id % NUM_Q_HEADS
            q_token_id = q_task_id // NUM_Q_HEADS
            if GQA_INTERLEAVE:
                kv_head_id = q_head_id % NUM_KV_HEADS
            else:
                kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

            q_abs = q_token_id + kv_computed_len
            offs_d = tl.arange(0, BLOCK_SIZE_D)
            q_ptrs = q_ptr + (q_start + q_token_id) * stride_qt + q_head_id * stride_qh + offs_d * stride_qd
            q = tl.load(q_ptrs, mask=offs_d < HEAD_DIM, other=0.0)

            m_i = -float("inf")
            l_i = 0.0
            acc = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

            num_global_window_blocks, non_global_window_start_block, num_total_blocks = _swa_split_blocks(
                q_abs,
                1,
                kv_seq_len,
                BLOCK_SIZE_N,
                IS_CAUSAL,
                GLOBAL_WINDOW,
                LOCAL_WINDOW,
            )

            for kv_block_id in range(num_global_window_blocks):
                kv_block_start = kv_block_id * BLOCK_SIZE_N
                kv_block_end = min(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
                kv_block_len = kv_block_end - kv_block_start
                logical_page_id = kv_block_start // PAGE_SIZE
                kv_block_start_in_page = kv_block_start % PAGE_SIZE
                kv_pos = kv_block_start + tl.arange(0, BLOCK_SIZE_N)
                kv_mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len

                mask = kv_mask
                if IS_CAUSAL:
                    global_mask = kv_pos < GLOBAL_WINDOW
                    if LOCAL_WINDOW is not None:
                        local_mask = (kv_pos + LOCAL_WINDOW) >= q_abs
                        global_mask = global_mask | local_mask
                    causal_mask = kv_pos <= q_abs
                    mask = kv_mask & global_mask & causal_mask

                physical_page_id = tl.load(
                    block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
                )
                k_block_ptr = tl.make_block_ptr(
                    base=k_cache_ptr
                    + physical_page_id * stride_k_block
                    + kv_head_id * stride_k_head
                    + kv_block_start_in_page * stride_k_blksz,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_k_blksz, stride_k_dim),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )
                v_block_ptr = tl.make_block_ptr(
                    base=v_cache_ptr
                    + physical_page_id * stride_v_block
                    + kv_head_id * stride_v_head
                    + kv_block_start_in_page * stride_v_blksz,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_v_blksz, stride_v_dim),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )

                acc, l_i, m_i = _sdpa_acc_fwd_1xN(
                    acc,
                    l_i,
                    m_i,
                    q,
                    k_block_ptr,
                    v_block_ptr,
                    mask,
                    softmax_scale,
                    HEAD_DIM,
                    BLOCK_SIZE_D,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_D,
                    v_cache_ptr.dtype.element_ty == tl.float8e5,
                )

            for kv_block_id in range(non_global_window_start_block, num_total_blocks):
                kv_block_start = kv_block_id * BLOCK_SIZE_N
                kv_block_end = min(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
                kv_block_len = kv_block_end - kv_block_start
                logical_page_id = kv_block_start // PAGE_SIZE
                kv_block_start_in_page = kv_block_start % PAGE_SIZE
                kv_pos = kv_block_start + tl.arange(0, BLOCK_SIZE_N)
                kv_mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len

                mask = kv_mask
                if IS_CAUSAL:
                    causal_mask = kv_pos <= q_abs
                    if LOCAL_WINDOW is not None:
                        local_mask = (kv_pos + LOCAL_WINDOW) >= q_abs
                        causal_mask = causal_mask & local_mask
                    mask = kv_mask & causal_mask

                physical_page_id = tl.load(
                    block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
                )
                k_block_ptr = tl.make_block_ptr(
                    base=k_cache_ptr
                    + physical_page_id * stride_k_block
                    + kv_head_id * stride_k_head
                    + kv_block_start_in_page * stride_k_blksz,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_k_blksz, stride_k_dim),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )
                v_block_ptr = tl.make_block_ptr(
                    base=v_cache_ptr
                    + physical_page_id * stride_v_block
                    + kv_head_id * stride_v_head
                    + kv_block_start_in_page * stride_v_blksz,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_v_blksz, stride_v_dim),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )

                acc, l_i, m_i = _sdpa_acc_fwd_1xN(
                    acc,
                    l_i,
                    m_i,
                    q,
                    k_block_ptr,
                    v_block_ptr,
                    mask,
                    softmax_scale,
                    HEAD_DIM,
                    BLOCK_SIZE_D,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_D,
                    v_cache_ptr.dtype.element_ty == tl.float8e5,
                )

            out = acc / l_i
            out_ptrs = o_ptr + (q_start + q_token_id) * stride_ot + q_head_id * stride_oh + offs_d * stride_od
            tl.store(out_ptrs, out.to(o_ptr.dtype.element_ty), mask=offs_d < HEAD_DIM)

        pid = (pid - num_tasks % n_progs + n_progs) % n_progs

@libentry()
@triton.jit
def _paged_decode_kernel(
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
    softmax_scale,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    OUT_T: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_SIZE_D, "HEAD_DIM should be <= BLOCK_SIZE_D")
    tl.static_assert(
        PAGE_SIZE % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must divide PAGE_SIZE for paged decode tiling"
    )
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    num_tasks = BATCH_SIZE * NUM_Q_HEADS

    for q_task_id in tl.range(pid, num_tasks, n_progs):
        q_head_id = q_task_id % NUM_Q_HEADS
        b_id = q_task_id // NUM_Q_HEADS
        if GQA_INTERLEAVE:
            kv_head_id = q_head_id % NUM_KV_HEADS
        else:
            kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

        kv_seq_len = tl.load(seqlens_ptr + b_id)


        offs_d = tl.arange(0, BLOCK_SIZE_D)
        q_ptrs = q_ptr + b_id * stride_qb + q_head_id * stride_qh + offs_d * stride_qd
        q = tl.load(q_ptrs, mask = offs_d < HEAD_DIM, other = 0.0)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

        num_global_window_blocks, non_global_window_start_block, num_total_blocks = _swa_split_blocks(
            kv_seq_len - 1,
            1,
            kv_seq_len,
            BLOCK_SIZE_N,
            True,
            GLOBAL_WINDOW,
            LOCAL_WINDOW,
        )

        for kv_block_id in tl.range(0, num_global_window_blocks):
            kv_block_start = kv_block_id * BLOCK_SIZE_N
            kv_block_end = tl.minimum(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
            kv_block_len = kv_block_end - kv_block_start
            logical_page_id = kv_block_start // PAGE_SIZE
            kv_block_start_in_page = kv_block_start % PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
            )
            k_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr
                + physical_page_id * stride_k_block
                + kv_head_id * stride_k_head
                + kv_block_start_in_page * stride_k_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_k_blksz, stride_k_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr
                + physical_page_id * stride_v_block
                + kv_head_id * stride_v_head
                + kv_block_start_in_page * stride_v_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )
            gw_mask = (kv_block_start + tl.arange(0, BLOCK_SIZE_N)) < GLOBAL_WINDOW
            if LOCAL_WINDOW is not None:
                sw_mask = (kv_block_start + tl.arange(0, BLOCK_SIZE_N) + LOCAL_WINDOW) >= (kv_seq_len - 1)
                gw_mask = gw_mask | sw_mask
            kv_mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len
            mask = gw_mask & kv_mask

            acc, l_i, m_i = _sdpa_acc_fwd_1xN(
                acc,
                l_i,
                m_i,
                q,
                k_block_ptr,
                v_block_ptr,
                mask,
                softmax_scale,
                HEAD_DIM,
                BLOCK_SIZE_D,
                BLOCK_SIZE_N,
                BLOCK_SIZE_D,
                v_cache_ptr.dtype.element_ty == tl.float8e5,
            )

        for kv_block_id in tl.range(non_global_window_start_block, num_total_blocks):
            kv_block_start = kv_block_id * BLOCK_SIZE_N
            kv_block_end = tl.minimum(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
            kv_block_len = kv_block_end - kv_block_start
            logical_page_id = kv_block_start // PAGE_SIZE
            kv_block_start_in_page = kv_block_start % PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
            )
            k_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr
                + physical_page_id * stride_k_block
                + kv_head_id * stride_k_head
                + kv_block_start_in_page * stride_k_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_k_blksz, stride_k_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr
                + physical_page_id * stride_v_block
                + kv_head_id * stride_v_head
                + kv_block_start_in_page * stride_v_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            kv_mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len
            if LOCAL_WINDOW is not None:
                sw_mask = (kv_block_start + tl.arange(0, BLOCK_SIZE_N) + LOCAL_WINDOW) >= (kv_seq_len - 1)
                mask = kv_mask & sw_mask
            else:
                mask = kv_mask

            acc, l_i, m_i = _sdpa_acc_fwd_1xN(
                acc,
                l_i,
                m_i,
                q,
                k_block_ptr,
                v_block_ptr,
                mask,
                softmax_scale,
                HEAD_DIM,
                BLOCK_SIZE_D,
                BLOCK_SIZE_N,
                BLOCK_SIZE_D,
                v_cache_ptr.dtype.element_ty == tl.float8e5,
            )

        # Match torch reference: padded rows (kv_seq_len == 0) keep zero output; avoid acc / 0.
        if kv_seq_len > 0:
            acc = acc / l_i

        o_ptrs = o_ptr + b_id * stride_ob + q_head_id * stride_oh + offs_d * stride_od
        tl.store(o_ptrs, acc.to(OUT_T), mask=offs_d < HEAD_DIM)


def _paged_decode_launch_config(head_dim: int, page_size: int) -> int:
    if head_dim <= 64:
        num_warps = 4
    else:
        num_warps = 8
    if page_size >= 128 and head_dim >= 128:
        num_warps = max(num_warps, 8)
    return num_warps


def swa_paged_decode_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    gqa_interleave: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    batch_size, num_q_heads, head_dim = q.shape
    num_total_blocks, num_kv_heads, block_size, head_dim_cache = key_cache.shape

    block_size_n = min(128, triton.next_power_of_2(block_size))
    if block_size % block_size_n != 0:
        raise ValueError(
            f"KV block_size ({block_size}) must be divisible by decode tile size ({block_size_n})."
        )
    max_num_blocks_per_seq = block_tables.shape[1]

    assert head_dim == head_dim_cache
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    o = torch.empty_like(q, memory_format=torch.contiguous_format)

    # One program per (batch, head): maximizes parallelism on NVIDIA (vs. a tiny grid + device loop).
    grid = (batch_size * num_q_heads,)
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)

    if q.dtype == torch.float16:
        out_t = tl.float16
    elif q.dtype == torch.bfloat16:
        out_t = tl.bfloat16
    else:
        out_t = tl.float32

    bt = block_tables if block_tables.dtype == torch.int32 else block_tables.to(torch.int32)
    num_warps = _paged_decode_launch_config(head_dim, block_size_n)

    _paged_decode_kernel[grid](
        q,
        key_cache,
        value_cache,
        o,
        seqlens,
        bt,
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
        bt.stride(0),
        bt.stride(1),
        softmax_scale,
        global_window_size,
        local_window_size,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        block_size,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_N=block_size_n,
        OUT_T=out_t,
        num_warps=num_warps,
    )
    return o
