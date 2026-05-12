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


def _swa_infer_autotune_configs() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=1),
    ]


def _swa_paged_prefill_autotune_configs() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 128}, num_warps=8, num_stages=1),
    ]


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
def _swa_acc_fwd_nomask_mxn(
    acc_ptr,
    l_i,
    m_i,
    q,
    k_block_ptr,
    v_block_ptr,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
    qk = tl.dot(q, tl.trans(k))
    qk = qk * qk_scale

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


@triton.autotune(
    configs=_swa_infer_autotune_configs(),
    key=["HEAD_DIM", "GLOBAL_WINDOW", "LOCAL_WINDOW", "NUM_Q_HEADS", "NUM_KV_HEADS"],
)
@libentry()
@triton.jit
def _swa_infer_kernel(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    bsz,
    cu_q_lens_ptr,
    cu_total_seq_lens_ptr,
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
    has_global_window = GLOBAL_WINDOW is not None
    has_local_window = LOCAL_WINDOW is not None

    cu_q_chunks = 0
    q_offsets = tl.arange(0, BLOCK_M)
    kv_offsets = tl.arange(0, BLOCK_N)

    for b_id in range(bsz):
        q_start = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        kv_start = tl.load(cu_total_seq_lens_ptr + b_id).to(tl.int32)
        kv_end = tl.load(cu_total_seq_lens_ptr + b_id + 1).to(tl.int32)
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
                    if has_global_window and has_local_window:
                        local_mask = q_abs[:, None] <= (kv_abs + LOCAL_WINDOW)[None, :]
                        global_mask = kv_abs[None, :] < GLOBAL_WINDOW
                        mask = mask & causal_mask & (global_mask | local_mask)
                    elif has_global_window:
                        mask = mask & causal_mask & (kv_abs[None, :] < GLOBAL_WINDOW)
                    elif has_local_window:
                        local_mask = q_abs[:, None] <= (kv_abs + LOCAL_WINDOW)[None, :]
                        mask = mask & causal_mask & local_mask
                    else:
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

            can_use_nomask_local = IS_CAUSAL and q_block_len == BLOCK_M
            full_local_start_block = non_global_window_start_block
            full_local_end_block = non_global_window_start_block - 1
            if can_use_nomask_local:
                q_abs_start = q_block_start + kv_computed_len
                q_abs_end = q_abs_start + BLOCK_M - 1
                last_causal_full_block = (q_abs_start - (BLOCK_N - 1)) // BLOCK_N
                first_local_full_block = non_global_window_start_block
                if has_local_window:
                    first_local_full_block = tl.maximum(
                        non_global_window_start_block,
                        tl.cdiv(tl.maximum(q_abs_end - LOCAL_WINDOW, 0), BLOCK_N),
                    )
                full_local_start_block = first_local_full_block
                full_local_end_block = tl.minimum(num_total_blocks - 1, last_causal_full_block)
                can_use_nomask_local = full_local_start_block <= full_local_end_block

            if can_use_nomask_local:
                for kv_block_id in range(non_global_window_start_block, full_local_start_block):
                    kv_block_start = kv_block_id * BLOCK_N
                    kv_abs = kv_block_start + kv_offsets
                    kv_valid = kv_abs < kv_seq_len
                    mask = q_valid[:, None] & kv_valid[None, :]
                    causal_mask = q_abs[:, None] >= kv_abs[None, :]
                    if has_local_window:
                        local_mask = q_abs[:, None] <= (kv_abs + LOCAL_WINDOW)[None, :]
                        mask = mask & causal_mask & local_mask
                    else:
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

                for kv_block_id in range(full_local_start_block, full_local_end_block + 1):
                    kv_block_start = kv_block_id * BLOCK_N
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
                    acc, l_i, m_i = _swa_acc_fwd_nomask_mxn(
                        acc,
                        l_i,
                        m_i,
                        q_block,
                        k_block_ptr,
                        v_block_ptr,
                        scale,
                        HEAD_DIM,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_D,
                    )

                for kv_block_id in range(full_local_end_block + 1, num_total_blocks):
                    kv_block_start = kv_block_id * BLOCK_N
                    kv_abs = kv_block_start + kv_offsets
                    kv_valid = kv_abs < kv_seq_len
                    mask = q_valid[:, None] & kv_valid[None, :]
                    causal_mask = q_abs[:, None] >= kv_abs[None, :]
                    if has_local_window:
                        local_mask = q_abs[:, None] <= (kv_abs + LOCAL_WINDOW)[None, :]
                        mask = mask & causal_mask & local_mask
                    else:
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
            else:
                for kv_block_id in range(non_global_window_start_block, num_total_blocks):
                    kv_block_start = kv_block_id * BLOCK_N
                    kv_abs = kv_block_start + kv_offsets
                    kv_valid = kv_abs < kv_seq_len
                    mask = q_valid[:, None] & kv_valid[None, :]
                    if IS_CAUSAL:
                        causal_mask = q_abs[:, None] >= kv_abs[None, :]
                        if has_local_window:
                            local_mask = q_abs[:, None] <= (kv_abs + LOCAL_WINDOW)[None, :]
                            mask = mask & causal_mask & local_mask
                        else:
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

            l_i_safe = tl.where(l_i > 0, l_i, 1.0)
            out_block = tl.where(l_i[:, None] > 0, acc / l_i_safe[:, None], 0.0)
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


@triton.autotune(
    configs=_swa_paged_prefill_autotune_configs(),
    key=["HEAD_DIM", "GLOBAL_WINDOW", "LOCAL_WINDOW", "NUM_Q_HEADS", "NUM_KV_HEADS"],
)
@libentry()
@triton.jit
def _swa_paged_prefill_kernel(
    o_ptr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    bsz,
    cu_q_lens_ptr,
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
    has_global_window = GLOBAL_WINDOW is not None
    has_local_window = LOCAL_WINDOW is not None

    cu_q_chunks = 0
    q_offsets = tl.arange(0, BLOCK_M)
    kv_offsets = tl.arange(0, BLOCK_N)

    for b_id in range(bsz):
        q_start = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
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
                kv_abs = kv_block_start + kv_offsets
                kv_valid = kv_abs < kv_seq_len
                mask = q_valid[:, None] & kv_valid[None, :]
                if IS_CAUSAL:
                    causal_mask = q_abs[:, None] >= kv_abs[None, :]
                    if has_global_window and has_local_window:
                        local_mask = q_abs[:, None] <= (kv_abs + LOCAL_WINDOW)[None, :]
                        global_mask = kv_abs[None, :] < GLOBAL_WINDOW
                        mask = mask & causal_mask & (global_mask | local_mask)
                    elif has_global_window:
                        mask = mask & causal_mask & (kv_abs[None, :] < GLOBAL_WINDOW)
                    elif has_local_window:
                        local_mask = q_abs[:, None] <= (kv_abs + LOCAL_WINDOW)[None, :]
                        mask = mask & causal_mask & local_mask
                    else:
                        mask = mask & causal_mask

                physical_page_id = tl.load(
                    block_table_ptr + b_id * stride_block_table_b + logical_page_id * stride_block_table_p
                )
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

            can_use_nomask_local = IS_CAUSAL and q_block_len == BLOCK_M
            full_local_start_block = non_global_window_start_block
            full_local_end_block = non_global_window_start_block - 1
            if can_use_nomask_local:
                q_abs_start = q_block_start + kv_computed_len
                q_abs_end = q_abs_start + BLOCK_M - 1
                last_causal_full_block = (q_abs_start - (BLOCK_N - 1)) // BLOCK_N
                first_local_full_block = non_global_window_start_block
                if has_local_window:
                    first_local_full_block = tl.maximum(
                        non_global_window_start_block,
                        tl.cdiv(tl.maximum(q_abs_end - LOCAL_WINDOW, 0), BLOCK_N),
                    )
                full_local_start_block = first_local_full_block
                full_local_end_block = tl.minimum(num_total_blocks - 1, last_causal_full_block)
                can_use_nomask_local = full_local_start_block <= full_local_end_block

            if can_use_nomask_local:
                for kv_block_id in range(non_global_window_start_block, full_local_start_block):
                    kv_block_start = kv_block_id * BLOCK_N
                    kv_block_end = min(kv_block_start + BLOCK_N, kv_seq_len)
                    kv_block_len = kv_block_end - kv_block_start
                    logical_page_id = kv_block_start // PAGE_SIZE
                    kv_block_start_in_page = kv_block_start % PAGE_SIZE
                    kv_abs = kv_block_start + kv_offsets
                    kv_valid = kv_abs < kv_seq_len
                    mask = q_valid[:, None] & kv_valid[None, :]
                    causal_mask = q_abs[:, None] >= kv_abs[None, :]
                    if has_local_window:
                        local_mask = q_abs[:, None] <= (kv_abs + LOCAL_WINDOW)[None, :]
                        mask = mask & causal_mask & local_mask
                    else:
                        mask = mask & causal_mask

                    physical_page_id = tl.load(
                        block_table_ptr + b_id * stride_block_table_b + logical_page_id * stride_block_table_p
                    )
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

                for kv_block_id in range(full_local_start_block, full_local_end_block + 1):
                    kv_block_start = kv_block_id * BLOCK_N
                    kv_block_end = min(kv_block_start + BLOCK_N, kv_seq_len)
                    kv_block_len = kv_block_end - kv_block_start
                    logical_page_id = kv_block_start // PAGE_SIZE
                    kv_block_start_in_page = kv_block_start % PAGE_SIZE
                    physical_page_id = tl.load(
                        block_table_ptr + b_id * stride_block_table_b + logical_page_id * stride_block_table_p
                    )
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
                    acc, l_i, m_i = _swa_acc_fwd_nomask_mxn(
                        acc,
                        l_i,
                        m_i,
                        q_block,
                        k_block_ptr,
                        v_block_ptr,
                        scale,
                        HEAD_DIM,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_D,
                    )

                for kv_block_id in range(full_local_end_block + 1, num_total_blocks):
                    kv_block_start = kv_block_id * BLOCK_N
                    kv_block_end = min(kv_block_start + BLOCK_N, kv_seq_len)
                    kv_block_len = kv_block_end - kv_block_start
                    logical_page_id = kv_block_start // PAGE_SIZE
                    kv_block_start_in_page = kv_block_start % PAGE_SIZE
                    kv_abs = kv_block_start + kv_offsets
                    kv_valid = kv_abs < kv_seq_len
                    mask = q_valid[:, None] & kv_valid[None, :]
                    causal_mask = q_abs[:, None] >= kv_abs[None, :]
                    if has_local_window:
                        local_mask = q_abs[:, None] <= (kv_abs + LOCAL_WINDOW)[None, :]
                        mask = mask & causal_mask & local_mask
                    else:
                        mask = mask & causal_mask

                    physical_page_id = tl.load(
                        block_table_ptr + b_id * stride_block_table_b + logical_page_id * stride_block_table_p
                    )
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
            else:
                for kv_block_id in range(non_global_window_start_block, num_total_blocks):
                    kv_block_start = kv_block_id * BLOCK_N
                    kv_block_end = min(kv_block_start + BLOCK_N, kv_seq_len)
                    kv_block_len = kv_block_end - kv_block_start
                    logical_page_id = kv_block_start // PAGE_SIZE
                    kv_block_start_in_page = kv_block_start % PAGE_SIZE
                    kv_abs = kv_block_start + kv_offsets
                    kv_valid = kv_abs < kv_seq_len
                    mask = q_valid[:, None] & kv_valid[None, :]
                    if IS_CAUSAL:
                        causal_mask = q_abs[:, None] >= kv_abs[None, :]
                        if has_local_window:
                            local_mask = q_abs[:, None] <= (kv_abs + LOCAL_WINDOW)[None, :]
                            mask = mask & causal_mask & local_mask
                        else:
                            mask = mask & causal_mask

                    physical_page_id = tl.load(
                        block_table_ptr + b_id * stride_block_table_b + logical_page_id * stride_block_table_p
                    )
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

            l_i_safe = tl.where(l_i > 0, l_i, 1.0)
            out_block = tl.where(l_i[:, None] > 0, acc / l_i_safe[:, None], 0.0)
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
    cu_q_lens: torch.Tensor,
    cu_total_seq_lens: torch.Tensor,
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
    batch_size = cu_q_lens.shape[0] - 1
    block_d = triton.next_power_of_2(head_dim)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]

    def grid(meta):
        block_m = meta["BLOCK_M"]
        total_q_chunks = int(torch.div(q_lens + block_m - 1, block_m, rounding_mode="floor").sum().item())
        return (ilu_grid_dim_from_row_tasks(total_q_chunks * num_q_heads),)

    _swa_infer_kernel[grid](
        outputs,
        q,
        k,
        v,
        batch_size,
        cu_q_lens,
        cu_total_seq_lens,
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
        BLOCK_D=block_d,
    )
    return outputs


@triton.jit
def _swa_paged_prefill_quant_kernel(
    Q,
    K_cache,
    V_cache,
    Out,
    K_qscale,
    V_qscale,
    cu_seqlens_q_ptr,
    seqlens_kv_ptr,
    block_tables_ptr,
    stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_bt_batch, stride_bt_block,
    stride_ks_h, stride_ks_d,
    stride_vs_h, stride_vs_d,
    sm_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    LOCAL_WINDOW_SIZE: tl.constexpr,
    GLOBAL_WINDOW_SIZE: tl.constexpr,
):
    """Paged prefill SWA attention with int8 KV cache (per-token scalar loop).

    Mirrors `_paged_prefill_quant_kernel` (commit 1130f65) but adds the SWA
    local / global window mask. Each program handles one
    (q_token_in_seq, q_head, batch) triple and walks all KV tokens linearly,
    accumulating online-softmax stats in fp32. We deliberately avoid tl.dot
    because the ILU triton compiler generates invalid bitcasts in the
    SharedToDotOperand layout pass when tl.dot's data provenance is an int8
    load.
    """
    q_token_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    b_id = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q_ptr + b_id).to(tl.int32)
    q_seq_len = tl.load(cu_seqlens_q_ptr + b_id + 1).to(tl.int32) - q_start
    kv_seq_len = tl.load(seqlens_kv_ptr + b_id).to(tl.int32)

    if q_token_id >= q_seq_len:
        return

    if GQA_INTERLEAVE:
        kv_head_id = q_head_id % NUM_KV_HEADS
    else:
        kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM

    q_vec = tl.load(
        Q + (q_start + q_token_id) * stride_qt + q_head_id * stride_qh + offs_d * stride_qd,
        mask=d_mask, other=0.0,
    ).to(tl.float32)

    k_scale_vec = tl.load(
        K_qscale + q_head_id * stride_ks_h + offs_d * stride_ks_d,
        mask=d_mask, other=0.0,
    )
    v_scale_vec = tl.load(
        V_qscale + q_head_id * stride_vs_h + offs_d * stride_vs_d,
        mask=d_mask, other=0.0,
    )

    # `kv_cache_len` = number of already-cached KV tokens that come BEFORE this
    # batch's prefill chunk. The j-th KV token corresponds to absolute
    # query position (q_token_id + kv_cache_len) for causal alignment.
    kv_cache_len = kv_seq_len - q_seq_len
    abs_q_pos = q_token_id + kv_cache_len

    if IS_CAUSAL:
        kv_loop_end = tl.minimum(kv_seq_len, abs_q_pos + 1)
    else:
        kv_loop_end = kv_seq_len

    m_max = tl.full((), -float("inf"), tl.float32)
    l_sum = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    num_kv_pages = tl.cdiv(kv_loop_end, PAGE_SIZE)
    for page_idx in tl.range(0, num_kv_pages):
        physical_block = tl.load(
            block_tables_ptr + b_id * stride_bt_batch + page_idx * stride_bt_block
        )
        kv_page_start = page_idx * PAGE_SIZE
        kv_page_end = tl.minimum(kv_page_start + PAGE_SIZE, kv_loop_end)

        for j in tl.range(kv_page_start, kv_page_end):
            offset_in_page = j - kv_page_start

            # Causal + SWA window. Match _generate_window_mask exactly:
            #   causal:  abs_q_pos >= j
            #   local:   abs_q_pos <= j + local_window_size  (when set)
            #   global:  j < global_window_size              (when set)
            #   final:   causal AND (local OR global) when either is set,
            #            else just causal.
            if IS_CAUSAL:
                causal_ok = j <= abs_q_pos
                if (LOCAL_WINDOW_SIZE is None) and (GLOBAL_WINDOW_SIZE is None):
                    allowed = causal_ok
                else:
                    if LOCAL_WINDOW_SIZE is not None:
                        local_ok = abs_q_pos <= j + LOCAL_WINDOW_SIZE
                    else:
                        local_ok = False
                    if GLOBAL_WINDOW_SIZE is not None:
                        global_ok = j < GLOBAL_WINDOW_SIZE
                    else:
                        global_ok = False
                    allowed = causal_ok and (local_ok or global_ok)
            else:
                allowed = True

            k_vec = tl.load(
                K_cache + physical_block * stride_kb + kv_head_id * stride_kh
                + offset_in_page * stride_kn + offs_d * stride_kd,
                mask=d_mask, other=0,
            ).to(tl.float32)
            k_vec = k_vec * k_scale_vec

            s = tl.sum(q_vec * k_vec) * sm_scale
            s = tl.where(allowed, s, -float("inf"))

            m_new = tl.maximum(m_max, s)
            row_is_all_masked = m_new == -float("inf")
            alpha = tl.math.exp(tl.where(row_is_all_masked, 0.0, m_max - m_new))
            alpha = tl.where(row_is_all_masked, 0.0, alpha)
            p = tl.math.exp(tl.where(row_is_all_masked, 0.0, s - m_new))
            p = tl.where(allowed, p, 0.0)
            p = tl.where(row_is_all_masked, 0.0, p)

            v_vec = tl.load(
                V_cache + physical_block * stride_vb + kv_head_id * stride_vh
                + offset_in_page * stride_vn + offs_d * stride_vd,
                mask=d_mask, other=0,
            ).to(tl.float32)
            v_vec = v_vec * v_scale_vec

            acc = acc * alpha + p * v_vec
            l_sum = l_sum * alpha + p
            m_max = m_new

    l_sum_safe = tl.where(l_sum > 0, l_sum, 1.0)
    out_vec = tl.where(l_sum > 0, acc / l_sum_safe, 0.0)

    tl.store(
        Out + (q_start + q_token_id) * stride_ot + q_head_id * stride_oh + offs_d * stride_od,
        out_vec.to(Out.dtype.element_ty),
        mask=d_mask,
    )


def swa_paged_prefill_quant_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    k_qscale: torch.Tensor,
    value_cache: torch.Tensor,
    v_qscale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqlens_kv: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
    max_seqlen_q: Optional[int] = None,
) -> torch.Tensor:
    """Paged prefill SWA attention with int8 KV cache and per-channel scales.

    Args:
        q:               (T, Hq, D) bf16/fp16 query tokens.
        key_cache:       (N_blocks, Hkv, page_size, D) int8 key cache.
        k_qscale:        (Hq, D) float32 per-channel key scale, **already
                         expanded to query-head count by the caller**.
        value_cache:     (N_blocks, Hkv, page_size, D) int8 value cache.
        v_qscale:        (Hq, D) float32 per-channel value scale, expanded.
        cu_seqlens_q:    (B+1,) int32 cumulative query lengths.
        seqlens_kv:      (B,) int32 KV lengths (or None -> use query lengths).
        block_tables:    (B, max_num_blocks) int32 block mapping.
        is_causal:       Whether to apply causal + window mask.
        local_window_size, global_window_size:
                         SWA window args; only effective when ``is_causal``.
        softmax_scale:   Attention scale, default 1/sqrt(D).
        gqa_interleave:  Whether to use ABAB GQA layout.
        max_seqlen_q:    Optional max query length hint to size grid_x.
    """
    total_q_tokens, num_q_heads, head_dim = q.shape
    _, num_kv_heads, page_size, _ = key_cache.shape
    batch_size = cu_seqlens_q.shape[0] - 1

    sm_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)

    if seqlens_kv is None:
        seqlens_kv = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)
    else:
        seqlens_kv = seqlens_kv.to(torch.int32)

    out = torch.empty(total_q_tokens, num_q_heads, head_dim, device=q.device, dtype=q.dtype)
    block_tables_i32 = block_tables.to(torch.int32)

    BLOCK_D = triton.next_power_of_2(head_dim)

    if max_seqlen_q is not None:
        max_q_len = int(max_seqlen_q)
    else:
        q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        max_q_len = int(q_lens.max().item()) if q_lens.numel() > 0 else 0

    if max_q_len == 0:
        return out

    grid = (max_q_len, num_q_heads, batch_size)

    _swa_paged_prefill_quant_kernel[grid](
        q,
        key_cache,
        value_cache,
        out,
        k_qscale,
        v_qscale,
        cu_seqlens_q,
        seqlens_kv,
        block_tables_i32,
        q.stride(0), q.stride(1), q.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        block_tables_i32.stride(0), block_tables_i32.stride(1),
        k_qscale.stride(0), k_qscale.stride(1),
        v_qscale.stride(0), v_qscale.stride(1),
        float(sm_scale),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        GQA_INTERLEAVE=gqa_interleave,
        HEAD_DIM=head_dim,
        BLOCK_D=BLOCK_D,
        PAGE_SIZE=page_size,
        IS_CAUSAL=is_causal,
        LOCAL_WINDOW_SIZE=local_window_size,
        GLOBAL_WINDOW_SIZE=global_window_size,
    )

    return out


def swa_paged_prefill_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
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
    batch_size = cu_q_lens.shape[0] - 1
    if seqlens_kv is None:
        seqlens_kv = cu_q_lens[1:] - cu_q_lens[:-1]
    block_n = min(128, triton.next_power_of_2(block_size))
    if block_size % block_n != 0:
        raise ValueError(
            f"KV block_size ({block_size}) must be divisible by Triton tile size ({block_n}); "
            "use a compatible page size (e.g. power of two, multiple of 128 for large pages)."
        )
    block_d = triton.next_power_of_2(head_dim)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]

    def grid(meta):
        block_m = meta["BLOCK_M"]
        total_q_chunks = int(torch.div(q_lens + block_m - 1, block_m, rounding_mode="floor").sum().item())
        return (ilu_grid_dim_from_row_tasks(total_q_chunks * num_q_heads),)

    _swa_paged_prefill_kernel[grid](
        outputs,
        q,
        key_cache,
        value_cache,
        batch_size,
        cu_q_lens,
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
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        PAGE_SIZE=block_size,
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
        qk = tl.where(mask, qk, -1.0e20)

    m_ij = tl.maximum(m_i, tl.max(qk, 0))
    qk = qk - m_ij

    p = tl.math.exp(qk)
    if mask is not None and mask is not True:
        p = tl.where(mask, p, 0.0)

    p_cast = p.to(k.dtype)

    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    l_ij = tl.sum(p, axis=0)
    alpha = tl.math.exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc_ptr = acc_ptr * alpha
    acc_ptr += tl.sum((p_cast[:, None] * v).to(tl.float32), axis=0)

    m_i = m_ij
    return acc_ptr, l_i, m_i


@triton.jit
def _sdpa_acc_fwd_1xT(
    acc_ptr,
    l_i,
    m_i,
    q,
    k,
    v,
    mask,
    qk_scale,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    fp8_v: tl.constexpr,
):
    if mask is False:
        return acc_ptr, l_i, m_i

    qk = tl.sum((q[None, :] * k).to(tl.float32), axis=1)
    qk = qk * qk_scale
    if mask is not None and mask is not True:
        qk = tl.where(mask, qk, -1.0e20)

    m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
    qk = qk - m_ij
    p = tl.math.exp(qk)
    if mask is not None and mask is not True:
        p = tl.where(mask, p, 0.0)

    p_cast = p.to(k.dtype)
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
    cu_q_lens_ptr,
    cu_total_seq_lens_ptr,
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
    has_global_window = GLOBAL_WINDOW is not None
    has_local_window = LOCAL_WINDOW is not None

    cu_q_tasks = 0
    for b_id in range(bsz):
        q_start = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        kv_start = tl.load(cu_total_seq_lens_ptr + b_id).to(tl.int32)
        kv_end = tl.load(cu_total_seq_lens_ptr + b_id + 1).to(tl.int32)
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
                    causal_mask = kv_pos <= q_abs
                    if has_global_window and has_local_window:
                        local_mask = (kv_pos + LOCAL_WINDOW) >= q_abs
                        global_mask = kv_pos < GLOBAL_WINDOW
                        mask = kv_mask & causal_mask & (global_mask | local_mask)
                    elif has_global_window:
                        mask = kv_mask & causal_mask & (kv_pos < GLOBAL_WINDOW)
                    elif has_local_window:
                        local_mask = (kv_pos + LOCAL_WINDOW) >= q_abs
                        mask = kv_mask & causal_mask & local_mask
                    else:
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

            for kv_block_id in range(non_global_window_start_block, num_total_blocks):
                kv_block_start = kv_block_id * BLOCK_SIZE_N
                kv_block_end = min(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
                kv_block_len = kv_block_end - kv_block_start
                kv_pos = kv_block_start + tl.arange(0, BLOCK_SIZE_N)
                kv_mask = tl.arange(0, BLOCK_SIZE_N) < kv_block_len

                mask = kv_mask
                if IS_CAUSAL:
                    causal_mask = kv_pos <= q_abs
                    if has_local_window:
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

            l_i_safe = tl.where(l_i > 0, l_i, 1.0)
            out = tl.where(l_i > 0, acc / l_i_safe, 0.0)
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
    cu_q_lens_ptr,
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
    has_global_window = GLOBAL_WINDOW is not None
    has_local_window = LOCAL_WINDOW is not None

    for b_id in range(bsz):
        q_start = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
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
                    causal_mask = kv_pos <= q_abs
                    if has_global_window and has_local_window:
                        local_mask = (kv_pos + LOCAL_WINDOW) >= q_abs
                        global_mask = kv_pos < GLOBAL_WINDOW
                        mask = kv_mask & causal_mask & (global_mask | local_mask)
                    elif has_global_window:
                        mask = kv_mask & causal_mask & (kv_pos < GLOBAL_WINDOW)
                    elif has_local_window:
                        local_mask = (kv_pos + LOCAL_WINDOW) >= q_abs
                        mask = kv_mask & causal_mask & local_mask
                    else:
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
                    if has_local_window:
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

            l_i_safe = tl.where(l_i > 0, l_i, 1.0)
            out = tl.where(l_i > 0, acc / l_i_safe, 0.0)
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

        l_i_safe = tl.where(l_i > 0, l_i, 1.0)
        acc = tl.where(l_i > 0, acc / l_i_safe, 0.0)

        o_ptrs = o_ptr + b_id * stride_ob + q_head_id * stride_oh + offs_d * stride_od
        tl.store(o_ptrs, acc.to(OUT_T), mask=offs_d < HEAD_DIM)


@libentry()
@triton.jit
def _paged_decode_kernel_tiny_global(
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
    TINY_GLOBAL_N: tl.constexpr,
    OUT_T: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_SIZE_D, "HEAD_DIM should be <= BLOCK_SIZE_D")
    tl.static_assert(
        PAGE_SIZE % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must divide PAGE_SIZE for paged decode tiling"
    )
    tl.static_assert(TINY_GLOBAL_N <= BLOCK_SIZE_N, "TINY_GLOBAL_N should be <= BLOCK_SIZE_N")
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
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        offs_t = tl.arange(0, TINY_GLOBAL_N)
        q_ptrs = q_ptr + b_id * stride_qb + q_head_id * stride_qh + offs_d * stride_qd
        q = tl.load(q_ptrs, mask=offs_d < HEAD_DIM, other=0.0)

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

        if num_global_window_blocks > 0:
            block0_fully_covered_by_local = False
            if LOCAL_WINDOW is not None:
                block0_fully_covered_by_local = (BLOCK_SIZE_N + LOCAL_WINDOW) >= kv_seq_len

            physical_page_id = tl.load(block_tables_ptr + b_id * stride_bt_batch)
            if block0_fully_covered_by_local:
                kv_block_len = tl.minimum(BLOCK_SIZE_N, kv_seq_len)
                k_block_ptr = tl.make_block_ptr(
                    base=k_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_k_blksz, stride_k_dim),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )
                v_block_ptr = tl.make_block_ptr(
                    base=v_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_v_blksz, stride_v_dim),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )
                kv_mask = offs_n < kv_block_len
                gw_mask = offs_n < GLOBAL_WINDOW
                if LOCAL_WINDOW is not None:
                    sw_mask = (offs_n + LOCAL_WINDOW) >= (kv_seq_len - 1)
                    gw_mask = gw_mask | sw_mask
                mask = kv_mask & gw_mask
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
            else:
                k_ptrs = (
                    k_cache_ptr
                    + physical_page_id * stride_k_block
                    + kv_head_id * stride_k_head
                    + offs_t[:, None] * stride_k_blksz
                    + offs_d[None, :] * stride_k_dim
                )
                v_ptrs = (
                    v_cache_ptr
                    + physical_page_id * stride_v_block
                    + kv_head_id * stride_v_head
                    + offs_t[:, None] * stride_v_blksz
                    + offs_d[None, :] * stride_v_dim
                )
                tiny_valid = (offs_t < GLOBAL_WINDOW) & (offs_t < kv_seq_len)
                tiny_load_mask = tiny_valid[:, None] & (offs_d[None, :] < HEAD_DIM)
                k_tiny = tl.load(k_ptrs, mask=tiny_load_mask, other=0.0)
                v_tiny = tl.load(v_ptrs, mask=tiny_load_mask, other=0.0)
                tiny_mask = tiny_valid
                acc, l_i, m_i = _sdpa_acc_fwd_1xT(
                    acc,
                    l_i,
                    m_i,
                    q,
                    k_tiny,
                    v_tiny,
                    tiny_mask,
                    softmax_scale,
                    TINY_GLOBAL_N,
                    BLOCK_SIZE_D,
                    v_cache_ptr.dtype.element_ty == tl.float8e5,
                )

        for kv_block_id in tl.range(1, num_global_window_blocks):
            kv_block_start = kv_block_id * BLOCK_SIZE_N
            kv_block_end = tl.minimum(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
            kv_block_len = kv_block_end - kv_block_start
            physical_page_id = tl.load(
                block_tables_ptr + b_id * stride_bt_batch + kv_block_id * stride_bt_block
            )
            k_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_k_blksz, stride_k_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )
            gw_mask = (kv_block_start + offs_n) < GLOBAL_WINDOW
            if LOCAL_WINDOW is not None:
                sw_mask = (kv_block_start + offs_n + LOCAL_WINDOW) >= (kv_seq_len - 1)
                gw_mask = gw_mask | sw_mask
            kv_mask = offs_n < kv_block_len
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


        num_full_pages = kv_seq_len // BLOCK_SIZE_N
        if LOCAL_WINDOW is not None:
            local_win_threshold = tl.maximum(0, kv_seq_len - 1 - LOCAL_WINDOW)
            first_fully_in_local = tl.cdiv(local_win_threshold, BLOCK_SIZE_N)
            nomask_start = tl.maximum(first_fully_in_local, non_global_window_start_block)
        else:
            nomask_start = non_global_window_start_block
        nomask_end = tl.maximum(nomask_start, num_full_pages)

        for kv_block_id in tl.range(non_global_window_start_block, nomask_start):
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
            kv_mask = offs_n < kv_block_len
            if LOCAL_WINDOW is not None:
                sw_mask = (kv_block_start + offs_n + LOCAL_WINDOW) >= (kv_seq_len - 1)
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

        for kv_block_id in tl.range(nomask_start, nomask_end):
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
            acc, l_i, m_i = _sdpa_acc_fwd_nomask_1xN(
                acc,
                l_i,
                m_i,
                q,
                k_block_ptr,
                v_block_ptr,
                softmax_scale,
                HEAD_DIM,
                BLOCK_SIZE_D,
                BLOCK_SIZE_N,
                BLOCK_SIZE_D,
                v_cache_ptr.dtype.element_ty == tl.float8e5,
            )

        for kv_block_id in tl.range(nomask_end, num_total_blocks):
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
            kv_mask = offs_n < kv_block_len
            if LOCAL_WINDOW is not None:
                sw_mask = (kv_block_start + offs_n + LOCAL_WINDOW) >= (kv_seq_len - 1)
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

        l_i_safe = tl.where(l_i > 0, l_i, 1.0)
        acc = tl.where(l_i > 0, acc / l_i_safe, 0.0)

        o_ptrs = o_ptr + b_id * stride_ob + q_head_id * stride_oh + offs_d * stride_od
        tl.store(o_ptrs, acc.to(OUT_T), mask=offs_d < HEAD_DIM)

@libentry()
@triton.jit
def _paged_decode_quant_kernel(
    q_ptr,           # [bsz, n_q_heads, head_dim] float
    k_cache_ptr,     # [n_pages, n_kv_heads, page_size, head_dim] int8
    k_qscale_ptr,    # [n_kv_heads, head_dim] float
    v_cache_ptr,     # [n_pages, n_kv_heads, page_size, head_dim] int8
    v_qscale_ptr,    # [n_kv_heads, head_dim] float
    o_ptr,           # [bsz, n_q_heads, head_dim] float
    seqlens_ptr,     # [bsz] int32
    block_tables_ptr,  # [bsz, max_num_blocks] int32
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
    stride_kqs_head,
    stride_kqs_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_vqs_head,
    stride_vqs_dim,
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
        offs_n = tl.arange(0, BLOCK_SIZE_N)

        # Load Q and k_qscale, compute scaled Q for quantization
        q_ptrs = q_ptr + b_id * stride_qb + q_head_id * stride_qh + offs_d * stride_qd
        q = tl.load(q_ptrs, mask=offs_d < HEAD_DIM, other=0.0).to(tl.float32)

        kqs_ptrs = k_qscale_ptr + kv_head_id * stride_kqs_head + offs_d * stride_kqs_dim
        k_qscale = tl.load(kqs_ptrs, mask=offs_d < HEAD_DIM, other=0.0).to(tl.float32)

        # Dynamic quantize Q * k_qscale -> q_int8, q_q_scale
        q_scaled = q * k_qscale
        q_amax = tl.max(tl.abs(q_scaled), axis=0)
        q_amax = tl.maximum(q_amax, 1e-12)
        q_q_scale = q_amax / 127.0
        q_scaled_norm = q_scaled / q_q_scale
        q_int8 = tl.where(q_scaled_norm < 0, q_scaled_norm - 0.5, q_scaled_norm + 0.5).to(tl.int8)

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

        # Global window blocks
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
            k_block = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            qk = tl.sum(q_int8.to(tl.float32)[None, :] * k_block, axis=1) * q_q_scale * softmax_scale

            gw_mask = (kv_block_start + offs_n) < GLOBAL_WINDOW
            if LOCAL_WINDOW is not None:
                sw_mask = (kv_block_start + offs_n + LOCAL_WINDOW) >= (kv_seq_len - 1)
                gw_mask = gw_mask | sw_mask
            kv_mask = offs_n < kv_block_len
            mask = gw_mask & kv_mask
            qk = tl.where(mask, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
            row_is_all_masked = m_ij == -float("inf")
            p = tl.math.exp(tl.where(row_is_all_masked, 0.0, qk - m_ij))
            p = tl.where(row_is_all_masked, 0.0, p)
            p = tl.where(mask, p, 0.0)
            l_ij = tl.sum(p, axis=0)
            alpha = tl.math.exp(tl.where(row_is_all_masked, 0.0, m_i - m_ij))
            alpha = tl.where(row_is_all_masked, 0.0, alpha)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha

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
            v_block = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            acc += tl.sum(p[:, None] * v_block, axis=0)
            m_i = m_ij

        # Local window blocks
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
            k_block = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            qk = tl.sum(q_int8.to(tl.float32)[None, :] * k_block, axis=1) * q_q_scale * softmax_scale

            kv_mask = offs_n < kv_block_len
            if LOCAL_WINDOW is not None:
                sw_mask = (kv_block_start + offs_n + LOCAL_WINDOW) >= (kv_seq_len - 1)
                mask = kv_mask & sw_mask
            else:
                mask = kv_mask
            qk = tl.where(mask, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
            row_is_all_masked = m_ij == -float("inf")
            p = tl.math.exp(tl.where(row_is_all_masked, 0.0, qk - m_ij))
            p = tl.where(row_is_all_masked, 0.0, p)
            p = tl.where(mask, p, 0.0)
            l_ij = tl.sum(p, axis=0)
            alpha = tl.math.exp(tl.where(row_is_all_masked, 0.0, m_i - m_ij))
            alpha = tl.where(row_is_all_masked, 0.0, alpha)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha

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
            v_block = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            acc += tl.sum(p[:, None] * v_block, axis=0)
            m_i = m_ij

        l_i_safe = tl.where(l_i > 0, l_i, 1.0)
        acc = tl.where(l_i > 0, acc / l_i_safe, 0.0)

        # Dequantize V output by multiplying with v_qscale
        vqs_ptrs = v_qscale_ptr + kv_head_id * stride_vqs_head + offs_d * stride_vqs_dim
        v_qscale = tl.load(vqs_ptrs, mask=offs_d < HEAD_DIM, other=0.0).to(tl.float32)
        acc = acc * v_qscale

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

    grid = (batch_size * num_q_heads,)
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)

    if q.dtype == torch.float16:
        out_t = tl.float16
    elif q.dtype == torch.bfloat16:
        out_t = tl.bfloat16
    else:
        out_t = tl.float32

    num_warps = _paged_decode_launch_config(head_dim, block_size)
    use_tiny_global = (
        global_window_size is not None
        and global_window_size <= 8
        and block_size >= 128
    )

    if use_tiny_global:
        _paged_decode_kernel_tiny_global[grid](
            q,
            key_cache,
            value_cache,
            o,
            seqlens,
            block_tables,
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
            global_window_size,
            local_window_size,
            num_q_heads,
            num_kv_heads,
            gqa_interleave,
            head_dim,
            block_size,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            BLOCK_SIZE_N=block_size_n,
            TINY_GLOBAL_N=8,
            OUT_T=out_t,
            num_warps=num_warps,
        )
    else:
        _paged_decode_kernel[grid](
            q,
            key_cache,
            value_cache,
            o,
            seqlens,
            block_tables,
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


def swa_paged_decode_quant_impl(
    q: torch.Tensor,           # [bsz, n_q_heads, head_dim] float
    key_cache: torch.Tensor,   # [n_pages, n_kv_heads, page_size, head_dim] int8
    k_qscale: torch.Tensor,    # [n_kv_heads, head_dim] float
    value_cache: torch.Tensor, # [n_pages, n_kv_heads, page_size, head_dim] int8
    v_qscale: torch.Tensor,    # [n_kv_heads, head_dim] float
    seqlens: torch.Tensor,     # [bsz] int32
    block_tables: torch.Tensor,  # [bsz, max_num_blocks] int32
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    gqa_interleave: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    batch_size, num_q_heads, head_dim = q.shape
    num_total_blocks, num_kv_heads, block_size, head_dim_cache = key_cache.shape

    assert head_dim == head_dim_cache
    assert key_cache.dtype == torch.int8, "key_cache must be int8"
    assert value_cache.dtype == torch.int8, "value_cache must be int8"

    block_size_n = min(128, triton.next_power_of_2(block_size))
    if block_size % block_size_n != 0:
        raise ValueError(
            f"KV block_size ({block_size}) must be divisible by decode tile size ({block_size_n})."
        )
    max_num_blocks_per_seq = block_tables.shape[1]

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    o = torch.empty_like(q, memory_format=torch.contiguous_format)

    grid = (batch_size * num_q_heads,)
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)

    if q.dtype == torch.float16:
        out_t = tl.float16
    elif q.dtype == torch.bfloat16:
        out_t = tl.bfloat16
    else:
        out_t = tl.float32

    num_warps = _paged_decode_launch_config(head_dim, block_size)

    _paged_decode_quant_kernel[grid](
        q,
        key_cache,
        k_qscale,
        value_cache,
        v_qscale,
        o,
        seqlens,
        block_tables,
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
        k_qscale.stride(0),
        k_qscale.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        v_qscale.stride(0),
        v_qscale.stride(1),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        block_tables.stride(0),
        block_tables.stride(1),
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
