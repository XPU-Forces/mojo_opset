import math

from typing import Optional

import torch
import triton
import triton.language as tl

from .utils import get_num_cores


@triton.jit
def _padded_window_attention_fwd_kernel(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    seqlens_ptr,
    scale,
    left_window,
    right_window,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.static_assert(head_dim <= BLOCK_D, "BLOCK_D must cover head_dim")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    q_blocks_per_head = tl.cdiv(seq_len, BLOCK_M)
    total_tasks = batch_size * num_heads * q_blocks_per_head

    for task_id in range(pid, total_tasks, n_programs):
        b_id = task_id // (num_heads * q_blocks_per_head)
        rem = task_id % (num_heads * q_blocks_per_head)
        h_id = rem // q_blocks_per_head
        q_block_id = rem % q_blocks_per_head

        cur_seq_len = tl.load(seqlens_ptr + b_id).to(tl.int32)
        q_block_start = q_block_id * BLOCK_M
        if q_block_start < cur_seq_len:
            q_offsets = q_block_start + tl.arange(0, BLOCK_M)
            q_valid = q_offsets < cur_seq_len

            q_block_ptr = tl.make_block_ptr(
                base=q_ptr + b_id * stride_qb + h_id * stride_qh,
                shape=(seq_len, head_dim),
                strides=(stride_qs, stride_qd),
                offsets=(q_block_start, 0),
                block_shape=(BLOCK_M, head_dim),
                order=(1, 0),
            )
            q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)

            q_block_last = tl.minimum(q_block_start + BLOCK_M - 1, cur_seq_len - 1)
            kv_start = tl.maximum(0, q_block_start - left_window)
            kv_end = tl.minimum(cur_seq_len, q_block_last + right_window + 1)
            kv_block_start_id = kv_start // BLOCK_N
            kv_block_end_id = tl.cdiv(kv_end, BLOCK_N)
            total_kv_blocks = tl.cdiv(seq_len, BLOCK_N)

            for kv_block_id in range(0, total_kv_blocks):
                do_work = (kv_block_id >= kv_block_start_id) & (kv_block_id < kv_block_end_id)
                kv_block_start = kv_block_id * BLOCK_N
                kv_offsets = kv_block_start + tl.arange(0, BLOCK_N)
                kv_valid = kv_offsets < cur_seq_len

                k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + b_id * stride_kb + h_id * stride_kh,
                    shape=(seq_len, head_dim),
                    strides=(stride_ks, stride_kd),
                    offsets=(kv_block_start, 0),
                    block_shape=(BLOCK_N, head_dim),
                    order=(1, 0),
                )
                v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + b_id * stride_vb + h_id * stride_vh,
                    shape=(seq_len, head_dim),
                    strides=(stride_vs, stride_vd),
                    offsets=(kv_block_start, 0),
                    block_shape=(BLOCK_N, head_dim),
                    order=(1, 0),
                )

                k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
                v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

                qk = tl.dot(q, tl.trans(k)) * scale

                left_bound = q_offsets[:, None] - left_window
                right_bound = q_offsets[:, None] + right_window
                window_mask = (kv_offsets[None, :] >= left_bound) & (kv_offsets[None, :] <= right_bound)
                pre_mask = do_work & q_valid[:, None] & kv_valid[None, :] & window_mask

                masked_scores = tl.where(pre_mask, qk, -1e6)
                m_candidate = tl.max(masked_scores, axis=1)
                m_new = tl.where(q_valid, tl.maximum(m_i, m_candidate), m_i)
                shifted = masked_scores - tl.where(q_valid, m_new, 0.0)[:, None]
                p = tl.where(pre_mask, tl.exp(shifted), 0.0)

                l_ij = tl.sum(p, axis=1)
                alpha = tl.where(q_valid, tl.exp(m_i - m_new), 1.0)
                acc_scaled = acc * alpha[:, None]
                acc_update = tl.dot(p.to(k.dtype), v)
                acc = tl.where(q_valid[:, None], acc_scaled + acc_update, acc)
                l_i = tl.where(q_valid, l_i * alpha + l_ij, l_i)
                m_i = m_new

            denom = tl.where(q_valid, l_i, 1.0)
            out = acc / denom[:, None]
            out = tl.where(q_valid[:, None], out, 0.0)

            o_block_ptr = tl.make_block_ptr(
                base=o_ptr + b_id * stride_ob + h_id * stride_oh,
                shape=(seq_len, head_dim),
                strides=(stride_os, stride_od),
                offsets=(q_block_start, 0),
                block_shape=(BLOCK_M, head_dim),
                order=(1, 0),
            )
            tl.store(o_block_ptr, out.to(o_ptr.type.element_ty), boundary_check=(0, 1))


def padded_window_attention_infer_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlens: torch.Tensor,
    left_window: int,
    right_window: int,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be [B, H, S, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v shape mismatch: {q.shape}, {k.shape}, {v.shape}")
    if seqlens.ndim != 1 or seqlens.shape[0] != q.shape[0]:
        raise ValueError(f"seqlens must be [B], got {seqlens.shape} for batch {q.shape[0]}")

    batch_size, num_heads, seq_len, head_dim = q.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    o = torch.zeros_like(q, memory_format=torch.contiguous_format)
    # Keep tiles conservative on Ascend to avoid UB pressure from
    # qk/p/acc temporaries, especially for bf16 + larger head dims.
    block_m = 64
    block_n = 64
    block_d = head_dim
    grid = (get_num_cores("cube"),)

    _padded_window_attention_fwd_kernel[grid](
        o,
        q,
        k,
        v,
        seqlens,
        softmax_scale,
        left_window,
        right_window,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        block_m,
        block_n,
        block_d,
    )
    return o
