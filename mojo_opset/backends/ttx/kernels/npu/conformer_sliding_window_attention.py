import math
import os

from typing import Optional

import torch
import triton
import triton.language as tl

from .swa import get_aux_mask
from .utils import get_num_cores

_AUX_MASK_SIZE = 256


@triton.jit
def _load_tril_mask(
    mask_ptr_tril,
    mask_size,
    mask_stride_m,
    mask_stride_n,
    M_BLOCK,
    N_BLOCK,
    m_start,
    n_start,
):
    """Load tril mask slice: n + n_start <= m + m_start  (i.e. kv_col <= q_row + offset)."""
    offset = min(max(n_start - m_start, -mask_size), mask_size)
    mask = tl.load(
        mask_ptr_tril
        + tl.arange(0, M_BLOCK)[:, None] * mask_stride_m
        + (offset + tl.arange(0, N_BLOCK))[None, :] * mask_stride_n
    )
    return mask.to(tl.int1)


@triton.jit
def _load_triu_mask(
    mask_ptr_triu,
    mask_size,
    mask_stride_m,
    mask_stride_n,
    M_BLOCK,
    N_BLOCK,
    m_start,
    n_start,
):
    """Load triu mask slice: n + n_start >= m + m_start  (i.e. kv_col >= q_row + offset)."""
    offset = min(max(n_start - m_start, -mask_size), mask_size)
    mask = tl.load(
        mask_ptr_triu
        + tl.arange(0, M_BLOCK)[:, None] * mask_stride_m
        + (offset + tl.arange(0, N_BLOCK))[None, :] * mask_stride_n
    )
    return mask.to(tl.int1)


@triton.jit
def _conformer_sliding_window_attention_kernel(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    cu_q_lens_ptr,
    cu_total_seq_lens_ptr,
    scale,
    left_window: tl.constexpr,
    right_window: tl.constexpr,
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
    aux_mask_ptr,
    aux_mask_size,
    aux_mask_stride_m,
    aux_mask_stride_n,
    aux_mask_ptr_triu,
    aux_mask_ptr_tril,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_MASK_LOOKUP: tl.constexpr,
):
    tl.static_assert(head_dim <= BLOCK_D, "BLOCK_D must cover head_dim")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    prev_q_chunks = 0
    for b_id in range(batch_size):
        q_start = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        kv_start = tl.load(cu_total_seq_lens_ptr + b_id).to(tl.int32)
        kv_end = tl.load(cu_total_seq_lens_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end - q_start
        kv_seq_len = kv_end - kv_start
        kv_computed_len = kv_seq_len - q_seq_len

        cur_q_chunks = tl.cdiv(q_seq_len, BLOCK_M)
        prev_q_tasks = prev_q_chunks * num_heads
        cur_q_tasks = cur_q_chunks * num_heads
        prev_q_chunks += cur_q_chunks

        for q_task_id in range((prev_q_tasks + pid) % n_programs, cur_q_tasks, n_programs):
            q_block_id = q_task_id // num_heads
            h_id = q_task_id % num_heads
            q_block_start = q_block_id * BLOCK_M
            q_offs = q_block_start + tl.arange(0, BLOCK_M)
            q_valid = q_offs < q_seq_len
            q_block_abs_start = kv_computed_len + q_block_start

            q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_start * stride_qt + h_id * stride_qh,
                shape=(q_seq_len, head_dim),
                strides=(stride_qt, stride_qd),
                offsets=(q_block_start, 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)

            q_block_last = tl.minimum(q_block_start + BLOCK_M - 1, q_seq_len - 1)
            kv_win_start = tl.maximum(0, kv_computed_len + q_block_start - left_window)
            kv_win_end = tl.minimum(kv_seq_len, kv_computed_len + q_block_last + right_window + 1)
            kv_block_start_id = kv_win_start // BLOCK_N
            kv_block_end_id = tl.cdiv(kv_win_end, BLOCK_N)

            # Precompute fp32 query positions once (shared across all KV blocks)
            q_abs_f32 = (kv_computed_len + q_offs).to(tl.float32)
            left_bound_f32 = q_abs_f32 - left_window
            right_bound_f32 = q_abs_f32 + right_window

            for kv_block_id in range(kv_block_start_id, kv_block_end_id):
                kv_block_start = kv_block_id * BLOCK_N
                kv_offsets = kv_block_start + tl.arange(0, BLOCK_N)
                kv_valid = kv_offsets < kv_seq_len

                k_t_block_ptr = tl.make_block_ptr(
                    base=k_ptr + kv_start * stride_kt + h_id * stride_kh,
                    shape=(head_dim, kv_seq_len),
                    strides=(stride_kd, stride_kt),
                    offsets=(0, kv_block_start),
                    block_shape=(BLOCK_D, BLOCK_N),
                    order=(0, 1),
                )
                v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + kv_start * stride_vt + h_id * stride_vh,
                    shape=(kv_seq_len, head_dim),
                    strides=(stride_vt, stride_vd),
                    offsets=(kv_block_start, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )

                k_t = tl.load(k_t_block_ptr, boundary_check=(0, 1), padding_option="zero")
                tl.multibuffer(k_t, 2)
                v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
                tl.multibuffer(v, 2)

                # QK^T
                qk = tl.dot(q, k_t) * scale

                # Window mask: kv_offset ∈ [q_abs - left_window, q_abs + right_window]
                if USE_MASK_LOOKUP:
                    mask_left = _load_triu_mask(
                        aux_mask_ptr_triu,
                        aux_mask_size,
                        aux_mask_stride_m,
                        aux_mask_stride_n,
                        BLOCK_M,
                        BLOCK_N,
                        q_block_abs_start - left_window,
                        kv_block_start,
                    )
                    mask_right = _load_tril_mask(
                        aux_mask_ptr_tril,
                        aux_mask_size,
                        aux_mask_stride_m,
                        aux_mask_stride_n,
                        BLOCK_M,
                        BLOCK_N,
                        q_block_abs_start + right_window,
                        kv_block_start,
                    )
                    win_mask = mask_left & mask_right
                else:
                    kv_f32 = kv_offsets.to(tl.float32)
                    win_mask = (kv_f32[None, :] >= left_bound_f32[:, None]) & (
                        kv_f32[None, :] <= right_bound_f32[:, None]
                    )

                pre_mask = q_valid[:, None] & kv_valid[None, :] & win_mask

                # Online softmax: mask → row-max → rescale → exp → reweight
                qk = tl.where(pre_mask, qk, float("-inf"))
                m_candidate = tl.max(qk, axis=1)
                m_new = tl.where(q_valid, tl.maximum(m_i, m_candidate), m_i)
                qk = qk - tl.where(q_valid, m_new, 0.0)[:, None]
                qk = tl.where(pre_mask, tl.exp(qk), 0.0)

                l_ij = tl.sum(qk, axis=1)
                alpha = tl.where(q_valid, tl.exp(m_i - m_new), 1.0)
                # Fused rescale + PV matmul
                acc_update = tl.dot(qk.to(k_t.dtype), v, acc * alpha[:, None])
                acc = tl.where(q_valid[:, None], acc_update, acc)
                l_i = tl.where(q_valid, l_i * alpha + l_ij, l_i)
                m_i = m_new

            out = acc / tl.where(q_valid, l_i, 1.0)[:, None]
            out = tl.where(q_valid[:, None], out, 0.0)
            o_block_ptr = tl.make_block_ptr(
                base=o_ptr + q_start * stride_ot + h_id * stride_oh,
                shape=(q_seq_len, head_dim),
                strides=(stride_ot, stride_od),
                offsets=(q_block_start, 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            tl.store(o_block_ptr, out.to(o_ptr.type.element_ty), boundary_check=(0, 1))


def _select_blocks(head_dim: int, dtype: torch.dtype) -> tuple[int, int, int, bool, bool]:
    """Heuristic tile selection for Ascend 910B2C with 192 KB UB.

    UB budget with double-buffered K_t/V:
        UB = BM*D*2 + 2*D*BN*2 + 2*BN*D*2 + BM*BN*4 + BM*D*4 + BM*8
           = BM*(6*D + 8) + BN*(8*D) + BM*BN*4

    Arithmetic intensity (AI) = 2*BM*BN / (BM + 2*BN).
    Roofline knee ≈ 197 FLOP/byte for Ascend 910B2C.
    """
    block_m_override = os.getenv("MOJO_CONFORMER_ATTN_BLOCK_M")
    block_n_override = os.getenv("MOJO_CONFORMER_ATTN_BLOCK_N")
    use_mask_lookup = os.getenv("MOJO_CONFORMER_MASK_LOOKUP", "0") == "1"

    if block_m_override is not None or block_n_override is not None:
        block_m = int(block_m_override) if block_m_override is not None else 128
        block_n = int(block_n_override) if block_n_override is not None else 64
        return block_m, block_n, head_dim, use_mask_lookup, True

    if dtype == torch.float32:
        return 64, 64, head_dim, use_mask_lookup, False

    enable_multibuf = True

    if head_dim >= 128:
        # D=128: BM=128, BN=48, UB=173056, AI≈55
        return 128, 32, head_dim, use_mask_lookup, enable_multibuf
    elif head_dim >= 96:
        # D=96: BM=192, BN=48, UB=185856, AI≈64
        return 64, 64, head_dim, use_mask_lookup, enable_multibuf
    else:
        # D<=64: BM=128, BN=128, UB=181248, AI≈85
        return 128, 128, head_dim, use_mask_lookup, enable_multibuf


def conformer_sliding_window_attention_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_total_seq_lens: Optional[torch.Tensor],
    left_window: int,
    right_window: int,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q/k/v must be [T, H, D]")
    if k.shape != v.shape or q.shape[1:] != k.shape[1:]:
        raise ValueError(f"q/k/v shape mismatch: {q.shape}, {k.shape}, {v.shape}")
    if cu_total_seq_lens is None:
        cu_total_seq_lens = cu_q_lens

    _, num_heads, head_dim = q.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    block_m, block_n, block_d, use_mask_lookup, enable_multibuf = _select_blocks(head_dim, q.dtype)

    aux_mask_size, aux_mask = get_aux_mask()
    aux_mask_stride_m = aux_mask.stride(0)
    aux_mask_stride_n = aux_mask.stride(1)
    aux_mask_ptr_triu = aux_mask.data_ptr() + aux_mask_size * aux_mask_stride_n
    aux_mask_ptr_tril = aux_mask.data_ptr() + 3 * aux_mask_size * aux_mask_stride_n

    o = torch.zeros_like(q, memory_format=torch.contiguous_format)
    grid = (get_num_cores("cube"),)

    _conformer_sliding_window_attention_kernel[grid](
        o,
        q,
        k,
        v,
        cu_q_lens,
        cu_total_seq_lens,
        softmax_scale,
        left_window,
        right_window,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        aux_mask,
        aux_mask_size,
        aux_mask_stride_m,
        aux_mask_stride_n,
        aux_mask_ptr_triu,
        aux_mask_ptr_tril,
        cu_q_lens.shape[0] - 1,
        num_heads,
        head_dim,
        block_m,
        block_n,
        block_d,
        USE_MASK_LOOKUP=use_mask_lookup,
        multibuffer=enable_multibuf,
    )
    return o
