import math
import os

from typing import Optional

import torch
import triton
import triton.language as tl

from .utils import get_num_cores


@triton.jit
def _conformer_chunk_attention_kernel(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    cu_q_lens_ptr,
    cu_total_seq_lens_ptr,
    scale,
    chunk_size: tl.constexpr,
    left_context_chunks: tl.constexpr,
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
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
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

            # Compute KV window for this query block
            # ctx_start = max(0, chunk_start - left_context_chunks * chunk_size)
            # chunk_end = min(chunk_start + chunk_size, kv_seq_len)
            q_block_abs_last_actual = tl.minimum(q_block_abs_start + BLOCK_M - 1, kv_seq_len - 1)
            chunk_start_first = (q_block_abs_start // chunk_size) * chunk_size
            chunk_start_last = (q_block_abs_last_actual // chunk_size) * chunk_size

            if left_context_chunks < 0:
                kv_win_start = 0
            else:
                kv_win_start = tl.maximum(0, chunk_start_first - left_context_chunks * chunk_size)

            kv_win_end = tl.minimum(chunk_start_last + chunk_size, kv_seq_len)
            kv_block_start_id = kv_win_start // BLOCK_N
            kv_block_end_id = tl.cdiv(kv_win_end, BLOCK_N)

            # Precompute fp32 query positions and chunk boundaries for mask
            q_abs_f32 = (kv_computed_len + q_offs).to(tl.float32)
            q_abs_i32 = kv_computed_len + q_offs
            chunk_start_i32 = (q_abs_i32 // chunk_size) * chunk_size
            chunk_end_f32 = tl.minimum((chunk_start_i32 + chunk_size).to(tl.float32), kv_seq_len.to(tl.float32))
            if left_context_chunks < 0:
                ctx_start_f32 = tl.zeros((BLOCK_M,), dtype=tl.float32)
            else:
                ctx_start_f32 = tl.maximum((chunk_start_i32 - left_context_chunks * chunk_size).to(tl.float32), 0.0)

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

                # Chunk mask: kv_idx ∈ [ctx_start[q], chunk_end[q])
                kv_f32 = kv_offsets.to(tl.float32)
                chunk_mask = (kv_f32[None, :] >= ctx_start_f32[:, None]) & (kv_f32[None, :] < chunk_end_f32[:, None])

                pre_mask = q_valid[:, None] & kv_valid[None, :] & chunk_mask

                # Online softmax
                qk = tl.where(pre_mask, qk, float("-inf"))
                m_candidate = tl.max(qk, axis=1)
                m_new = tl.maximum(m_i, m_candidate)
                # Guard against -inf - (-inf) → NaN when a query has no valid KV
                # in the first KV block it encounters (cross-chunk boundary case).
                has_valid = m_candidate > float("-inf")
                m_shift = tl.where(q_valid & (has_valid | (m_i > float("-inf"))), m_new, 0.0)
                qk = qk - m_shift[:, None]
                qk = tl.where(pre_mask, tl.exp(qk), 0.0)

                l_ij = tl.sum(qk, axis=1)
                # alpha rescales old accumulator: exp(m_i - m_new) if m_new > m_i, else 1.
                # When m_i == -inf (first valid block): alpha = 0 (no prior accumulator).
                m_delta = tl.where(has_valid & (m_i > float("-inf")), m_new - m_i, 0.0)
                alpha = tl.where(m_i > float("-inf"), tl.exp(-m_delta), 0.0)
                acc_update = tl.dot(qk.to(k_t.dtype), v, acc * alpha[:, None])
                acc = acc_update
                l_i = l_i * alpha + l_ij
                m_i = m_new

            out = acc / tl.where(q_valid, l_i, 1.0)[:, None]
            # out = tl.where(q_valid[:, None], out, 0.0)
            o_block_ptr = tl.make_block_ptr(
                base=o_ptr + q_start * stride_ot + h_id * stride_oh,
                shape=(q_seq_len, head_dim),
                strides=(stride_ot, stride_od),
                offsets=(q_block_start, 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            tl.store(o_block_ptr, out.to(o_ptr.type.element_ty), boundary_check=(0, 1))


def _select_blocks(head_dim: int, dtype: torch.dtype) -> tuple[int, int, int, bool]:
    """Heuristic tile selection for Ascend 910B2C with 192 KB UB.

    UB budget accounting for explicit tl.multibuffer + auto-multibuffer:
        UB_total = BM*(12*D + 8) + BN*(8*D) + BM*BN*8

    All tile combinations below are verified to fit within ~196 KB UB.
    """
    block_m_override = os.getenv("MOJO_CHUNK_ATTN_BLOCK_M")
    block_n_override = os.getenv("MOJO_CHUNK_ATTN_BLOCK_N")

    if block_m_override is not None or block_n_override is not None:
        block_m = int(block_m_override) if block_m_override is not None else 128
        block_n = int(block_n_override) if block_n_override is not None else 64
        return block_m, block_n, head_dim, True

    if dtype == torch.float32:
        return 64, 64, head_dim, False

    enable_multibuf = True

    if head_dim >= 128:
        # D=128: BM=80, BN=36, UB≈183KB, AI≈38
        return 80, 36, head_dim, enable_multibuf
    elif head_dim >= 96:
        # D=96: BM=64, BN=64, UB≈153KB, AI≈43
        return 64, 64, head_dim, enable_multibuf
    else:
        # D<=64: BM=96, BN=64, UB≈156KB, AI≈55
        return 96, 64, head_dim, enable_multibuf


def conformer_chunk_attention_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_total_seq_lens: Optional[torch.Tensor],
    chunk_size: int,
    left_context_chunks: int,
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

    block_m, block_n, block_d, enable_multibuf = _select_blocks(head_dim, q.dtype)

    o = torch.zeros_like(q, memory_format=torch.contiguous_format)
    grid = (get_num_cores("cube"),)

    _conformer_chunk_attention_kernel[grid](
        o,
        q,
        k,
        v,
        cu_q_lens,
        cu_total_seq_lens,
        softmax_scale,
        chunk_size,
        left_context_chunks,
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
        cu_q_lens.shape[0] - 1,
        num_heads,
        head_dim,
        block_m,
        block_n,
        block_d,
        multibuffer=enable_multibuf,
    )
    return o
