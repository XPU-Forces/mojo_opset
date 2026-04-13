import torch
import triton
import triton.language as tl
from typing import Optional
from .utils import get_mlu_total_cores


@triton.jit
def causal_mask_fn(
    mask_ptr,
    mask_size,
    mask_stride_m,
    mask_stride_n,
    q_start,
    kv_start,
    Q_BLOCK,
    KV_BLOCK,
):
    offset_causal = min(max(kv_start - q_start, -mask_size), mask_size)
    offsets_mask_causal = (tl.arange(0, Q_BLOCK)[:, None]) * mask_stride_m + (
        mask_size + offset_causal + tl.arange(0, KV_BLOCK)[None, :]
    ) * mask_stride_n
    mask_causal = tl.load(mask_ptr + offsets_mask_causal).to(tl.int1)

    return mask_causal


@triton.jit
def _infer_single_block(
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
    tl.static_assert(
        HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM"
    )
    # -- Compute qk ----

    # Load (transposed) K block
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    k_T = tl.trans(k)
    qk = tl.dot(q, k_T)
    # tl.compile_hint(qk, "tile_cube_loop")

    qk = qk * qk_scale
    if mask is not None:
        qk = tl.where(mask, qk, float("-inf"))  # 32B # bool

    m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Scaled max
    qk = qk - m_ij[:, None]  # Stabilize

    # Softmax weights p = exp(qk)
    p = tl.math.exp(qk)

    p_cast = p.to(k_T.dtype)

    # Load corresponding V block
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Softmax denominator (sum of each row)
    l_ij = tl.sum(p, 1)
    # -- Update m_i and l_i
    alpha = tl.math.exp(
        m_i - m_ij
    )  # Update factor: exp difference between old and new max
    l_i = l_i * alpha + l_ij  # Update softmax denominator
    # -- Update output accumulator --
    acc_ptr = acc_ptr * alpha[:, None]
    acc_ptr = tl.dot(p_cast, v, acc_ptr)
    # tl.compile_hint(acc_ptr, "tile_cube_loop")

    # Update current block max
    m_i = m_ij

    # NOTE(zhangjihang): for training
    # Return accumulated output acc_ptr, softmax denominator l_i, and max value m_i
    return acc_ptr, l_i, m_i


@triton.jit
def paged_prefill_kernel(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
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
    aux_mask_ptr,
    stride_mask_m,
    stride_mask_n,
    AUX_MASK_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    tl.static_assert(PAGE_SIZE % BLOCK_N == 0, "BLOCK_N must be a divisor of PAGE_SIZE")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    cu_q_chunks = 0
    for b_id in range(bsz):
        q_start = tl.load(cu_seqlens_q_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_seqlens_q_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end - q_start

        if kv_lens_ptr is None:
            kv_seq_len = q_seq_len
        else:
            kv_seq_len = tl.load(kv_lens_ptr + b_id).to(tl.int32)
        kv_computed_len = kv_seq_len - q_seq_len

        num_q_chunks = tl.cdiv(q_seq_len, BLOCK_M)
        new_q_tasks = num_q_chunks * NUM_Q_HEADS
        prev_q_tasks = cu_q_chunks * NUM_Q_HEADS
        cu_q_chunks += num_q_chunks

        for q_task_id in range(
            (prev_q_tasks + pid) % n_programs, new_q_tasks, n_programs
        ):
            q_block_id = q_task_id // NUM_Q_HEADS
            q_head_id = q_task_id % NUM_Q_HEADS
            if GQA_INTERLEAVE:
                kv_head_id = q_head_id % NUM_KV_HEADS
            else:
                kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

            q_block_start = q_block_id * BLOCK_M
            q_block_end = min(q_block_start + BLOCK_M, q_seq_len)
            q_block_len = q_block_end - q_block_start

            cur_q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_qt, stride_qd),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            cur_q_block = tl.load(
                cur_q_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

            m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

            num_kv_blocks = tl.cdiv(kv_computed_len + q_block_end, BLOCK_N)

            for kv_block_id in range(0, num_kv_blocks):
                kv_block_start = kv_block_id * BLOCK_N
                kv_block_end = min(kv_block_start + BLOCK_N, kv_seq_len)
                kv_block_len = kv_block_end - kv_block_start
                logical_page_id = kv_block_start // PAGE_SIZE
                kv_block_start_in_page = kv_block_start % PAGE_SIZE
                physical_page_id = tl.load(
                    block_table_ptr
                    + b_id * stride_block_table_b
                    + logical_page_id * stride_block_table_p
                )

                cur_k_block_ptr = tl.make_block_ptr(
                    base=k_ptr
                    + physical_page_id * stride_kp
                    + kv_head_id * stride_kh
                    + kv_block_start_in_page * stride_kt,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                cur_v_block_ptr = tl.make_block_ptr(
                    base=v_ptr
                    + physical_page_id * stride_vp
                    + kv_head_id * stride_vh
                    + kv_block_start_in_page * stride_vt,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )

                mask = causal_mask_fn(
                    aux_mask_ptr,
                    AUX_MASK_SIZE,
                    stride_mask_m,
                    stride_mask_n,
                    kv_computed_len + q_block_start,
                    kv_block_start,
                    BLOCK_M,
                    BLOCK_N,
                )

                acc, l_i, m_i = _infer_single_block(
                    acc,
                    l_i,
                    m_i,
                    cur_q_block,
                    cur_k_block_ptr,
                    cur_v_block_ptr,
                    mask,
                    scale,
                    HEAD_DIM,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_D,
                    v_ptr.dtype.element_ty == tl.float8e5,
                )

            # cur_o_block_ptr = tl.advance(o_block_ptr, (q_block_start.to(tl.int32), 0))
            cur_o_block_ptr = tl.make_block_ptr(
                base=o_ptr + q_start * stride_ot + q_head_id * stride_oh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_ot, stride_od),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            accumulator = acc / l_i[:, None]
            tl.store(
                cur_o_block_ptr,
                accumulator.to(o_ptr.type.element_ty),
                boundary_check=(0, 1),
            )


def paged_attention_prefill_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,  # [bsz + 1]
    seqlens_kv: torch.Tensor,  # [bsz + 1]
    block_tables: torch.Tensor,  # [bsz, num_kv_blocks]
    gqa_interleave: bool,
    softmax_scale: Optional[float] = None,
    aux_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bsz = cu_seqlens_q.shape[0] - 1
    _, num_q_heads, head_dim = q.shape
    _, num_kv_heads, page_size, _ = key_cache.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    if aux_mask is None:
        aux_mask = torch.ones(1024, 1024 * 3, dtype=torch.bool).tril(1024).mlu()

    o = torch.zeros_like(q, memory_format=torch.contiguous_format)

    if q.dtype == torch.float32:
        BLOCK_M = 64
        BLOCK_N = min(64, triton.next_power_of_2(page_size))
    else:
        BLOCK_M = 128
        BLOCK_N = min(128, triton.next_power_of_2(page_size))

    BLOCK_D = head_dim
    job_num = get_mlu_total_cores()
    grid = (job_num,)

    paged_prefill_kernel[grid](
        o,
        q,
        key_cache,
        value_cache,
        bsz,
        cu_seqlens_q,
        seqlens_kv,
        block_tables,
        softmax_scale,
        o.stride(0),
        o.stride(1),
        o.stride(2),
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
        aux_mask,
        aux_mask.stride(0),
        aux_mask.stride(1),
        aux_mask.shape[0],
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        page_size,
        num_warps=1,
        num_stages=1,
        force_use_shared_memory=True,
        bottleneck="simd",
    )
    return o
