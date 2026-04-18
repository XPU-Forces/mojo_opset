import torch
import math
from typing import Optional, Tuple

from .utils import get_mlu_total_cores

AUX_MASK_SIZE = 256
AUX_MASK = None


def get_aux_mask():
    global AUX_MASK
    global AUX_MASK_SIZE
    if AUX_MASK is None:
        AUX_MASK = torch.cat(
            [
                torch.cat(
                    [
                        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
                        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE).tril().bool(),
                        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
                        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE).triu().bool(),
                        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
                        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE).bool(),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        ).mlu()
    return AUX_MASK_SIZE, AUX_MASK


import triton
import triton.language as tl


@triton.jit
def _swa_split_blocks(
    q_block_start_id, 
    q_block_len, 
    kv_seq_len, 
    BLOCK_SIZE_N, 
    IS_CAUSAL, 
    GLOBAL_WINDOW_SIZE, 
    LOCAL_WINDOW_SIZE
):
    if not IS_CAUSAL:
        return 0, 0, tl.cdiv(kv_seq_len, BLOCK_SIZE_N)

    num_total_blocks = tl.cdiv(q_block_start_id + q_block_len, BLOCK_SIZE_N)
    if GLOBAL_WINDOW_SIZE is None and LOCAL_WINDOW_SIZE is None:
        return 0, 0, num_total_blocks
    
    if GLOBAL_WINDOW_SIZE is not None:
        num_global_window_blocks = min(tl.cdiv(GLOBAL_WINDOW_SIZE, BLOCK_SIZE_N), num_total_blocks)
    else:
        num_global_window_blocks = 0
    
    if LOCAL_WINDOW_SIZE is not None:
        local_window_start_id = max(q_block_start_id - LOCAL_WINDOW_SIZE, 0)
        local_window_start_block = local_window_start_id // BLOCK_SIZE_N
    else:
        local_window_start_block = num_total_blocks
    
    non_global_window_start_block = max(num_global_window_blocks, local_window_start_block)
    
    return num_global_window_blocks, non_global_window_start_block, num_total_blocks

@triton.jit
def _swa_transposed_range_blocks(
    kv_block_start_id, 
    kv_block_len, 
    kv_computed_len, 
    q_seq_len, 
    BLOCK_SIZE_M, 
    IS_CAUSAL, 
    GLOBAL_WINDOW_SIZE, 
    LOCAL_WINDOW_SIZE
):
    if IS_CAUSAL:
        cur_q_start = max(kv_block_start_id - kv_computed_len, 0)
        if GLOBAL_WINDOW_SIZE is None and LOCAL_WINDOW_SIZE is None:
            # vanilla attention, iterate over all queries
            cur_q_end = q_seq_len
        else:
            if LOCAL_WINDOW_SIZE is not None:
                # otherwise, it can only be attented as sliding window tokens
                cur_q_end = max(kv_block_start_id + kv_block_len + LOCAL_WINDOW_SIZE - kv_computed_len, 0)
            if GLOBAL_WINDOW_SIZE is not None:
                if kv_block_start_id < GLOBAL_WINDOW_SIZE:
                    # sink token is attended by all succeeding tokens
                    cur_q_end = q_seq_len
                elif LOCAL_WINDOW_SIZE is None:
                    # Not attended
                    cur_q_start = 0
                    cur_q_end = 0
    else:
        # full attention, iterate over all queries
        cur_q_start = 0
        cur_q_end = q_seq_len

    start_block = cur_q_start // BLOCK_SIZE_M
    end_block = tl.cdiv(cur_q_end, BLOCK_SIZE_M)
    return start_block, end_block



@triton.jit
def gen_mask_n_right_bound(mask_10_ptr, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, n_start, right):
    # tl.arange(n_start, n_start + N_BLOCK)[None, :] < right
    offset = min(max(n_start - right, -mask_size), 0)
    mask = tl.load(
        mask_10_ptr
        + tl.arange(0, M_BLOCK)[None, :] * mask_stride_m
        + (offset + tl.arange(0, N_BLOCK))[:, None] * mask_stride_n
    )
    return mask.to(tl.int1)


@triton.jit
def gen_mask_n_left_bound(mask_01_ptr, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, n_start, left):
    # tl.arange(n_start, n_start + N_BLOCK)[None, :] >= left
    offset = min(max(n_start - left, -mask_size), 0)
    mask = tl.load(
        mask_01_ptr
        + tl.arange(0, M_BLOCK)[None, :] * mask_stride_m
        + (offset + tl.arange(0, N_BLOCK))[:, None] * mask_stride_n
    )
    return mask.to(tl.int1)


@triton.jit
def gen_mask_m_right_bound(mask_10t_ptr, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, m_start, right):
    # tl.arange(m_start, m_start + M_BLOCK)[:, None] < right
    offset = min(max(m_start - right, -mask_size), 0)
    mask = tl.load(
        mask_10t_ptr
        + (offset + tl.arange(0, M_BLOCK)[None, :]) * mask_stride_m
        + tl.arange(0, N_BLOCK)[:, None] * mask_stride_n
    )
    return mask.to(tl.int1)


@triton.jit
def gen_mask_m_left_bound(mask_01t_ptr, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, m_start, left):
    # tl.arange(m_start, m_start + M_BLOCK)[:, None] >= left
    offset = min(max(m_start - left, -mask_size), 0)
    mask = tl.load(
        mask_01t_ptr
        + (offset + tl.arange(0, M_BLOCK)[None, :]) * mask_stride_m
        + tl.arange(0, N_BLOCK)[:, None] * mask_stride_n
    )
    return mask.to(tl.int1)


@triton.jit
def gen_mask_tril(mask_ptr_tril, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, m_start, n_start):
    # tl.arange(n_start, n_start + N_BLOCK)[None, :] <= tl.arange(m_start, m_start + M_BLOCK)[:, None]
    offset = min(max(n_start - m_start, -mask_size), mask_size)
    mask = tl.load(
        mask_ptr_tril
        + tl.arange(0, M_BLOCK)[None, :] * mask_stride_m
        + (offset + tl.arange(0, N_BLOCK))[:, None] * mask_stride_n
    )
    return mask.to(tl.int1)


@triton.jit
def gen_mask_triu(mask_ptr_triu, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, m_start, n_start):
    # tl.arange(n_start, n_start + N_BLOCK)[None, :] >= tl.arange(m_start, m_start + M_BLOCK)[:, None]
    len_offset = min(max(n_start - m_start, -mask_size), mask_size)
    mask = tl.load(
        mask_ptr_triu
        + tl.arange(0, M_BLOCK)[None, :] * mask_stride_m
        + (len_offset + tl.arange(0, N_BLOCK))[:, None] * mask_stride_n
    )
    return mask.to(tl.int1)


@triton.jit
def _sdpa_acc_fwd_MxN(
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
    # -- Compute qk ----

    # Load (transposed) K block
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    qk = tl.dot(k, q)

    qk = qk * qk_scale
    if mask is not None and mask is not True:
        qk = tl.where(mask, qk, -1e6)  # 32B # bool

    m_ij = tl.maximum(m_i, tl.max(qk, 0))  # Scaled max
    qk = qk - m_ij[None, :]  # Stabilize

    # Softmax weights p = exp(qk)
    p = tl.math.exp(qk)

    p_cast = p.to(k.dtype)

    # Load corresponding V block
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Softmax denominator (sum of each row)
    l_ij = tl.sum(p, 0)
    # -- Update m_i and l_i
    alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
    l_i = l_i * alpha + l_ij  # Update softmax denominator
    # -- Update output accumulator --
    acc_ptr = acc_ptr * alpha[None, :]
    acc_ptr = tl.dot(v, p_cast, acc_ptr)

    # Update current block max
    m_i = m_ij

    # Return accumulated output acc_ptr, softmax denominator l_i, and max value m_i
    return acc_ptr, l_i, m_i

@triton.jit
def _swa_paged_prefill_kernel(
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
    aux_mask_size,
    stride_mask_m,
    stride_mask_n,
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
    tl.static_assert(HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM")
    tl.static_assert(PAGE_SIZE % BLOCK_N == 0, "BLOCK_N must be a divisor of PAGE_SIZE")

    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    # Hint(chenyifan):
    #   the prepared aux_mask is [[empty, triu, full, tril, empty],
    #                             [full, empty, empty, full, empty]]
    #   every mask [BLOCK_M, BLOCK_N] can be sliced from the aux_mask and further combined
    aux_mask_ptr_01 = aux_mask_ptr + aux_mask_size * 1 * stride_mask_m + aux_mask_size * 3 * stride_mask_n
    aux_mask_ptr_10 = aux_mask_ptr + aux_mask_size * 1 * stride_mask_m + aux_mask_size * 1 * stride_mask_n
    aux_mask_ptr_triu = aux_mask_ptr + aux_mask_size * 1 * stride_mask_n
    aux_mask_ptr_tril = aux_mask_ptr + aux_mask_size * 3 * stride_mask_n
    aux_mask_ptr_01t = aux_mask_ptr + aux_mask_size * 1 * stride_mask_m
    aux_mask_ptr_10t = aux_mask_ptr + aux_mask_size * 1 * stride_mask_m + aux_mask_size * 2 * stride_mask_n

    cu_q_chunks = 0
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

            # q_block_ptr = tl.make_block_ptr(
            #     base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
            #     shape=(q_seq_len, HEAD_DIM),
            #     strides=(stride_qt, stride_qd),
            #     offsets=(0, 0),
            #     block_shape=(BLOCK_M, BLOCK_D),
            #     order=(1, 0),
            # )
            # o_block_ptr = tl.make_block_ptr(
            #     base=o_ptr + q_start * stride_ot + q_head_id * stride_oh,
            #     shape=(q_seq_len, HEAD_DIM),
            #     strides=(stride_ot, stride_od),
            #     offsets=(0, 0),
            #     block_shape=(BLOCK_M, BLOCK_D),
            #     order=(1, 0),
            # )
            q_mask = gen_mask_m_right_bound(
                aux_mask_ptr_10t,
                aux_mask_size,
                stride_mask_m,
                stride_mask_n,
                BLOCK_M,
                BLOCK_N,
                q_block_id * BLOCK_M,
                q_seq_len,
            )
            q_block_start = q_block_id * BLOCK_M
            q_block_end = min(q_block_start + BLOCK_M, q_seq_len)
            q_block_len = q_block_end - q_block_start
            # cur_q_block_ptr = tl.advance(q_block_ptr, (q_block_start.to(tl.int32), 0))
            cur_q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
                shape=(HEAD_DIM, q_seq_len),
                strides=(stride_qd, stride_qt),
                offsets=(0, q_block_start.to(tl.int32)),
                block_shape=(BLOCK_D, BLOCK_M),
                order=(0, 1),
            )
            cur_q_block = tl.load(cur_q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((HEAD_DIM, BLOCK_M), dtype=tl.float32)

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
                kv_mask = gen_mask_n_right_bound(
                    aux_mask_ptr_10,
                    aux_mask_size,
                    stride_mask_m,
                    stride_mask_n,
                    BLOCK_M,
                    BLOCK_N,
                    kv_block_start,
                    kv_seq_len,
                )
                if IS_CAUSAL:
                    # actually, it must be true for global window blocks
                    mask_gw = gen_mask_n_right_bound(
                        aux_mask_ptr_10,
                        aux_mask_size,
                        stride_mask_m,
                        stride_mask_n,
                        BLOCK_M,
                        BLOCK_N,
                        kv_block_start,
                        GLOBAL_WINDOW,
                    )
                    if LOCAL_WINDOW is not None:
                        mask_sw = gen_mask_triu(
                            aux_mask_ptr_triu,
                            aux_mask_size,
                            stride_mask_m,
                            stride_mask_n,
                            BLOCK_M,
                            BLOCK_N,
                            q_block_start + kv_computed_len,
                            kv_block_start + LOCAL_WINDOW,
                        )
                        mask_gw = mask_gw | mask_sw
                    mask_causal = gen_mask_tril(
                        aux_mask_ptr_tril,
                        aux_mask_size,
                        stride_mask_m,
                        stride_mask_n,
                        BLOCK_M,
                        BLOCK_N,
                        q_block_start + kv_computed_len,
                        kv_block_start,
                    )
                    mask_causal = mask_gw & mask_causal
                    mask = mask_causal & q_mask & kv_mask
                else:
                    mask = q_mask & kv_mask
                cur_k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + physical_page_id * stride_kp + kv_head_id * stride_kh + kv_block_start_in_page * stride_kt,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                cur_v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + physical_page_id * stride_vp + kv_head_id * stride_vh + kv_block_start_in_page * stride_vt,
                    shape=(HEAD_DIM, kv_block_len),
                    strides=(stride_vd, stride_vt),
                    offsets=(0, 0),
                    block_shape=(BLOCK_D, BLOCK_N),
                    order=(0, 1),
                )
                acc, l_i, m_i = _sdpa_acc_fwd_MxN(
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

            for kv_block_id in range(non_global_window_start_block, num_total_blocks):
                kv_block_start = kv_block_id * BLOCK_N
                kv_block_end = min(kv_block_start + BLOCK_N, kv_seq_len)
                kv_block_len = kv_block_end - kv_block_start
                logical_page_id = kv_block_start // PAGE_SIZE
                kv_block_start_in_page = kv_block_start % PAGE_SIZE
                physical_page_id = tl.load(
                    block_table_ptr + b_id * stride_block_table_b + logical_page_id * stride_block_table_p
                )
                kv_mask = gen_mask_n_right_bound(
                    aux_mask_ptr_10,
                    aux_mask_size,
                    stride_mask_m,
                    stride_mask_n,
                    BLOCK_M,
                    BLOCK_N,
                    kv_block_start,
                    kv_seq_len,
                )
                if IS_CAUSAL:
                    mask_causal = gen_mask_tril(
                        aux_mask_ptr_tril,
                        aux_mask_size,
                        stride_mask_m,
                        stride_mask_n,
                        BLOCK_M,
                        BLOCK_N,
                        q_block_start + kv_computed_len,
                        kv_block_start,
                    )
                    if LOCAL_WINDOW is not None:
                        mask_sw = gen_mask_triu(
                            aux_mask_ptr_triu,
                            aux_mask_size,
                            stride_mask_m,
                            stride_mask_n,
                            BLOCK_M,
                            BLOCK_N,
                            q_block_start + kv_computed_len,
                            kv_block_start + LOCAL_WINDOW,
                        )
                        mask_causal = mask_causal & mask_sw
                    mask = mask_causal & q_mask & kv_mask
                else:
                    mask = q_mask & kv_mask

                cur_k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + physical_page_id * stride_kp + kv_head_id * stride_kh + kv_block_start_in_page * stride_kt,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                cur_v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + physical_page_id * stride_vp + kv_head_id * stride_vh + kv_block_start_in_page * stride_vt,
                    shape=(HEAD_DIM, kv_block_len),
                    strides=(stride_vd, stride_vt),
                    offsets=(0, 0),
                    block_shape=(BLOCK_D, BLOCK_N),
                    order=(0, 1),
                )
                acc, l_i, m_i = _sdpa_acc_fwd_MxN(
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
            accumulator = acc / l_i[None, :]
            accumulator = tl.trans(accumulator)
            tl.store(cur_o_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1))


def swa_paged_prefill_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,  # [bsz + 1]
    kvlens: torch.Tensor,  # [bsz + 1]
    block_table: torch.Tensor,  # [bsz, num_kv_blocks]
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_size, mask = get_aux_mask()
    bsz = cu_seqlens_q.shape[0] - 1
    tot_q_toks, num_q_heads, head_dim = q.shape
    _, num_kv_heads, page_size, _ = k_cache.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    o = torch.empty_like(q, memory_format=torch.contiguous_format)
    if q.dtype == torch.float32:
        BLOCK_M = 64
        BLOCK_N = min(64, triton.next_power_of_2(page_size))
    else:
        BLOCK_M = 128
        BLOCK_N = min(128, triton.next_power_of_2(page_size))

    BLOCK_D = head_dim
    job_num = get_mlu_total_cores()

    grid = (job_num,)

    _swa_paged_prefill_kernel[grid](
        o,
        q,
        k_cache,
        v_cache,
        bsz,
        cu_seqlens_q,
        kvlens,
        block_table,
        softmax_scale,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        mask,
        mask_size,
        mask.stride(1),
        mask.stride(0),
        IS_CAUSAL=is_causal,
        GLOBAL_WINDOW=global_window_size,
        LOCAL_WINDOW=local_window_size,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        GQA_INTERLEAVE=gqa_interleave,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        PAGE_SIZE=page_size,
        num_warps=1, num_stages=5, force_use_shared_memory=True, bottleneck="simd",
    )
    return o

@triton.jit
def _decode_acc_fwd_MxN(
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
    # -- Compute qk ----

    # Load (transposed) K block
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    k_T = tl.trans(k)
    qk = tl.dot(q, k_T)

    qk = qk * qk_scale
    if mask is not None and mask is not True:
        qk = tl.where(mask, qk, -1e6)  # 32B # bool

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
    alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
    l_i = l_i * alpha + l_ij  # Update softmax denominator
    # -- Update output accumulator --
    acc_ptr = acc_ptr * alpha[:, None]
    acc_ptr = tl.dot(p_cast, v, acc_ptr)

    # Update current block max
    m_i = m_ij

    # Return accumulated output acc_ptr, softmax denominator l_i, and max value m_i
    return acc_ptr, l_i, m_i


@triton.jit
def _swa_paged_decode_kernel(
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
    BLOCK_SIZE_Q_HEADS: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_SIZE_D, "HEAD_DIM should be <= BLOCK_SIZE_D")
    tl.static_assert(PAGE_SIZE % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must be a divisor of PAGE_SIZE")
    GQA_GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    tl.static_assert(GQA_GROUP_SIZE % BLOCK_SIZE_Q_HEADS == 0, "BLOCK_SIZE_Q_HEADS must be a divisor of GQA_GROUP_SIZE")
    GQA_GROUP_STRIDE: tl.constexpr = 1 if GQA_INTERLEAVE else GQA_GROUP_SIZE
    GQA_HEAD_STRIDE: tl.constexpr = NUM_KV_HEADS if GQA_INTERLEAVE else 1
    NUM_Q_HEAD_BLOCKS_PER_KV_HEAD: tl.constexpr = tl.cdiv(GQA_GROUP_SIZE, BLOCK_SIZE_Q_HEADS)


    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    num_tasks = BATCH_SIZE * NUM_KV_HEADS * NUM_Q_HEAD_BLOCKS_PER_KV_HEAD

    for q_task_id in range(pid, num_tasks, n_progs):
        b_id = q_task_id // NUM_Q_HEAD_BLOCKS_PER_KV_HEAD // NUM_KV_HEADS
        kv_head_id = q_task_id // NUM_Q_HEAD_BLOCKS_PER_KV_HEAD % NUM_KV_HEADS
        q_head_block_id = q_task_id % NUM_Q_HEAD_BLOCKS_PER_KV_HEAD

        kv_seq_len = tl.load(seqlens_ptr + b_id)

        offs_d = tl.arange(0, BLOCK_SIZE_D)
        offs_gqa_block = q_head_block_id * BLOCK_SIZE_Q_HEADS + tl.arange(0, BLOCK_SIZE_Q_HEADS)
        offs_head_block = kv_head_id * GQA_GROUP_STRIDE + offs_gqa_block * GQA_HEAD_STRIDE
        q_ptrs = q_ptr + b_id * stride_qb + offs_head_block[:, None] * stride_qh + offs_d[None, :] * stride_qd

        q = tl.load(q_ptrs, mask = (offs_d[None, :] < HEAD_DIM) & (offs_gqa_block[:, None] < GQA_GROUP_SIZE), other = 0.0)

        m_i = tl.zeros((BLOCK_SIZE_Q_HEADS,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((BLOCK_SIZE_Q_HEADS,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_SIZE_Q_HEADS, BLOCK_SIZE_D), dtype=tl.float32)

        num_global_window_blocks, non_global_window_start_block, num_total_blocks = _swa_split_blocks(
            kv_seq_len - 1,
            1,
            kv_seq_len,
            BLOCK_SIZE_N,
            True,
            GLOBAL_WINDOW,
            LOCAL_WINDOW,
        )
        

        for kv_block_id in range(num_global_window_blocks):
            kv_block_start = kv_block_id * BLOCK_SIZE_N
            kv_block_end = min(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
            kv_block_len = kv_block_end - kv_block_start
            logical_page_id = kv_block_start // PAGE_SIZE
            kv_block_start_in_page = kv_block_start % PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
            )
            k_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_k_blksz, stride_k_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head + kv_block_start_in_page * stride_v_blksz,
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
            
            acc, l_i, m_i = _decode_acc_fwd_MxN(
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
            physical_page_id = tl.load(
                block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
            )
            k_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_k_blksz, stride_k_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head + kv_block_start_in_page * stride_v_blksz,
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
            
            acc, l_i, m_i = _decode_acc_fwd_MxN(
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

        if kv_seq_len > 0:
            # avoid division by zero
            acc = acc / l_i[:, None]

        o_ptrs = o_ptr + b_id * stride_ob + offs_head_block[:, None] * stride_oh + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=(offs_d[None, :] < HEAD_DIM) & (offs_gqa_block[:, None] < GQA_GROUP_SIZE))


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
    num_total_blocks, num_kv_heads, page_size, head_dim_cache = key_cache.shape

    max_num_blocks_per_seq = block_tables.shape[1]

    assert head_dim == head_dim_cache
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    o = torch.empty_like(q, memory_format=torch.contiguous_format)
    
    job_num = get_mlu_total_cores()
    grid = (job_num, )
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    BLOCK_SIZE_N = min(256, triton.next_power_of_2(page_size))
    BLOCK_SIZE_Q_HEADS = 1 if gqa_interleave else min(16, num_q_heads // num_kv_heads)
    BLOCK_SIZE_Q_HEADS = math.gcd(BLOCK_SIZE_Q_HEADS, num_q_heads // num_kv_heads)

    _swa_paged_decode_kernel[grid](
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
        page_size,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_Q_HEADS=BLOCK_SIZE_Q_HEADS,
        num_warps=1, num_stages=3,
        pipeline_strategies=["reduce_delay"],
    )
    return o
