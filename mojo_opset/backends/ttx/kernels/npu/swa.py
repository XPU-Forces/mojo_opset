import torch
from typing import Optional, Tuple

import triton
import triton.language as tl

from .utils import get_num_cores

AUX_MASK_SIZE = 256
AUX_MASK = None

_GLOBAL_WINDOW_SIZE = None
_LOCAL_WINDOW_SIZE = None
_COMPRESSED_MASK = None


def get_aux_mask():
    global AUX_MASK
    global AUX_MASK_SIZE
    if AUX_MASK is None:
        AUX_MASK = torch.cat(
            [
                torch.cat(
                    [
                        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE, device="npu", dtype=torch.bool),
                        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE, device="npu", dtype=torch.bool).triu(),
                        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE, device="npu", dtype=torch.bool),
                        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE, device="npu", dtype=torch.bool).tril(),
                        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE, device="npu", dtype=torch.bool),
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE, device="npu", dtype=torch.bool),
                        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE, device="npu", dtype=torch.bool),
                        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE, device="npu", dtype=torch.bool),
                        torch.ones(AUX_MASK_SIZE, AUX_MASK_SIZE, device="npu", dtype=torch.bool),
                        torch.zeros(AUX_MASK_SIZE, AUX_MASK_SIZE, device="npu", dtype=torch.bool),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        )
    return AUX_MASK_SIZE, AUX_MASK


def get_mask_causal_with_window(
        BLOCK_M: int,
        BLOCK_N: int,
        local_window_size: Optional[int] = None,
        global_window_size: Optional[int] = None,
        device: str = "npu",
):
    global _GLOBAL_WINDOW_SIZE
    global _LOCAL_WINDOW_SIZE
    global _COMPRESSED_MASK
    if (
        _GLOBAL_WINDOW_SIZE == global_window_size
        and _LOCAL_WINDOW_SIZE == local_window_size
        and _COMPRESSED_MASK is not None
    ):
        return _COMPRESSED_MASK

    if local_window_size is None:
        local_window_size = 0
    if global_window_size is None:
        global_window_size = 0

    M = (global_window_size + local_window_size + 2 * max(BLOCK_M, BLOCK_N) + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    N = (global_window_size + local_window_size + 3 * max(BLOCK_M, BLOCK_N) + BLOCK_N - 1) // BLOCK_N * BLOCK_N

    causal = torch.ones(M, N, dtype=torch.bool).tril()

    sink_band = torch.zeros(M, N, dtype=torch.bool)
    sink_band[:, :global_window_size] = True

    local_band = torch.ones(M, N, dtype=torch.bool).triu(diagonal=-local_window_size)

    mask = causal & (sink_band | local_band)

    M_boundary = M + BLOCK_M
    N_boundary = N + BLOCK_N
    mask_boundary = torch.zeros(M_boundary, N_boundary, dtype=torch.bool)
    mask_boundary[:M, :N] = mask
    mask_boundary = mask_boundary.to(device=device)

    _GLOBAL_WINDOW_SIZE = global_window_size
    _LOCAL_WINDOW_SIZE = local_window_size
    _COMPRESSED_MASK = mask_boundary
    return _COMPRESSED_MASK


@triton.jit
def gen_mask_causal_with_window(mask_ptr_causal, mask_size_m, mask_size_n, M_BLOCK, N_BLOCK, m_start, n_start,
                                global_window_size, local_windows_size, q_seq_len, kv_seq_len, GEN_MASK_BLOCK_SIZE=256):
    if local_windows_size is None:
        local_windows_size = 0
    if global_window_size is None:
        global_window_size = 0

    actual_mask_m = mask_size_m - GEN_MASK_BLOCK_SIZE
    is_q_oob = (m_start >= kv_seq_len).to(tl.int32)
    valid_rows = max(0, min(kv_seq_len - m_start, M_BLOCK))
    is_tail = (valid_rows < M_BLOCK).to(tl.int32)

    m_pos_normal = min(m_start, actual_mask_m - M_BLOCK)
    m_pos_tail = actual_mask_m - valid_rows
    m_pos = (1 - is_q_oob) * ((1 - is_tail) * m_pos_normal + is_tail * m_pos_tail) + is_q_oob * actual_mask_m

    shift = m_start - m_pos
    need_adjust = (shift != 0).to(tl.int32)
    is_global_block = (n_start < global_window_size).to(tl.int32)
    can_compensate = ((m_start - n_start) <= local_windows_size).to(tl.int32)
    need_adjust = need_adjust * ((1 - is_global_block) + is_global_block * can_compensate)
    n_pos = ((1 - need_adjust) * n_start + need_adjust * max(global_window_size + 1, n_start - shift)) * (1 - is_q_oob)

    mask = tl.load(
        mask_ptr_causal
        + (m_pos + tl.arange(0, M_BLOCK)[:, None]) * mask_size_n
        + (n_pos + tl.arange(0, N_BLOCK))[None, :]
    )
    return mask


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
        + tl.arange(0, M_BLOCK)[:, None] * mask_stride_m
        + (offset + tl.arange(0, N_BLOCK))[None, :] * mask_stride_n
    )
    return mask


@triton.jit
def gen_mask_n_left_bound(mask_01_ptr, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, n_start, left):
    # tl.arange(n_start, n_start + N_BLOCK)[None, :] >= left
    offset = min(max(n_start - left, -mask_size), 0)
    mask = tl.load(
        mask_01_ptr
        + tl.arange(0, M_BLOCK)[:, None] * mask_stride_m
        + (offset + tl.arange(0, N_BLOCK))[None, :] * mask_stride_n
    )
    return mask


@triton.jit
def gen_mask_m_right_bound(mask_10t_ptr, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, m_start, right):
    # tl.arange(m_start, m_start + M_BLOCK)[:, None] < right
    offset = min(max(m_start - right, -mask_size), 0)
    mask = tl.load(
        mask_10t_ptr
        + (offset + tl.arange(0, M_BLOCK)[:, None]) * mask_stride_m
        + tl.arange(0, N_BLOCK)[None, :] * mask_stride_n
    )
    return mask


@triton.jit
def gen_mask_m_left_bound(mask_01t_ptr, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, m_start, left):
    # tl.arange(m_start, m_start + M_BLOCK)[:, None] >= left
    offset = min(max(m_start - left, -mask_size), 0)
    mask = tl.load(
        mask_01t_ptr
        + (offset + tl.arange(0, M_BLOCK)[:, None]) * mask_stride_m
        + tl.arange(0, N_BLOCK)[None, :] * mask_stride_n
    )
    return mask


@triton.jit
def gen_mask_tril(mask_ptr_tril, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, m_start, n_start):
    # tl.arange(n_start, n_start + N_BLOCK)[None, :] <= tl.arange(m_start, m_start + M_BLOCK)[:, None]
    offset = min(max(n_start - m_start, -mask_size), mask_size)
    mask = tl.load(
        mask_ptr_tril
        + tl.arange(0, M_BLOCK)[:, None] * mask_stride_m
        + (offset + tl.arange(0, N_BLOCK))[None, :] * mask_stride_n
    )
    return mask


@triton.jit
def gen_mask_triu(mask_ptr_triu, mask_size, mask_stride_m, mask_stride_n, M_BLOCK, N_BLOCK, m_start, n_start):
    # tl.arange(n_start, n_start + N_BLOCK)[None, :] >= tl.arange(m_start, m_start + M_BLOCK)[:, None]
    len_offset = min(max(n_start - m_start, -mask_size), mask_size)
    mask = tl.load(
        mask_ptr_triu
        + tl.arange(0, M_BLOCK)[:, None] * mask_stride_m
        + (len_offset + tl.arange(0, N_BLOCK))[None, :] * mask_stride_n
    )
    return mask


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
    k_T = tl.trans(k)
    qk = tl.dot(q, k_T)
    # tl.compile_hint(qk, "tile_cube_loop")

    qk = qk * qk_scale
    if mask is not None and mask is not True:
        qk = tl.where(mask, qk, -1e6)  # 32B # bool

    m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)  # Scaled max
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
    # tl.compile_hint(acc_ptr, "tile_cube_loop")

    # Update current block max
    m_i = m_ij

    # NOTE(zhangjihang): for training
    # Return accumulated output acc_ptr, softmax denominator l_i, and max value m_i
    return acc_ptr, l_i, m_i


@triton.jit
def _sdpa_infer_kernel(
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
):
    tl.static_assert(HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM")
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
            # k_block_ptr = tl.make_block_ptr(
            #     base=k_ptr + kv_start * stride_kt + kv_head_id * stride_kh,
            #     shape=(kv_seq_len, HEAD_DIM),
            #     strides=(stride_kt, stride_kd),
            #     offsets=(0, 0),
            #     block_shape=(BLOCK_N, BLOCK_D),
            #     order=(1, 0),
            # )
            # v_block_ptr = tl.make_block_ptr(
            #     base=v_ptr + kv_start * stride_vt + kv_head_id * stride_vh,
            #     shape=(kv_seq_len, HEAD_DIM),
            #     strides=(stride_vt, stride_vd),
            #     offsets=(0, 0),
            #     block_shape=(BLOCK_N, BLOCK_D),
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
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_qt, stride_qd),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            cur_q_block = tl.load(cur_q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

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
                # cur_k_block_ptr = tl.advance(k_block_ptr, (kv_block_start.to(tl.int32), 0))
                cur_k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + kv_start * stride_kt + kv_head_id * stride_kh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                # cur_v_block_ptr = tl.advance(v_block_ptr, (kv_block_start.to(tl.int32), 0))
                cur_v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + kv_start * stride_vt + kv_head_id * stride_vh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
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
                # cur_k_block_ptr = tl.advance(k_block_ptr, (kv_block_start.to(tl.int32), 0))
                cur_k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + kv_start * stride_kt + kv_head_id * stride_kh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                # cur_v_block_ptr = tl.advance(v_block_ptr, (kv_block_start.to(tl.int32), 0))
                cur_v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + kv_start * stride_vt + kv_head_id * stride_vh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
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
            accumulator = acc / l_i[:, None]
            tl.store(cur_o_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1))


def swa_infer_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,  # [bsz + 1]
    cu_total_seq_lens: torch.Tensor,  # [bsz + 1]
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_size, mask = get_aux_mask()
    bsz = cu_q_lens.shape[0] - 1
    tot_q_toks, num_q_heads, head_dim = q.shape
    tot_kv_toks, num_kv_heads, _ = k.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    o = torch.zeros_like(q, memory_format=torch.contiguous_format)

    if q.dtype == torch.float32:
        BLOCK_M = 64
        BLOCK_N = 64
    else:
        BLOCK_M = 128
        BLOCK_N = 128
    BLOCK_D = head_dim

    cube_num = get_num_cores("cube")
    grid = (cube_num,)

    _sdpa_infer_kernel[grid](
        o,
        q,
        k,
        v,
        bsz,
        cu_q_lens,
        cu_total_seq_lens,
        softmax_scale,
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
        mask,
        mask_size,
        mask.stride(0),
        mask.stride(1),
        is_causal,
        global_window_size,
        local_window_size,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
    )
    return o


@triton.jit
def _swa_paged_prefill_kernel(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
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
    causal_mask_ptr,
    causal_mask_m_size: tl.constexpr,
    causal_mask_n_size: tl.constexpr,
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

    cu_q_chunks = 0
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
        for q_task_id in range((n_programs - prev_q_tasks % n_programs + pid) % n_programs, new_q_tasks, n_programs):
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

            q_block_start = q_block_id * BLOCK_M
            q_block_end = min(q_block_start + BLOCK_M, q_seq_len)
            q_block_len = q_block_end - q_block_start
            # cur_q_block_ptr = tl.advance(q_block_ptr, (q_block_start.to(tl.int32), 0))
            cur_q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_qt, stride_qd),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            cur_q_block = tl.load(cur_q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

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

                if IS_CAUSAL:
                    mask = gen_mask_causal_with_window(
                        causal_mask_ptr,
                        causal_mask_m_size,
                        causal_mask_n_size,
                        BLOCK_M,
                        BLOCK_N,
                        q_block_start + kv_computed_len,
                        kv_block_start,
                        GLOBAL_WINDOW,
                        LOCAL_WINDOW,
                        q_seq_len,
                        kv_seq_len,
                        GEN_MASK_BLOCK_SIZE=128,
                    )
                else:
                    mask = tl.full((BLOCK_M, BLOCK_N), 1,  dtype=tl.int1)

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
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
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

                if IS_CAUSAL:
                    mask = gen_mask_causal_with_window(
                        causal_mask_ptr,
                        causal_mask_m_size,
                        causal_mask_n_size,
                        BLOCK_M,
                        BLOCK_N,
                        q_block_start + kv_computed_len,
                        kv_block_start,
                        GLOBAL_WINDOW,
                        LOCAL_WINDOW,
                        q_seq_len,
                        kv_seq_len,
                        GEN_MASK_BLOCK_SIZE=128,
                    )
                else:
                    mask = tl.full((BLOCK_M, BLOCK_N), 1,  dtype=tl.int1)

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
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
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
            accumulator = acc / l_i[:, None]
            tl.store(cur_o_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1))


@triton.jit
def _swa_paged_prefill_small_kernel(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
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
    causal_mask_ptr,
    causal_mask_m_size: tl.constexpr,
    causal_mask_n_size: tl.constexpr,
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


    cu_q_chunks = 0
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
        for q_task_id in range((n_programs - prev_q_tasks % n_programs + pid) % n_programs, new_q_tasks, n_programs):
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
            q_block_start = q_block_id * BLOCK_M
            q_block_end = min(q_block_start + BLOCK_M, q_seq_len)
            q_block_len = q_block_end - q_block_start
            # cur_q_block_ptr = tl.advance(q_block_ptr, (q_block_start.to(tl.int32), 0))
            cur_q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_qt, stride_qd),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            cur_q_block = tl.load(cur_q_block_ptr, boundary_check=(0,), padding_option="zero")

            m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

            num_global_window_blocks, non_global_window_start_block, num_total_blocks = _swa_split_blocks(
                q_block_start + kv_computed_len,
                q_block_len,
                kv_seq_len,
                BLOCK_N,
                IS_CAUSAL,
                GLOBAL_WINDOW,
                LOCAL_WINDOW,
            )

            for kv_block_id in range(num_total_blocks):
                kv_block_start = kv_block_id * BLOCK_N
                kv_block_end = min(kv_block_start + BLOCK_N, kv_seq_len)
                kv_block_len = kv_block_end - kv_block_start
                logical_page_id = kv_block_start // PAGE_SIZE
                kv_block_start_in_page = kv_block_start % PAGE_SIZE
                physical_page_id = tl.load(
                    block_table_ptr + b_id * stride_block_table_b + logical_page_id * stride_block_table_p
                )

                if IS_CAUSAL:
                    mask = gen_mask_causal_with_window(
                        causal_mask_ptr,
                        causal_mask_m_size,
                        causal_mask_n_size,
                        BLOCK_M,
                        BLOCK_N,
                        q_block_start + kv_computed_len,
                        kv_block_start,
                        GLOBAL_WINDOW,
                        LOCAL_WINDOW,
                        q_seq_len,
                        kv_seq_len,
                        GEN_MASK_BLOCK_SIZE=128,
                    )
                else:
                    mask = tl.full((BLOCK_M, BLOCK_N), 1,  dtype=tl.int1)

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
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(0, 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                if mask is not False:
                    k = tl.load(cur_k_block_ptr, boundary_check=(0,), padding_option="zero")
                    k_T = tl.trans(k)
                    qk = tl.dot(cur_q_block, k_T)
                    qk = qk * scale
                    if mask is not None and mask is not True:
                        qk = tl.where(mask, qk, -1e6)
                    m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
                    qk = qk - m_ij[:, None]
                    p = tl.math.exp(qk)
                    p_cast = p.to(k_T.dtype)
                    v = tl.load(cur_v_block_ptr, boundary_check=(0, ), padding_option="zero")
                    l_ij = tl.sum(p, 1)
                    alpha = tl.math.exp(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, None]
                    acc = tl.dot(p_cast, v, acc)
                    m_i = m_ij


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
            tl.store(cur_o_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, ))



def swa_paged_prefill_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,  # [bsz + 1]
    kvlens: torch.Tensor,  # [bsz + 1]
    block_table: torch.Tensor,  # [bsz, num_kv_blocks]
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:

    bsz = cu_q_lens.shape[0] - 1
    tot_q_toks, num_q_heads, head_dim = q.shape
    _, num_kv_heads, page_size, _ = k_cache.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    o = torch.zeros_like(q, memory_format=torch.contiguous_format)
    if q.dtype == torch.float32:
        BLOCK_M = min(64, triton.next_power_of_2(tot_q_toks))
        BLOCK_N = min(64, triton.next_power_of_2(page_size))
    else:
        BLOCK_M = min(128, triton.next_power_of_2(tot_q_toks))
        BLOCK_N = min(128, triton.next_power_of_2(page_size))

    BLOCK_D = head_dim
    cube_num = get_num_cores("cube")

    num_q_chunks = triton.cdiv(tot_q_toks, BLOCK_M)
    num_tasks = num_q_chunks * num_q_heads
    grid = (min(cube_num, num_tasks),)
    use_swa_paged_prefill_kernel = _swa_paged_prefill_kernel
    max_kv_len = kvlens.max()
    if max_kv_len < global_window_size + local_window_size + BLOCK_N * 4:
        use_swa_paged_prefill_kernel = _swa_paged_prefill_small_kernel

    if global_window_size is None:
        global_window_size = 0

    causal_mask = get_mask_causal_with_window(
        BLOCK_M,
        BLOCK_N,
        local_window_size,
        global_window_size
    )
    causal_mask_m_size, causal_mask_n_size = causal_mask.shape

    use_swa_paged_prefill_kernel[grid](
        o,
        q,
        k_cache,
        v_cache,
        bsz,
        cu_q_lens,
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
        causal_mask,
        causal_mask_m_size,
        causal_mask_n_size,
        is_causal,
        global_window_size,
        local_window_size,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        page_size,
    )
    return o


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
    # -- Compute qk ----
                    
    # Load K block
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    qk = tl.sum((q[None, :] * k).to(tl.float32), axis=1)

    qk = qk * qk_scale
    if mask is not None and mask is not True:
        qk = tl.where(mask, qk, float("-inf"))  # 32B # bool

    m_ij = tl.maximum(m_i, tl.max(qk, 0))  # Scaled max
    qk = qk - m_ij  # Stabilize

    # Softmax weights p = exp(qk)
    p = tl.math.exp(qk)

    p_cast = p.to(k.dtype)

    # Load corresponding V block
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Softmax denominator (sum of each row)
    l_ij = tl.sum(p, axis=0)
    # -- Update m_i and l_i
    alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
    l_i = l_i * alpha + l_ij  # Update softmax denominator
    # -- Update output accumulator --
    acc_ptr = acc_ptr * alpha
    acc_ptr += tl.sum((p_cast[:, None] * v).to(tl.float32), axis=0)

    # Update current block max
    m_i = m_ij

    # NOTE(zhangjihang): for training
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
):
    GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    tl.static_assert(HEAD_DIM <= BLOCK_SIZE_D, "HEAD_DIM should be <= BLOCK_SIZE_D")
    tl.static_assert(PAGE_SIZE % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must be a divisor of PAGE_SIZE")

    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    num_tasks = BATCH_SIZE * NUM_KV_HEADS

    for kv_task_id in range(pid, num_tasks, n_progs):
        kv_head_id = kv_task_id % NUM_KV_HEADS
        b_id = kv_task_id // NUM_KV_HEADS

        kv_seq_len = tl.load(seqlens_ptr + b_id)

        g_offsets = tl.arange(0, GROUP_SIZE)
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + g_offsets * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GROUP_SIZE + g_offsets

        offs_d = tl.arange(0, BLOCK_SIZE_D)
        q_ptrs = q_ptr + b_id * stride_qb + q_head_ids[:, None] * stride_qh + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=offs_d[None, :] < HEAD_DIM, other=0.0)

        m_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        acc = tl.zeros((GROUP_SIZE, BLOCK_SIZE_D), dtype=tl.float32)

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
            K_T_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
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

            k_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k_T)
            qk = qk * softmax_scale + tl.where(mask[None, :], 0.0, -2.0**30)

            m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
            p = tl.math.exp(qk - m_ij[:, None])

            pv = tl.dot(p.to(k_T.dtype), v)

            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + pv

            m_i = m_ij

        for kv_block_id in range(non_global_window_start_block, num_total_blocks):
            kv_block_start = kv_block_id * BLOCK_SIZE_N
            kv_block_end = min(kv_block_start + BLOCK_SIZE_N, kv_seq_len)
            kv_block_len = kv_block_end - kv_block_start
            logical_page_id = kv_block_start // PAGE_SIZE
            kv_block_start_in_page = kv_block_start % PAGE_SIZE
            physical_page_id = tl.load(
                block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
            )
            K_T_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
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

            k_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k_T)
            qk = qk * softmax_scale + tl.where(mask[None, :], 0.0, -2.0**30)

            m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
            p = tl.math.exp(qk - m_ij[:, None])

            pv = tl.dot(p.to(k_T.dtype), v)

            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None] + pv

            m_i = m_ij

        if kv_seq_len > 0:
            acc = acc / l_i[:, None]

        o_ptrs = o_ptr + b_id * stride_ob + q_head_ids[:, None] * stride_oh + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_d[None, :] < HEAD_DIM)


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

    cube_num = get_num_cores("cube")
    grid = (cube_num, )
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    BLOCK_SIZE_N = min(128, triton.next_power_of_2(page_size))

    # Note(chenyifan): 
    #   under swa, the kv workload is rather evenly across diffrent queries,
    #   so we have low necessity to apply split-kv strategy             

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
    )
    return o


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN, "multibuffer": MF})
        for BM in [64, 128]
        for BN in [64, 128]
        for MF in [True, False]
    ],
    key=["HEAD_DIM"],
)
@triton.jit
def _swa_fwd_kernel(
    o_ptr,
    o_f32_ptr,
    lse_ptr,
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
    stride_ot_f32,
    stride_oh_f32,
    stride_od_f32,
    stride_lse_h,
    stride_lse_t,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    causal_mask_ptr,
    causal_mask_m_size: tl.constexpr,
    causal_mask_n_size: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUTPUT_F32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM")
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    cu_q_chunks = 0
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

            lse_i_ptr = lse_ptr + q_head_id * stride_lse_h + q_start * stride_lse_t
            q_block_start = q_block_id * BLOCK_M
            q_block_end = min(q_block_start + BLOCK_M, q_seq_len)
            q_block_len = q_block_end - q_block_start
            q_mask = (q_block_start + tl.arange(0, BLOCK_M)[:, None]) < q_seq_len
            # cur_q_block_ptr = tl.advance(q_block_ptr, (q_block_start.to(tl.int32), 0))
            cur_q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_qt, stride_qd),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            cur_q_block = tl.load(cur_q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            num_global_window_blocks, non_global_window_start_block, num_total_blocks = _swa_split_blocks(
                q_block_start + kv_computed_len,
                q_block_len,
                kv_seq_len,
                BLOCK_N,
                IS_CAUSAL,
                GLOBAL_WINDOW,
                LOCAL_WINDOW,
            )

            m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

            # 计算需要处理的 block 总数
            num_blocks_to_process = num_global_window_blocks + (num_total_blocks - non_global_window_start_block)

            for idx in range(0, num_blocks_to_process):
                # 映射 idx 到实际的 kv_block_id（无分支，避免 scf.if 在 NPU 后端的 codegen 问题）
                is_non_global = (idx >= num_global_window_blocks).to(tl.int32)
                kv_block_id = idx + is_non_global * (non_global_window_start_block - num_global_window_blocks)
                kv_block_start = kv_block_id * BLOCK_N
                kv_mask = (kv_block_start + tl.arange(0, BLOCK_N)[None, :]) < kv_seq_len

                if IS_CAUSAL:
                    q_pos = q_block_start + kv_computed_len
                    # kv_block_end = kv_block_start + BLOCK_N
                    # if q_pos < GLOBAL_WINDOW and kv_block_end <= q_pos + 1:
                    #     is_full_mask = True
                    # else:
                    mask = gen_mask_causal_with_window(
                        causal_mask_ptr,
                        causal_mask_m_size,
                        causal_mask_n_size,
                        BLOCK_M,
                        BLOCK_N,
                        q_pos,
                        kv_block_start,
                        GLOBAL_WINDOW,
                        LOCAL_WINDOW,
                        q_seq_len,
                        kv_seq_len,
                    )
                    # mask = q_mask & kv_mask
                else:
                    mask = q_mask & kv_mask
                cur_k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + kv_start * stride_kt + kv_head_id * stride_kh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                cur_v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + kv_start * stride_vt + kv_head_id * stride_vh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=(kv_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
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

            lse_i = m_i + tl.math.log(l_i)
            lse_i_offs = (q_block_start + tl.arange(0, BLOCK_M)).to(tl.int32)
            lse_i_mask = lse_i_offs < q_seq_len
            tl.store(lse_i_ptr + lse_i_offs * stride_lse_t, lse_i, mask=lse_i_mask)
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
            tl.store(cur_o_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1))
            if OUTPUT_F32:
                # cur_o_f32_block_ptr = tl.advance(o_f32_block_ptr, (q_block_start.to(tl.int32), 0))
                cur_o_f32_block_ptr = tl.make_block_ptr(
                    base=o_f32_ptr + q_start * stride_ot_f32 + q_head_id * stride_oh_f32,
                    shape=(q_seq_len, HEAD_DIM),
                    strides=(stride_ot_f32, stride_od_f32),
                    offsets=(q_block_start.to(tl.int32), 0),
                    block_shape=(BLOCK_M, BLOCK_D),
                    order=(1, 0),
                )
                tl.store(cur_o_f32_block_ptr, accumulator, boundary_check=(0, 1))


def swa_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,  # [bsz + 1]
    cu_total_seq_lens: torch.Tensor,  # [bsz + 1]
    is_causal: bool = True,
    local_window_size: Optional[int] = None,
    global_window_size: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
    output_f32: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # mask_size, mask = get_aux_mask()

    if global_window_size is None:
        global_window_size = 0

    causal_mask = get_mask_causal_with_window(
        256,
        256,
        local_window_size,
        global_window_size
    )
    causal_mask_m_size, causal_mask_n_size = causal_mask.shape

    bsz = cu_q_lens.shape[0] - 1
    tot_q_toks, num_q_heads, head_dim = q.shape
    tot_kv_toks, num_kv_heads, _ = k.shape
    o = torch.zeros_like(q, memory_format=torch.contiguous_format)
    softmax_lse = torch.zeros((num_q_heads, tot_q_toks), dtype=torch.float32, device=q.device) + float("-inf")

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    if output_f32:
        o_f32 = torch.zeros_like(q, dtype=torch.float32)
        of32_stride_t, of32_stride_h, of32_stride_d = o_f32.stride(0), o_f32.stride(1), o_f32.stride(2)
    else:
        o_f32 = None
        of32_stride_t, of32_stride_h, of32_stride_d = 0, 0, 0

    BLOCK_D = head_dim

    cube_num = get_num_cores("cube")
    grid = (cube_num,)

    _swa_fwd_kernel[grid](
        o,
        o_f32,
        softmax_lse,
        q,
        k,
        v,
        bsz,
        cu_q_lens,
        cu_total_seq_lens,
        softmax_scale,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        of32_stride_t,
        of32_stride_h,
        of32_stride_d,
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        causal_mask,
        causal_mask_m_size,
        causal_mask_n_size,
        is_causal,
        global_window_size,
        local_window_size,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        BLOCK_D,
        output_f32,
        limit_auto_multi_buffer_buffer="no-limit",
        hfusion_enable_multiple_consumer_fusion=True
    )
    if output_f32:
        return o, softmax_lse, o_f32
    else:
        return o, softmax_lse


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 32}),
        triton.Config(kwargs={"BLOCK_SIZE": 64}),
        triton.Config(kwargs={"BLOCK_SIZE": 128}),
        triton.Config(kwargs={"BLOCK_SIZE": 256}),
        triton.Config(kwargs={"BLOCK_SIZE": 512}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}),
    ],
    key=["HEAD_DIM"],
)
@triton.jit
def _swa_bwd_preprocess(
    d_ptr: torch.Tensor,
    o_ptr: torch.Tensor,
    do_ptr: torch.Tensor,
    num_tokens: int,
    d_stride_h: int,
    d_stride_t: int,
    o_stride_t: int,
    o_stride_h: int,
    o_stride_d: int,
    do_stride_t: int,
    do_stride_h: int,
    do_stride_d: int,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)
    num_blocks = tl.cdiv(num_tokens, BLOCK_SIZE)
    num_tasks = num_blocks * NUM_HEADS
    tl.static_assert(d_ptr.type.element_ty == tl.float32)
    for task_id in range(pid, num_tasks, n_progs):
        block_id = task_id // NUM_HEADS
        head_id = task_id % NUM_HEADS
        t_offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        t_mask = t_offs < num_tokens
        o = tl.load(
            o_ptr + t_offs[:, None] * o_stride_t + head_id * o_stride_h + tl.arange(0, HEAD_DIM)[None, :] * o_stride_d,
            mask=t_mask[:, None],
            other=0.0,
        )
        do = tl.load(
            do_ptr
            + t_offs[:, None] * do_stride_t
            + head_id * do_stride_h
            + tl.arange(0, HEAD_DIM)[None, :] * do_stride_d,
            mask=t_mask[:, None],
            other=0.0,
        )
        delta = tl.sum(o.cast(tl.float32) * do.cast(tl.float32), axis=-1)
        tl.store(d_ptr + head_id * d_stride_h + t_offs * d_stride_t, delta, mask=t_mask)


@triton.jit
def _sdpa_single_block_bwd_dkdv(
    dk_ptr,
    dv_ptr,
    d,
    lse,
    Q_block_ptr,
    DO_block_ptr,
    k,
    v,
    mask,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    fp8_v: tl.constexpr,
):
    if mask is False:
        return dk_ptr, dv_ptr
    # -- Compute qk ----

    # Load (transposed) K block
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    # q_T = tl.trans(q)
    # qkT = tl.dot(k, q_T)  # [BLOCK_N, BLOCK_M]
    k_T = tl.trans(k)
    qk = tl.dot(q, k_T)
    # tl.compile_hint(qk, "tile_cube_loop")
    qk = qk * qk_scale

    # -- Compute p ----
    # Softmax weights p = exp(qk - lse.unsqueeze(1))
    # a.k.a. pT = exp(qkT - lse.unsqueeze(0))
    # scale according to recorded logsumexp
    p = tl.math.exp(qk - lse[:, None])
    if mask is not None and mask is not True:
        p = tl.where(mask, p, 0.0)  # 32B # bool
    p_cast = p.to(q.dtype)

    # -- Compute dV ----
    # dv = pT @ do
    do = tl.load(DO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dv = tl.dot(tl.trans(p_cast), do, dv_ptr)

    # -- Compute dS ----
    # dpT = v @ doT
    # doT = tl.trans(do)
    dp = tl.dot(do, tl.trans(v))

    # dsT = pT * (dpT - dT)
    ds = p * (dp - d[:, None]) * qk_scale
    ds_cast = ds.to(q.dtype)

    # -- Compute dK ----
    # dk = dsT @ q
    dk = tl.dot(tl.trans(ds_cast), q, dk_ptr)

    return dk, dv


@triton.jit
def _sdpa_single_block_bwd_dq(
    dq_ptr,
    d,
    lse,
    q,
    do,
    K_block_ptr,
    V_block_ptr,
    mask,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    fp8_v: tl.constexpr,
):
    if mask is False:
        return dq_ptr
    # -- Compute qk ----

    # Load (transposed) K block
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    k_T = tl.trans(k)
    qk = tl.dot(q, k_T)  # [BLOCK_M, BLOCK_N]
    # tl.compile_hint(qk, "tile_cube_loop")
    qk = qk * qk_scale

    # -- Compute p ----
    # Softmax weights p = exp(qk - lse)
    # scale according to recorded logsumexp
    p = tl.math.exp(qk - lse[:, None])
    if mask is not None and mask is not True:
        p = tl.where(mask, p, 0.0)  # 32B # bool

    # -- Compute dS ----
    # dp = do @ v.T
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    v_T = tl.trans(v)
    dp = tl.dot(do, v_T)

    # ds = p * (dp - d)
    ds = p * (dp - d[:, None]) * qk_scale
    ds_cast = ds.to(q.dtype)

    # -- Compute dK ----
    # dq = ds @ k
    dq = tl.dot(ds_cast, k, dq_ptr)

    return dq


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN, "multibuffer": MF})
        for BM in [64, 128, 256]
        for BN in [64, 128]
        for MF in [True, False]
    ],
    key=["HEAD_DIM", "LOCAL_WINDOW", "GLOBAL_WINDOW"],
)
@triton.jit
def _swa_bwd_dkdv_kernel(
    dk_ptr,
    dv_ptr,
    do_ptr,
    delta_ptr,
    lse_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    bsz,
    cu_q_lens_ptr,
    cu_total_seq_lens_ptr,
    scale,
    stride_dkt,
    stride_dkh,
    stride_dkd,
    stride_dvt,
    stride_dvh,
    stride_dvd,
    stride_dot,
    stride_doh,
    stride_dod,
    stride_delta_h,
    stride_delta_t,
    stride_lse_h,
    stride_lse_t,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    causal_mask_ptr,
    causal_mask_m_size: tl.constexpr,
    causal_mask_n_size: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM")
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    cu_kv_chunks = 0
    for b_id in range(bsz):
        kv_start = tl.load(cu_total_seq_lens_ptr + b_id).to(tl.int32)
        kv_end = tl.load(cu_total_seq_lens_ptr + b_id + 1).to(tl.int32)
        q_start = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)

        q_seq_len = q_end - q_start
        kv_seq_len = kv_end - kv_start
        kv_computed_len = kv_seq_len - q_seq_len

        num_kv_chunks = tl.cdiv(kv_seq_len, BLOCK_N)
        balance_ratio = 1
        gw_kv_chunks = num_kv_chunks
        if GLOBAL_WINDOW is not None and LOCAL_WINDOW is not None:
            # if 2*LOCAL_WINDOW < q_seq_len:
            gw_task = tl.cdiv(q_seq_len, BLOCK_M)
            lw_task = tl.cdiv(LOCAL_WINDOW, BLOCK_M) + 1
            balance_ratio_ = gw_task // lw_task
            if balance_ratio_ >= 2:
                gw_kv_chunks = tl.cdiv(GLOBAL_WINDOW, BLOCK_N)
                gw_kv_chunks = min(gw_kv_chunks, num_kv_chunks)
                lw_kv_chunks = tl.cdiv(num_kv_chunks - gw_kv_chunks, balance_ratio_)
                num_kv_chunks = gw_kv_chunks + lw_kv_chunks
                balance_ratio = balance_ratio_

        prev_kv_tasks = cu_kv_chunks * NUM_KV_HEADS
        cu_kv_chunks += num_kv_chunks
        new_kv_tasks = num_kv_chunks * NUM_KV_HEADS
        for kv_task_id in range((prev_kv_tasks + pid) % n_programs, new_kv_tasks, n_programs):
            _kv_block_id = kv_task_id // NUM_KV_HEADS
            kv_head_id = kv_task_id % NUM_KV_HEADS
            is_global_task = _kv_block_id < gw_kv_chunks
            inner_loops = tl.where(is_global_task, 1, balance_ratio)
            # inner_loops = 1
            # if _kv_block_id >= gw_kv_chunks:
            #     inner_loops = balance_ratio
            for inner_id in range(inner_loops):
                # kv_block_id = _kv_block_id
                # if _kv_block_id >= gw_kv_chunks:
                #     kv_block_id = gw_kv_chunks + (_kv_block_id - gw_kv_chunks) * inner_loops + inner_id

                kv_block_id = tl.where(is_global_task, _kv_block_id,
                                       gw_kv_chunks + (_kv_block_id - gw_kv_chunks) * inner_loops + inner_id)
                kv_block_start = kv_block_id * BLOCK_N
                kv_block_end = min(kv_block_start + BLOCK_N, kv_seq_len)
                kv_block_len = kv_block_end - kv_block_start
                if kv_block_len > 0:
                    # cur_k_block_ptr = tl.advance(k_block_ptr, (kv_block_start.to(tl.int32), 0))
                    cur_k_block_ptr = tl.make_block_ptr(
                        base=k_ptr + kv_start * stride_kt + kv_head_id * stride_kh,
                        shape=(kv_seq_len, HEAD_DIM),
                        strides=(stride_kt, stride_kd),
                        offsets=(kv_block_start.to(tl.int32), 0),
                        block_shape=(BLOCK_N, BLOCK_D),
                        order=(1, 0),
                    )
                    cur_k_block = tl.load(cur_k_block_ptr, boundary_check=(0, 1), padding_option="zero")
                    # cur_v_block_ptr = tl.advance(v_block_ptr, (kv_block_start.to(tl.int32), 0))
                    cur_v_block_ptr = tl.make_block_ptr(
                        base=v_ptr + kv_start * stride_vt + kv_head_id * stride_vh,
                        shape=(kv_seq_len, HEAD_DIM),
                        strides=(stride_vt, stride_vd),
                        offsets=(kv_block_start.to(tl.int32), 0),
                        block_shape=(BLOCK_N, BLOCK_D),
                        order=(1, 0),
                    )
                    cur_v_block = tl.load(cur_v_block_ptr, boundary_check=(0, 1), padding_option="zero")

                    start_block, end_block = _swa_transposed_range_blocks(
                        kv_block_start,
                        kv_block_len,
                        kv_computed_len,
                        q_seq_len,
                        BLOCK_M,
                        IS_CAUSAL,
                        GLOBAL_WINDOW,
                        LOCAL_WINDOW,
                    )

                    dk = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)
                    dv = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)

                    # For GQA, iterate over all q_heads
                    for q_head_rpt in range(NUM_Q_HEADS // NUM_KV_HEADS):
                        if GQA_INTERLEAVE:
                            q_head_id = NUM_KV_HEADS * q_head_rpt + kv_head_id
                        else:
                            q_head_id = q_head_rpt + kv_head_id * (NUM_Q_HEADS // NUM_KV_HEADS)

                        lse_i_ptr = lse_ptr + q_head_id * stride_lse_h + q_start * stride_lse_t
                        delta_i_ptr = delta_ptr + q_head_id * stride_delta_h + q_start * stride_delta_t

                        for q_block_id in range(start_block, end_block):
                            q_block_start = q_block_id * BLOCK_M
                            q_mask = (q_block_start + tl.arange(0, BLOCK_M)[:, None]) < q_seq_len
                            kv_mask = (kv_block_start + tl.arange(0, BLOCK_N)[None, :]) < kv_seq_len

                            if IS_CAUSAL:
                                q_pos = q_block_start + kv_computed_len
                                mask = gen_mask_causal_with_window(
                                    causal_mask_ptr,
                                    causal_mask_m_size,
                                    causal_mask_n_size,
                                    BLOCK_M,
                                    BLOCK_N,
                                    q_pos,
                                    kv_block_start,
                                    GLOBAL_WINDOW,
                                    LOCAL_WINDOW,
                                    q_seq_len,
                                    kv_seq_len,
                                )
                            else:
                                mask = q_mask & kv_mask

                            cur_q_block_ptr = tl.make_block_ptr(
                                base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
                                shape=(q_seq_len, HEAD_DIM),
                                strides=(stride_qt, stride_qd),
                                offsets=(q_block_start.to(tl.int32), 0),
                                block_shape=(BLOCK_M, BLOCK_D),
                                order=(1, 0),
                            )
                            cur_do_block_ptr = tl.make_block_ptr(
                                base=do_ptr + q_start * stride_dot + q_head_id * stride_doh,
                                shape=(q_seq_len, HEAD_DIM),
                                strides=(stride_dot, stride_dod),
                                offsets=(q_block_start.to(tl.int32), 0),
                                block_shape=(BLOCK_M, BLOCK_D),
                                order=(1, 0),
                            )
                            q_offs = q_block_start + tl.arange(0, BLOCK_M)

                            cur_delta = tl.load(delta_i_ptr + q_offs * stride_delta_t, mask=q_offs < q_seq_len,
                                                other=0.0)
                            tl.static_assert(cur_delta.dtype == tl.float32)
                            cur_lse = tl.load(lse_i_ptr + q_offs * stride_lse_t, mask=q_offs < q_seq_len,
                                              other=-float("inf"))
                            tl.static_assert(cur_lse.dtype == tl.float32)
                            dk, dv = _sdpa_single_block_bwd_dkdv(
                                dk,
                                dv,
                                cur_delta,
                                cur_lse,
                                cur_q_block_ptr,
                                cur_do_block_ptr,
                                cur_k_block,
                                cur_v_block,
                                mask,
                                scale,
                                HEAD_DIM,
                                BLOCK_M,
                                BLOCK_N,
                                BLOCK_D,
                                v_ptr.dtype.element_ty == tl.float8e5,
                            )
                    # tl.extra.cann.extension.compile_hint(dk, "matmul_at_least_once")
                    # tl.extra.cann.extension.compile_hint(dv, "matmul_at_least_once")
                    # cur_dk_block_ptr = tl.advance(dk_block_ptr, (kv_block_start.to(tl.int32), 0))
                    cur_dk_block_ptr = tl.make_block_ptr(
                        base=dk_ptr + kv_start * stride_dkt + kv_head_id * stride_dkh,
                        shape=(kv_seq_len, HEAD_DIM),
                        strides=(stride_dkt, stride_dkd),
                        offsets=(kv_block_start.to(tl.int32), 0),
                        block_shape=(BLOCK_N, BLOCK_D),
                        order=(1, 0),
                    )
                    tl.store(cur_dk_block_ptr, dk.to(dk_ptr.type.element_ty), boundary_check=(0, 1))
                    # cur_dv_block_ptr = tl.advance(dv_block_ptr, (kv_block_start.to(tl.int32), 0))
                    cur_dv_block_ptr = tl.make_block_ptr(
                        base=dv_ptr + kv_start * stride_dvt + kv_head_id * stride_dvh,
                        shape=(kv_seq_len, HEAD_DIM),
                        strides=(stride_dvt, stride_dvd),
                        offsets=(kv_block_start.to(tl.int32), 0),
                        block_shape=(BLOCK_N, BLOCK_D),
                        order=(1, 0),
                    )
                    tl.store(cur_dv_block_ptr, dv.to(dv_ptr.type.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN, "multibuffer": MF})
        for BM in [64, 128]
        for BN in [64, 128]
        for MF in [True, False]
    ],
    key=["HEAD_DIM"],
)
@triton.jit
def _swa_bwd_dq_kernel(
    dq_ptr,
    do_ptr,
    delta_ptr,
    lse_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    bsz,
    cu_q_lens_ptr,
    cu_total_seq_lens_ptr,
    scale,
    stride_dqt,
    stride_dqh,
    stride_dqd,
    stride_dot,
    stride_doh,
    stride_dod,
    stride_delta_h,
    stride_delta_t,
    stride_lse_h,
    stride_lse_t,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    causal_mask_ptr,
    causal_mask_m_size: tl.constexpr,
    causal_mask_n_size: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GLOBAL_WINDOW: tl.constexpr,
    LOCAL_WINDOW: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM")
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    cu_q_chunks = 0
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

            delta_i_ptr = delta_ptr + q_start * stride_delta_t + q_head_id * stride_delta_h
            lse_i_ptr = lse_ptr + q_head_id * stride_lse_h + q_start * stride_lse_t

            q_block_start = q_block_id * BLOCK_M
            q_block_end = min(q_block_start + BLOCK_M, q_seq_len)
            q_block_len = q_block_end - q_block_start
            q_mask = (q_block_start + tl.arange(0, BLOCK_M)[:, None]) < q_seq_len
            # cur_q_block_ptr = tl.advance(q_block_ptr, (q_block_start.to(tl.int32), 0))
            cur_q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_start * stride_qt + q_head_id * stride_qh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_qt, stride_qd),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            cur_q_block = tl.load(cur_q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # cur_do_block_ptr = tl.advance(do_block_ptr, (q_block_start.to(tl.int32), 0))
            cur_do_block_ptr = tl.make_block_ptr(
                base=do_ptr + q_start * stride_dot + q_head_id * stride_doh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_dot, stride_dod),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            cur_do_block = tl.load(cur_do_block_ptr, boundary_check=(0, 1), padding_option="zero")

            q_offs = q_block_start + tl.arange(0, BLOCK_M)
            cur_delta = tl.load(delta_i_ptr + q_offs, q_offs < q_seq_len, other=0.0)
            cur_lse = tl.load(lse_i_ptr + q_offs, q_offs < q_seq_len, other=-float("inf"))

            num_global_window_blocks, non_global_window_start_block, num_total_blocks = _swa_split_blocks(
                q_block_start + kv_computed_len,
                q_block_len,
                kv_seq_len,
                BLOCK_N,
                IS_CAUSAL,
                GLOBAL_WINDOW,
                LOCAL_WINDOW,
            )
            dq = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

            # 计算需要处理的 block 总数
            num_blocks_to_process = num_global_window_blocks + (num_total_blocks - non_global_window_start_block)

            for idx in range(0, num_blocks_to_process):
                # 映射 idx 到实际的 kv_block_id（无分支，避免 scf.if 在 NPU 后端的 codegen 问题）
                is_non_global = (idx >= num_global_window_blocks).to(tl.int32)
                kv_block_id = idx + is_non_global * (non_global_window_start_block - num_global_window_blocks)

                # for kv_block_id in range(num_global_window_blocks):
                kv_block_start = kv_block_id * BLOCK_N
                kv_mask = (kv_block_start + tl.arange(0, BLOCK_N)[None, :]) < kv_seq_len
                if IS_CAUSAL:
                    q_pos = q_block_start + kv_computed_len
                    mask = gen_mask_causal_with_window(
                        causal_mask_ptr,
                        causal_mask_m_size,
                        causal_mask_n_size,
                        BLOCK_M,
                        BLOCK_N,
                        q_pos,
                        kv_block_start,
                        GLOBAL_WINDOW,
                        LOCAL_WINDOW,
                        q_seq_len,
                        kv_seq_len,
                    )
                else:
                    mask = q_mask & kv_mask
                cur_k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + kv_start * stride_kt + kv_head_id * stride_kh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_kt, stride_kd),
                    offsets=((kv_block_id * BLOCK_N).to(tl.int32), 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                # cur_v_block_ptr = tl.advance(v_block_ptr, ((kv_block_id * BLOCK_N).to(tl.int32), 0))
                cur_v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + kv_start * stride_vt + kv_head_id * stride_vh,
                    shape=(kv_seq_len, HEAD_DIM),
                    strides=(stride_vt, stride_vd),
                    offsets=((kv_block_id * BLOCK_N).to(tl.int32), 0),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(1, 0),
                )
                dq = _sdpa_single_block_bwd_dq(
                    dq,
                    cur_delta,
                    cur_lse,
                    cur_q_block,
                    cur_do_block,
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

            # cur_dq_block_ptr = tl.advance(dq_block_ptr, (q_block_start.to(tl.int32), 0))
            cur_dq_block_ptr = tl.make_block_ptr(
                base=dq_ptr + q_start * stride_dqt + q_head_id * stride_dqh,
                shape=(q_seq_len, HEAD_DIM),
                strides=(stride_dqt, stride_dqd),
                offsets=(q_block_start.to(tl.int32), 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            tl.store(cur_dq_block_ptr, dq.to(dq_ptr.type.element_ty), boundary_check=(0, 1))


def swa_bwd_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_total_seq_lens: torch.Tensor,
    is_causal: bool,
    local_window_size: Optional[int],
    global_window_size: Optional[int],
    softmax_scale: Optional[float],
    gqa_interleave: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # mask_size, mask = get_aux_mask()

    causal_mask = get_mask_causal_with_window(
        256,
        256,
        local_window_size,
        global_window_size
    )
    causal_mask_m_size, causal_mask_n_size = causal_mask.shape

    bsz = cu_q_lens.shape[0] - 1
    tot_q_toks, num_q_heads, head_dim = q.shape
    tot_kv_toks, num_kv_heads, _ = k.shape

    delta = torch.zeros((num_q_heads, tot_q_toks), dtype=torch.float32, device=q.device)
    o = o.contiguous()
    do = do.contiguous()

    num_vecs = get_num_cores("vector")
    _swa_bwd_preprocess[(num_vecs,)](
        delta,
        o,
        do,
        tot_q_toks,
        delta.stride(0),
        delta.stride(1),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        num_q_heads,
        head_dim,
    )
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    dq = torch.zeros_like(q, memory_format=torch.contiguous_format)
    dk = torch.zeros_like(k, memory_format=torch.contiguous_format)
    dv = torch.zeros_like(v, memory_format=torch.contiguous_format)

    BLOCK_D = head_dim
    cube_num = get_num_cores("cube")

    grid = (cube_num,)

    _swa_bwd_dkdv_kernel[grid](
        dk,
        dv,
        do,
        delta,
        softmax_lse,
        q,
        k,
        v,
        bsz,
        cu_q_lens,
        cu_total_seq_lens,
        softmax_scale,
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        delta.stride(0),
        delta.stride(1),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        causal_mask,
        causal_mask_m_size,
        causal_mask_n_size,
        is_causal,
        global_window_size,
        local_window_size,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        BLOCK_D,
    )
    _swa_bwd_dq_kernel[grid](
        dq,
        do,
        delta,
        softmax_lse,
        q,
        k,
        v,
        bsz,
        cu_q_lens,
        cu_total_seq_lens,
        softmax_scale,
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        delta.stride(0),
        delta.stride(1),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        causal_mask,
        causal_mask_m_size,
        causal_mask_n_size,
        is_causal,
        global_window_size,
        local_window_size,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        BLOCK_D,
    )

    return dq, dk, dv
