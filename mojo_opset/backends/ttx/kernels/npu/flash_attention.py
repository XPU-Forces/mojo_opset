# Copyright 2026, The FlagOS Contributors.

import math

from typing import Optional

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores
from mojo_opset.backends.ttx.kernels.utils import prepare_chunk_indices

try:
    import triton.experimental.tle as tle
    HAS_TLE = True
except Exception:
    HAS_TLE = False


@triton.jit
def causal_mask_fn(mask_ptr, mask_size, mask_stride_m, mask_stride_n, q_start, kv_start, Q_BLOCK, KV_BLOCK):
    offset_causal = min(max(kv_start - q_start, -mask_size), mask_size)
    offsets_mask_causal = (
        (tl.arange(0, Q_BLOCK)[:, None]) * mask_stride_m
        + (mask_size + offset_causal + tl.arange(0, KV_BLOCK)[None, :]) * mask_stride_n
    )
    mask_causal = tl.load(mask_ptr + offsets_mask_causal).to(tl.int1)

    return mask_causal


@triton.jit
def _sdpa_infer_single_block(
    acc_ptr,
    l_i,
    m_i,
    q,  # Accumulator, local l, local m, query vector
    K_T_block_ptr,
    V_block_ptr,  # Key and value block pointers for current stage
    qk_scale,
    mask,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    fp8_v: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_D, "BLOCK_SIZE_D should not be less than HEAD_DIM")
    # -- Compute qk ----

    # Load (transposed) K block
    k_T = tl.load(K_T_block_ptr, boundary_check=(0, 1), padding_option="zero")
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
def paged_prefill_kernel(
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    o_ptr,
    aux_mask_ptr,
    batch_size,
    cu_q_lens_ptr,
    seqlens_kv_ptr,
    block_tables_ptr,
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
    stride_mask_m,
    stride_mask_n,
    softmax_scale,
    AUX_MASK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    tl.static_assert(PAGE_SIZE % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must be a divisor of PAGE_SIZE")

    prev_q_chunks = 0

    for b_id in range(batch_size):
        q_start_loc = tl.load(cu_q_lens_ptr + b_id).to(tl.int32)
        q_end_loc = tl.load(cu_q_lens_ptr + b_id + 1).to(tl.int32)
        q_seq_len = q_end_loc - q_start_loc

        if seqlens_kv_ptr is None:
            kv_seq_len = q_seq_len
        else:
            kv_seq_len = tl.load(seqlens_kv_ptr + b_id)
        kv_cache_len = kv_seq_len - q_seq_len

        cur_q_chunks = tl.cdiv(q_seq_len, BLOCK_SIZE_M)
        cur_q_tasks = cur_q_chunks * NUM_Q_HEADS
        prev_q_tasks = prev_q_chunks * NUM_Q_HEADS
        prev_q_chunks += cur_q_chunks
        for q_task_id in range((prev_q_tasks + pid) % n_progs, cur_q_tasks, n_progs):
            q_block_id = q_task_id // NUM_Q_HEADS
            q_head_id = q_task_id % NUM_Q_HEADS

            if GQA_INTERLEAVE:
                kv_head_id = q_head_id % NUM_KV_HEADS
            else:
                kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

            q_block_start_in_seq = q_block_id * BLOCK_SIZE_M
            q_block_end_in_seq = min(q_block_start_in_seq + BLOCK_SIZE_M, q_seq_len)
            q_block_len = q_block_end_in_seq - q_block_start_in_seq

            Q_block_ptr = tl.make_block_ptr(
                base=q_ptr + (q_start_loc + q_block_start_in_seq) * stride_qt + q_head_id * stride_qh,
                shape=(q_block_len, HEAD_DIM),
                strides=(stride_qt, stride_qd),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
                order=(1, 0),
            )
            O_block_ptr = tl.make_block_ptr(
                base=o_ptr + (q_start_loc + q_block_start_in_seq) * stride_ot + q_head_id * stride_oh,
                shape=(q_block_len, HEAD_DIM),
                strides=(stride_ot, stride_od),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
                order=(1, 0),
            )

            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            m_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) - float("inf")
            l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)

            num_kv_blocks = tl.cdiv(kv_cache_len + q_block_end_in_seq, BLOCK_SIZE_N)

            for kv_block_id in range(0, num_kv_blocks):
                kv_block_start_in_seq = kv_block_id * BLOCK_SIZE_N
                kv_block_end_in_seq = min(kv_block_start_in_seq + BLOCK_SIZE_N, kv_seq_len)
                kv_block_len = kv_block_end_in_seq - kv_block_start_in_seq
                
                logical_page_id = kv_block_start_in_seq // PAGE_SIZE
                kv_block_start_in_page = kv_block_start_in_seq % PAGE_SIZE
                physical_page_id = tl.load(
                    block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block
                )

                K_T_block_ptr = tl.make_block_ptr(
                    base=key_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                    shape=(HEAD_DIM, kv_block_len),
                    strides=(stride_k_dim, stride_k_blksz),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                    order=(0, 1),
                )
                V_block_ptr = tl.make_block_ptr(
                    base=value_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head + kv_block_start_in_page * stride_v_blksz,
                    shape=(kv_block_len, HEAD_DIM),
                    strides=(stride_v_blksz, stride_v_dim),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                    order=(1, 0),
                )

                mask = causal_mask_fn(
                    aux_mask_ptr,
                    AUX_MASK_SIZE,
                    stride_mask_m,
                    stride_mask_n,
                    kv_cache_len + q_block_start_in_seq,
                    kv_block_start_in_seq,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                )

                acc, l_i, m_i = _sdpa_infer_single_block(
                    acc,
                    l_i,
                    m_i,
                    q,
                    K_T_block_ptr,
                    V_block_ptr,
                    softmax_scale,
                    mask,
                    HEAD_DIM,
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_D,
                    value_cache_ptr.dtype.element_ty == tl.float8e5,
                )

            m_i += tl.math.log(l_i)
            accumulator = acc / l_i[:, None]

            # NOTE(zhangjihang): for training
            # m_ptrs = M + task_bn_idx * sub_kv_len + offs_m
            # tl.store(m_ptrs, m_i)
            tl.store(O_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1))


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
    max_q_len: Optional[int] = None,
    max_total_seq_len: Optional[int] = None,
) -> torch.Tensor:
    _, num_q_heads, head_dim = q.shape
    _, num_kv_heads, page_size, _ = key_cache.shape
    batch_size = cu_q_lens.shape[0] - 1

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if aux_mask is None:
        aux_mask = torch.ones(1024, 1024 * 3, device="npu", dtype=torch.bool).tril(1024)

    # Note(chenyifan):
    #   In general, this paged attention kernel works in a `split-q` style.
    #   "bsz * query * q_head" is splited into tasks of shape [BLOCK_SIZE_M, HEAD_DIM]
    #   and then attributed to one program.
    #
    #   Currently, we chunk the queries manually according to a magic CHUNK_SIZE to split queries
    #   It should be better with a autotuned BLOCK_SIZE_M and a pre-configured max_seq_len

    o = torch.empty_like(q)

    CHUNK_SIZE = 128
    BLOCK_SIZE_N = min(128, triton.next_power_of_2(page_size))
    cube_num = get_num_cores("cube")
    grid = (cube_num,)

    paged_prefill_kernel[grid](
        q,
        key_cache,
        value_cache,
        o,
        aux_mask,
        batch_size,
        cu_q_lens,
        seqlens_kv,
        block_tables.to(torch.int32),
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
        aux_mask.stride(0),
        aux_mask.stride(1),
        softmax_scale,
        aux_mask.shape[0],
        page_size,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        BLOCK_SIZE_M=CHUNK_SIZE,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=head_dim,
        limit_auto_multi_buffer_only_for_local_buffer=False,
        set_workspace_multibuffer=4,
    )
    return o


@triton.jit
def paged_decode_vector_kernel(
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
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(HEAD_DIM <= BLOCK_D, "HEAD_DIM should be less than BLOCK_D")
    tl.static_assert(PAGE_SIZE % BLOCK_N == 0, "BLOCK_N must be a divisor of PAGE_SIZE")
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    num_tasks = BATCH_SIZE * NUM_KV_HEADS

    for task_id in range(pid, num_tasks, n_progs):
        kv_head_id = task_id % NUM_KV_HEADS
        b_id = task_id // NUM_KV_HEADS

        kv_seq_len = tl.load(seqlens_ptr + b_id)

        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        row_mask = offs_m < GQA_RATIO

        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + offs_m * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GQA_RATIO + offs_m

        q_ptrs = q_ptr + b_id * stride_qb + q_head_ids[:, None] * stride_qh + offs_d[None, :] * stride_qd
        q_mask = row_mask[:, None] & (offs_d[None, :] < HEAD_DIM)
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        num_kv_blocks = tl.cdiv(kv_seq_len, BLOCK_N)

        for kv_block_id in range(0, num_kv_blocks):
            kv_block_start_in_seq = kv_block_id * BLOCK_N
            kv_block_end_in_seq = min(kv_block_start_in_seq + BLOCK_N, kv_seq_len)
            kv_block_len = kv_block_end_in_seq - kv_block_start_in_seq

            logical_page_id = kv_block_start_in_seq // PAGE_SIZE
            kv_block_start_in_page = kv_block_start_in_seq % PAGE_SIZE
            physical_page_id = tl.load(block_tables_ptr + b_id * stride_bt_batch + logical_page_id * stride_bt_block)

            k_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr + physical_page_id * stride_k_block + kv_head_id * stride_k_head + kv_block_start_in_page * stride_k_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_k_blksz, stride_k_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr + physical_page_id * stride_v_block + kv_head_id * stride_v_head + kv_block_start_in_page * stride_v_blksz,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(1, 0),
            )

            n_mask = tl.arange(0, BLOCK_N) < kv_block_len

            k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # QK: [BLOCK_M, BLOCK_N]
            qk = tl.sum(q[:, None, :] * k[None, :, :], axis=2).to(tl.float32)
            qk *= softmax_scale
            qk = tl.where(row_mask[:, None] & n_mask[None, :], qk, float("-inf"))

            # Online softmax
            m_j = tl.max(qk, axis=1)
            m_ij = tl.maximum(m_i, m_j)
            qk = qk - m_ij[:, None]
            p = tl.exp(qk)

            l_ij = tl.sum(p, axis=1)
            alpha = tl.math.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            # PV: [BLOCK_M, BLOCK_D]
            p_cast = p.to(v.dtype)
            acc += tl.sum(p_cast[:, :, None] * v[None, :, :], axis=1).to(tl.float32)
            m_i = m_ij

        # Normalize
        if kv_seq_len > 0:
            acc = acc / l_i[:, None]

        # Store output
        o_ptrs = o_ptr + b_id * stride_ob + q_head_ids[:, None] * stride_oh + offs_d[None, :] * stride_od
        o_mask = row_mask[:, None] & (offs_d[None, :] < HEAD_DIM)
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=o_mask)



@triton.jit
def _paged_decode_cube_attention(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    seqlens_ptr,
    block_tables_ptr,
    b_id,
    kv_head_id,
    kv_block_start,
    kv_block_end,
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
    stride_bt_batch,
    stride_bt_block,
    softmax_scale,
    NUM_KV_HEADS: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    kv_seq_len = tl.load(seqlens_ptr + b_id)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    if GQA_INTERLEAVE:
        q_head_ids = kv_head_id + offs_m * NUM_KV_HEADS
    else:
        q_head_ids = kv_head_id * GQA_RATIO + offs_m

    q_ptrs = (q_ptr
              + b_id * stride_qb
              + q_head_ids[:, None] * stride_qh
              + offs_d[None, :] * stride_qd)
    q_mask = (offs_m[:, None] < GQA_RATIO) & (offs_d[None, :] < HEAD_DIM)
    q_packed = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize online softmax state
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Iterate over KV blocks in the given range
    for kv_block_id in range(kv_block_start, kv_block_end):
        kv_start_in_seq = kv_block_id * BLOCK_N
        kv_len = tl.minimum(BLOCK_N, kv_seq_len - kv_start_in_seq)

        # Page address translation
        logical_page = kv_start_in_seq // PAGE_SIZE
        offset_in_page = kv_start_in_seq % PAGE_SIZE
        physical_page = tl.load(
            block_tables_ptr + b_id * stride_bt_batch + logical_page * stride_bt_block
        )

        # Load K^T: [BLOCK_D, BLOCK_N]
        k_T_block_ptr = tl.make_block_ptr(
            base=(k_cache_ptr
                  + physical_page * stride_k_block
                  + kv_head_id * stride_k_head
                  + offset_in_page * stride_k_blksz),
            shape=(HEAD_DIM, kv_len),
            strides=(stride_k_dim, stride_k_blksz),
            offsets=(0, 0),
            block_shape=(BLOCK_D, BLOCK_N),
            order=(0, 1),
        )
        k_T = tl.load(k_T_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # QK = Q_packed @ K^T -> [BLOCK_M, BLOCK_N]
        qk = tl.dot(q_packed, k_T)

        qk = qk * softmax_scale

        # Mask out-of-range positions
        kv_mask = tl.arange(0, BLOCK_N) < kv_len
        qk = tl.where(kv_mask[None, :], qk, float("-inf"))

        # Online softmax update
        m_j = tl.max(qk, axis=1)
        m_ij = tl.maximum(m_i, m_j)
        alpha = tl.math.exp(m_i - m_ij)
        p = tl.math.exp(qk - m_ij[:, None])

        l_ij = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # P @ V -> [BLOCK_M, BLOCK_D]
        v_block_ptr = tl.make_block_ptr(
            base=(v_cache_ptr
                  + physical_page * stride_v_block
                  + kv_head_id * stride_v_head
                  + offset_in_page * stride_v_blksz),
            shape=(kv_len, HEAD_DIM),
            strides=(stride_v_blksz, stride_v_dim),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        p_cast = p.to(v.dtype)

        # PV = P @ V -> [BLOCK_M, BLOCK_D]
        pv = tl.dot(p_cast, v)
        acc += pv
        m_i = m_ij

    return acc, m_i, l_i



@triton.jit
def paged_decode_cube_kernel(
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
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    num_tasks = BATCH_SIZE * NUM_KV_HEADS

    for task_id in range(pid, num_tasks, n_progs):
        kv_head_id = task_id % NUM_KV_HEADS
        b_id = task_id // NUM_KV_HEADS

        kv_seq_len = tl.load(seqlens_ptr + b_id)
        total_kv_blocks = tl.cdiv(kv_seq_len, BLOCK_N)

        # GQA: BLOCK_M == GQA_RATIO, no padding waste
        acc, m_i, l_i = _paged_decode_cube_attention(
            q_ptr, k_cache_ptr, v_cache_ptr, seqlens_ptr, block_tables_ptr,
            b_id, kv_head_id,
            0, total_kv_blocks,
            stride_qb, stride_qh, stride_qd,
            stride_k_block, stride_k_head, stride_k_blksz, stride_k_dim,
            stride_v_block, stride_v_head, stride_v_blksz, stride_v_dim,
            stride_bt_batch, stride_bt_block,
            softmax_scale,
            NUM_KV_HEADS=NUM_KV_HEADS,
            GQA_RATIO=GQA_RATIO,
            GQA_INTERLEAVE=GQA_INTERLEAVE,
            HEAD_DIM=HEAD_DIM,
            PAGE_SIZE=PAGE_SIZE,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )

        # Normalize and store output
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + offs_m * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GQA_RATIO + offs_m

        if kv_seq_len > 0:
            acc = acc / l_i[:, None]

        o_ptrs = (o_ptr
                    + b_id * stride_ob
                    + q_head_ids[:, None] * stride_oh
                    + offs_d[None, :] * stride_od)
        o_mask = (offs_m[:, None] < GQA_RATIO) & (offs_d[None, :] < HEAD_DIM)
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=o_mask)



@triton.jit
def paged_decode_splitkv_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    seqlens_ptr,
    block_tables_ptr,
    acc_ws_ptr,
    m_ws_ptr,
    l_ws_ptr,
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
    stride_ws_b,
    stride_ws_h,
    stride_ws_s,
    stride_ws_m,
    stride_ws_d,
    stride_ml_b,
    stride_ml_h,
    stride_ml_s,
    stride_ml_m,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    # Stage 1: Split-KV computation
    num_stage1_tasks = BATCH_SIZE * NUM_KV_HEADS * NUM_SPLITS

    for task_id in range(pid, num_stage1_tasks, n_progs):
        split_id = task_id % NUM_SPLITS
        remainder = task_id // NUM_SPLITS
        kv_head_id = remainder % NUM_KV_HEADS
        b_id = remainder // NUM_KV_HEADS

        kv_seq_len = tl.load(seqlens_ptr + b_id)
        total_kv_blocks = tl.cdiv(kv_seq_len, BLOCK_N)
        blocks_per_split = tl.cdiv(total_kv_blocks, NUM_SPLITS)
        kv_block_start = split_id * blocks_per_split
        kv_block_end = tl.minimum(kv_block_start + blocks_per_split, total_kv_blocks)

        acc, m_i, l_i = _paged_decode_cube_attention(
            q_ptr, k_cache_ptr, v_cache_ptr, seqlens_ptr, block_tables_ptr,
            b_id, kv_head_id,
            kv_block_start, kv_block_end,
            stride_qb, stride_qh, stride_qd,
            stride_k_block, stride_k_head, stride_k_blksz, stride_k_dim,
            stride_v_block, stride_v_head, stride_v_blksz, stride_v_dim,
            stride_bt_batch, stride_bt_block,
            softmax_scale,
            NUM_KV_HEADS=NUM_KV_HEADS,
            GQA_RATIO=GQA_RATIO,
            GQA_INTERLEAVE=GQA_INTERLEAVE,
            HEAD_DIM=HEAD_DIM,
            PAGE_SIZE=PAGE_SIZE,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )

        # Store partial results to workspace
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        ws_acc_base = (acc_ws_ptr
                       + b_id * stride_ws_b
                       + kv_head_id * stride_ws_h
                       + split_id * stride_ws_s)
        ws_acc_offs = offs_m[:, None] * stride_ws_m + offs_d[None, :] * stride_ws_d
        ws_acc_mask = (offs_m[:, None] < GQA_RATIO) & (offs_d[None, :] < HEAD_DIM)
        tl.store(ws_acc_base + ws_acc_offs, acc, mask=ws_acc_mask)

        ws_ml_base_m = (m_ws_ptr
                        + b_id * stride_ml_b
                        + kv_head_id * stride_ml_h
                        + split_id * stride_ml_s)
        ws_ml_base_l = (l_ws_ptr
                        + b_id * stride_ml_b
                        + kv_head_id * stride_ml_h
                        + split_id * stride_ml_s)
        ml_offs = offs_m * stride_ml_m
        ml_mask = offs_m < GQA_RATIO
        tl.store(ws_ml_base_m + ml_offs, m_i, mask=ml_mask)
        tl.store(ws_ml_base_l + ml_offs, l_i, mask=ml_mask)


    # Global sync
    tle.dsa.ascend.sync_block_all("all", 0)

    # Stage 2: Merge partial results
    num_stage2_tasks = BATCH_SIZE * NUM_KV_HEADS

    for task_id in range(pid, num_stage2_tasks, n_progs):
        kv_head_id = task_id % NUM_KV_HEADS
        b_id = task_id // NUM_KV_HEADS

        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        row_mask = offs_m < GQA_RATIO

        # Find global max across all splits
        m_global = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        for s in range(NUM_SPLITS):
            m_s_base = (m_ws_ptr
                        + b_id * stride_ml_b
                        + kv_head_id * stride_ml_h
                        + s * stride_ml_s)
            m_s = tl.load(m_s_base + offs_m * stride_ml_m, mask=row_mask, other=float("-inf"))
            m_global = tl.maximum(m_global, m_s)

        # Weighted merge
        l_global = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc_global = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for s in range(NUM_SPLITS):
            m_s_base = (m_ws_ptr
                        + b_id * stride_ml_b
                        + kv_head_id * stride_ml_h
                        + s * stride_ml_s)
            l_s_base = (l_ws_ptr
                        + b_id * stride_ml_b
                        + kv_head_id * stride_ml_h
                        + s * stride_ml_s)
            acc_s_base = (acc_ws_ptr
                          + b_id * stride_ws_b
                          + kv_head_id * stride_ws_h
                          + s * stride_ws_s)

            m_s = tl.load(m_s_base + offs_m * stride_ml_m, mask=row_mask, other=float("-inf"))
            l_s = tl.load(l_s_base + offs_m * stride_ml_m, mask=row_mask, other=0.0)

            alpha_s = tl.where(m_s == float("-inf"), 0.0, tl.exp(m_s - m_global))
            l_global += l_s * alpha_s

            acc_s_offs = offs_m[:, None] * stride_ws_m + offs_d[None, :] * stride_ws_d
            acc_s_mask = row_mask[:, None] & (offs_d[None, :] < HEAD_DIM)
            acc_s = tl.load(acc_s_base + acc_s_offs, mask=acc_s_mask, other=0.0)
            acc_global += acc_s * alpha_s[:, None]

        # Normalize
        kv_seq_len = tl.load(seqlens_ptr + b_id)
        if kv_seq_len > 0:
            acc_global = acc_global / l_global[:, None]

        # Store output
        if GQA_INTERLEAVE:
            q_head_ids = kv_head_id + offs_m * NUM_KV_HEADS
        else:
            q_head_ids = kv_head_id * GQA_RATIO + offs_m

        o_ptrs = (o_ptr
                  + b_id * stride_ob
                  + q_head_ids[:, None] * stride_oh
                  + offs_d[None, :] * stride_od)
        o_mask = row_mask[:, None] & (offs_d[None, :] < HEAD_DIM)
        tl.store(o_ptrs, acc_global.to(o_ptr.dtype.element_ty), mask=o_mask)



def paged_attention_decode_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    batch_size, num_q_heads, head_dim = q.shape
    num_total_blocks, num_kv_heads, page_size, head_dim_cache = key_cache.shape

    max_num_blocks_per_seq = block_tables.shape[1]

    assert head_dim == head_dim_cache
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    o = torch.empty_like(q)

    assert num_q_heads % num_kv_heads == 0
    gqa_ratio = num_q_heads // num_kv_heads

    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_N = min(128, triton.next_power_of_2(page_size))
    BLOCK_M = triton.next_power_of_2(gqa_ratio)

    # Vector kernel UB capacity limit for N dimension
    VECTOR_BLOCK_N_MAX = 64

    if BLOCK_N >= 64 and gqa_ratio > 1:
        num_cores = get_num_cores("cube")
        num_tasks_base = batch_size * num_kv_heads
        max_seq = seqlens.max().item()

        num_kv_blocks = (max_seq + BLOCK_N - 1) // BLOCK_N
        MIN_BLOCKS_PER_SPLIT = 4
        BLOCK_M = max(4, BLOCK_M)

        # Compute natural num_splits: only split when cores are underutilized
        # and seq is long enough to support meaningful splits
        if num_tasks_base >= num_cores or num_kv_blocks < MIN_BLOCKS_PER_SPLIT * 2:
            num_splits = 1
        else:
            num_splits = min(
                num_cores // num_tasks_base,
                num_kv_blocks // MIN_BLOCKS_PER_SPLIT,
            )

        grid = (num_cores,)

        if num_splits >= 2 and HAS_TLE:
            # Split-KV flash kernel: workspace + sync + merge
            acc_ws = torch.zeros(
                (batch_size, num_kv_heads, num_splits, gqa_ratio, head_dim),
                dtype=torch.float32, device=q.device,
            )
            m_ws = torch.full(
                (batch_size, num_kv_heads, num_splits, gqa_ratio),
                float("-inf"), dtype=torch.float32, device=q.device,
            )
            l_ws = torch.zeros(
                (batch_size, num_kv_heads, num_splits, gqa_ratio),
                dtype=torch.float32, device=q.device,
            )

            paged_decode_splitkv_kernel[grid](
                q,
                key_cache,
                value_cache,
                o,
                seqlens,
                block_tables,
                acc_ws,
                m_ws,
                l_ws,
                batch_size,
                num_total_blocks,
                max_num_blocks_per_seq,
                q.stride(0), q.stride(1), q.stride(2),
                key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
                value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
                o.stride(0), o.stride(1), o.stride(2),
                block_tables.stride(0), block_tables.stride(1),
                acc_ws.stride(0), acc_ws.stride(1), acc_ws.stride(2), acc_ws.stride(3), acc_ws.stride(4),
                m_ws.stride(0), m_ws.stride(1), m_ws.stride(2), m_ws.stride(3),
                softmax_scale,
                NUM_Q_HEADS=num_q_heads,
                NUM_KV_HEADS=num_kv_heads,
                GQA_RATIO=gqa_ratio,
                GQA_INTERLEAVE=gqa_interleave,
                HEAD_DIM=head_dim,
                PAGE_SIZE=page_size,
                NUM_SPLITS=num_splits,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_D=BLOCK_D,
            )
        else:
            paged_decode_cube_kernel[grid](
                q,
                key_cache,
                value_cache,
                o,
                seqlens,
                block_tables,
                batch_size,
                num_total_blocks,
                max_num_blocks_per_seq,
                q.stride(0), q.stride(1), q.stride(2),
                key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
                value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
                o.stride(0), o.stride(1), o.stride(2),
                block_tables.stride(0), block_tables.stride(1),
                softmax_scale,
                NUM_Q_HEADS=num_q_heads,
                NUM_KV_HEADS=num_kv_heads,
                GQA_RATIO=gqa_ratio,
                GQA_INTERLEAVE=gqa_interleave,
                HEAD_DIM=head_dim,
                PAGE_SIZE=page_size,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_D=BLOCK_D,
            )
    else:
        num_vec_cores = get_num_cores("vector")
        grid_vec = (num_vec_cores,)
        BLOCK_N_VEC = min(BLOCK_N, VECTOR_BLOCK_N_MAX)

        paged_decode_vector_kernel[grid_vec](
            q,
            key_cache,
            value_cache,
            o,
            seqlens,
            block_tables,
            batch_size,
            num_total_blocks,
            max_num_blocks_per_seq,
            q.stride(0), q.stride(1), q.stride(2),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
            o.stride(0), o.stride(1), o.stride(2),
            block_tables.stride(0), block_tables.stride(1),
            softmax_scale,
            NUM_Q_HEADS=num_q_heads,
            NUM_KV_HEADS=num_kv_heads,
            GQA_RATIO=gqa_ratio,
            GQA_INTERLEAVE=gqa_interleave,
            HEAD_DIM=head_dim,
            PAGE_SIZE=page_size,
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            BLOCK_N=BLOCK_N_VEC,
        )

    return o
