import math

from typing import Optional

import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores


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
    cu_seqlens_q_ptr,
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
    sm_scale,
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

    prev_q_chunks = 0

    for b_id in range(batch_size):
        q_start_loc = tl.load(cu_seqlens_q_ptr + b_id).to(tl.int32)
        q_end_loc = tl.load(cu_seqlens_q_ptr + b_id + 1).to(tl.int32)
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

            num_logical_blocks = tl.cdiv(kv_cache_len + q_block_end_in_seq, PAGE_SIZE)

            tl.static_assert(PAGE_SIZE == BLOCK_SIZE_N, "Currently only BLOCK_SIZE_N==PAGE_SIZE supported")

            for logical_block_idx in range(0, num_logical_blocks):
                physical_block_id = tl.load(
                    block_tables_ptr + b_id * stride_bt_batch + logical_block_idx * stride_bt_block
                )

                kv_block_start_in_seq = logical_block_idx * PAGE_SIZE
                kv_block_end_in_seq = min(kv_block_start_in_seq + PAGE_SIZE, kv_seq_len)
                kv_block_len = kv_block_end_in_seq - kv_block_start_in_seq
                K_T_block_ptr = tl.make_block_ptr(
                    base=key_cache_ptr + physical_block_id * stride_k_block + kv_head_id * stride_k_head,
                    shape=(HEAD_DIM, kv_block_len),
                    strides=(stride_k_dim, stride_k_blksz),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                    order=(0, 1),
                )
                V_block_ptr = tl.make_block_ptr(
                    base=value_cache_ptr + physical_block_id * stride_v_block + kv_head_id * stride_v_head,
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
                    sm_scale,
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
    cu_seqlens_q: torch.Tensor,
    seqlens_kv: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    sm_scale: Optional[float] = None,
    aux_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _, num_q_heads, head_dim = q.shape
    _, num_kv_heads, block_size, _ = key_cache.shape
    batch_size = cu_seqlens_q.shape[0] - 1

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    if aux_mask is None:
        aux_mask = torch.ones(1024, 1024 * 3, dtype=torch.bool).tril(1024).npu()

    # Note(chenyifan):
    #   In general, this paged attention kernel works in a `split-q` style.
    #   "bsz * query * q_head" is splited into tasks of shape [BLOCK_SIZE_M, HEAD_DIM]
    #   and then attributed to one program.
    #
    #   Currently, we chunk the queries manually according to a magic CHUNK_SIZE to split queries
    #   It should be better with a autotuned BLOCK_SIZE_M and a pre-configured max_seq_len

    o = torch.empty_like(q)

    CHUNK_SIZE = 128
    cube_num = get_num_cores("cube")
    grid = (cube_num,)

    paged_prefill_kernel[grid](
        q,
        key_cache,
        value_cache,
        o,
        aux_mask,
        batch_size,
        cu_seqlens_q,
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
        sm_scale,
        aux_mask.shape[0],
        block_size,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        BLOCK_SIZE_M=CHUNK_SIZE,
        BLOCK_SIZE_N=block_size,
        BLOCK_SIZE_D=head_dim,
        limit_auto_multi_buffer_only_for_local_buffer=False,
        set_workspace_multibuffer=4,
    )
    return o


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
):
    tl.static_assert(HEAD_DIM <= BLOCK_SIZE_D, "HEAD_DIM should be less than BLOCK_SIZE_D")
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    num_tasks = BATCH_SIZE * NUM_Q_HEADS

    for q_task_id in range(pid, num_tasks, n_progs):
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

        num_logical_blocks = tl.cdiv(kv_seq_len, PAGE_SIZE)
        
        tl.static_assert(PAGE_SIZE == BLOCK_SIZE_N, "PAGE_SIZE should be equal to BLOCK_SIZE_N")

        for logical_block_idx in range(0, num_logical_blocks):
            physical_block_id = tl.load(block_tables_ptr + b_id * stride_bt_batch + logical_block_idx * stride_bt_block)
            
            kv_block_start_in_seq = logical_block_idx * PAGE_SIZE
            kv_block_end_in_seq = min(kv_block_start_in_seq + PAGE_SIZE, kv_seq_len)
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

            # p = p.to(v.dtype)

            acc += tl.sum((p_cast[:, None] * v).to(tl.float32), axis=0)

            m_i = m_ij

        m_i += tl.math.log(l_i)
        acc = acc / l_i

        o_ptrs = o_ptr + b_id * stride_ob + q_head_id * stride_oh + offs_d * stride_od
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_d < HEAD_DIM)


def paged_attention_decode_impl(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_interleave: bool,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    batch_size, num_q_heads, head_dim = q.shape
    num_total_blocks, num_kv_heads, block_size, head_dim_cache = key_cache.shape

    assert block_size <= 128, f"temp: only support block_size <= 128, but got {block_size}"
    max_num_blocks_per_seq = block_tables.shape[1]

    assert head_dim == head_dim_cache
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    o = torch.empty_like(q)
    
    num_vectors = get_num_cores("vector")
    grid = (num_vectors, )
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)

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
        sm_scale,
        num_q_heads,
        num_kv_heads,
        gqa_interleave,
        head_dim,
        block_size,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_N=block_size,
        multibuffer=False,
    )
    return o
