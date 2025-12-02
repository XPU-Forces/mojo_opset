import math

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def paged_prefill_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    cu_seqlens_q_ptr,
    block_tables_ptr,
    BATCH_SIZE,
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
    NUM_TOTAL_BLOCKS,
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
    sm_scale,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    q_start_loc = tl.load(cu_seqlens_q_ptr + pid_b)
    q_end_loc = tl.load(cu_seqlens_q_ptr + pid_b + 1)
    q_seq_len = q_end_loc - q_start_loc

    k_seq_len = q_seq_len

    q_block_start_offset_in_seq = pid_m * BLOCK_SIZE_M
    if q_block_start_offset_in_seq >= q_seq_len:
        return

    pid_kh = pid_h // (NUM_Q_HEADS // NUM_KV_HEADS)

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)

    q_tokens_global_idx = q_start_loc + q_block_start_offset_in_seq + offs_m

    q_ptrs = q_ptr + (q_tokens_global_idx[:, None] * stride_qt) + (pid_h * stride_qh) + (offs_d[None, :] * stride_qd)

    q_mask = (q_block_start_offset_in_seq + offs_m) < q_seq_len
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    m_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)

    num_logical_blocks = (k_seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

    for logical_block_idx in range(0, num_logical_blocks):
        physical_block_id = tl.load(block_tables_ptr + pid_b * stride_bt_batch + logical_block_idx)

        k_block_ptr = tl.make_block_ptr(
            base=k_cache_ptr + pid_kh * stride_k_head,
            shape=(NUM_TOTAL_BLOCKS, BLOCK_SIZE_N, HEAD_DIM),
            strides=(stride_k_block, stride_k_blksz, stride_k_dim),
            offsets=(physical_block_id, 0, 0),
            block_shape=(1, BLOCK_SIZE_N, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        k = tl.load(k_block_ptr)
        k = tl.reshape(k, (BLOCK_SIZE_N, BLOCK_SIZE_D))

        s_ij = tl.dot(q, tl.trans(k))

        query_pos = q_block_start_offset_in_seq + offs_m

        key_pos = logical_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        causal_mask = query_pos[:, None] >= key_pos[None, :]
        key_padding_mask = key_pos[None, :] < k_seq_len
        final_mask = causal_mask & key_padding_mask

        s_ij = tl.where(final_mask, s_ij * sm_scale, -float("inf"))

        m_j = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_j)
        p_ij = tl.exp(s_ij - m_new[:, None])
        l_j = tl.sum(p_ij, axis=1)
        alpha = tl.exp(m_i - m_new)
        l_new = alpha * l_i + l_j
        acc_o = acc_o * alpha[:, None]

        v_block_ptr = tl.make_block_ptr(
            base=v_cache_ptr + pid_kh * stride_v_head,
            shape=(NUM_TOTAL_BLOCKS, BLOCK_SIZE_N, HEAD_DIM),
            strides=(stride_v_block, stride_v_blksz, stride_v_dim),
            offsets=(physical_block_id, 0, 0),
            block_shape=(1, BLOCK_SIZE_N, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        v = tl.load(v_block_ptr)
        v = tl.reshape(v, (BLOCK_SIZE_N, BLOCK_SIZE_D))

        p_ij = p_ij.to(v.dtype)
        acc_o += tl.dot(p_ij, v)

        m_i = m_new
        l_i = l_new

    l_i_rcp = 1.0 / l_i
    acc_o = acc_o * l_i_rcp[:, None]

    o_ptrs = o_ptr + (q_tokens_global_idx[:, None] * stride_ot) + (pid_h * stride_oh) + (offs_d[None, :] * stride_od)

    tl.store(o_ptrs, acc_o, mask=q_mask[:, None])


@torch.library.custom_op("ttx::paged_attention_prefill", mutates_args={})
def paged_attention_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_tables: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    total_tokens, num_q_heads, head_dim = q.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    num_total_blocks, num_kv_heads, block_size, _ = k_cache.shape
    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    max_seqlen_q = int(q_lens.max().item())

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    o = torch.empty_like(q)
    BLOCK_SIZE_M = 16

    grid = (batch_size, num_q_heads, triton.cdiv(max_seqlen_q, BLOCK_SIZE_M))

    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)

    paged_prefill_kernel[grid](
        q,
        k_cache,
        v_cache,
        o,
        cu_seqlens_q,
        block_tables.to(torch.int32),
        batch_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        num_total_blocks,
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
        o.stride(0),
        o.stride(1),
        o.stride(2),
        block_tables.stride(0),
        block_tables.stride(1),
        sm_scale,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=block_size,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    return o


@paged_attention_prefill.register_fake
def paged_attention_prefill_fake(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_tables: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    return torch.empty_like(q)


@triton.jit
def paged_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    seqlens_ptr,
    block_tables_ptr,
    BATCH_SIZE,
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
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
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    NUM_SHARE_Q_HEADS = NUM_Q_HEADS // NUM_KV_HEADS
    pid_kh = pid_h // NUM_SHARE_Q_HEADS

    kv_len = tl.load(seqlens_ptr + pid_b)

    num_logical_blocks = (kv_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

    q_offset = pid_b * stride_qb + pid_h * stride_qh

    offs_d = tl.arange(0, BLOCK_SIZE_D)
    q_ptrs = q_ptr + q_offset + offs_d * stride_qd
    q = tl.load(q_ptrs)

    m_i = -float("inf")
    l_i = 0.0
    acc_o = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

    for logical_block_idx in range(0, num_logical_blocks):
        bt_offset = pid_b * stride_bt_batch + logical_block_idx * stride_bt_block
        physical_block_id = tl.load(block_tables_ptr + bt_offset)

        k_block_ptr = tl.make_block_ptr(
            base=k_cache_ptr + pid_kh * stride_k_head,
            shape=(NUM_TOTAL_BLOCKS, BLOCK_SIZE_N, HEAD_DIM),
            strides=(stride_k_block, stride_k_blksz, stride_k_dim),
            offsets=(physical_block_id, 0, 0),
            block_shape=(1, BLOCK_SIZE_N, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_cache_ptr + pid_kh * stride_v_head,
            shape=(NUM_TOTAL_BLOCKS, BLOCK_SIZE_N, HEAD_DIM),
            strides=(stride_v_block, stride_v_blksz, stride_v_dim),
            offsets=(physical_block_id, 0, 0),
            block_shape=(1, BLOCK_SIZE_N, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )

        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)

        k = tl.reshape(k, (BLOCK_SIZE_N, BLOCK_SIZE_D))
        v = tl.reshape(v, (BLOCK_SIZE_N, BLOCK_SIZE_D))

        qk = tl.sum(q[None, :] * k, axis=1)

        current_logical_offset = logical_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = current_logical_offset < kv_len

        qk = tl.where(mask, qk, -float("inf"))
        qk *= sm_scale

        m_j = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, m_j)

        p = tl.exp(qk - m_new)
        l_j = tl.sum(p, axis=0)

        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_j - m_new)

        l_new = alpha * l_i + l_j

        acc_o = acc_o * alpha

        p = p.to(v.dtype)

        acc_o += tl.sum(p[:, None] * v, axis=0)

        l_i = l_new
        m_i = m_new

    acc_o = acc_o / l_i

    o_offset = pid_b * stride_ob + pid_h * stride_oh
    o_ptrs = o_ptr + o_offset + offs_d * stride_od
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty))


@torch.library.custom_op("ttx::paged_attention_decode", mutates_args={})
def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    batch_size, num_q_heads, head_dim = q.shape
    num_total_blocks, num_kv_heads, block_size, head_dim_cache = k_cache.shape
    max_num_blocks_per_seq = block_tables.shape[1]

    assert head_dim == head_dim_cache
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    o = torch.empty_like(q)
    grid = (batch_size, num_q_heads)
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)

    paged_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        o,
        seqlens,
        block_tables.to(torch.int32),
        batch_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        num_total_blocks,
        max_num_blocks_per_seq,
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
        o.stride(0),
        o.stride(1),
        o.stride(2),
        block_tables.stride(0),
        block_tables.stride(1),
        sm_scale,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_N=block_size,
        multibuffer=False,
    )
    return o


@paged_attention_decode.register_fake
def paged_attention_decode_fake(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    return torch.empty_like(q)
