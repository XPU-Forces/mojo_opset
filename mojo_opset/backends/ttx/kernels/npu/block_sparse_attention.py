import torch
import math

import triton
import triton.language as tl
from triton.runtime.libentry import libentry
from .utils import get_num_cores


@triton.jit
def get_mask_ptr_offset_n_cross(q_start, q_block, q_len, kv_start, kv_block, kv_len):
    offset_kv = min(max(kv_start - kv_len, -kv_block), 0)
    return offset_kv


@triton.jit
def get_mask_ptr_offset_n_causal(q_start, q_block, q_len, kv_start, kv_block, kv_len):
    offset_kv = min(max(kv_start - q_start, -kv_block), kv_block)
    return offset_kv


@triton.jit
def mask_mod_fn(mask_ptr, mask_size, mask_stride_m, mask_stride_n, q_start, Q_BLOCK, q_end, kv_start, KV_BLOCK, kv_end):

    offset_len = min(max(kv_start - kv_end, -KV_BLOCK), 0)
    offsets_mask_len = (tl.arange(0, Q_BLOCK)[:, None]) * mask_stride_m + (
        mask_size + offset_len + tl.arange(0, KV_BLOCK)[None, :]
    ) * mask_stride_n
    mask_len = tl.load(mask_ptr + offsets_mask_len).to(tl.int1)

    offset_causal = min(max(kv_start - q_start, -KV_BLOCK), KV_BLOCK)
    offsets_mask_causal = (tl.arange(0, Q_BLOCK)[:, None]) * mask_stride_m + (
        3 * mask_size + offset_causal + tl.arange(0, KV_BLOCK)[None, :]
    ) * mask_stride_n
    mask_causal = tl.load(mask_ptr + offsets_mask_causal).to(tl.int1)

    mask = mask_len & mask_causal
    return mask


@triton.jit
def _sdpa_infer_inner(
    acc_ptr,
    l_i,
    m_i,
    q,  # Accumulator, local l, local m, query vector
    K_block_ptr,
    V_block_ptr,  # Key and value block pointers for current stage
    mask_base_ptr,
    stride_mask_base_ptr_m,
    stride_mask_base_ptr_n,
    mask_size,
    qk_scale,  # Starting position of current query block, qk scale factor
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  # Block size constants
    offs_m,
    offs_n,
    SEQ,
    fp8_v: tl.constexpr,
):
    n_iters = tl.cdiv(SEQ, BLOCK_N)
    # Iterate over all k, v blocks in the current stage and accumulate the output
    for kv_block_idx in range(n_iters):  # Process BLOCK_N columns at a time
        start_n = kv_block_idx * BLOCK_N
        start_n = tl.multiple_of(start_n, BLOCK_N)  # Align column start position

        # -- Compute qk ----
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # Modify K
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        # tl.compile_hint(qk, "tile_cube_loop")

        # NOTE(zhangjihang): tl.where will introduce ub overflow
        qk = qk * qk_scale

        # mask = tl.load(mask_ptr)

        # qk += (1 - mask.to(tl.float32)) * (-1e6)
        # TODO(zhangjihang): tl.where with a non-boolean condition is deprecated and will error out in a future triton release. Got int8
        if mask_base_ptr is not None:
            mask = mask_mod_fn(
                mask_base_ptr,
                mask_size,
                stride_mask_base_ptr_m,
                stride_mask_base_ptr_n,
                offs_m,
                BLOCK_M,
                offs_m + BLOCK_M,
                offs_n + start_n,
                BLOCK_N,
                offs_n + SEQ,
            )
            qk = tl.where(mask, qk, float("-inf"))  # 32B # bool

        m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Scaled max
        qk = qk - m_ij[:, None]  # Stabilize

        # Softmax weights p = exp(qk)
        p = tl.math.exp(qk)

        p_cast = p.to(k.dtype)

        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")  # Load corresponding V block
        # v = tl.where(kv_mask[:, None], v, 0)  # 32B # bool

        l_ij = tl.sum(p, 1)  # Softmax denominator (sum of each row)
        # -- Update m_i and l_i
        alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
        l_i = l_i * alpha + l_ij  # Update softmax denominator
        # -- Update output accumulator --
        acc_ptr = acc_ptr * alpha[:, None]
        acc_ptr = tl.dot(p_cast, v, acc_ptr)
        # tl.compile_hint(acc_ptr, "tile_cube_loop")

        m_i = m_ij  # Update current block max
        # Advance V and K block pointers to next BLOCK_N range
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))

    # NOTE(zhangjihang): for training
    # Return accumulated output acc_ptr, softmax denominator l_i, and max value m_i
    return acc_ptr, l_i, m_i


@triton.jit
def block_sparse_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    scale,
    topk_page_indices_ptr,
    o_ptr,
    session_mask_ptr,
    q_seg_start,
    q_seg_len,
    kv_cache_len,
    q_chunk_size,
    n_topk_pages,
    stride_q_head,
    stride_q_m,
    stride_q_k,
    stride_k_head,
    stride_k_n,
    stride_k_k,
    stride_v_head,
    stride_v_n,
    stride_v_k,
    stride_o_head,
    stride_o_m,
    stride_o_k,
    stride_session_mask_m,
    stride_session_mask_n,
    topk_page_indices_stride_head,
    topk_page_indices_stride_idx,
    Q_HEAD_NUM: tl.constexpr,
    KV_HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_SEG_SIZE: tl.constexpr,
    KV_PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_MASK: tl.constexpr,
):
    """
    Block sparse prefill attention kernel
    Args:
        q_ptr: Pointer to query tensor (Q), shape [Q_HEAD_NUM, q_seg_len, HEAD_DIM]
        k_ptr: Pointer to key tensor (K), shape [KV_HEAD_NUM, kv_cache_len+q_chunk_size, HEAD_DIM]
        v_ptr: Pointer to value tensor (V), shape [KV_HEAD_NUM, kv_cache_len+q_chunk_size, HEAD_DIM]
        topk_page_indices_ptr: Pointer to topk page indices, shape [Q_HEAD_NUM, n_topk_pages]
        o_ptr: Pointer to output tensor (O), shape [Q_HEAD_NUM, Q_SEG_SIZE, HEAD_DIM]
        session_mask_ptr: Pointer to session mask, shape [q_seg_len, q_chunk_size]

        KV_PAGE_SIZE: size of each kv page
        BLOCK_SIZE_M: tile size along the query tensor
        BLOCK_SIZE_N: tile size along the key/value tensor
    """

    # split threads along query tensor
    # Total number of blocks in sequence dimension (M)
    NUM_BLOCKS_M = tl.cdiv(Q_SEG_SIZE, BLOCK_SIZE_M)
    # Total tasks = number of sequence blocks × number of attention heads (Q_HEAD_NUM)
    NUM_BLOCKS = NUM_BLOCKS_M * Q_HEAD_NUM

    # Current M-dimension block index
    pid = tl.program_id(0)
    core_step = tl.num_programs(0)
    for block_idx in range(pid, NUM_BLOCKS, core_step):
        q_block_idx = block_idx % NUM_BLOCKS_M
        q_head_idx = block_idx // NUM_BLOCKS_M
        kv_head_idx = q_head_idx // (Q_HEAD_NUM // KV_HEAD_NUM)

        q_head_offset = q_head_idx * stride_q_head
        k_head_offset = kv_head_idx * stride_k_head
        v_head_offset = kv_head_idx * stride_v_head
        o_head_offset = q_head_idx * stride_o_head
        # Create block pointers for Q, K, V, Output
        Q_block_ptr = tl.make_block_ptr(
            base=q_ptr + q_head_offset,
            shape=(q_seg_len, HEAD_DIM),
            strides=(stride_q_m, stride_q_k),
            offsets=(q_block_idx * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=o_ptr + o_head_offset,
            shape=(q_seg_len, HEAD_DIM),
            strides=(stride_o_m, stride_o_k),
            offsets=(q_block_idx * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, HEAD_DIM),
            order=(1, 0),
        )

        m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) + 1.0

        # Initialize accumulator
        acc = tl.zeros([BLOCK_SIZE_M, HEAD_DIM], dtype=tl.float32)

        # load q: it will stay in SRAM throughout
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # accumulate over kv cache pages
        # tl.device_print("n_topk_pages %d", n_topk_pages)
        for i in range(n_topk_pages):
            topk_page_idx = tl.load(
                topk_page_indices_ptr + topk_page_indices_stride_head * q_head_idx + topk_page_indices_stride_idx * i
            )
            kv_offset = (topk_page_idx * KV_PAGE_SIZE).to(tl.int32)
            K_block_ptr = tl.make_block_ptr(
                base=k_ptr + k_head_offset,
                shape=(kv_cache_len, HEAD_DIM),
                strides=(stride_k_n, stride_k_k),
                offsets=(kv_offset, 0),
                block_shape=(BLOCK_SIZE_N, HEAD_DIM),
                order=(1, 0),
            )
            V_block_ptr = tl.make_block_ptr(
                base=v_ptr + v_head_offset,
                shape=(kv_cache_len, HEAD_DIM),
                strides=(stride_v_n, stride_v_k),
                offsets=(kv_offset, 0),
                block_shape=(BLOCK_SIZE_N, HEAD_DIM),
                order=(1, 0),
            )

            sub_seq = max(min(KV_PAGE_SIZE, kv_cache_len - kv_offset), 0)

            acc, l_i, m_i = _sdpa_infer_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
                session_mask_ptr,
                stride_session_mask_m,
                stride_session_mask_n,
                BLOCK_SIZE_MASK,
                scale,
                BLOCK_SIZE_M,
                HEAD_DIM,
                BLOCK_SIZE_N,
                kv_cache_len + q_seg_start + q_block_idx * BLOCK_SIZE_M,
                kv_offset,
                sub_seq,
                v_ptr.dtype.element_ty == tl.float8e5,
            )

        # accumulate over session kv
        K_block_ptr = tl.make_block_ptr(
            base=k_ptr + k_head_offset + kv_cache_len * stride_k_n,
            shape=(q_chunk_size, HEAD_DIM),
            strides=(stride_k_n, stride_k_k),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_N, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=v_ptr + v_head_offset + kv_cache_len * stride_v_n,
            shape=(q_chunk_size, HEAD_DIM),
            strides=(stride_v_n, stride_v_k),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_N, HEAD_DIM),
            order=(1, 0),
        )

        acc, l_i, m_i = _sdpa_infer_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            session_mask_ptr,
            stride_session_mask_m,
            stride_session_mask_n,
            BLOCK_SIZE_MASK,
            scale,
            BLOCK_SIZE_M,
            HEAD_DIM,
            BLOCK_SIZE_N,
            q_seg_start + q_block_idx * BLOCK_SIZE_M,
            0,
            q_seg_start + (q_block_idx + 1) * BLOCK_SIZE_M,
            v_ptr.dtype.element_ty == tl.float8e5,
        )

        m_i += tl.math.log(l_i)
        accumulator = acc / l_i[:, None]

        # NOTE(zhangjihang): for training
        # m_ptrs = M + task_bn_idx * SEQ + offs_m
        # tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1))


def block_sparse_attention_impl(
    curr_query_seg,
    key,
    value,
    scale,
    whole_causal_mask,
    topk_page_indices,
    q_seg_id,
    q_chunk_size,
    q_seg_size: int,
    page_size: int,
) -> torch.Tensor:
    # topk_page_indices = torch.sort(topk_page_indices, axis=1).values
    num_q_heads, n_topk_pages = topk_page_indices.shape

    _, q_seg_len, head_dim = curr_query_seg.shape
    num_kv_heads, kv_len, _ = key.shape
    kv_cache_len = kv_len - q_chunk_size
    q_seg_start = q_seg_id * q_seg_size

    print("page_idx:", topk_page_indices * page_size)
    print(f"{kv_len=} {kv_cache_len=} {q_chunk_size=}")

    o = torch.zeros_like(curr_query_seg, memory_format=torch.contiguous_format)
    assert o.is_contiguous()

    cube_num = get_num_cores("cube")
    num_cores = cube_num

    grid = (num_cores,)

    assert scale == 1 / math.sqrt(head_dim)
    assert q_seg_start + q_seg_len <= q_chunk_size
    print(curr_query_seg.shape, curr_query_seg.stride())
    # curr_seg_causal = whole_causal_mask[q_seg_start : q_seg_start + q_seg_len, -q_chunk_size:]
    # curr_seg_causal = whole_causal_mask[-q_chunk_size:, -q_chunk_size:]
    # print(f"{curr_seg_causal.shape=} {curr_seg_causal.stride(0)} {curr_seg_causal.stride(1)} {curr_seg_causal}")
    curr_seg_causal = whole_causal_mask

    mask_block_size = curr_seg_causal.size(0)

    block_sparse_fwd_kernel[grid](
        curr_query_seg,
        key,
        value,
        scale,
        topk_page_indices,
        o,
        curr_seg_causal,
        q_seg_start,
        q_seg_len,
        kv_cache_len,
        q_chunk_size,
        n_topk_pages,
        curr_query_seg.stride(0),
        curr_query_seg.stride(1),
        curr_query_seg.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        curr_seg_causal.stride(0),
        curr_seg_causal.stride(1),
        topk_page_indices.stride(0),
        topk_page_indices.stride(1),
        num_q_heads,
        num_kv_heads,
        head_dim,
        q_seg_size,
        page_size,
        BLOCK_SIZE_M=64,  # dummy, for test
        BLOCK_SIZE_N=128,  # dummy, for test
        BLOCK_SIZE_MASK=mask_block_size,
    )
    return o


@triton.jit
def block_sparse_attention_paged_prefill_kernel(
    o_ptr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    scale,
    block_table_ptr,
    mask_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    total_q_chunks,
    q_chunk_indices_ptr,
    selected_kv_page_indices_ptr,
    cu_num_kv_pages_per_chunk_ptr,
    o_stride_h,
    o_stride_l,
    o_stride_e,
    q_stride_h,
    q_stride_l,
    q_stride_e,
    k_stride_p,
    k_stride_h,
    k_stride_l,
    k_stride_e,
    v_stride_p,
    v_stride_h,
    v_stride_l,
    v_stride_e,
    block_table_stride_q,
    block_table_stride_p,
    mask_stride_q,
    mask_stride_kv,
    selected_kv_page_indices_stride_h,
    selected_kv_page_indices_stride_p,
    cu_num_kv_pages_per_chunk_stride_c,
    Q_HEAD_NUM: tl.constexpr,
    KV_HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_CHUNK_SIZE: tl.constexpr,
    KV_PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    AUX_MASK_SIZE: tl.constexpr,
):
    """
    Block sparse prefill attention kernel
    Args:
        q_ptr: Pointer to query tensor (Q), shape [Q_HEAD_NUM, q_seg_len, HEAD_DIM]
        k_ptr: Pointer to key tensor (K), shape [KV_HEAD_NUM, kv_cache_len+q_chunk_size, HEAD_DIM]
        v_ptr: Pointer to value tensor (V), shape [KV_HEAD_NUM, kv_cache_len+q_chunk_size, HEAD_DIM]
        topk_page_indices_ptr: Pointer to topk page indices, shape [Q_HEAD_NUM, n_topk_pages]
        o_ptr: Pointer to output tensor (O), shape [Q_HEAD_NUM, Q_SEG_SIZE, HEAD_DIM]
        session_mask_ptr: Pointer to session mask, shape [q_seg_len, q_chunk_size]

        KV_PAGE_SIZE: size of each kv page
        BLOCK_SIZE_M: tile size along the query tensor
        BLOCK_SIZE_N: tile size along the key/value tensor
    """

    # split threads along query tensor
    # Total number of blocks in sequence dimension (M)
    # Total tasks = number of sequence blocks × number of attention heads (Q_HEAD_NUM)
    NUM_BLOCKS = total_q_chunks * Q_HEAD_NUM

    # Current M-dimension block index
    pid = tl.program_id(0)
    core_step = tl.num_programs(0)
    for task_idx in range(pid, NUM_BLOCKS, core_step):
        q_chunk_idx = task_idx % total_q_chunks
        q_head_idx = task_idx // total_q_chunks
        kv_head_idx = q_head_idx // (Q_HEAD_NUM // KV_HEAD_NUM)
        q_id = tl.load(q_chunk_indices_ptr + q_chunk_idx * 2)
        seg_id = tl.load(q_chunk_indices_ptr + q_chunk_idx * 2 + 1)
        q_start = tl.load(cu_seqlens_q_ptr + q_id)
        q_end = tl.load(cu_seqlens_q_ptr + q_id + 1)
        q_len = q_end - q_start
        kv_start = tl.load(cu_seqlens_k_ptr + q_id)
        kv_end = tl.load(cu_seqlens_k_ptr + q_id + 1)
        kv_len = kv_end - kv_start
        kv_cache_len = kv_len - q_len

        q_seg_start = q_start + seg_id * Q_CHUNK_SIZE
        q_seg_len = min(q_end - q_seg_start, Q_CHUNK_SIZE)

        q_head_offset = q_head_idx * q_stride_h
        o_head_offset = q_head_idx * o_stride_h
        q_token_offset = q_seg_start * q_stride_l
        o_token_offset = q_seg_start * o_stride_l
        selected_pages_l = tl.load(cu_num_kv_pages_per_chunk_ptr + q_chunk_idx * cu_num_kv_pages_per_chunk_stride_c)
        selected_pages_r = tl.load(
            cu_num_kv_pages_per_chunk_ptr + (q_chunk_idx + 1) * cu_num_kv_pages_per_chunk_stride_c
        )
        for q_block_start in range(0, q_seg_len, BLOCK_SIZE_M):
            q_seg_offset = q_block_start.to(tl.int32)
            # Create block pointers for Q, K, V, Output
            Q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_head_offset + q_token_offset,
                shape=(q_seg_len, HEAD_DIM),
                strides=(q_stride_l, q_stride_e),
                offsets=(q_seg_offset, 0),
                block_shape=(BLOCK_SIZE_M, HEAD_DIM),
                order=(1, 0),
            )
            O_block_ptr = tl.make_block_ptr(
                base=o_ptr + o_head_offset + o_token_offset,
                shape=(q_seg_len, HEAD_DIM),
                strides=(o_stride_l, o_stride_e),
                offsets=(q_seg_offset, 0),
                block_shape=(BLOCK_SIZE_M, HEAD_DIM),
                order=(1, 0),
            )

            m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) + 1.0

            # Initialize accumulator
            acc = tl.zeros([BLOCK_SIZE_M, HEAD_DIM], dtype=tl.float32)

            # load q: it will stay in SRAM throughout
            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # accumulate over kv cache pages
            # tl.device_print("n_topk_pages %d", n_topk_pages)
            for selected_page_idx in range(selected_pages_l, selected_pages_r):
                logical_page_idx = tl.load(
                    selected_kv_page_indices_ptr
                    + q_head_idx * selected_kv_page_indices_stride_h
                    + selected_page_idx * selected_kv_page_indices_stride_p
                )
                physical_page_idx = tl.load(
                    block_table_ptr + block_table_stride_q * q_id + block_table_stride_p * logical_page_idx
                )
                k_page_offset = physical_page_idx * k_stride_p + kv_head_idx * k_stride_h
                v_page_offset = physical_page_idx * v_stride_p + kv_head_idx * v_stride_h
                page_start = 0
                page_end = max(min(KV_PAGE_SIZE, kv_cache_len - logical_page_idx * KV_PAGE_SIZE), 0)
                page_len = page_end - page_start
                page_offset = page_start

                K_block_ptr = tl.make_block_ptr(
                    base=k_cache_ptr + k_page_offset,
                    shape=(page_end, HEAD_DIM),
                    strides=(k_stride_l, k_stride_e),
                    offsets=(page_offset, 0),
                    block_shape=(BLOCK_SIZE_N, HEAD_DIM),
                    order=(1, 0),
                )
                V_block_ptr = tl.make_block_ptr(
                    base=v_cache_ptr + v_page_offset,
                    shape=(page_end, HEAD_DIM),
                    strides=(v_stride_l, v_stride_e),
                    offsets=(page_offset, 0),
                    block_shape=(BLOCK_SIZE_N, HEAD_DIM),
                    order=(1, 0),
                )

                acc, l_i, m_i = _sdpa_infer_inner(
                    acc,
                    l_i,
                    m_i,
                    q,
                    K_block_ptr,
                    V_block_ptr,
                    mask_ptr,
                    mask_stride_q,
                    mask_stride_kv,
                    AUX_MASK_SIZE,
                    scale,
                    BLOCK_SIZE_M,
                    HEAD_DIM,
                    BLOCK_SIZE_N,
                    kv_cache_len + seg_id * Q_CHUNK_SIZE + q_block_start,
                    logical_page_idx * KV_PAGE_SIZE + page_start,
                    page_len,
                    v_cache_ptr.dtype.element_ty == tl.float8e5,
                )

            new_kv_page_idx_start = kv_cache_len // KV_PAGE_SIZE
            new_kv_page_idx_end = tl.cdiv(
                kv_len,
                KV_PAGE_SIZE,
                # min(kv_cache_len + seg_id * Q_CHUNK_SIZE + q_block_start + BLOCK_SIZE_M, kv_len), KV_PAGE_SIZE
            )

            for logical_page_idx in range(new_kv_page_idx_start, new_kv_page_idx_end):
                physical_page_idx = tl.load(
                    block_table_ptr + block_table_stride_q * q_id + block_table_stride_p * logical_page_idx
                )
                k_page_offset = physical_page_idx * k_stride_p + kv_head_idx * k_stride_h
                v_page_offset = physical_page_idx * v_stride_p + kv_head_idx * v_stride_h

                page_start = max(kv_cache_len, logical_page_idx * KV_PAGE_SIZE) % KV_PAGE_SIZE
                page_end = max(min(KV_PAGE_SIZE, kv_len - logical_page_idx * KV_PAGE_SIZE), 0)
                page_len = page_end - page_start
                # accumulate over session kv
                page_offset = page_start.to(tl.int32)
                K_block_ptr = tl.make_block_ptr(
                    base=k_cache_ptr + k_page_offset,
                    shape=(page_end, HEAD_DIM),
                    strides=(k_stride_l, k_stride_e),
                    offsets=(page_offset, 0),
                    block_shape=(BLOCK_SIZE_N, HEAD_DIM),
                    order=(1, 0),
                )
                V_block_ptr = tl.make_block_ptr(
                    base=v_cache_ptr + v_page_offset,
                    shape=(page_end, HEAD_DIM),
                    strides=(v_stride_l, v_stride_e),
                    offsets=(page_offset, 0),
                    block_shape=(BLOCK_SIZE_N, HEAD_DIM),
                    order=(1, 0),
                )

                acc, l_i, m_i = _sdpa_infer_inner(
                    acc,
                    l_i,
                    m_i,
                    q,
                    K_block_ptr,
                    V_block_ptr,
                    mask_ptr,
                    mask_stride_q,
                    mask_stride_kv,
                    AUX_MASK_SIZE,
                    scale,
                    BLOCK_SIZE_M,
                    HEAD_DIM,
                    BLOCK_SIZE_N,
                    kv_cache_len + seg_id * Q_CHUNK_SIZE + q_block_start,
                    logical_page_idx * KV_PAGE_SIZE + page_start,
                    page_len,
                    v_cache_ptr.dtype.element_ty == tl.float8e5,
                )

            m_i += tl.math.log(l_i)
            accumulator = acc / l_i[:, None]

            # NOTE(zhangjihang): for training
            # m_ptrs = M + task_bn_idx * SEQ + offs_m
            # tl.store(m_ptrs, m_i)
            tl.store(O_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1))


def block_sparse_attention_paged_prefill_impl(
    query,
    key_cache,
    value_cache,
    scale,
    cu_seqlens_q,
    cu_seqlens_k,
    whole_causal_mask,
    kv_cache_indices,
    q_chunk_indices,
    kv_page_indices,
    cu_num_kv_pages_per_chunk,
    q_chunk_size: int,
    kv_page_size: int,
) -> torch.Tensor:
    # topk_page_indices = torch.sort(topk_page_indices, axis=1).values
    q_head_num = query.shape[0]
    head_dim = query.shape[2]
    kv_head_num = key_cache.shape[1]

    o = torch.zeros_like(query, memory_format=torch.contiguous_format)
    assert o.is_contiguous()
    total_q_chunks = q_chunk_indices.size(0)

    cube_num = get_num_cores("cube")
    num_cores = cube_num

    grid = (num_cores,)

    assert scale == 1 / math.sqrt(head_dim)
    # curr_seg_causal = whole_causal_mask[q_seg_start : q_seg_start + q_seg_len, -q_chunk_size:]
    # curr_seg_causal = whole_causal_mask[-q_chunk_size:, -q_chunk_size:]
    # print(f"{curr_seg_causal.shape=} {curr_seg_causal.stride(0)} {curr_seg_causal.stride(1)} {curr_seg_causal}")
    mask_block_size = whole_causal_mask.size(0)

    block_sparse_attention_paged_prefill_kernel[grid](
        o,
        query,
        key_cache,
        value_cache,
        scale,
        kv_cache_indices,
        whole_causal_mask,
        cu_seqlens_q,
        cu_seqlens_k,
        total_q_chunks,
        q_chunk_indices,
        kv_page_indices,
        cu_num_kv_pages_per_chunk,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        kv_cache_indices.stride(0),
        kv_cache_indices.stride(1),
        whole_causal_mask.stride(0),
        whole_causal_mask.stride(1),
        kv_page_indices.stride(0),
        kv_page_indices.stride(1),
        cu_num_kv_pages_per_chunk.stride(0),
        q_head_num,
        kv_head_num,
        head_dim,
        q_chunk_size,
        kv_page_size,
        BLOCK_SIZE_M=64,  # dummy, for test
        BLOCK_SIZE_N=128,  # dummy, for test
        AUX_MASK_SIZE=mask_block_size,
    )
    return o
