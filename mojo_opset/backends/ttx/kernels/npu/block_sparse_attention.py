import torch
import math

import triton
import triton.language as tl
from triton.runtime.libentry import libentry


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_M": 1, "BLOCK_N": 32}),
        triton.Config(kwargs={"BLOCK_M": 1, "BLOCK_N": 64}),
        triton.Config(kwargs={"BLOCK_M": 1, "BLOCK_N": 128}),
        triton.Config(kwargs={"BLOCK_M": 1, "BLOCK_N": 256}),
        triton.Config(kwargs={"BLOCK_M": 16, "BLOCK_N": 16}),
        triton.Config(kwargs={"BLOCK_M": 16, "BLOCK_N": 32}),
        triton.Config(kwargs={"BLOCK_M": 16, "BLOCK_N": 64}),
    ],
    key=["NUM_PAGES", "Q_LEN"],
)
@libentry()
@triton.jit
def quest_reduce_kernel(
    Curr_query_seg,
    Mins,
    Maxs,
    Page_score,
    stride_query_h,
    stride_query_l,
    stride_minmax_h,
    stride_minmax_p,
    stride_score_h,
    stride_score_l,
    NUM_PAGES: tl.constexpr,
    Q_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    head_num = pid

    offset_block_len = tl.arange(0, BLOCK_M)
    offset_block_page = tl.arange(0, BLOCK_N)
    offset_d = tl.arange(0, HEAD_DIM)

    for start_len in range(0, Q_LEN, BLOCK_M):
        offset_len = start_len + offset_block_len
        offset_query = head_num * stride_query_h + (offset_len * stride_query_l)[:, None] + offset_d[None, :]
        curr_query_seg = tl.load(Curr_query_seg + offset_query, mask=offset_len[:, None] < Q_LEN, other=0.0)
        curr_query_seg = curr_query_seg.expand_dims(axis=1)  # [BLOCK_M, 1, HEAD_DIM]

        for start_page in range(0, NUM_PAGES, BLOCK_N):
            offset_page = start_page + offset_block_page
            offset_minmax = head_num * stride_minmax_h + (offset_page * stride_minmax_p)[:, None] + offset_d[None, :]

            mins = tl.load(Mins + offset_minmax, mask=offset_page[:, None] < NUM_PAGES, other=0.0)
            mins = mins.expand_dims(axis=0)  # [1, BLOCK_N, HEAD_DIM]
            q_min_k = curr_query_seg * mins

            maxs = tl.load(Maxs + offset_minmax, mask=offset_page[:, None] < NUM_PAGES, other=0.0)
            maxs = maxs.expand_dims(axis=0)  # [1, BLOCK_N, HEAD_DIM]
            q_max_k = curr_query_seg * maxs

            max_qk = tl.maximum(q_min_k, q_max_k)
            page_score_tile = tl.sum(max_qk, axis=-1)

            offset_score = head_num * stride_score_h + (offset_len * stride_score_l)[:, None] + offset_page[None, :]
            tl.store(
                Page_score + offset_score,
                page_score_tile,
                mask=(offset_len[:, None] < Q_LEN) & (offset_page[None, :] < NUM_PAGES),
            )


def block_quest_impl(
    curr_query_seg: torch.Tensor,
    mins: torch.Tensor,
    maxs: torch.Tensor,
    top_k_page: int,
):
    curr_query_seg = curr_query_seg[:, 0]
    num_heads, head_dim = curr_query_seg.shape
    q_len = 1

    assert mins.shape[0] == maxs.shape[0]
    assert mins.shape[1] == maxs.shape[1]
    num_pages = mins.shape[1]
    assert top_k_page <= num_pages

    page_score = torch.zeros(num_heads, num_pages, device=mins.device, dtype=mins.dtype)

    grid = (num_heads,)
    quest_reduce_kernel[grid](
        curr_query_seg,
        mins,
        maxs,
        page_score,
        curr_query_seg.stride(0),
        curr_query_seg.stride(1),
        mins.stride(0),
        mins.stride(1),
        page_score.stride(0),
        page_score.stride(1),
        NUM_PAGES=num_pages,
        Q_LEN=q_len,
        HEAD_DIM=head_dim,
    )
    _, topk_page_indices = page_score.topk(top_k_page, dim=-1)

    return topk_page_indices


@triton.jit
def get_mask_ptr_offset_n_kvcache(q_start, q_block, q_len, kv_start, kv_block, kv_len):
    offset_kv = min(max(kv_start + kv_block - kv_len, 0), kv_block)
    return offset_kv


@triton.jit
def get_mask_ptr_offset_n_causal(q_start, q_block, q_len, kv_start, kv_block, kv_len):
    offset_kv = min(max(kv_start - q_start, -kv_block), kv_block)
    return offset_kv


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
    mask_mod_fn,
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
        kv_mask = tl.arange(0, BLOCK_N) + start_n < SEQ
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
            offset_n = mask_mod_fn(offs_m, BLOCK_M, BLOCK_M, start_n + offs_n, BLOCK_N, SEQ)
            mask_ptr = (
                mask_base_ptr
                + tl.arange(0, BLOCK_M)[:, None] * stride_mask_base_ptr_m
                + (offset_n + tl.arange(0, BLOCK_N))[None, :] * stride_mask_base_ptr_n
            )
            # tl.device_print("start_m %d", start_m)
            # tl.device_print("start_n %d", start_n)
            mask = tl.load(mask_ptr).to(tl.int1)
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
def paged_sparse_prefill_kernel(
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
    # BLOCK_SIZE_D: tl.constexpr,
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
                session_mask_ptr + q_block_idx * BLOCK_SIZE_M * stride_session_mask_m,
                stride_session_mask_m,
                stride_session_mask_n,
                get_mask_ptr_offset_n_kvcache,
                scale,
                BLOCK_SIZE_M,
                HEAD_DIM,
                BLOCK_SIZE_N,
                0,
                0,
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
            session_mask_ptr
            + q_block_idx * BLOCK_SIZE_M * stride_session_mask_m
            + (KV_PAGE_SIZE * 2 + Q_SEG_SIZE + q_block_idx * BLOCK_SIZE_M) * stride_session_mask_n,
            stride_session_mask_m,
            stride_session_mask_n,
            get_mask_ptr_offset_n_causal,
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


def block_sparse_paged_attention_prefill_impl(
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

    # cube_num, vector_num = get_device_properties()
    # num_cores = cube_num
    num_cores = 20

    grid = (num_cores,)

    assert scale == 1 / math.sqrt(head_dim)
    assert q_seg_start + q_seg_len <= q_chunk_size
    print(curr_query_seg.shape, curr_query_seg.stride())
    # curr_seg_causal = whole_causal_mask[q_seg_start : q_seg_start + q_seg_len, -q_chunk_size:]
    # curr_seg_causal = whole_causal_mask[-q_chunk_size:, -q_chunk_size:]
    # print(f"{curr_seg_causal.shape=} {curr_seg_causal.stride(0)} {curr_seg_causal.stride(1)} {curr_seg_causal}")
    curr_seg_causal = whole_causal_mask

    print(f"{topk_page_indices.stride()=}")
    paged_sparse_prefill_kernel[grid](
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
        BLOCK_SIZE_N=64,  # dummy, for test
    )
    return o
