import torch

import triton
import triton.language as tl
from triton.runtime.libentry import libentry
from .utils import get_num_cores
from ..utils import prepare_chunk_indices, prepare_lens


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SEQ": 1, "BLOCK_PAGE": 32}),
        triton.Config(kwargs={"BLOCK_SEQ": 1, "BLOCK_PAGE": 64}),
        triton.Config(kwargs={"BLOCK_SEQ": 1, "BLOCK_PAGE": 128}),
        triton.Config(kwargs={"BLOCK_SEQ": 1, "BLOCK_PAGE": 256}),
        triton.Config(kwargs={"BLOCK_SEQ": 16, "BLOCK_PAGE": 16}),
        triton.Config(kwargs={"BLOCK_SEQ": 16, "BLOCK_PAGE": 32}),
        triton.Config(kwargs={"BLOCK_SEQ": 16, "BLOCK_PAGE": 64}),
    ],
    key=["Q_LEN"],
)
@libentry()
@triton.jit
def quest_reduce_kernel(
    Page_score,
    Curr_query_seg,
    Mins,
    Maxs,
    num_pages,
    q_len,
    STRIDE_SCORE_H: tl.constexpr,
    STRIDE_SCORE_L: tl.constexpr,
    STRIDE_SCORE_P: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_Q_L: tl.constexpr,
    STRIDE_Q_E: tl.constexpr,
    STRIDE_MIN_H: tl.constexpr,
    STRIDE_MIN_P: tl.constexpr,
    STRIDE_MIN_E: tl.constexpr,
    STRIDE_MAX_H: tl.constexpr,
    STRIDE_MAX_P: tl.constexpr,
    STRIDE_MAX_E: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_PAGE: tl.constexpr,
):
    pid = tl.program_id(0)
    core_step = tl.num_programs(0)
    num_q_blocks = tl.cdiv(q_len, BLOCK_SEQ)
    num_tasks = NUM_HEADS * num_q_blocks

    for task_id in range(pid, num_tasks, core_step):
        head_idx = task_id % NUM_HEADS
        offset_q = task_id // NUM_HEADS * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
        mask_q = offset_q < q_len
        curr_query_seg = tl.load(
            Curr_query_seg
            + head_idx * STRIDE_Q_H
            + offset_q[:, None] * STRIDE_Q_L
            + tl.arange(0, HEAD_DIM)[None, :] * STRIDE_Q_E,
            mask=mask_q[:, None],
        )  # [BLOCK_SEQ, HEAD_DIM]
        curr_query_seg = curr_query_seg.cast(tl.float32).expand_dims(1)  # [BLOCK_SEQ, 1, HEAD_DIM]

        for start_page in range(0, num_pages, BLOCK_PAGE):
            offset_page = start_page + tl.arange(0, BLOCK_PAGE)
            mask_page = offset_page < num_pages

            mins = tl.load(
                Mins
                + head_idx * STRIDE_MIN_H
                + offset_page[:, None] * STRIDE_MIN_P
                + tl.arange(0, HEAD_DIM) * STRIDE_MIN_E,
                mask=mask_page[:, None],
                other=0.0,
            )  # [BLOCK_PAGE, HEAD_DIM]
            mins = mins.cast(tl.float32).expand_dims(0)  # [1, BLOCK_PAGE, HEAD_DIM]
            q_min_k = curr_query_seg * mins  # [BLOCK_SEQ, BLOCK_PAGE, HEAD_DIM]

            maxs = tl.load(
                Maxs
                + head_idx * STRIDE_MAX_H
                + offset_page[:, None] * STRIDE_MAX_P
                + tl.arange(0, HEAD_DIM) * STRIDE_MAX_E,
                mask=mask_page[:, None],
                other=0.0,
            )  # [BLOCK_PAGE, HEAD_DIM]
            maxs = maxs.cast(tl.float32).expand_dims(0)  # [1, BLOCK_PAGE, HEAD_DIM]
            q_max_k = curr_query_seg * maxs  # [BLOCK_SEQ, BLOCK_PAGE, HEAD_DIM]

            max_qk = tl.maximum(q_min_k, q_max_k)
            page_score_tile = tl.sum(max_qk, axis=-1)  # [BLOCK_SEQ, BLOCK_PAGE]

            tl.store(
                Page_score
                + head_idx * STRIDE_SCORE_H
                + offset_q[:, None] * STRIDE_SCORE_L
                + offset_page[None, :] * STRIDE_SCORE_P,
                page_score_tile,
                mask=mask_q[:, None] & mask_page[None, :],
            )


def quest_impl(
    curr_query_seg: torch.Tensor,
    mins: torch.Tensor,
    maxs: torch.Tensor,
    top_k_page: int,
):
    num_heads, q_len, head_dim = curr_query_seg.shape

    assert mins.shape[0] == maxs.shape[0]
    assert mins.shape[1] == maxs.shape[1]
    num_pages = mins.shape[1]
    assert top_k_page <= num_pages

    page_score = torch.zeros(num_heads, q_len, num_pages, device=curr_query_seg.device, dtype=torch.float32)

    num_vec = get_num_cores("vector")
    grid = (num_vec,)
    quest_reduce_kernel[grid](
        page_score,
        curr_query_seg,
        mins,
        maxs,
        num_pages,
        q_len,
        page_score.stride(0),
        page_score.stride(1),
        page_score.stride(2),
        curr_query_seg.stride(0),
        curr_query_seg.stride(1),
        curr_query_seg.stride(2),
        mins.stride(0),
        mins.stride(1),
        mins.stride(2),
        maxs.stride(0),
        maxs.stride(1),
        maxs.stride(2),
        num_heads,
        head_dim,
    )
    _, topk_page_indices = page_score.topk(top_k_page, dim=-1)

    return topk_page_indices


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_PAGE": 32}),
        triton.Config(kwargs={"BLOCK_PAGE": 64}),
        triton.Config(kwargs={"BLOCK_PAGE": 128}),
        triton.Config(kwargs={"BLOCK_PAGE": 256}),
    ],
    key=["Q_LEN"],
)
@libentry()
@triton.jit
def paged_prefill_block_quest_reduce_kernel(
    Page_score,
    Query,
    cu_seqlens_q,
    q_chunk_indices,
    total_num_q_chunks,
    cu_tot_valid_pages_per_query,
    num_valid_pages_per_query,
    Page_k_mins,
    Page_k_maxs,
    kv_cache_indices,
    PAGE_SCORE_STRIDE_H: tl.constexpr,
    PAGE_SCORE_STRIDE_P: tl.constexpr,
    QUERY_STRIDE_H: tl.constexpr,
    QUERY_STRIDE_L: tl.constexpr,
    QUERY_STRIDE_E: tl.constexpr,
    PAGE_K_MINS_STRIDE_P: tl.constexpr,
    PAGE_K_MINS_STRIDE_H: tl.constexpr,
    PAGE_K_MINS_STRIDE_E: tl.constexpr,
    PAGE_K_MAXS_STRIDE_P: tl.constexpr,
    PAGE_K_MAXS_STRIDE_H: tl.constexpr,
    PAGE_K_MAXS_STRIDE_E: tl.constexpr,
    KV_CACHE_INDICES_STRIDE_Q: tl.constexpr,
    KV_CACHE_INDICES_STRIDE_P: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_PAGE: tl.constexpr,
):
    pid = tl.program_id(0)
    core_step = tl.num_programs(0)
    num_tasks = NUM_Q_HEADS * total_num_q_chunks

    for task_id in range(pid, num_tasks, core_step):
        q_head_idx = task_id % NUM_Q_HEADS
        q_chunk_idx = task_id // NUM_Q_HEADS
        q_id = tl.load(q_chunk_indices + q_chunk_idx * 2)
        seg_id = tl.load(q_chunk_indices + q_chunk_idx * 2 + 1)
        q_start = tl.load(cu_seqlens_q + q_id)
        q_end = tl.load(cu_seqlens_q + q_id + 1)
        seg_start = q_start + seg_id * CHUNK_SIZE
        curr_query = tl.load(
            Query + q_head_idx * QUERY_STRIDE_H + seg_start * QUERY_STRIDE_L + tl.arange(0, HEAD_DIM) * QUERY_STRIDE_E,
        )  # [HEAD_DIM]
        curr_query = curr_query.cast(tl.float32)

        kv_head_idx = q_head_idx // (NUM_Q_HEADS // NUM_KV_HEADS)

        num_pages = tl.load(num_valid_pages_per_query + q_id)

        curr_page_score_start = tl.load(cu_tot_valid_pages_per_query + q_id)

        for logical_page_id in range(0, num_pages):
            physical_page_id = tl.load(
                kv_cache_indices + q_id * KV_CACHE_INDICES_STRIDE_Q + logical_page_id * KV_CACHE_INDICES_STRIDE_P
            )

            mins = tl.load(
                Page_k_mins
                + physical_page_id * PAGE_K_MINS_STRIDE_P
                + kv_head_idx * PAGE_K_MINS_STRIDE_H
                + tl.arange(0, HEAD_DIM) * PAGE_K_MINS_STRIDE_E,
            )  # [HEAD_DIM]

            maxs = tl.load(
                Page_k_maxs
                + physical_page_id * PAGE_K_MAXS_STRIDE_P
                + kv_head_idx * PAGE_K_MAXS_STRIDE_H
                + tl.arange(0, HEAD_DIM) * PAGE_K_MAXS_STRIDE_E,
            )  # [HEAD_DIM]

            mins = mins.cast(tl.float32)  # [HEAD_DIM]
            q_min_k = curr_query * mins  # [HEAD_DIM]
            maxs = maxs.cast(tl.float32)  # [HEAD_DIM]
            q_max_k = curr_query * maxs  # [HEAD_DIM]

            max_qk = tl.maximum(q_min_k, q_max_k)
            page_score_tile = tl.sum(max_qk, axis=-1)  # [1]

            tl.store(
                Page_score
                + q_head_idx * PAGE_SCORE_STRIDE_H
                + (curr_page_score_start + seg_id * num_pages + logical_page_id) * PAGE_SCORE_STRIDE_P,
                page_score_tile,
            )


def paged_prefill_block_quest_impl(
    query: torch.Tensor,
    cu_seqlens_q: torch.LongTensor,
    page_k_mins: torch.Tensor,
    page_k_maxs: torch.Tensor,
    kv_cache_indices: torch.Tensor,
    cu_seqlens_k: torch.LongTensor,
    num_topk_pages: torch.LongTensor,
    chunk_size: int,
    page_size: int,
    recent_window: int,
):
    num_q_heads, q_len, head_dim = query.shape
    num_kv_heads = page_k_mins.shape[1]
    q_lengths = prepare_lens(cu_seqlens_q)
    num_q_chunks = (q_lengths + chunk_size - 1) // chunk_size

    q_chunk_indices = prepare_chunk_indices(cu_seqlens_q, chunk_size)
    total_num_q_chunks = q_chunk_indices.shape[0]

    kv_lengths = prepare_lens(cu_seqlens_k)
    num_valid_pages = torch.clamp(kv_lengths - q_lengths - recent_window, min=0) // page_size
    tot_valid_pages_per_query = num_valid_pages * num_q_chunks
    cu_tot_valid_pages_per_query = torch.nn.functional.pad(
        torch.cumsum(tot_valid_pages_per_query, dim=0), (1, 0), value=0
    )

    num_total_pages = cu_tot_valid_pages_per_query[-1].item()

    page_score = torch.zeros((num_q_heads, num_total_pages), device=query.device, dtype=torch.float32)

    num_vec = get_num_cores("vector")
    grid = (num_vec,)

    paged_prefill_block_quest_reduce_kernel[grid](
        page_score,
        query,
        cu_seqlens_q,
        q_chunk_indices,
        total_num_q_chunks,
        cu_tot_valid_pages_per_query,
        num_valid_pages,
        page_k_mins,
        page_k_maxs,
        kv_cache_indices,
        page_score.stride(0),
        page_score.stride(1),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        page_k_mins.stride(0),
        page_k_mins.stride(1),
        page_k_mins.stride(2),
        page_k_maxs.stride(0),
        page_k_maxs.stride(1),
        page_k_maxs.stride(2),
        kv_cache_indices.stride(0),
        kv_cache_indices.stride(1),
        num_q_heads,
        num_kv_heads,
        chunk_size,
        head_dim,
    )
    num_topk_pages = torch.minimum(num_topk_pages, num_valid_pages)
    num_topk_pages_per_seg = num_topk_pages.repeat_interleave(num_q_chunks)
    num_sparse_pages_per_seg = num_valid_pages.repeat_interleave(num_q_chunks)

    cu_num_topk_pages_per_seg = torch.cumsum(num_topk_pages_per_seg, dim=0)
    cu_num_topk_pages_per_seg = torch.nn.functional.pad(cu_num_topk_pages_per_seg, (1, 0), value=0)

    topk_page_indices = []
    for i in range(len(kv_cache_indices)):
        curr_q_page_scores = page_score[:, cu_tot_valid_pages_per_query[i] : cu_tot_valid_pages_per_query[i + 1]]
        curr_q_page_scores = curr_q_page_scores.reshape(num_q_heads, num_q_chunks[i].item(), -1)
        _, chunk_topk_page_indices = curr_q_page_scores.topk(num_topk_pages[i].item(), dim=-1)
        topk_page_indices.append(chunk_topk_page_indices.reshape(num_q_heads, -1))

    return (
        torch.cat(topk_page_indices, dim=-1),
        q_chunk_indices,
        num_sparse_pages_per_seg,
        cu_num_topk_pages_per_seg,
    )
