import torch
import triton
import triton.language as tl
from triton.runtime.libentry import libentry

@triton.autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_M': 1, 'BLOCK_N': 32}),
        triton.Config(kwargs={'BLOCK_M': 1, 'BLOCK_N': 64}),
        triton.Config(kwargs={'BLOCK_M': 1, 'BLOCK_N': 128}),
        triton.Config(kwargs={'BLOCK_M': 1, 'BLOCK_N': 256}),
        triton.Config(kwargs={'BLOCK_M': 16, 'BLOCK_N': 16}),
        triton.Config(kwargs={'BLOCK_M': 16, 'BLOCK_N': 32}),
        triton.Config(kwargs={'BLOCK_M': 16, 'BLOCK_N': 64}),
    ],
    key=['NUM_PAGES', 'Q_LEN']
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
        curr_query_seg = curr_query_seg.expand_dims(axis=1) # [BLOCK_M, 1, HEAD_DIM]

        for start_page in range(0, NUM_PAGES, BLOCK_N):
            offset_page = start_page + offset_block_page
            offset_minmax = head_num * stride_minmax_h + (offset_page * stride_minmax_p)[:, None] + offset_d[None, :]

            mins = tl.load(Mins + offset_minmax, mask=offset_page[:, None] < NUM_PAGES, other=0.0)
            mins = mins.expand_dims(axis=0) # [1, BLOCK_N, HEAD_DIM]
            q_min_k = curr_query_seg * mins

            maxs = tl.load(Maxs + offset_minmax, mask=offset_page[:, None] < NUM_PAGES, other=0.0)
            maxs = maxs.expand_dims(axis=0)  # [1, BLOCK_N, HEAD_DIM]
            q_max_k = curr_query_seg * maxs

            max_qk = tl.maximum(q_min_k, q_max_k)
            page_score_tile = tl.sum(max_qk, axis=-1)

            offset_score = head_num * stride_score_h + (offset_len * stride_score_l)[:, None] + offset_page[None, :]
            tl.store(Page_score + offset_score, page_score_tile, mask=(offset_len[:, None] < Q_LEN) & (offset_page[None, :] < NUM_PAGES))


def ttx_quest(
    curr_query_seg: torch.Tensor,
    mins: torch.Tensor,
    maxs: torch.Tensor,
    top_k_page: torch.uint8,
):
    num_heads, q_len, head_dim = curr_query_seg.shape
    assert top_k_page <= q_len

    num_pages = mins.shape[1]
    assert mins.shape[0] == maxs.shape[0]
    assert mins.shape[1] == maxs.shape[1]

    page_score = torch.zeros(num_heads, q_len, num_pages, device=mins.device, dtype=mins.dtype)

    grid = (num_heads, )
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