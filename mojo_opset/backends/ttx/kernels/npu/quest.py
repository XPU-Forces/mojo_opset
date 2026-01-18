import torch

import triton
import triton.language as tl
from triton.runtime.libentry import libentry
from .utils import get_num_cores


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
