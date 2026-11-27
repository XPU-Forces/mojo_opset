"""Ascend A5 (950) Triton short-context top-k fill.

Semantics (vllm DeepSeek V4 CPU indexer ``_fill_short_context_topk``)::

    num_compressed = (positions[r] + 1) // compress_ratio
    out[r, i] = i   if i < num_compressed
    out[r, i] = -1  otherwise

Pure integer arithmetic, so the result is bit-exact against the torch reference.

Rows are merged ``RPP=4`` per program once the batch reaches 1024 rows; below the
threshold the block tail waste costs more than the per-row launch overhead, so
the single-row kernel is used. Both constants come from a cold-cache calibration on
this arch and are plain module constants (no autotune, no runtime
probe). The column mask folds away at compile time when ``top_k`` is a power of
two.
"""

import torch
import triton
import triton.language as tl

_RPP = 4
_ROW_THRESHOLD = 1024
# The merged tile is [RPP, next_pow2(top_k)] int32; 4 x 8192 overflows UB.
_ROW_MERGE_MAX_PADDED = 4096
# Ascend 950 supports the SIMT lowering; on this memcpy-shaped kernel it is the
# faster schedule for the small-batch path (the only place num_warps matters
# on the SIMD path).
_SMALL_BATCH_SIMT = True


@triton.jit
def _fill_short_context_topk_indices(
    output,
    positions,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, PADDED_TOP_K)
    num_compressed = (tl.load(positions + row) + 1) // COMPRESS_RATIO
    tl.store(
        output + row * TOP_K + offsets,
        tl.where(offsets < num_compressed, offsets, -1),
        mask=offsets < TOP_K,
    )


@triton.jit
def _fill_short_context_topk_indices_rpp(
    output,
    positions,
    num_rows,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
    RPP: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * RPP + tl.arange(0, RPP)
    row_mask = rows < num_rows
    pos = tl.load(positions + rows, mask=row_mask, other=0)
    num_compressed = (pos + 1) // COMPRESS_RATIO
    offsets = tl.arange(0, PADDED_TOP_K)
    vals = tl.where(offsets[None, :] < num_compressed[:, None], offsets[None, :], -1)
    ptrs = output + rows[:, None] * TOP_K + offsets[None, :]
    if TOP_K == PADDED_TOP_K:
        tl.store(ptrs, vals, mask=row_mask[:, None])
    else:
        tl.store(ptrs, vals, mask=row_mask[:, None] & (offsets[None, :] < TOP_K))


def _fill_short_context_topk_impl(topk_indices_buffer, positions, top_k, compress_ratio):
    num_tokens = positions.numel()
    if num_tokens == 0:
        return
    padded_top_k = triton.next_power_of_2(top_k)
    if num_tokens >= _ROW_THRESHOLD and padded_top_k <= _ROW_MERGE_MAX_PADDED:
        grid = ((num_tokens + _RPP - 1) // _RPP,)
        _fill_short_context_topk_indices_rpp[grid](
            topk_indices_buffer,
            positions,
            num_tokens,
            TOP_K=top_k,
            COMPRESS_RATIO=compress_ratio,
            PADDED_TOP_K=padded_top_k,
            RPP=_RPP,
        )
    else:
        launch = dict(
            TOP_K=top_k,
            COMPRESS_RATIO=compress_ratio,
            PADDED_TOP_K=padded_top_k,
        )
        if _SMALL_BATCH_SIMT:
            launch.update(num_warps=8, compile_mode="simt_only")
        _fill_short_context_topk_indices[(num_tokens,)](
            topk_indices_buffer,
            positions,
            **launch,
        )


@torch.library.custom_op(
    "mojo_npu_triton_a5::fill_short_context_topk_fwd",
    mutates_args=("topk_indices_buffer",),
)
def fill_short_context_topk_fwd(
    topk_indices_buffer: torch.Tensor,
    positions: torch.Tensor,
    top_k: int,
    compress_ratio: int,
) -> None:
    _fill_short_context_topk_impl(topk_indices_buffer, positions.contiguous(), top_k, compress_ratio)


@fill_short_context_topk_fwd.register_fake
def _fill_short_context_topk_fake(topk_indices_buffer, positions, top_k, compress_ratio) -> None:
    return None
