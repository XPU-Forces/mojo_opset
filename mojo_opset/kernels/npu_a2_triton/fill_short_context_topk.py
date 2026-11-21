"""Ascend A2 (910B) Triton short-context top-k fill.

Semantics (vllm DeepSeek V4 CPU indexer ``_fill_short_context_topk``)::

    num_compressed = (positions[r] + 1) // compress_ratio
    out[r, i] = i   if i < num_compressed
    out[r, i] = -1  otherwise

Pure integer arithmetic, so the result is bit-exact against the torch reference.

A2 keeps the one-program-per-row structure: the row-merge (RPP) tile exceeds the
per-core UB budget at ``[8, 2048]`` ("ub overflow") and the feasible tiles were
not competitive, so RPP is not compiled for this arch.
"""

import torch
import triton
import triton.language as tl


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


def _fill_short_context_topk_impl(topk_indices_buffer, positions, top_k, compress_ratio):
    num_tokens = positions.numel()
    if num_tokens == 0:
        return
    _fill_short_context_topk_indices[(num_tokens,)](
        topk_indices_buffer,
        positions,
        TOP_K=top_k,
        COMPRESS_RATIO=compress_ratio,
        PADDED_TOP_K=triton.next_power_of_2(top_k),
    )


@torch.library.custom_op(
    "mojo_npu_triton_a2::fill_short_context_topk_fwd",
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
