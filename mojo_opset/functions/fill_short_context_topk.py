"""Short-context shortcut for the DeepSeek V4 sparse-MLA indexer (in-place top-k fill)."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def fill_short_context_topk(
    topk_indices_buffer: torch.Tensor,
    positions: torch.Tensor,
    top_k: int,
    compress_ratio: int,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Select every candidate in order while the sequence fits in ``top_k`` slots.

    ``output[r, i] = i`` when ``i < (positions[r] + 1) // compress_ratio``, else ``-1``.
    ``topk_indices_buffer`` is int32 ``[num_tokens, top_k]`` (a persistent buffer in
    production) and is written in place and returned; ``positions`` is a 1D
    int32/int64 ``[num_tokens]`` tensor whose range is
    ``positions <= top_k * compress_ratio - 1``.
    """
    _require_inference("fill_short_context_topk", topk_indices_buffer, positions)
    if top_k <= 0 or compress_ratio <= 0:
        raise ValueError("top_k and compress_ratio must be positive")
    if topk_indices_buffer.dtype != torch.int32 or topk_indices_buffer.ndim != 2:
        raise ValueError("topk_indices_buffer must be an int32 [num_tokens, top_k] tensor")
    if topk_indices_buffer.shape[1] != top_k or not topk_indices_buffer.is_contiguous():
        raise ValueError("topk_indices_buffer must be contiguous with exactly top_k columns")
    if positions.ndim != 1 or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be a 1D int32/int64 tensor")
    if positions.numel() != topk_indices_buffer.shape[0]:
        raise ValueError("positions must hold one entry per token")
    if positions.device != topk_indices_buffer.device:
        raise ValueError("positions and topk_indices_buffer must be on the same device")
    forward, _ = load_impl("fill_short_context_topk", implementation)
    forward(topk_indices_buffer, positions, int(top_k), int(compress_ratio))
    return topk_indices_buffer
