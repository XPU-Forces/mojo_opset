"""DSA candidate-block selection and masking over packed logits.

Both operators are inference-only and write in place: ``select_candidate_blocks``
fills ``out`` and ``apply_candidate_mask`` masks ``logits`` itself.
"""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def _check_bounds(op, rows, row_ks, row_ke, row_repeat):
    if row_repeat < 1:
        raise ValueError(f"{op} requires row_repeat >= 1, got {row_repeat}")
    if row_ke.ndim != 1 or row_ke.dtype != torch.int32:
        raise ValueError(f"{op} requires row_ke to be a 1D int32 tensor")
    if row_ks is not None:
        if row_ks.ndim != 1 or row_ks.dtype != torch.int32:
            raise ValueError(f"{op} requires row_ks to be a 1D int32 tensor")
        if row_ks.numel() != row_ke.numel():
            raise ValueError(f"{op} requires row_ks and row_ke to have the same length")
    packed = -(-rows // row_repeat)
    if packed > row_ke.numel():
        raise ValueError(f"{op} requires row_ke to cover {packed} packed rows, got {row_ke.numel()}")


def select_candidate_blocks(
    logits: torch.Tensor,
    row_ks: Optional[torch.Tensor],
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Select the highest scoring block ids of every row into ``out``.

    Block ``b`` of a row covers the packed columns
    ``[start + b * block_size, start + (b + 1) * block_size)`` where ``start``
    is the row's ``row_ks`` entry (0 when ``row_ks`` is None). The row's newest
    block is pinned to +inf so it always survives the top-k, and blocks with no
    in-bounds column score -inf. Ties keep ``torch.topk`` order.

    Args:
        logits: 2D scores ``[rows, width]``; ``width`` is a packed column span.
        row_ks: 1D int32 packed row starts, or None for all-zero starts.
        row_ke: 1D int32 packed row ends.
        topk_blocks: number of block ids written per row.
        block_size: logical block width in packed columns.
        out: 2D int32 ``[rows, topk_blocks]`` block ids, padded with -1.
        row_repeat: number of expanded rows sharing one packed bound.
        implementation: select a specific implementation instead of the
            configured one.

    Returns:
        ``out``.
    """
    _require_inference("select_candidate_blocks", logits, out)
    if logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("logits must be a 2D floating-point tensor")
    rows = logits.shape[0]
    _check_bounds("select_candidate_blocks", rows, row_ks, row_ke, row_repeat)
    if topk_blocks < 1 or block_size < 1:
        raise ValueError("topk_blocks and block_size must be >= 1")
    if out.ndim != 2 or out.dtype != torch.int32:
        raise ValueError("out must be a 2D int32 tensor")
    if out.shape != (rows, topk_blocks):
        raise ValueError(f"out must have shape {(rows, topk_blocks)}, got {tuple(out.shape)}")

    forward, _ = load_impl("select_candidate_blocks", implementation)
    forward(logits, row_ks, row_ke, topk_blocks, block_size, out, row_repeat)
    return out


def apply_candidate_mask(
    logits: torch.Tensor,
    row_ks: Optional[torch.Tensor],
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int = 1,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Mask packed logits to -inf outside the causal bounds and candidates.

    Columns outside ``[start, end)`` are masked, and so is every column whose
    block does not appear in ``candidate_blocks``. Candidates past the packed
    width clamp onto the row's last column, which keeps the newest block alive.

    Args:
        logits: 2D scores ``[rows, width]``, modified in place.
        row_ks: 1D int32 packed row starts, or None for all-zero starts.
        row_ke: 1D int32 packed row ends.
        candidate_blocks: 2D int32 block ids; negative entries are ignored.
        block_size: logical block width in packed columns.
        row_repeat: number of expanded rows sharing one packed bound.
        implementation: select a specific implementation instead of the
            configured one.

    Returns:
        ``logits``.
    """
    _require_inference("apply_candidate_mask", logits)
    if logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError("logits must be a 2D floating-point tensor")
    rows = logits.shape[0]
    _check_bounds("apply_candidate_mask", rows, row_ks, row_ke, row_repeat)
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if candidate_blocks.ndim != 2 or candidate_blocks.dtype != torch.int32:
        raise ValueError("candidate_blocks must be a 2D int32 tensor")
    if candidate_blocks.shape[0] != rows:
        raise ValueError(f"candidate_blocks must have {rows} rows, got {candidate_blocks.shape[0]}")

    forward, _ = load_impl("apply_candidate_mask", implementation)
    forward(logits, row_ks, row_ke, candidate_blocks, block_size, row_repeat)
    return logits
