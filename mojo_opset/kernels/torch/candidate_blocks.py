"""Torch reference for DSA candidate-block selection over packed logits.

The two entry points mirror the NPU Triton kernels bit for bit: every output is
either an integer id or a max/select over already-materialized values, so there
is no rounding path that would justify a tolerance.
"""

from typing import Optional

import torch


def _expand_bounds(
    rows: int,
    row_ks: Optional[torch.Tensor],
    row_ke: torch.Tensor,
    row_repeat: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map packed row bounds onto the expanded rows the kernels iterate."""
    group = torch.arange(rows, device=device) // row_repeat
    ends = row_ke[group].to(torch.int64)
    if row_ks is None:
        starts = torch.zeros(rows, dtype=torch.int64, device=device)
    else:
        starts = row_ks[group].to(torch.int64)
    return starts, ends


def select_candidate_blocks_fwd(
    logits: torch.Tensor,
    row_ks: Optional[torch.Tensor],
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int,
) -> None:
    """Write the ``topk_blocks`` best block ids per row into ``out``."""
    rows, width = logits.shape
    if not rows:
        return
    if not width:
        out.fill_(-1)
        return

    nblocks = -(-width // block_size)
    starts, ends = _expand_bounds(rows, row_ks, row_ke, row_repeat, logits.device)
    cols = torch.arange(width, device=logits.device, dtype=torch.int64)
    valid = (cols >= starts[:, None]) & (cols < ends[:, None])

    # Per-block max over the row's in-bounds columns; blocks without any valid
    # column keep the -inf fill, which the top-k turns into padding.
    scores = logits.masked_fill(~valid, -torch.inf)
    blocks = ((cols - starts[:, None]) // block_size).clamp(min=0)
    reduced = logits.new_full((rows, nblocks), -torch.inf)
    reduced.scatter_reduce_(1, blocks, scores, reduce="amax", include_self=True)

    # The newest block is pinned to +inf so it always survives the top-k. The
    # block id uses the raw end (not clipped to width), matching the kernel.
    lengths = ends - starts
    last = (lengths - 1).div(block_size, rounding_mode="floor")
    pinned = (lengths > 0) & (last < nblocks)
    reduced[pinned, last[pinned]] = torch.inf

    top = reduced.topk(min(topk_blocks, nblocks), dim=-1)
    out.fill_(-1)
    out[:, : top.indices.shape[1]] = torch.where(
        top.values != -torch.inf, top.indices, -1
    ).to(out.dtype)


def apply_candidate_mask_fwd(
    logits: torch.Tensor,
    row_ks: Optional[torch.Tensor],
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int,
) -> None:
    """Mask ``logits`` to -inf outside the causal bounds and candidate blocks."""
    rows, width = logits.shape
    if not rows or not width:
        return

    starts, ends = _expand_bounds(rows, row_ks, row_ke, row_repeat, logits.device)
    cols = torch.arange(width, device=logits.device, dtype=torch.int64)
    valid = (cols >= starts[:, None]) & (cols < ends[:, None])

    # Every candidate block contributes its whole column window. Blocks past
    # the packed width clamp onto the last column, which is what the kernel's
    # ``nblocks`` overflow bucket plus its last-column edge flag express.
    # Padded (negative) ids are routed to a sink column so they cannot clobber
    # a real candidate that clamped onto column 0.
    offsets = torch.arange(block_size, device=logits.device, dtype=torch.int64)
    kept = (candidate_blocks >= 0)[:, :, None].expand(-1, -1, block_size)
    positions = (
        starts[:, None, None]
        + candidate_blocks.to(torch.int64)[:, :, None] * block_size
        + offsets[None, None, :]
    ).clamp(0, width - 1)
    positions = torch.where(kept, positions, width)
    keep = torch.zeros((rows, width + 1), dtype=torch.int8, device=logits.device)
    keep.scatter_(1, positions.reshape(rows, -1), 1)
    keep = keep[:, :width]

    logits.masked_fill_((keep == 0) | ~valid, -torch.inf)
