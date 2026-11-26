"""Bilinear learned position embeddings in spatial-merge order."""

import operator

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


class _InvalidGridError(ValueError, AssertionError):
    """Invalid spatial dimensions, compatible with legacy assertion handlers."""


def _validated_grids(embed_weight, grid_thw, num_grid_per_side, m_size, dtype):
    if embed_weight.ndim != 2 or embed_weight.shape[1] < 1:
        raise ValueError("embed_weight must have shape [G*G, hidden_dim] with hidden_dim > 0")
    if embed_weight.dtype not in (torch.float32, torch.bfloat16) or dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("position interpolation supports FP32 and BF16")
    g, m = operator.index(num_grid_per_side), operator.index(m_size)
    if g < 1 or m < 1 or embed_weight.shape[0] != g * g:
        raise ValueError("G and m_size must be positive and embed_weight.shape[0] must be G*G")
    flat = []
    for grid in grid_thw:
        if len(grid) != 3:
            raise ValueError("each grid must contain exactly (t, h, w)")
        t, h, w = (operator.index(value) for value in grid)
        if min(t, h, w) < 1:
            raise ValueError("t, h and w must be positive")
        if h % m or w % m:
            raise _InvalidGridError(f"h={h} and w={w} must be divisible by m_size={m}")
        flat.extend((t, h, w))
    return flat, g, m


def vision_pos_embed_interp_batch(embed_weight, grid_thw, num_grid_per_side, m_size, dtype, *, implementation=None):
    """Interpolate all grids into one contiguous output with one Triton launch.

    ``grid_thw`` is a host sequence of integer (t,h,w) triples. Each image's
    [t*h*w,D] output occupies its list-ordered slice. An empty list returns [0,D].
    No intermediate per-image outputs or device metadata copies are created.
    """
    _require_inference("vision_pos_embed_interp_batch", embed_weight)
    flat, g, m = _validated_grids(embed_weight, grid_thw, num_grid_per_side, m_size, dtype)
    forward, _ = load_impl("vision_pos_embed_interp_batch", implementation)
    return forward(embed_weight, flat, g, m, dtype)


def vision_pos_embed_interp(embed_weight, t, h, w, num_grid_per_side, m_size, dtype, *, implementation=None):
    """Return [t*h*w,D] interpolated position embeddings in spatial-merge order."""
    _require_inference("vision_pos_embed_interp", embed_weight)
    flat, g, m = _validated_grids(embed_weight, ((t, h, w),), num_grid_per_side, m_size, dtype)
    forward, _ = load_impl("vision_pos_embed_interp", implementation)
    return forward(embed_weight, *flat, g, m, dtype)


def triton_pos_embed_interpolate(embed_weight, t, h, w, num_grid_per_side, m_size, dtype):
    """vLLM-compatible seven-argument API; keep its public signature unchanged."""
    return vision_pos_embed_interp(embed_weight, t, h, w, num_grid_per_side, m_size, dtype, implementation="triton")
