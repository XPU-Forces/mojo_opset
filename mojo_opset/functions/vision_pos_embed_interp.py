"""Bilinear vision position-embedding interpolation for Qwen3-VL style towers.

Public entry points
-------------------
``vision_pos_embed_interp`` mirrors the upstream vLLM
``triton_pos_embed_interpolate(embed_weight, t, h, w, num_grid_per_side,
m_size, dtype)`` contract and returns ``[t * h * w, hidden_dim]`` in
spatial-merge order.

``vision_pos_embed_interp_grids`` is the merged entry for a ``grid_thw`` list.
It produces exactly ``torch.cat([single(grid) for grid in grids])`` while
issuing one interpolation kernel launch.
"""

from __future__ import annotations

from typing import Optional
from typing import Sequence

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def _normalize_grids(grid_thw: Sequence[Sequence[int]]) -> list[tuple[int, int, int]]:
    try:
        raw = list(grid_thw)
    except TypeError:
        raise TypeError("grid_thw must be a sequence of (t, h, w) triples") from None
    grids: list[tuple[int, int, int]] = []
    for entry in raw:
        values = list(entry)
        if len(values) != 3:
            raise ValueError(f"each grid_thw entry must have three values, got {entry!r}")
        t, h, w = (int(value) for value in values)
        if t <= 0:
            raise ValueError(f"grid_thw requires t > 0, got {entry!r}")
        grids.append((t, h, w))
    return grids


def _coerce_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        try:
            return getattr(torch, dtype)
        except AttributeError:
            pass
    raise TypeError(f"dtype must be a torch.dtype, got {dtype!r}")


def _validate(
    embed_weight: torch.Tensor,
    grids: list[tuple[int, int, int]],
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> None:
    if embed_weight.ndim != 2:
        raise ValueError("embed_weight must be a 2D [G*G, hidden_dim] tensor")
    if embed_weight.shape[0] != num_grid_per_side * num_grid_per_side:
        raise ValueError(
            "embed_weight rows must equal num_grid_per_side ** 2, "
            f"got {embed_weight.shape[0]} and {num_grid_per_side}"
        )
    if embed_weight.shape[1] <= 0:
        raise ValueError("hidden_dim must be positive")
    if num_grid_per_side < 2:
        raise ValueError("num_grid_per_side must be at least 2")
    if m_size < 1:
        raise ValueError("m_size must be positive")
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"dtype must be float32 or bfloat16, got {dtype}")
    for t, h, w in grids:
        if h <= 0 or w <= 0:
            raise ValueError(f"h and w must be positive, got {(t, h, w)!r}")
        if h % m_size != 0 or w % m_size != 0:
            raise ValueError(f"h={h} and w={w} must be divisible by m_size={m_size}")


def vision_pos_embed_interp(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Interpolate one ``(t, h, w)`` grid; upstream-compatible signature and layout."""
    _require_inference("vision_pos_embed_interp", embed_weight)
    dtype = _coerce_dtype(dtype)
    grids = _normalize_grids([(t, h, w)])
    _validate(embed_weight, grids, int(num_grid_per_side), int(m_size), dtype)
    forward, _ = load_impl("vision_pos_embed_interp", implementation)
    return forward(embed_weight, grids, int(num_grid_per_side), int(m_size), dtype)


def vision_pos_embed_interp_grids(
    embed_weight: torch.Tensor,
    grid_thw: Sequence[Sequence[int]],
    num_grid_per_side: int,
    m_size: int,
    dtype: Optional[torch.dtype] = None,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Merged multi-image entry: one interpolation kernel for the whole list."""
    _require_inference("vision_pos_embed_interp_grids", embed_weight)
    if dtype is None:
        dtype = embed_weight.dtype
    dtype = _coerce_dtype(dtype)
    grids = _normalize_grids(grid_thw)
    _validate(embed_weight, grids, int(num_grid_per_side), int(m_size), dtype)
    forward, _ = load_impl("vision_pos_embed_interp", implementation)
    return forward(embed_weight, grids, int(num_grid_per_side), int(m_size), dtype)


def vision_pos_embed_interp_grids_add(
    embed_weight: torch.Tensor,
    grid_thw: Sequence[Sequence[int]],
    num_grid_per_side: int,
    m_size: int,
    residual: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Fused interpolation + residual add: out = interp(embed) + residual.

    Model-integration entry only. It removes one full write+read of the
    intermediate position embedding, but it is NOT part of the standalone
    benchmark (the benchmark contract times the interpolation kernel alone).
    """
    _require_inference("vision_pos_embed_interp_grids_add", embed_weight)
    if dtype is None:
        dtype = embed_weight.dtype
    dtype = _coerce_dtype(dtype)
    grids = _normalize_grids(grid_thw)
    _validate(embed_weight, grids, int(num_grid_per_side), int(m_size), dtype)
    forward, _ = load_impl("vision_pos_embed_interp", implementation)
    try:
        return forward(embed_weight, grids, int(num_grid_per_side), int(m_size), dtype, add=residual)
    except TypeError:
        out = forward(embed_weight, grids, int(num_grid_per_side), int(m_size), dtype)
        return out + residual


# Short alias used by some call sites; same semantics as ``_grids``.
vision_pos_embed_interp_multi = vision_pos_embed_interp_grids
