"""Bilinear positional-embedding interpolation for vision inputs (Qwen3-VL)."""

from typing import Optional

import torch

from ._dispatch import load_impl


def vision_pos_embed_interp(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: Optional[torch.dtype] = None,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Interpolate ``embed_weight`` (``[num_grid**2, hidden]``) onto the ``h x w``
    patch grid, tiling across ``t`` frames; output is ``[t*h*w, hidden]``.

    ``h`` and ``w`` must be divisible by ``m_size`` (the spatial merge size).
    ``dtype`` defaults to ``embed_weight.dtype``.
    """
    dtype = embed_weight.dtype if dtype is None else dtype
    if embed_weight.ndim != 2 or embed_weight.shape[0] != num_grid_per_side**2:
        raise ValueError(f"embed_weight must be [{num_grid_per_side**2}, hidden]")
    if h % m_size or w % m_size:
        raise ValueError(f"h={h} and w={w} must be divisible by m_size={m_size}")
    forward, _ = load_impl("vision_pos_embed_interp", implementation)
    return forward(embed_weight, int(t), int(h), int(w), int(num_grid_per_side), int(m_size), dtype)
