"""Bilinear positional-embedding interpolation for vision inputs (Qwen3-VL).

Public entry under the op id ``vision_pos_embed_interp``. The kernel and its validation
live in :mod:`mojo_opset.functions.position_embedding`; this module only fixes the public
name so the acceptance suite's ``functions.vision_pos_embed_interp`` mark resolves.
"""

from typing import Optional

import torch

from ._dispatch import load_impl
from .position_embedding import _validate_grid
from .position_embedding import _validate_pos_embed


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
    """Interpolate ``embed_weight`` (``[num_grid**2, hidden]``) onto the ``h x w`` patch grid,
    tiling across ``t`` frames; the output is ``[t*h*w, hidden]`` in spatial-merge order.

    ``h`` and ``w`` must be divisible by ``m_size`` (the spatial merge size). ``dtype``
    defaults to ``embed_weight.dtype``.
    """
    dtype = embed_weight.dtype if dtype is None else dtype
    _validate_pos_embed(embed_weight, num_grid_per_side, m_size, "vision_pos_embed_interp")
    _validate_grid(t, h, w, m_size)
    forward, _ = load_impl("vision_pos_embed_interp", implementation)
    return forward(embed_weight, int(t), int(h), int(w), int(num_grid_per_side), int(m_size), dtype)
