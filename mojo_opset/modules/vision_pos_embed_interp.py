"""Parameterized wrapper for vision position-embedding interpolation."""

from __future__ import annotations

from typing import Optional

import torch

from mojo_opset import functions


class VisionPosEmbedInterp(torch.nn.Module):
    """Interpolate a learned ``G x G`` position table onto vision grids.

    ``forward`` accepts either the upstream single-grid form
    ``(embed_weight, t, h, w)`` or the merged multi-grid form
    ``(embed_weight, grid_thw)``.  The latter issues one interpolation kernel
    for the whole list.
    """

    def __init__(
        self,
        num_grid_per_side: int = 48,
        m_size: int = 2,
        dtype: Optional[torch.dtype] = None,
        *,
        implementation: Optional[str] = None,
    ):
        super().__init__()
        self.num_grid_per_side = int(num_grid_per_side)
        self.m_size = int(m_size)
        self.dtype = dtype
        self.implementation = implementation

    def forward(self, embed_weight, t, h=None, w=None):
        if h is None and w is None:
            return functions.vision_pos_embed_interp_grids(
                embed_weight,
                t,
                self.num_grid_per_side,
                self.m_size,
                self.dtype,
                implementation=self.implementation,
            )
        if h is None or w is None:
            raise TypeError("pass either (embed_weight, t, h, w) or (embed_weight, grid_thw)")
        return functions.vision_pos_embed_interp(
            embed_weight,
            t,
            h,
            w,
            self.num_grid_per_side,
            self.m_size,
            self.dtype or embed_weight.dtype,
            implementation=self.implementation,
        )

    def extra_repr(self):
        return (
            f"num_grid_per_side={self.num_grid_per_side}, m_size={self.m_size}, "
            f"dtype={self.dtype}, implementation={self.implementation!r}"
        )
