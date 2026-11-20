"""Parameter-configured position-table interpolation (inference only)."""

import torch

from mojo_opset import functions


class VisionPosEmbedInterp(torch.nn.Module):
    def __init__(self, num_grid_per_side=48, m_size=2, *, implementation=None):
        super().__init__()
        self.num_grid_per_side = num_grid_per_side
        self.m_size = m_size
        self.implementation = implementation

    def forward(self, embed_weight, grid_thw, dtype=None):
        return functions.vision_pos_embed_interp_batch(
            embed_weight,
            grid_thw,
            self.num_grid_per_side,
            self.m_size,
            embed_weight.dtype if dtype is None else dtype,
            implementation=self.implementation,
        )
