"""Stateless modules for the original inference RoPE interfaces."""

import torch

from mojo_opset import functions


class RoPEInfer(torch.nn.Module):
    def __init__(self, *, head_first=True, keep_cos_sin_dtype=False, implementation=None):
        super().__init__()
        self.head_first = head_first
        self.keep_cos_sin_dtype = keep_cos_sin_dtype
        self.implementation = implementation

    def forward(self, q, k, cos, sin):
        return functions.rope_infer(
            q,
            k,
            cos,
            sin,
            head_first=self.head_first,
            keep_cos_sin_dtype=self.keep_cos_sin_dtype,
            implementation=self.implementation,
        )


class VisionRoPE2DInfer(torch.nn.Module):
    def __init__(self, *, implementation=None):
        super().__init__()
        self.implementation = implementation

    def forward(self, q, k, cos, sin):
        return functions.vision_rope_2d_infer(q, k, cos, sin, implementation=self.implementation)
