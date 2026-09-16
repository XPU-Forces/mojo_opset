import torch

from mojo_opset import functions


class MRoPE(torch.nn.Module):
    def __init__(self, *, implementation=None):
        super().__init__()
        self.implementation = implementation

    def forward(self, q, k, cos, sin, mrope_section, is_interleaved=False, head_dim=None):
        return functions.mrope(
            q, k, cos, sin, mrope_section, is_interleaved, head_dim, implementation=self.implementation
        )


class MRoPEInplace(torch.nn.Module):
    def __init__(self, inplace=False, *, implementation=None):
        super().__init__()
        self.inplace, self.implementation = inplace, implementation

    def forward(self, q, k, cos, sin, mrope_section, is_interleaved=False, head_dim=None):
        return functions.mrope(
            q,
            k,
            cos,
            sin,
            mrope_section,
            is_interleaved,
            head_dim,
            inplace=self.inplace,
            implementation=self.implementation,
        )
