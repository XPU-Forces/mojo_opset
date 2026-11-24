"""State-only SDPA module; execution belongs to functions."""

from torch import nn

from mojo_opset.functions.sdpa import sdpa_infer


class Sdpa(nn.Module):
    def __init__(self, scale=None, enable_gqa=False, *, implementation=None):
        super().__init__()
        self.scale, self.enable_gqa, self.implementation = scale, enable_gqa, implementation

    def forward(self, query, key, value, attn_mask=None):
        return sdpa_infer(
            query,
            key,
            value,
            attn_mask,
            scale=self.scale,
            enable_gqa=self.enable_gqa,
            implementation=self.implementation,
        )
