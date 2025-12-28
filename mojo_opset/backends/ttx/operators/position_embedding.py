import torch

from mojo_opset.backends.ttx.kernels import rope_fwd
from mojo_opset.core import MojoRoPE


class TTXRoPE(MojoRoPE):
    supported_platforms_list = ["npu"]

    def forward(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        return rope_fwd(q, k, cos, sin)
