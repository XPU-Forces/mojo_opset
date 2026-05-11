import torch

from mojo_opset.core import MojoGelu

from ._utils import run_unary_kernel


class UCGelu(MojoGelu):
    supported_platforms_list = ["npu"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_unary_kernel("mojo_gelu", x)
