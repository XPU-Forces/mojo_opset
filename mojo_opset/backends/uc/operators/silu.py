import torch

from mojo_opset.core import MojoSilu

from ._utils import run_unary_kernel


class UCSilu(MojoSilu):
    supported_platforms_list = ["npu"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_unary_kernel("mojo_silu", x)
