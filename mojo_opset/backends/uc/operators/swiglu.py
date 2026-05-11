import torch

from mojo_opset.core import MojoSwiGLU

from ._utils import run_binary_kernel


class UCSwiGLU(MojoSwiGLU):
    supported_platforms_list = ["npu"]

    def forward(self, gate_out: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
        if self.swiglu_limit > 0:
            return super().forward(gate_out, up_out)
        return run_binary_kernel("mojo_swiglu", gate_out, up_out)
