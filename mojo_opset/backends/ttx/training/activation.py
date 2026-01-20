import torch

from mojo_opset.backends.ttx.kernels import silu_bwd
from mojo_opset.backends.ttx.kernels import silu_fwd
from mojo_opset.training.operators.activation import MojoSiluKernel


class TTXSiluKernel(MojoSiluKernel):
    supported_platforms_list = ["npu"]

    @staticmethod
    def forward(
        input: torch.Tensor,
    ) -> torch.Tensor:
        output = silu_fwd(input)
        return output

    @staticmethod
    def backward(
        grad_output: torch.Tensor,
        input: torch.Tensor,
    ) -> torch.Tensor:
        grad_input = silu_bwd(grad_output, input)
        return grad_input
