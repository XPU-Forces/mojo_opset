import torch

from mojo_opset.backends.ttx.kernels import silu_bwd
from mojo_opset.backends.ttx.kernels import silu_fwd
from mojo_opset.core import MojoSiluFunction


class TTXSiluFunction(MojoSiluFunction):
    supported_platforms_list = ["npu"]

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
    ) -> torch.Tensor:
        y = silu_fwd(input)
        ctx.save_for_backward(input)
        return y

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        dx = silu_bwd(grad_output, input)
        return dx
