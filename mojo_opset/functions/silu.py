from typing import Optional

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl


class SiluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, implementation: Optional[str]) -> torch.Tensor:
        forward, backward = load_impl("silu", implementation, require_backward=True)
        ctx.backward_kernel = backward
        ctx.save_for_backward(x)
        return forward(x)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        return ctx.backward_kernel(grad_output.contiguous(), x), None


def silu(
    x: torch.Tensor,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """SiLU dispatched to one exact provider implementation."""

    return SiluFunction.apply(x, implementation)
