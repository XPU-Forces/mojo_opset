from typing import Optional

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl


class GeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, approximate: str, implementation: Optional[str]) -> torch.Tensor:
        forward, backward = load_impl("gelu", implementation, require_backward=True)
        ctx.backward_kernel = backward
        ctx.approximate = approximate
        ctx.save_for_backward(x)
        return forward(x, approximate)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        return ctx.backward_kernel(grad_output.contiguous(), x, ctx.approximate), None, None


def gelu(
    x: torch.Tensor,
    *,
    approximate: str = "tanh",
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """GELU with explicit tanh or exact-erf semantics; Triton currently supports tanh only."""

    if approximate not in ("tanh", "none"):
        raise ValueError("approximate must be 'tanh' or 'none'")
    return GeluFunction.apply(x, approximate, implementation)
