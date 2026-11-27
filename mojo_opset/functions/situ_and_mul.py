from typing import Optional

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl


class SituAndMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta, linear_beta, implementation):
        forward, backward = load_impl("situ_and_mul", implementation, require_backward=True)
        ctx.backward_kernel = backward
        ctx.beta = beta
        ctx.linear_beta = linear_beta
        ctx.save_for_backward(x)
        return forward(x, beta, linear_beta)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = ctx.backward_kernel(grad_output.contiguous(), x, ctx.beta, ctx.linear_beta)
        return grad_x, None, None, None


def situ_and_mul(
    x: torch.Tensor,
    *,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """SiTU gated activation from Kimi-K3 on concatenated ``[gate | up]`` inputs
    (Kimi-K3: ``beta=4.0``, ``linear_beta=25.0``)."""

    if x.shape[-1] % 2 != 0:
        raise ValueError(f"situ_and_mul expects an even last dimension, got {x.shape[-1]}.")
    return SituAndMulFunction.apply(x, beta, linear_beta, implementation)
