from typing import Optional

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, scales, implementation):
        forward, backward = load_impl("swiglu", implementation, require_backward=True)
        ctx.backward_kernel = backward
        ctx.has_scales = scales is not None
        if scales is None:
            ctx.save_for_backward(x1, x2)
        else:
            ctx.save_for_backward(x1, x2, scales)
        return forward(x1, x2, scales)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if ctx.has_scales:
            x1, x2, scales = ctx.saved_tensors
        else:
            x1, x2 = ctx.saved_tensors
            scales = None
        grad_x1, grad_x2, grad_scales = ctx.backward_kernel(
            grad_output.contiguous(), x1, x2, scales
        )
        return grad_x1, grad_x2, grad_scales if ctx.has_scales else None, None


def swiglu(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scales: Optional[torch.Tensor] = None,
    *,
    swiglu_limit: float = 0.0,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """SwiGLU with optional per-row scales and the original asymmetric clamp."""

    if swiglu_limit > 0:
        x1 = x1.clamp(max=swiglu_limit)
        x2 = x2.clamp(min=-swiglu_limit, max=swiglu_limit)
    return SwiGLUFunction.apply(x1, x2, scales, implementation)
