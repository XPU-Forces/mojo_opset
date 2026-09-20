from typing import Optional

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl


class RmsNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, implementation):
        forward, backward = load_impl("rms_norm", implementation, require_backward=True)
        output, rstd = forward(x, weight, eps)
        ctx.backward_kernel = backward
        ctx.save_for_backward(x, weight, rstd)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight, rstd = ctx.saved_tensors
        grad_x, grad_weight = ctx.backward_kernel(grad_output.contiguous(), x, weight, rstd)
        return grad_x, grad_weight, None, None


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """RMS normalization with output dtype matching ``x``."""

    if weight.dim() != 1:
        raise ValueError("RMSNorm weight must be one-dimensional.")
    if x.shape[-1] != weight.numel():
        raise ValueError(f"RMSNorm weight has {weight.numel()} elements, expected {x.shape[-1]}.")
    return RmsNormFunction.apply(x, weight, float(eps), implementation)
