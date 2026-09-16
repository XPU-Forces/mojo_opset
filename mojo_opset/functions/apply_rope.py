from typing import Optional

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl


class ApplyRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, unsqueeze_dim, implementation):
        forward, backward = load_impl("apply_rope", implementation, require_backward=True)
        ctx.backward_kernel = backward
        ctx.unsqueeze_dim = unsqueeze_dim
        ctx.save_for_backward(cos, sin)
        return forward(q, k, cos, sin, unsqueeze_dim)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_q, grad_k):
        cos, sin = ctx.saved_tensors
        grad_q, grad_k = ctx.backward_kernel(
            grad_q.contiguous(), grad_k.contiguous(), cos, sin, ctx.unsqueeze_dim
        )
        return grad_q, grad_k, None, None, None, None


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply partial or full rotary position embedding."""

    return ApplyRoPEFunction.apply(q, k, cos, sin, unsqueeze_dim, implementation)
