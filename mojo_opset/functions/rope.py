from typing import Optional

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl


class RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, unsqueeze_dim, implementation):
        forward, backward = load_impl("rope", implementation, require_backward=True)
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


def rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q/K using supplied cos/sin; partial RoPE rotates the suffix.

    Returns rotated Q/K and supports their gradients. Tables are supplied by
    the caller, optionally prepared with rope_cos_sin.
    """

    return RoPEFunction.apply(q, k, cos, sin, unsqueeze_dim, implementation)
