"""Dense inference SDPA and the original diffusion-training contract."""

from typing import Optional

import torch

from torch.autograd.function import once_differentiable

from ._checks import _require_inference
from ._dispatch import load_impl


def sdpa_infer(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    *,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Dense SDPA. The inherited Triton specialization requires S % 512 == 0 and D in {64, 128}."""
    _require_inference("sdpa_infer", query, key, value)
    forward, _ = load_impl("sdpa_infer", implementation)
    # Make the public torch SDPA default explicit; the old Triton launcher used 1.0 for None.
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    return forward(query, key, value, attn_mask, scale, enable_gqa)


class DiffusionAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, mask, scale, enable_gqa, implementation):
        forward, ctx.backward_kernel = load_impl("diffusion_attention", implementation, require_backward=True)
        output, output_fp32, lse = forward(query, key, value, mask, scale, enable_gqa)
        ctx.save_for_backward(query, key, value, mask, output_fp32, lse)
        ctx.scale, ctx.enable_gqa = scale, enable_gqa
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v, mask, output_fp32, lse = ctx.saved_tensors
        grads = ctx.backward_kernel(
            output_fp32, grad_output.contiguous(), q, k, v, lse, mask, ctx.scale, ctx.enable_gqa
        )
        return *grads, None, None, None, None


def diffusion_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    scale: float = 1.0,
    enable_gqa: bool = False,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Text diffusion attention on concatenated noisy/clean sequences.

    The inherited Triton implementation specializes a 32-token block mask:
    noisy queries see their noisy block and earlier clean blocks; clean queries
    see their current and earlier clean blocks. The supplied boolean mask must
    match this pattern, and the total sequence length must be divisible by 512
    for Triton. For arbitrary masks use flex_attention instead.
    """
    return DiffusionAttentionFunction.apply(query, key, value, mask, scale, enable_gqa, implementation)
