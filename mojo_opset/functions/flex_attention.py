"""Original Mojo FlexAttention interface and its no-dropout autograd wrapper."""

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl
from ._flex_attention import _restore_block_mask
from ._flex_attention import _save_block_mask
from ._flex_attention import _validate


class FlexAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, block_mask, scale, implementation, op_id, enable_gqa=True, score_mod=None, return_lse=False
    ):
        if score_mod is not None or return_lse:
            raise NotImplementedError("FlexAttention does not support score_mod or return_lse")
        _validate(q, k, v, block_mask)
        if not enable_gqa and q.shape[1] != k.shape[1]:
            raise ValueError("enable_gqa=False requires equal Q/K/V head counts")
        forward, backward = load_impl(op_id, implementation, require_backward=True)
        output, lse = forward(q, k, v, block_mask, scale)
        mask_tensors = _save_block_mask(ctx, block_mask)
        ctx.save_for_backward(q, k, v, output, lse, *mask_tensors)
        ctx.scale, ctx.backward_kernel = scale, backward
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v, output, lse, *mask_tensors = ctx.saved_tensors
        block_mask = _restore_block_mask(ctx, mask_tensors)
        dq, dk, dv = ctx.backward_kernel(grad_output.contiguous(), q, k, v, output, lse, block_mask, ctx.scale)
        return dq, dk, dv, None, None, None, None, None, None, None


def flex_attention(q, k, v, block_mask=None, sm_scale=None, *, implementation=None):
    """Mojo interface: grouped [B,H,S,D] attention with a prebuilt BlockMask."""
    return FlexAttentionFunction.apply(q, k, v, block_mask, sm_scale, implementation, "flex_attention")
