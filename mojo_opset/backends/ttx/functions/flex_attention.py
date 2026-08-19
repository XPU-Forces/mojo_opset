from typing import Optional
from typing import Tuple

import torch

from mojo_opset.backends.ttx.kernels import flex_attention_bwd
from mojo_opset.backends.ttx.kernels import flex_attention_fwd
from mojo_opset.experimental import MojoFlexAttentionFunction


class TTXFlexAttentionFunction(MojoFlexAttentionFunction):
    """TTX (Triton NPU) backend for FlexAttention.

    Uses the optimized Triton kernels for both forward and backward passes
    with BlockMask sparse attention.
    """

    supported_platforms_list = ["npu"]

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask=None,
        sm_scale: Optional[float] = None,
    ) -> torch.Tensor:
        output, lse = flex_attention_fwd(
            q,
            k,
            v,
            block_mask,
            sm_scale,
        )
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.block_mask = block_mask
        ctx.sm_scale = sm_scale
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
        grad_lse: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        q, k, v, output, lse = ctx.saved_tensors
        block_mask = ctx.block_mask
        sm_scale = ctx.sm_scale

        dq, dk, dv = flex_attention_bwd(
            grad_output,
            q,
            k,
            v,
            output,
            lse,
            block_mask,
            sm_scale,
        )
        return dq, dk, dv, None, None
