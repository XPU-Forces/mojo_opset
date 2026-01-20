from typing import Optional
from typing import Tuple

import torch

from mojo_opset.backends.ttx.kernels import diffusion_attention_bwd
from mojo_opset.backends.ttx.kernels import diffusion_attention_fwd
from mojo_opset.training.operators.attention import MojoDiffusionAttentionKernel


class TTXDiffusionAttentionKernel(MojoDiffusionAttentionKernel):
    supported_platforms_list = ["npu"]

    @staticmethod
    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        output, output_f32, lse = diffusion_attention_fwd(query, key, value, mask, scale)
        return output, output_f32, lse

    @staticmethod
    def backward(
        grad_output: torch.Tensor,
        output_f32: torch.Tensor,
        lse: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        grad_query, grad_key, grad_value = diffusion_attention_bwd(
            output_f32, grad_output, query, key, value, lse, mask, scale
        )
        return grad_query, grad_key, grad_value
