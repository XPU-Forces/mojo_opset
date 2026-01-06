from typing import Optional

import torch

from mojo_opset.core import MojoCausalConv1dFunction


class RefCausalConv1dFunction(MojoCausalConv1dFunction):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        activation: str = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        pass

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dht: Optional[torch.Tensor]):
        pass
