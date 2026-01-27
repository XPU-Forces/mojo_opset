from typing import Optional
from typing import Tuple

import torch

from mojo_opset.backends.ttx.kernels.npu.convolution import causal_conv1d_bwd
from mojo_opset.backends.ttx.kernels.npu.convolution import causal_conv1d_fwd
from mojo_opset.backends.ttx.kernels.utils import input_guard
from mojo_opset.core import MojoCausalConv1dFunction


class TTXCausalConv1dFunction(MojoCausalConv1dFunction):
    @staticmethod
    @input_guard(make_contiguous=True, auto_to_device=True)
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        activation: str = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        weight = weight.permute(1, 0).contiguous()
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.save_for_backward(x, weight, bias, residual, initial_state)

        y, final_state = causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            output_final_state=output_final_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
        )
        return y, final_state

    @staticmethod
    @input_guard(make_contiguous=True, auto_to_device=True)
    def backward(
        ctx,
        dy: torch.Tensor,
        dht: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        x, weight, bias, residual, initial_state = ctx.saved_tensors

        dx, dw, db, dr, dh0 = causal_conv1d_bwd(
            x=x,
            dy=dy,
            dht=dht,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=ctx.activation,
            cu_seqlens=ctx.cu_seqlens,
        )
        return dx, dw, db, dr, dh0, None, None, None
