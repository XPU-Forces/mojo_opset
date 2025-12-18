from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels.ascend.convolution import causal_conv1d_bwd
from mojo_opset.backends.ttx.kernels.ascend.convolution import causal_conv1d_fwd
from mojo_opset.backends.ttx.kernels.utils import input_guard
from mojo_opset.core import MojoCausalConv1dFunction


class TTXCausalConv1dFunction(MojoCausalConv1dFunction):
    @staticmethod
    @input_guard(make_contiguous=True, auto_to_device=True)
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
    def backward(ctx, dy: torch.Tensor, dht: Optional[torch.Tensor]):
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
