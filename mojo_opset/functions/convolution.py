from typing import Optional

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl


def causal_conv1d_update_state_infer(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Depthwise decode convolution on [B,D,T], updating [B,D,S] state in place.

    S must be at least width-1. Return the last T convolution outputs in the
    input dtype; state keeps its dtype and storage. Compute in the weight dtype.
    This is inference-only and distinct from causal_conv1d's returned state.
    """
    x = hidden_states
    if x.ndim != 3 or conv_state.ndim != 3 or weight.ndim != 2:
        raise ValueError("expected hidden_states[B,D,T], conv_state[B,D,S], weight[D,W]")
    if (conv_state.shape[:2] != x.shape[:2] or weight.shape[0] != x.shape[1]
            or weight.shape[1] < 1 or conv_state.shape[-1] < weight.shape[1] - 1):
        raise ValueError("incompatible convolution state or weight shape")
    if activation not in (None, "silu", "swish"):
        raise ValueError("activation must be None, 'silu', or 'swish'")
    if bias is not None and (bias.shape != (x.shape[1],) or bias.dtype != weight.dtype):
        raise ValueError("bias must have shape [D] and the weight dtype")
    for tensor in (x, conv_state, weight, bias):
        if tensor is None:
            continue
        if tensor.device != x.device or tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("convolution tensors must be floating and on the input device")
        if torch.is_grad_enabled() and tensor.requires_grad:
            raise RuntimeError("causal_conv1d_update_state_infer does not support autograd; use torch.no_grad()")
    forward, _ = load_impl("causal_conv1d_update_state_infer", implementation)
    return forward(x, conv_state, weight, bias, activation)


class CausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        residual,
        initial_state,
        output_final_state,
        activation,
        cu_seqlens,
        implementation,
    ):
        forward, backward = load_impl("causal_conv1d", implementation, require_backward=True)
        output, final_state = forward(
            x,
            weight,
            bias,
            residual,
            initial_state,
            output_final_state,
            activation,
            cu_seqlens,
        )
        ctx.backward_kernel = backward
        ctx.has_bias = bias is not None
        ctx.has_residual = residual is not None
        ctx.has_initial_state = initial_state is not None
        ctx.output_final_state = output_final_state
        ctx.activation = activation
        ctx.save_for_backward(x, weight, bias, residual, initial_state, cu_seqlens)
        return output, final_state if output_final_state else None

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_final_state=None):
        x, weight, bias, residual, initial_state, cu_seqlens = ctx.saved_tensors
        grad_x, grad_weight, grad_bias, grad_residual, grad_initial_state = ctx.backward_kernel(
            grad_output.contiguous(),
            grad_final_state,
            x,
            weight,
            bias,
            residual,
            initial_state,
            ctx.output_final_state,
            ctx.activation,
            cu_seqlens,
        )
        return (
            grad_x,
            grad_weight,
            grad_bias if ctx.has_bias else None,
            grad_residual if ctx.has_residual else None,
            grad_initial_state if ctx.has_initial_state else None,
            None,
            None,
            None,
            None,
        )


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Depthwise causal convolution over ``[batch, time, channels]`` inputs."""

    if x.ndim != 3 or weight.ndim != 2 or x.shape[-1] != weight.shape[0]:
        raise ValueError("x must be [batch, time, channels] and weight must be [channels, width]")
    if activation not in {None, "silu", "swish"}:
        raise ValueError("activation must be None, 'silu', or 'swish'")
    if weight.shape[1] < 1:
        raise ValueError("convolution width must be positive")
    batches = x.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
    if initial_state is not None and initial_state.shape != (batches, x.shape[-1], weight.shape[1] - 1):
        raise ValueError("initial_state must have shape [batch, channels, width - 1]")
    if cu_seqlens is not None and (cu_seqlens.ndim != 1 or x.shape[0] != 1):
        raise ValueError("varlen inputs require x [1, tokens, channels] and 1D cu_seqlens")
    if residual is not None and residual.shape != x.shape:
        raise ValueError("residual must have the same shape as x")
    return CausalConv1dFunction.apply(
        x,
        weight,
        bias,
        residual,
        initial_state,
        bool(output_final_state),
        activation,
        cu_seqlens,
        implementation,
    )
