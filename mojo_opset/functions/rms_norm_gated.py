from typing import Optional

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl


class RMSNormGatedFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        activation: str,
        residual: Optional[torch.Tensor],
        eps: float,
        prenorm: bool,
        residual_in_fp32: bool,
        implementation: Optional[str],
    ):
        op_id = "rms_norm_gated"
        forward, backward = load_impl(op_id, implementation, require_backward=True)

        x_shape_og = x.shape
        g_shape_og = g.shape
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        g_2d = g.reshape(-1, g.shape[-1]).contiguous()
        if g_2d.shape != x_2d.shape:
            raise ValueError("g must have the same shape as x")
        if residual is not None:
            if residual.shape != x_shape_og:
                raise ValueError("residual must have the same shape as x")
            residual = residual.reshape(-1, residual.shape[-1]).contiguous()
        residual_dtype = residual.dtype if residual is not None else (torch.float if residual_in_fp32 else None)

        y, _, rstd, residual_out = forward(
            x=x_2d,
            g=g_2d,
            weight=weight,
            bias=bias,
            activation=activation,
            eps=float(eps),
            residual=residual,
            residual_dtype=residual_dtype,
        )
        ctx.save_for_backward(residual_out, g_2d, weight, bias, rstd)
        ctx.backward_kernel = backward
        ctx.x_shape_og = x_shape_og
        ctx.g_shape_og = g_shape_og
        ctx.activation = activation
        ctx.eps = float(eps)
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x_2d.dtype
        y = y.reshape(x_shape_og)
        return y if not prenorm else (y, residual_out.reshape(x_shape_og))

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, *args):
        x, g, weight, bias, rstd = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1]).contiguous()
        if ctx.prenorm:
            dresidual = args[0].reshape(-1, args[0].shape[-1]).contiguous()
        else:
            dresidual = None
        dx, dg, dw, db, dresidual_in = ctx.backward_kernel(
            dy=dy,
            x=x,
            g=g,
            weight=weight,
            bias=bias,
            activation=ctx.activation,
            eps=ctx.eps,
            rstd=rstd,
            dresidual=dresidual,
            has_residual=ctx.has_residual,
            x_dtype=ctx.x_dtype,
        )[:5]
        return (
            dx.reshape(ctx.x_shape_og),
            dg.reshape(ctx.g_shape_og),
            dw,
            db,
            None,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )


def rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    activation: str = "swish",
    residual: Optional[torch.Tensor] = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """RMSNorm over the last dimension followed by a silu/sigmoid gate."""

    return RMSNormGatedFunction.apply(
        x, g, weight, bias, activation, residual, eps, prenorm, residual_in_fp32, implementation
    )
