"""Gated LayerNorm / RMSNorm fused forward (``layer_norm_gated``)."""

from typing import Optional

import torch

from ._dispatch import load_impl


def _run_forward(
    x,
    weight,
    bias,
    eps,
    z,
    out,
    group_size,
    norm_before_gate,
    is_rms_norm,
    activation,
    quant_heads,
    implementation,
):
    forward, _ = load_impl("layer_norm_gated", implementation)
    return forward(
        x,
        weight,
        bias,
        float(eps),
        z=z,
        out=out,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
        quant_heads=quant_heads,
    )


class LayerNormGatedFunction(torch.autograd.Function):
    """Forward-only operator: the gated normalization ships without a backward."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps, z, out, group_size, norm_before_gate, is_rms_norm, activation, quant_heads, implementation):
        if out is not None and not quant_heads:
            # The kernel writes in place; marking the buffer dirty keeps apply()
            # returning the caller's tensor object rather than a new view of it.
            ctx.mark_dirty(out)
        return _run_forward(
            x,
            weight,
            bias,
            eps,
            z,
            out,
            group_size,
            norm_before_gate,
            is_rms_norm,
            activation,
            quant_heads,
            implementation,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("layer_norm_gated provides the forward pass only.")


def layer_norm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float = 1e-6,
    *,
    z: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
    quant_heads: int = 0,
    implementation: Optional[str] = None,
):
    """Normalize ``x`` (LayerNorm or RMSNorm), optionally gated by ``z``.

    Mirrors ``sglang``'s ``_layer_norm_fwd``: returns ``((q, s) if quant_heads else out, mean, rstd)``.
    ``mean`` is ``None`` for RMSNorm; both statistics are FP32 and group-major.
    """

    return LayerNormGatedFunction.apply(
        x,
        weight,
        bias,
        eps,
        z,
        out,
        group_size,
        norm_before_gate,
        is_rms_norm,
        activation,
        quant_heads,
        implementation,
    )


def _layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
    quant_heads: int = 0,
):
    """Drop-in name for the upstream wrapper (same positional signature)."""

    return layer_norm_gated(
        x,
        weight,
        bias,
        eps,
        z=z,
        out=out,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
        quant_heads=quant_heads,
    )


def rms_norm_gated(
    *,
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
    implementation: Optional[str] = None,
):
    """Upstream convenience wrapper: returns only the normalized output."""

    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        if z.shape == x_shape_og:
            z = z.reshape(-1, z.shape[-1])
            if z.stride(-1) != 1:
                z = z.contiguous()
        else:
            assert len(x_shape_og) == 2
            assert z.ndim == 3
            assert z.shape[0] * z.shape[1] == x.shape[0]
            assert z.shape[2] == x.shape[1]
            assert z.stride(-1) == 1
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y, _, _ = layer_norm_gated(
        x,
        weight,
        bias,
        eps,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
        implementation=implementation,
    )
    return y.reshape(x_shape_og)
