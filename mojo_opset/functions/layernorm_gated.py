"""Fused LayerNorm/RMSNorm with optional gating and a per-token FP8 epilogue."""

from typing import Optional

import torch

from ._dispatch import load_impl

_ACTIVATIONS = ("swish", "silu", "sigmoid")


def _normalize_gate(z, x_shape, x_2d_shape):
    """Return a gate that shares the flattened row order of ``x``."""

    if z is None:
        return None
    if tuple(z.shape) == tuple(x_shape):
        z = z.reshape(x_2d_shape)
        return z if z.stride(-1) == 1 else z.contiguous()
    if len(x_shape) != 2:
        raise ValueError("3D gated views require a 2D input")
    if z.ndim != 3:
        raise ValueError("z must be 2D [M, N] or 3D [T, H, D]")
    if z.shape[0] * z.shape[1] != x_2d_shape[0] or z.shape[2] != x_2d_shape[1]:
        raise ValueError("3D z must satisfy T*H=M and D=N")
    if z.stride(-1) != 1:
        raise ValueError("the last stride of z must be 1")
    return z


def layernorm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    z: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
    quant_heads: int = 0,
    *,
    implementation: Optional[str] = None,
):
    """Return ``(result, mean, rstd)``.

    ``result`` is the normalized output, or ``(q, s)`` when ``quant_heads > 0``.
    ``mean`` is ``None`` for RMSNorm. Statistics have shape ``[ngroups*M]``.
    With ``z`` the result is ``norm(x) * act(z)`` when ``norm_before_gate`` and
    ``norm(x * act(z))`` otherwise; 3D gates address ``[T, H, D]`` views whose
    last dimension is contiguous. Forward only: this operator is an inference
    primitive and has no backward wrapper.
    """

    if activation not in _ACTIVATIONS:
        raise ValueError("activation must be 'swish', 'silu', or 'sigmoid'")
    if x.ndim < 1 or not x.is_floating_point():
        raise ValueError("x must be a floating tensor")
    if x.stride(-1) != 1:
        raise ValueError("the last stride of x must be 1")
    if weight.shape != (x.shape[-1],) or weight.device != x.device:
        raise ValueError("weight must have shape [N] on the input device")
    if bias is not None and (bias.shape != (x.shape[-1],) or bias.device != x.device):
        raise ValueError("bias must have shape [N] on the input device")
    if out is not None:
        if out.shape != x.shape:
            raise ValueError("out must have the same shape as x")
        if out.stride(-1) != 1:
            raise ValueError("the last stride of out must be 1")

    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    z = _normalize_gate(z, original_shape, x_2d.shape)
    if out is not None:
        out = out.reshape(x_2d.shape)
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    forward, _ = load_impl("layernorm_gated", implementation)
    result, mean, rstd = forward(
        x_2d,
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

    if not quant_heads:
        result = result.reshape(original_shape)
    return result, mean, rstd


def rms_norm_gated(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    group_size: Optional[int] = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
    implementation: Optional[str] = None,
):
    """SGLang-compatible wrapper returning only the normalized output."""

    result, _, _ = layernorm_gated(
        x,
        weight,
        bias,
        eps=eps,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
        implementation=implementation,
    )
    return result


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
    """SGLang-compatible 2D entry point dispatched through the target config."""

    if x.ndim != 2:
        raise ValueError("_layer_norm_fwd expects a 2D x")
    forward, _ = load_impl("layernorm_gated")
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


__all__ = ["_layer_norm_fwd", "layernorm_gated", "rms_norm_gated"]
