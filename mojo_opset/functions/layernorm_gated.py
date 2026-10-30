"""Front end for SGLang's gated LayerNorm / RMSNorm.

The operator is a port of SGLang v0.5.22
``python/sglang/kernels/ops/attention/fla/layernorm_gated.py``.  The public
surface mirrors ``rms_norm_gated`` and ``_layer_norm_fwd``: a 2-D ``[M, N]``
input, an optional gate that is either flat ``[M, N]`` or a strided
``[T, H, D]`` view with ``T * H == M`` and ``D == N``, and an optional
per-token FP8 epilogue.

This module only prepares tensors and hands them to the selected provider; the
numerics live in ``mojo_opset/kernels``.
"""

from typing import Optional

import torch

from ._dispatch import load_impl

_ACTIVATION_NAMES = frozenset({"swish", "silu", "sigmoid"})


def _row_vector(name, tensor, width, device):
    """Validate a per-column parameter and return a contiguous copy."""

    if tuple(tensor.shape) != (width,):
        raise ValueError(f"{name} must have shape [{width}], got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must live on the same device as x")
    return tensor.contiguous()


def _output_view(out, x, flat_shape):
    """Validate an optional output buffer and return its flattened view."""

    if out is None:
        return None
    if tuple(out.shape) != tuple(x.shape):
        raise ValueError("out must have the same shape as x")
    if out.stride(-1) != 1:
        raise ValueError("out must be contiguous along its last dimension")
    return out.reshape(flat_shape)


def _gate_for_kernel(z, x_shape, flat_shape):
    """Return the gate the kernel should read, or ``None`` when ungated.

    A gate shaped exactly like ``x`` is flattened together with it.  A 3-D gate
    keeps its tensor, because the kernel walks token and head strides directly
    instead of reading a contiguous copy.
    """

    if z is None:
        return None
    if tuple(z.shape) == tuple(x_shape):
        flat = z.reshape(flat_shape)
        return flat if flat.stride(-1) == 1 else flat.contiguous()
    if len(x_shape) != 2:
        raise ValueError("a 3-D gate is only accepted for a 2-D input")
    if z.dim() != 3:
        raise ValueError(f"z must be [M, N] or [T, H, D], got {tuple(z.shape)}")
    if z.shape[0] * z.shape[1] != flat_shape[0] or z.shape[2] != flat_shape[1]:
        raise ValueError(f"gate {tuple(z.shape)} does not tile a {flat_shape[0]}x{flat_shape[1]} input")
    if z.stride(-1) != 1:
        raise ValueError("z must be contiguous along its last dimension")
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

    ``result`` holds the normalized output, or ``(q, s)`` when ``quant_heads``
    is set.  ``mean`` is ``None`` for RMSNorm.  Both statistics are FP32 and
    have shape ``[ngroups * M]``.  With a gate the result is
    ``norm(x) * act(z)`` when ``norm_before_gate`` is set and
    ``norm(x * act(z))`` otherwise.

    Forward only: this operator is an inference primitive and has no backward
    wrapper.
    """

    if activation not in _ACTIVATION_NAMES:
        raise ValueError("activation must be 'swish', 'silu', or 'sigmoid'")
    if x.ndim < 1 or not x.is_floating_point():
        raise ValueError("x must be a floating tensor")
    if x.stride(-1) != 1:
        raise ValueError("x must be contiguous along its last dimension")

    width = x.shape[-1]
    flat_shape = (-1, width)
    x_2d = x.reshape(flat_shape)
    gate = _gate_for_kernel(z, x.shape, x_2d.shape)
    output = _output_view(out, x, flat_shape)
    weight = _row_vector("weight", weight, width, x.device)
    if bias is not None:
        bias = _row_vector("bias", bias, width, x.device)

    forward, _ = load_impl("layernorm_gated", implementation)
    result, mean, rstd = forward(
        x_2d,
        weight,
        bias,
        float(eps),
        z=gate,
        out=output,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
        quant_heads=quant_heads,
    )

    if not quant_heads:
        result = result.reshape(x.shape)
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
    """SGLang-compatible entry point that returns only the normalized output."""

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
    """SGLang-compatible 2-D entry point dispatched through the target config."""

    if x.ndim != 2:
        raise ValueError("_layer_norm_fwd expects a 2-D x")
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
