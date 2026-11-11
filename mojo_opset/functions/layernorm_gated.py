"""Gated LayerNorm/RMSNorm forward migrated from the SGLang FLA operator."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl

_ACTIVATIONS = ("swish", "silu", "sigmoid")
# The fused kernel keeps one row tile in registers: feature dim <= 64KB.
_MAX_FUSED_SIZE_BYTES = 65536


def _validate_gated_norm(x, weight, bias, z, out, eps, group_size, quant_heads, activation):
    if x.ndim != 2 or not x.is_floating_point() or x.stride(-1) != 1:
        raise ValueError("x must be a floating 2D [M, N] tensor with a contiguous last dimension")
    M, N = x.shape
    if eps <= 0:
        raise ValueError("eps must be positive")
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {_ACTIVATIONS}")
    if weight is None:
        raise ValueError("layer_norm_gated requires weight")
    for name, tensor in (("weight", weight), ("bias", bias)):
        if tensor is not None:
            if tensor.shape != (N,) or tensor.device != x.device or not tensor.is_floating_point():
                raise ValueError(f"{name} must be a floating [N] tensor on the input device")
            if tensor.stride(-1) != 1:
                raise ValueError(f"{name} must have a contiguous last dimension")
    if z is not None:
        if z.dtype != x.dtype or z.device != x.device or z.stride(-1) != 1:
            raise ValueError("z must share x dtype/device with a contiguous last dimension")
        if z.ndim == 3:
            if z.shape[0] * z.shape[1] != M or z.shape[2] != N:
                raise ValueError("3D z must have shape [T, H, D] with T*H == M and D == N")
        elif z.ndim == 2:
            if z.shape != (M, N):
                raise ValueError("2D z must have shape [M, N]")
        else:
            raise ValueError("z must be 2D [M, N] or 3D [T, H, D]")
    if out is not None:
        if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device or out.stride(-1) != 1:
            raise ValueError("out must match x shape/dtype/device with a contiguous last dimension")
    if group_size is not None:
        if group_size <= 0 or N % group_size != 0:
            raise ValueError("group_size must divide N")
        if group_size > _MAX_FUSED_SIZE_BYTES // x.element_size():
            raise ValueError("group_size exceeds the 64KB feature-dimension limit")
    if quant_heads:
        if group_size not in (None, N):
            raise ValueError("FP8 quantization requires a single normalization group")
        if quant_heads <= 0 or M % quant_heads != 0:
            raise ValueError("quant_heads must be positive and divide M")
    _require_inference("layer_norm_gated", x, weight, bias, z)


def layer_norm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
    quant_heads: int = 0,
    implementation: Optional[str] = None,
):
    """norm(x) * act(z) when norm_before_gate else norm(x * act(z)).

    Returns ((q, s) when quant_heads else out), mean, rstd; mean is None for
    RMSNorm and mean/rstd are FP32 with the group-major layout [ngroups * M].
    """
    _validate_gated_norm(x, weight, bias, z, out, eps, group_size, quant_heads, activation)
    forward, _ = load_impl("layer_norm_gated", implementation)
    return forward(
        x, weight, bias, z, float(eps), out, group_size,
        norm_before_gate, is_rms_norm, activation, int(quant_heads),
    )
