"""Forward-only normalization APIs migrated from the original operator layer."""

import math

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def _validate_norm(x, weight, bias, eps, op):
    if x.ndim < 1 or x.shape[-1] <= 0 or not x.is_floating_point():
        raise ValueError("normalization requires floating input with a nonempty last dimension")
    if eps <= 0:
        raise ValueError("eps must be positive")
    for name, tensor in (("weight", weight), ("bias", bias)):
        if tensor is not None:
            if tensor.shape != (x.shape[-1],) or tensor.device != x.device:
                raise ValueError(f"{name} must have shape ({x.shape[-1]},) on the input device")
            if not tensor.is_floating_point():
                raise ValueError(f"{name} must be floating point")
    _require_inference(op, x, weight, bias)


def layer_norm_infer(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """LayerNorm over the last dimension; returns the input shape and dtype."""
    _validate_norm(x, weight, bias, eps, "layer_norm_infer")
    forward, _ = load_impl("layer_norm_infer", implementation)
    return forward(x, weight, bias, float(eps))


def rms_norm_infer(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Forward-only RMSNorm; use rms_norm() for training."""
    if weight is None:
        raise ValueError("rms_norm_infer requires weight")
    _validate_norm(x, weight, None, eps, "rms_norm_infer")
    forward, _ = load_impl("rms_norm_infer", implementation)
    return forward(x, weight, float(eps))


def group_rms_norm_infer(
    input_groups: list[torch.Tensor],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    *,
    implementation: Optional[str] = None,
) -> list[torch.Tensor]:
    """RMSNorm on independent groups with optional [groups, dim] weight.

    Groups share their last dimension, dtype and device; leading shapes may differ.
    Every returned group has its input shape/dtype and contiguous layout.
    """
    if not isinstance(input_groups, (list, tuple)) or not input_groups:
        raise ValueError("input_groups must be a nonempty list or tuple")
    first = input_groups[0]
    if first.ndim < 1:
        raise ValueError("each group must have at least one dimension")
    if weight is not None and weight.shape != (len(input_groups), first.shape[-1]):
        raise ValueError("weight must have shape [num_groups, dim]")
    for index, x in enumerate(input_groups):
        if x.ndim < 1 or x.shape[-1] != first.shape[-1] or x.dtype != first.dtype or x.device != first.device:
            raise ValueError("groups must share last dimension, dtype and device")
        _validate_norm(x, None if weight is None else weight[index], None, eps, "group_rms_norm_infer")
    forward, _ = load_impl("group_rms_norm_infer", implementation)
    return forward(list(input_groups), weight, float(eps))


def indexed_rms_norm_scale_shift_infer(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float = 1e-5,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Apply RMSNorm and a per-row indexed affine transform in FP32.

    BF16 inputs are x [rows, hidden], weight [hidden], and shift/scale
    [conditions, hidden]. All inputs share a device. The transform is
    rms_norm(x) * weight * (1 + scale[indices]) + shift[indices],
    rounded to BF16 only at the output. The output is contiguous.
    Hidden must be a positive multiple of 16 up to 16384, and conditions
    must be between 1 and 64. Tensor offsets must fit signed int32.

    Indices are int64 [rows] with values in [0, conditions), a caller
    precondition. Row strides, index strides and shared condition-table
    storage are supported. A5 requires unit feature stride. Empty rows are
    supported. This API is forward-only.
    """
    if x.ndim != 2 or not 16 <= x.shape[1] <= 16384 or x.shape[1] % 16:
        raise ValueError("x must be [rows, hidden] with hidden a multiple of 16 in [16, 16384]")
    rows, hidden = x.shape
    if weight.shape != (hidden,):
        raise ValueError("weight must have shape [hidden]")
    if shift.ndim != 2 or shift.shape[1] != hidden or not 1 <= shift.shape[0] <= 64 or scale.shape != shift.shape:
        raise ValueError("shift and scale must share shape [conditions, hidden] with 1 <= conditions <= 64")
    tensors = (x, weight, shift, scale)
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise ValueError("x, weight, shift and scale must be BF16")
    if indices.shape != (rows,) or indices.dtype != torch.int64:
        raise ValueError("indices must be int64 [rows]")
    if any(tensor.device != x.device for tensor in (*tensors, indices)):
        raise ValueError("all inputs must share a device")
    if rows * hidden >= 2**31 or any(
        sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())) >= 2**31
        for tensor in (*tensors, indices)
    ):
        raise ValueError("tensor offsets must fit signed int32")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    _require_inference("indexed_rms_norm_scale_shift_infer", *tensors)
    forward, _ = load_impl("indexed_rms_norm_scale_shift_infer", implementation)
    return forward(x, weight, shift, scale, indices, eps)


def _validate_residual(x, residual, norm_pos):
    if norm_pos not in ("pre", "post"):
        raise ValueError("norm_pos must be 'pre' or 'post'")
    if residual.shape != x.shape or residual.dtype != x.dtype or residual.device != x.device:
        raise ValueError("residual must have the same shape, dtype and device as input")
    _require_inference("residual normalization", residual)


def residual_add_layer_norm_infer(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    *,
    norm_pos: str = "pre",
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (layer_norm(x + residual), sum) for pre, or (normed, normed) for post."""
    _validate_norm(x, weight, bias, eps, "residual_add_layer_norm_infer")
    _validate_residual(x, residual, norm_pos)
    if weight is None or bias is None:
        raise ValueError("residual_add_layer_norm_infer requires weight and bias")
    forward, _ = load_impl("residual_add_layer_norm_infer", implementation)
    output, summed = forward(x, residual, weight, bias, float(eps))
    return output, summed if norm_pos == "pre" else output


def residual_add_rms_norm_infer(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    *,
    norm_pos: str = "pre",
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (rms_norm(x + residual), sum) for pre, or (normed, normed) for post."""
    _validate_norm(x, weight, None, eps, "residual_add_rms_norm_infer")
    _validate_residual(x, residual, norm_pos)
    if weight is None:
        raise ValueError("residual_add_rms_norm_infer requires weight")
    forward, _ = load_impl("residual_add_rms_norm_infer", implementation)
    output, summed = forward(x, residual, weight, float(eps))
    return output, summed if norm_pos == "pre" else output
