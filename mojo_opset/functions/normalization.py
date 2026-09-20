"""Forward-only normalization APIs migrated from the original operator layer."""

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
