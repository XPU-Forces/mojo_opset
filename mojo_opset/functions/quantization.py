"""Forward-only quantization; scales follow the original Mojo contracts."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def static_quant(
    input: torch.Tensor,
    scale: torch.Tensor,
    *,
    quant_dtype: torch.dtype = torch.int8,
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (round(input / scale), scale), retaining the caller's scale tensor."""
    if quant_dtype not in (torch.int8, torch.float8_e4m3fn):
        raise NotImplementedError("static quantization supports int8 and float8_e4m3fn")
    if scale.ndim == 0 or input.ndim < scale.ndim or input.shape[-scale.ndim :] != scale.shape:
        raise ValueError("scale shape must match input trailing dimensions")
    if not input.is_floating_point() or not scale.is_floating_point() or input.device != scale.device:
        raise ValueError("input and scale must be floating tensors on the same device")
    _require_inference("static_quant", input, scale)
    forward, _ = load_impl("static_quant", implementation)
    return forward(input, scale, quant_dtype), scale


def dequant(
    input: torch.Tensor,
    scale: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Multiply in FP32 using broadcast scales, then cast to output_dtype."""
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("dequant output must be float16, bfloat16 or float32")
    if input.device != scale.device or not scale.is_floating_point():
        raise ValueError("scale must be floating and on the input device")
    if torch.broadcast_shapes(input.shape, scale.shape) != input.shape:
        raise ValueError("scale must broadcast to the input shape")
    _require_inference("dequant", input, scale)
    forward, _ = load_impl("dequant", implementation)
    return forward(input, scale, output_dtype)


def dynamic_quant(
    input: torch.Tensor,
    inv_smooth_scale: Optional[torch.Tensor] = None,
    *,
    quant_dtype: torch.dtype = torch.int8,
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row int8 quantization, returning scales with a trailing singleton dim.

    As in Mojo master, rows with max(abs(smoothed input))/127 < 1e-6 use scale=1.
    """
    if quant_dtype != torch.int8:
        raise NotImplementedError("dynamic quantization supports int8 only")
    if input.ndim < 1 or input.shape[-1] <= 0 or not input.is_floating_point():
        raise ValueError("input must be floating with nonempty last dimension")
    if inv_smooth_scale is not None and (
        inv_smooth_scale.shape != (input.shape[-1],)
        or inv_smooth_scale.device != input.device
        or not inv_smooth_scale.is_floating_point()
    ):
        raise ValueError("inv_smooth_scale must be a floating [D] tensor on the input device")
    _require_inference("dynamic_quant", input, inv_smooth_scale)
    forward, _ = load_impl("dynamic_quant", implementation)
    return forward(input, inv_smooth_scale)
