"""Inference-only normalization and grouped quantization from Mojo master."""

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def _check_quant_dtype(dtype, *, fp8=True):
    if dtype != torch.int8 and not (fp8 and dtype == torch.float8_e4m3fn):
        raise NotImplementedError("quant_dtype must be int8" + (" or float8_e4m3fn" if fp8 else ""))


def _check_norm(x, weight, bias, smooth_scale, quant_dtype):
    _check_quant_dtype(quant_dtype)
    if x.ndim < 1 or x.shape[-1] == 0 or not x.is_floating_point():
        raise ValueError("input must be floating with a nonempty last dimension")
    for name, tensor in (("weight", weight), ("bias", bias), ("smooth_scale", smooth_scale)):
        if tensor is not None and (
            tensor.shape != (x.shape[-1],) or tensor.device != x.device or not tensor.is_floating_point()
        ):
            raise ValueError(f"{name} must be a floating [D] tensor on the input device")


def _check_residual(x, residual, norm_pos):
    if norm_pos not in ("pre", "post"):
        raise ValueError("norm_pos must be 'pre' or 'post'")
    if residual.shape != x.shape or residual.dtype != x.dtype or residual.device != x.device:
        raise ValueError("residual must have the input shape, dtype and device")


def rms_norm_quant(
    x, weight, smooth_scale=None, *, eps=1e-5, quant_dtype=torch.int8, symmetric=True, implementation=None
):
    """RMS-normalize in FP32, smooth, then quantize per row; return (quantized, scale)."""
    _check_norm(x, weight, None, smooth_scale, quant_dtype)
    _require_inference("rms_norm_quant", x, weight, smooth_scale)
    forward, _ = load_impl("rms_norm_quant", implementation)
    return forward(x, weight, smooth_scale, eps, quant_dtype, symmetric)


def layer_norm_quant(
    x,
    weight=None,
    bias=None,
    smooth_scale=None,
    *,
    eps=1e-5,
    quant_dtype=torch.int8,
    symmetric=True,
    implementation=None,
):
    """Layer-normalize in FP32, smooth, then quantize per row; scale has shape (*, 1)."""
    _check_norm(x, weight, bias, smooth_scale, quant_dtype)
    _require_inference("layer_norm_quant", x, weight, bias, smooth_scale)
    forward, _ = load_impl("layer_norm_quant", implementation)
    return forward(x, weight, bias, smooth_scale, eps, quant_dtype, symmetric)


def residual_add_rms_norm_quant(
    x,
    residual,
    weight,
    smooth_scale=None,
    *,
    eps=1e-5,
    norm_pos="pre",
    quant_dtype=torch.int8,
    symmetric=True,
    implementation=None,
):
    """Return (quantized, residual, scale); post residual is the unquantized FP32 norm."""
    _check_norm(x, weight, None, smooth_scale, quant_dtype)
    _check_residual(x, residual, norm_pos)
    _require_inference("residual_add_rms_norm_quant", x, residual, weight, smooth_scale)
    forward, _ = load_impl("residual_add_rms_norm_quant", implementation)
    return forward(x, residual, weight, smooth_scale, eps, norm_pos, quant_dtype, symmetric)


def residual_add_layer_norm_quant(
    x,
    residual,
    weight=None,
    bias=None,
    smooth_scale=None,
    *,
    eps=1e-5,
    norm_pos="pre",
    quant_dtype=torch.int8,
    symmetric=True,
    implementation=None,
):
    """Return (quantized, x + residual, scale), for both original pre/post modes."""
    _check_norm(x, weight, bias, smooth_scale, quant_dtype)
    _check_residual(x, residual, norm_pos)
    _require_inference("residual_add_layer_norm_quant", x, residual, weight, bias, smooth_scale)
    forward, _ = load_impl("residual_add_layer_norm_quant", implementation)
    return forward(x, residual, weight, bias, smooth_scale, eps, norm_pos, quant_dtype, symmetric)


def _check_counts(token_count, groups, device):
    if (
        token_count.ndim != 1
        or token_count.numel() != groups
        or token_count.dtype not in (torch.int32, torch.int64)
        or token_count.device != device
    ):
        raise ValueError("token_count must be an int32/int64 [experts] tensor on the input device")
    # Counts are runtime data. They must be nonnegative and sum to the number of
    # rows; do not copy them to the host on every dispatch or graph capture.


def moe_dynamic_quant(input, token_count, inv_smooth_scale, *, quant_dtype=torch.int8, implementation=None):
    """Group-wise smoothing followed by per-token int8 quantization.

    token_count holds nonnegative expert row counts whose sum is the flattened
    row count; CPU counts are copied to the input device. inv_smooth_scale is
    [experts, D]; output scales retain a trailing 1.
    """
    _check_quant_dtype(quant_dtype, fp8=False)
    if input.ndim < 2 or not input.is_floating_point():
        raise ValueError("input must be a floating tensor with at least two dimensions")
    if (
        inv_smooth_scale.ndim != 2
        or inv_smooth_scale.shape[1] != input.shape[-1]
        or inv_smooth_scale.device != input.device
        or not inv_smooth_scale.is_floating_point()
    ):
        raise ValueError("inv_smooth_scale must be floating [experts, D] on the input device")
    token_count = token_count.to(device=input.device)
    _check_counts(token_count, inv_smooth_scale.shape[0], input.device)
    _require_inference("moe_dynamic_quant", input, inv_smooth_scale)
    forward, _ = load_impl("moe_dynamic_quant", implementation)
    return forward(input, token_count, inv_smooth_scale)


def dequant_swiglu_quant(
    x,
    weight_scale,
    quant_scale,
    activation_scale=None,
    bias=None,
    quant_offset=None,
    token_count=None,
    *,
    quant_dtype=torch.int8,
    activate_left=False,
    quant_mode=1,
    implementation=None,
):
    """Grouped dequantization -> SwiGLU -> smoothing -> per-token int8 quantization.

    x is [tokens, 2H]; weight_scale and quant_scale are [experts, 2H/H].
    Without token_count the scale rows must broadcast to tokens. quant_offset is
    reserved, as in the original reference; only dynamic quant_mode=1 is defined.
    """
    _check_quant_dtype(quant_dtype, fp8=False)
    if quant_mode != 1 or quant_offset is not None:
        raise NotImplementedError("only dynamic quant_mode=1 without quant_offset is supported")
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[1] % 2:
        raise ValueError("x must have shape [tokens, 2H]")
    for name, tensor, width in (
        ("weight_scale", weight_scale, x.shape[1]),
        ("quant_scale", quant_scale, x.shape[1] // 2),
    ):
        if tensor.ndim != 2 or tensor.shape[1] != width or tensor.device != x.device or not tensor.is_floating_point():
            raise ValueError(f"{name} must be floating [experts, {width}] on the input device")
    if weight_scale.shape[0] != quant_scale.shape[0]:
        raise ValueError("weight_scale and quant_scale must have the same number of experts")
    if token_count is not None:
        _check_counts(token_count, weight_scale.shape[0], x.device)
    elif weight_scale.shape[0] not in (1, x.shape[0]):
        raise ValueError("without token_count scale rows must be 1 or tokens")
    if activation_scale is not None and activation_scale.shape != (x.shape[0],):
        raise ValueError("activation_scale must have shape [tokens]")
    _require_inference("dequant_swiglu_quant", x, weight_scale, quant_scale, activation_scale, bias)
    forward, _ = load_impl("dequant_swiglu_quant", implementation)
    return forward(x, weight_scale, quant_scale, activation_scale, bias, token_count, activate_left, quant_mode)
