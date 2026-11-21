"""Forward-only indexed RMSNorm and affine modulation."""

import math

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def rms_norm_indexed_scale_shift(
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
    Hidden must be a multiple of 16, up to 8192 for A2 Triton and 16384
    for A5 Triton and Torch. Conditions must be between 1 and 64.
    Tensor offsets must fit signed int32.

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
    _require_inference("rms_norm_indexed_scale_shift", *tensors)
    forward, _ = load_impl("rms_norm_indexed_scale_shift", implementation)
    return forward(x, weight, shift, scale, indices, eps)
