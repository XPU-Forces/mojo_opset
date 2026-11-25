# SPDX-License-Identifier: Apache-2.0
"""Public inference API for indexed gated RMSNorm."""

import math

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def indexed_gate_rms_norm_scale_shift_infer(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float = 1e-6,
    *,
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fresh ``(updated_residual, modulated)`` BF16 tensors.

    ``residual`` and ``branch`` are [rows, hidden], ``weight`` is [hidden],
    and ``gate``, ``shift``, ``scale`` are [conditions, hidden]. All are BF16
    on the same device; int64 ``indices`` is [rows] on that device. Supports
    1 <= hidden <= 8192, 1 <= conditions <= 12, and rows * hidden < 2**31.
    The Ascend A2 Triton provider requires hidden == 5376. Both Triton
    providers read tables with unit column stride directly. Other strided
    inputs may be copied within the call. Outputs are contiguous and
    do not alias inputs. Empty rows are supported. Autograd is unsupported.

    Each index must be in [0, conditions); values are a caller precondition
    and are not read back to the host. First round the FP32 gated residual
    to BF16, then compute RMSNorm and indexed scale/shift in FP32, rounding
    only the final modulated output. Inputs are never modified.
    """
    if residual.ndim != 2 or not 1 <= residual.shape[1] <= 8192:
        raise ValueError("residual must be [rows, hidden] with 1 <= hidden <= 8192")
    rows, hidden = residual.shape
    if rows * hidden >= 2**31:
        raise ValueError("rows * hidden must fit signed int32 addressing")
    if branch.shape != residual.shape or weight.shape != (hidden,):
        raise ValueError("branch must match residual and weight must be [hidden]")
    if gate.ndim != 2 or gate.shape[1] != hidden or not 1 <= gate.shape[0] <= 12:
        raise ValueError("gate must be [conditions, hidden] with 1 <= conditions <= 12")
    if shift.shape != gate.shape or scale.shape != gate.shape:
        raise ValueError("shift and scale must match gate")
    tensors = (residual, gate, branch, weight, shift, scale)
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise ValueError("indexed_gate_rms_norm_scale_shift_infer requires BF16 floating inputs")
    if indices.dtype != torch.int64 or indices.shape != (rows,):
        raise ValueError("indices must be int64 [rows]")
    if any(t.device != residual.device for t in (*tensors, indices)):
        raise ValueError("all inputs must share the residual device")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    _require_inference("indexed_gate_rms_norm_scale_shift_infer", *tensors)
    forward, _ = load_impl("indexed_gate_rms_norm_scale_shift_infer", implementation)
    return forward(residual, gate, branch, weight, shift, scale, indices, eps)
