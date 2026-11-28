# SPDX-License-Identifier: Apache-2.0
"""Public indexed residual gate for inference."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def indexed_gate_infer(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Return a fresh contiguous BF16 tensor with FP32 indexed multiply and add.

    x and other are BF16 [R, H], gate is BF16 or FP32 [C, H], and indices
    is int64 [R]. Inputs share a device and may be strided. R and H can be
    zero. H is at most 8192 and R * H must fit signed int32. For nonempty
    output, each index is in [0, C), a caller precondition. This function
    does not modify inputs and does not support autograd.
    """
    if x.ndim != 2 or x.shape[1] > 8192 or x.numel() >= 2**31:
        raise ValueError("x must be [R, H], with H <= 8192 and R * H < 2**31")
    rows, hidden = x.shape
    if other.shape != x.shape or gate.ndim != 2 or gate.shape[1] != hidden:
        raise ValueError("other must match x and gate must be [C, H]")
    if x.numel() > 0 and gate.shape[0] == 0:
        raise ValueError("gate must contain a condition for nonempty output")
    if x.dtype != torch.bfloat16 or other.dtype != torch.bfloat16:
        raise ValueError("x and other must be BF16")
    if gate.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("gate must be BF16 or FP32")
    if indices.dtype != torch.int64 or indices.shape != (rows,):
        raise ValueError("indices must be int64 [R]")
    if any(t.device != x.device for t in (gate, other, indices)):
        raise ValueError("all inputs must share x's device")
    _require_inference("indexed_gate_infer", x, gate, other)
    forward, _ = load_impl("indexed_gate_infer", implementation)
    return forward(x, gate, other, indices)
