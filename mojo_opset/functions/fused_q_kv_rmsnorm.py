"""Fused Q/KV RMSNorm (optionally MXFP8-quantizing Q) for DeepSeek V4.1 attention."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl

# The native kernel's BLOCK is next_pow2(width); wider rows do not compile.
_MAX_WIDTH = 8192
_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _check_inputs(qr, kv, q_weight, kv_weight, eps):
    if qr.ndim != 2 or kv.ndim != 2 or qr.shape[0] != kv.shape[0]:
        raise ValueError("qr and kv must be 2D with a shared token dimension")
    if not (qr.is_floating_point() and kv.is_floating_point()):
        raise ValueError("qr and kv must be floating point")
    if qr.dtype != kv.dtype:
        raise ValueError("qr and kv must share a dtype")
    if qr.dtype not in _DTYPES:
        raise ValueError("qr and kv must be bfloat16, float16 or float32")
    if qr.shape[1] > _MAX_WIDTH or kv.shape[1] > _MAX_WIDTH:
        raise ValueError(f"q_size and kv_size must not exceed {_MAX_WIDTH}")
    for name, x in (("qr", qr), ("kv", kv)):
        if x.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous in the last dimension")
    for name, weight, width in (
        ("q_weight", q_weight, qr.shape[1]),
        ("kv_weight", kv_weight, kv.shape[1]),
    ):
        if weight.shape != (width,) or weight.device != qr.device or not weight.is_floating_point():
            raise ValueError(f"{name} must be a floating [{width}] tensor on the input device")
        if not weight.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if weight.dtype != qr.dtype:
            raise ValueError(f"{name} must share the input dtype")
    if eps < 0:
        raise ValueError("eps must be non-negative")


def fused_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    *,
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMS-normalise ``qr`` and ``kv`` independently in FP32, then cast to the input dtype.

    ``qr`` is ``[tokens, q_size]`` and ``kv`` is ``[tokens, kv_size]``; the weights are
    ``[q_size]`` / ``[kv_size]``. Both inputs are bfloat16, float16 or float32 and
    at most 8192 columns wide. Returns ``(q_norm, kv_norm)`` with the input shapes
    and dtype.
    """
    _check_inputs(qr, kv, q_weight, kv_weight, eps)
    _require_inference("fused_q_kv_rmsnorm", qr, kv, q_weight, kv_weight)
    forward, _ = load_impl("fused_q_kv_rmsnorm", implementation)
    return forward(qr, kv, q_weight, kv_weight, float(eps))


def fused_q_kv_rmsnorm_quant(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    *,
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalise both halves, then MXFP8-quantise Q while KV stays in the input dtype.

    Returns ``(q_fp8, q_scale, kv_norm)``: ``q_fp8`` is ``[tokens, q_size]``
    ``float8_e4m3fn``, ``q_scale`` the ``uint8`` F8_128x4-swizzled UE8M0 scale buffer
    (``ceil(tokens/128)*128*ceil(q_size/32/4)*4`` bytes) and ``kv_norm`` the RMSNorm
    of ``kv`` in the input dtype. ``q_size`` must be a multiple of 32, and both
    widths are limited to 8192 columns.
    """
    _check_inputs(qr, kv, q_weight, kv_weight, eps)
    if qr.shape[1] % 32:
        raise ValueError("q_size must be a multiple of 32 for MXFP8 grouping")
    _require_inference("fused_q_kv_rmsnorm_quant", qr, kv, q_weight, kv_weight)
    forward, _ = load_impl("fused_q_kv_rmsnorm_quant", implementation)
    return forward(qr, kv, q_weight, kv_weight, float(eps))
