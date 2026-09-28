from typing import Optional

import torch

from ._dispatch import load_impl


def _validate_inputs(
    input: torch.Tensor,
    axis: int,
    round_mode: str,
    dst_type: int,
    block_size: int,
    scale_alg: int,
    dst_type_max: float,
    max_low_bound: float,
) -> None:
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if input.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("input must have dtype float32 or bfloat16")
    if input.ndim < 1 or input.numel() == 0:
        raise ValueError("input must be nonempty and have at least one dimension")
    if type(axis) is not int:
        raise TypeError("axis must be an integer")
    if not -input.ndim <= axis < input.ndim:
        raise ValueError("axis is out of range for input dimensions")
    if round_mode != "stochastic":
        raise NotImplementedError("dynamic_mx_quant_sr requires round_mode='stochastic'")
    if dst_type not in (23, 24):
        raise NotImplementedError("dst_type must be 23 (E5M2) or 24 (E4M3FN)")
    if not 0 < block_size <= 1024 or block_size % 32:
        raise ValueError("block_size must be a positive multiple of 32, at most 1024")
    if block_size != 32:
        raise NotImplementedError("dynamic_mx_quant_sr requires block_size=32")
    if scale_alg not in (0, 1):
        raise NotImplementedError("dynamic_mx_quant_sr requires scale_alg=0 or 1")
    if dst_type_max != 0.0:
        raise NotImplementedError("dynamic_mx_quant_sr requires dst_type_max=0.0")
    if (
        type(max_low_bound) not in (float, int)
        or max_low_bound != max_low_bound
        or max_low_bound in (float("inf"), float("-inf"))
        or max_low_bound < 0
    ):
        raise ValueError("max_low_bound must be a finite non-negative number")
    if scale_alg != 1 and max_low_bound != 0:
        raise ValueError("max_low_bound must be 0 when scale_alg != 1")


def dynamic_mx_quant_sr(
    input: torch.Tensor,
    *,
    axis: int = -1,
    round_mode: str = "stochastic",
    dst_type: int = 24,
    block_size: int = 32,
    scale_alg: int = 1,
    dst_type_max: float = 0.0,
    max_low_bound: float = 0.0,
    implementation: Optional[str] = None,
):
    """Stochastic MXFP8 quantization with dynamic per-block E8M0 scales.

    MQA-style MX block quantization: per-32-element absmax scales with
    stochastic rounding seeded by a fixed Philox (0,0) stream, so repeated
    calls replay identical randomness.  ``dst_type`` uses Torch integer
    codes: 23=E5M2, 24=E4M3FN.

    Args:
        input: contiguous nonempty FP32/BF16 tensor; the quantization axis
            must be a multiple of 16 (at least 32 for FP32).
        axis: quantization axis.
        round_mode: only ``"stochastic"`` is supported.
        dst_type: 24 (E4M3FN, default) or 23 (E5M2).
        block_size: only 32 is supported.
        scale_alg: E8M0 scale algorithm, 0 (OCP) or 1 (ceil, default).
        dst_type_max: only 0.0 is supported.
        max_low_bound: optional non-negative floor for nonzero groups
            (scale_alg=1 only); extends the original Torch API.
        implementation: optional implementation override (e.g. "cannbotdsl").

    Returns:
        ``(y, mx_scale)``: ``y`` has the input shape and the selected FP8
        dtype; ``mx_scale`` is ``float8_e8m0fnu`` shaped like the input with
        the quantization axis replaced by ``ceil(K/64)`` plus a trailing
        dimension of 2.
    """
    _validate_inputs(
        input, axis, round_mode, dst_type, block_size, scale_alg, dst_type_max, max_low_bound
    )
    forward, _ = load_impl("dynamic_mx_quant_sr", implementation)
    return forward(
        input,
        axis % input.ndim,
        round_mode,
        dst_type,
        block_size,
        scale_alg,
        dst_type_max,
        float(max_low_bound),
    )
