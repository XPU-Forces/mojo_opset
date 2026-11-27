"""Inverse RoPE on the attention output tail followed by block FP8 quantisation."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def fused_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    tma_aligned_scales: bool = False,
    quantize: bool = True,
    *,
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the inverse RoPE to the last ``rope_dim`` dims, then quantise per block.

    ``o`` is ``[tokens, n_groups * heads_per_group, nope_dim + rope_dim]`` contiguous,
    bfloat16/float16, or float8_e4m3fn (no-quantize entry only, accelerator with
    FP8 storage); ``positions`` is int32/int64 ``[tokens]``; ``cos_sin_cache`` is FP32
    ``[max_pos, rope_dim]`` holding ``cos | sin`` halves. Rotation uses GPT-J pairings
    (``offset ^ 1``) with positive signs (inverse of the forward rotation).

    Quantisation uses one 2-power scale per ``quant_group_size`` block:
    ``scale = 2 ** ceil(log2(max(absmax, 1e-10) / fp8_max))`` and
    ``out = clamp(x / scale, -448, 448)`` cast round-to-nearest-even to
    ``float8_e4m3fn``.

    Returns ``(output, scale)``: ``output`` is ``[tokens, n_groups, heads_per_group *
    (nope_dim + rope_dim)]`` laid out as a transpose of a ``[n_groups, tokens, d]``
    buffer; ``scale`` is ``[tokens, n_groups, d // quant_group_size]`` FP32 and
    MN-major (token stride 1, block stride ``align(tokens, 4)``), with the padding
    rows of that alignment written as zero. With ``tma_aligned_scales`` the scales
    are ``int32`` UE8M0 bytes packed four to a word. ``quantize=False`` returns the
    rotated values in the input dtype and an empty scale tensor.
    """
    if o.ndim != 3 or not o.is_contiguous():
        raise ValueError("o must be a contiguous [tokens, heads, head_dim] tensor")
    if o.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
        raise ValueError("o must be a bfloat16/float16/float8_e4m3fn tensor")
    if positions.ndim != 1 or positions.numel() != o.shape[0] or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("positions must be a 1D int32/int64 tensor with one entry per token")
    if positions.device != o.device:
        raise ValueError("positions must be on the input device")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache must be a 2D FP32 tensor")
    if cos_sin_cache.shape[-1] != rope_dim:
        raise ValueError("cos_sin_cache must have rope_dim columns")
    if cos_sin_cache.device != o.device:
        raise ValueError("cos_sin_cache must be on the input device")
    if n_groups <= 0 or heads_per_group <= 0 or o.shape[1] != n_groups * heads_per_group:
        raise ValueError("o must have n_groups * heads_per_group heads")
    if o.shape[2] != nope_dim + rope_dim:
        raise ValueError("head_dim must equal nope_dim + rope_dim")
    if nope_dim < 0 or rope_dim <= 0 or rope_dim % 2:
        raise ValueError("rope_dim must be positive and even")
    if quant_group_size <= 0 or o.shape[2] % quant_group_size:
        raise ValueError("head_dim must be a multiple of quant_group_size")
    if tma_aligned_scales and (o.shape[2] // quant_group_size) % 4:
        raise ValueError("tma_aligned_scales needs chunks_per_head divisible by 4")
    _require_inference("fused_inv_rope_fp8_quant", o, cos_sin_cache)
    forward, _ = load_impl("fused_inv_rope_fp8_quant", implementation)
    return forward(
        o,
        positions,
        cos_sin_cache,
        int(n_groups),
        int(heads_per_group),
        int(nope_dim),
        int(rope_dim),
        int(quant_group_size),
        bool(tma_aligned_scales),
        bool(quantize),
    )
