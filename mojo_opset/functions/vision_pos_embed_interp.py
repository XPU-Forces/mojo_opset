# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL vision position-embed bilinear interpolation (Ascend).

Fused triton-ascend implementation of the vllm ``triton_pos_embed_interp*``
operators (vllm/model_executor/models/qwen3_vl.py:186-253, 301-368,
744-763) with the multi-image merged single-launch path. The kernel is the
r57b production baseline: segmented hidden blocks (exact pow2
decomposition + masked tail block), temporal frames replicated in-kernel,
2D frame-chunked stores, BODIES=2 amortized vbody for w_scale == 1.0
mid-size shapes. Accuracy benchmark and non-triton fallback:
``pos_embed_interpolate_native`` (upstream vllm native reference,
device-agnostic).
"""

from typing import Optional
from typing import Sequence

import torch

from ._dispatch import load_impl

__all__ = [
    "pos_embed_interpolate_native",
    "triton_pos_embed_interpolate",
    "triton_pos_embed_interpolate_merged",
]


def triton_pos_embed_interpolate(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Interpolate one (t, h, w) grid of a ViT position-embed weight.

    ``embed_weight`` is ``[num_grid_per_side**2, hidden_dim]``; the output
    has ``t * h * w * m_size**2`` rows of ``hidden_dim``. ``h``/``w`` must
    be divisible by ``m_size``. Signature mirrors the upstream vllm kernel
    (qwen3_vl.py:186-253).
    """
    forward, _ = load_impl("vision_pos_embed_interp", implementation)
    return forward(embed_weight, t, h, w, num_grid_per_side, m_size, dtype)


def triton_pos_embed_interpolate_merged(
    embed_weight: torch.Tensor,
    grid_thw_list: Sequence[Sequence[int]],
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Single-launch merged interpolation over a list of (t, h, w) grids.

    Bitwise identical to per-image ``triton_pos_embed_interpolate`` +
    ``torch.cat`` (gated every iteration round). Replaces the vllm-ascend
    dispatch at qwen3_vl.py:744-763.
    """
    forward, _ = load_impl("vision_pos_embed_interp_merged", implementation)
    return forward(
        embed_weight,
        [tuple(g) for g in grid_thw_list],
        num_grid_per_side,
        m_size,
        dtype,
    )


def pos_embed_interpolate_native(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Reference (upstream vllm native) implementation, device-agnostic.

    The accuracy benchmark for the triton kernels and the fallback when
    triton is unavailable; the same torch operations as vllm
    ``pos_embed_interpolate`` (qwen3_vl.py:301-368).
    """
    forward, _ = load_impl("vision_pos_embed_interp_native", implementation)
    return forward(embed_weight, t, h, w, num_grid_per_side, m_size, dtype)
