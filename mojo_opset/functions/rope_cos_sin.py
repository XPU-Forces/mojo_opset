"""Position-table generation and extraction, separate from applying RoPE."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def rope_cos_sin(
    x: torch.Tensor,
    inv_freq: torch.Tensor,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    cu_q_lens: Optional[torch.Tensor] = None,
    total_seq_lens: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    attention_scaling: float = 1.0,
    *,
    implementation: Optional[str] = None,
):
    """Return cos/sin tables for the requested positions without rotating x.

    The Triton implementation gathers precomputed tables. The reference can
    also generate them from inv_freq. x supplies shape and device metadata.
    """
    _require_inference("rope_cos_sin", inv_freq, cos, sin)
    if cu_q_lens is not None and position_ids is not None:
        raise ValueError("provide at most one of cu_q_lens and position_ids")
    for indices in (cu_q_lens, total_seq_lens, position_ids):
        if indices is not None and indices.dtype != torch.int32:
            raise ValueError("position indices and sequence lengths must be int32")
    forward, _ = load_impl("rope_cos_sin", implementation)
    # x contributes only shape metadata, not values or a gradient edge.
    return forward(x.detach(), inv_freq, cos, sin, cu_q_lens, total_seq_lens, position_ids, attention_scaling)


def vision_rope_cos_sin_2d(
    inv_freq: torch.Tensor,
    grid_hw: torch.Tensor,
    rope_dim: int,
    adapooling_factor: int = 1,
    *,
    implementation: Optional[str] = None,
):
    _require_inference("vision_rope_cos_sin_2d", inv_freq)
    if adapooling_factor < 1 or rope_dim % 4:
        raise ValueError("adapooling_factor must be positive and rope_dim must be divisible by 4")
    forward, _ = load_impl("vision_rope_cos_sin_2d", implementation)
    return forward(inv_freq, grid_hw, rope_dim, adapooling_factor)
