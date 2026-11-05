"""Per-head RMSNorm followed by RoPE for Q/K (vllm-omni fused_qk_norm_rope)."""

from typing import Optional

import torch

from ._dispatch import load_impl


def qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float = 1e-6,
    *,
    interleaved: bool = False,
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm each head of ``q``/``k`` (``[tokens, heads, head_dim]``), then apply
    half-split RoPE (``interleaved=False``) or interleaved RoPE. ``rope_table`` is
    ``[tokens, rotary_dim]`` holding concatenated cos/sin halves.
    """
    head_dim, rotary_dim = q.shape[-1], rope_table.shape[-1]
    if q.ndim != 3 or k.ndim != 3 or q.shape[0] != k.shape[0] or head_dim != k.shape[-1]:
        raise ValueError("q and k must be [tokens, heads, head_dim] with shared tokens and head_dim")
    if (
        q.shape[0] != rope_table.shape[0]
        or rope_table.shape[-1] != rotary_dim
        or rotary_dim % 2
        or head_dim < rotary_dim
    ):
        raise ValueError("rope_table must be [tokens, rotary_dim] with even rotary_dim <= head_dim")
    for name, x, weight in (("q", q, q_weight), ("k", k, k_weight)):
        if weight.shape != (head_dim,) or weight.device != x.device:
            raise ValueError(f"{name} weight must have shape ({head_dim},) on the input device")
        if not x.is_floating_point() or not weight.is_floating_point() or not rope_table.is_floating_point():
            raise ValueError("q, k, weights and rope_table must be floating point")
    if eps < 0:
        raise ValueError("eps must be non-negative")
    forward, _ = load_impl("qk_norm_rope", implementation)
    return forward(q, k, q_weight, k_weight, rope_table, float(eps), interleaved)
