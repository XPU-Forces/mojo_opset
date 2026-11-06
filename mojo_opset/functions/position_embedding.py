"""Inference RoPE contracts from Mojo operators (distinct from training RoPE)."""

from typing import List
from typing import Optional
from typing import Tuple

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def rope_infer(q, k, cos, sin, *, head_first=True, keep_cos_sin_dtype=False, implementation=None):
    """Rotate the trailing cos.shape[-1] channels; preserve leading non-RoPE channels."""
    if q.ndim != k.ndim or q.ndim not in (3, 4) or cos.shape != sin.shape:
        raise ValueError("q/k must both be 3D or 4D and cos/sin shapes must match")
    if cos.shape[-1] > q.shape[-1] or cos.shape[-1] % 2:
        raise ValueError("rotary dimensions must be even and no larger than head_dim")
    _require_inference("rope_infer", q, k, cos, sin)
    forward, _ = load_impl("rope_infer", implementation)
    return forward(q, k, cos, sin, head_first, keep_cos_sin_dtype)


def vision_rope_2d_infer(q, k, cos, sin, *, implementation=None):
    """Full-head half-rotation of packed [T,H,D] q/k using [T,D] cos/sin."""
    if q.ndim != 3 or k.ndim != 3 or cos.ndim != 2 or cos.shape != sin.shape:
        raise ValueError("vision RoPE expects q/k [T,H,D] and matching cos/sin [T,D]")
    if (
        q.shape[0] != cos.shape[0]
        or k.shape[0] != cos.shape[0]
        or q.shape[-1] != cos.shape[-1]
        or k.shape[-1] != cos.shape[-1]
    ):
        raise ValueError("vision RoPE rotates the full head dimension and requires matching token counts")
    _require_inference("vision_rope_2d_infer", q, k, cos, sin)
    forward, _ = load_impl("vision_rope_2d_infer", implementation)
    return forward(q, k, cos, sin)


def _validate_pos_embed(embed_weight, num_grid_per_side, m_size, op):
    if embed_weight.ndim != 2 or embed_weight.shape[0] <= 0 or embed_weight.shape[1] <= 0:
        raise ValueError(
            f"embed_weight must be a nonempty 2D table [G*G, hidden], got {tuple(embed_weight.shape)}"
        )
    if not embed_weight.is_floating_point():
        raise ValueError("embed_weight must be floating point")
    if num_grid_per_side < 2:
        raise ValueError(f"num_grid_per_side={num_grid_per_side} must be >= 2")
    if num_grid_per_side * num_grid_per_side != embed_weight.shape[0]:
        raise ValueError(
            f"embed_weight rows ({embed_weight.shape[0]}) must equal "
            f"num_grid_per_side^2 ({num_grid_per_side * num_grid_per_side})"
        )
    if m_size <= 0:
        raise ValueError(f"m_size={m_size} must be positive")
    _require_inference(op, embed_weight)


def _validate_grid(t, h, w, m_size):
    for name, value in (("t", t), ("h", h), ("w", w)):
        if value <= 0:
            raise ValueError(f"{name}={value} must be positive")
    if h % m_size or w % m_size:
        raise ValueError(f"h={h} and w={w} must be divisible by m_size={m_size}")
    if (t * h * w) % (m_size * m_size):
        raise ValueError(f"t*h*w={t * h * w} must be divisible by m_size^2={m_size * m_size}")


def pos_embed_interp_infer(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: Optional[torch.dtype] = None,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Bilinear interpolation of a ``G*G`` position table onto a ``(t, h, w)`` grid.

    Returns ``[t*h*w, hidden]`` in spatial-merge order: the ``m_size x m_size``
    neighbourhood of each merge block is written adjacently, matching the permute in the
    native PyTorch path. ``dtype`` defaults to ``embed_weight.dtype``.
    """
    _validate_pos_embed(embed_weight, num_grid_per_side, m_size, "pos_embed_interp_infer")
    _validate_grid(t, h, w, m_size)
    forward, _ = load_impl("pos_embed_interp_infer", implementation)
    return forward(embed_weight, int(t), int(h), int(w), int(num_grid_per_side), int(m_size), dtype)


def pos_embed_interp_multi_infer(
    embed_weight: torch.Tensor,
    grids,
    num_grid_per_side: int,
    m_size: int,
    dtype: Optional[torch.dtype] = None,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Single-launch variant of :func:`pos_embed_interp_infer` for a list of grids.

    ``grids`` is a list of ``(t, h, w)`` tuples or an int tensor of shape ``[N, 3]``.
    The result concatenates the per-grid outputs along dim 0 and is bit-exactly equal to
    the per-grid call followed by ``torch.cat``.
    """
    _validate_pos_embed(embed_weight, num_grid_per_side, m_size, "pos_embed_interp_multi_infer")
    normalized: List[Tuple[int, int, int]] = []
    if isinstance(grids, torch.Tensor):
        if grids.ndim != 2 or grids.shape[1] != 3:
            raise ValueError(f"grids tensor must be [N, 3], got {tuple(grids.shape)}")
        rows = grids.to(dtype=torch.int64, device="cpu").tolist()
    else:
        rows = [[int(v) for v in item] for item in grids]
    if not rows:
        raise ValueError("grids must be non-empty")
    for item in rows:
        if len(item) != 3:
            raise ValueError(f"each grid must be (t, h, w), got {item!r}")
        t, h, w = item
        _validate_grid(t, h, w, m_size)
        normalized.append((t, h, w))
    forward, _ = load_impl("pos_embed_interp_multi_infer", implementation)
    return forward(embed_weight, normalized, int(num_grid_per_side), int(m_size), dtype)

