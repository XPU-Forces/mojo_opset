"""Vision position-embedding interpolation (Qwen3-VL style)."""

from ._checks import _require_inference
from ._dispatch import load_impl


def vision_pos_embed_interp(
    embed_weight,
    t,
    h,
    w,
    num_grid_per_side,
    m_size,
    dtype=None,
    *,
    implementation=None,
):
    """Bilinear-interpolate a G x G positional table onto a (t, h, w) grid.

    Args:
        embed_weight: ``[num_grid_per_side**2, hidden_dim]`` FP32/BF16 table.
        t, h, w: grid dims; ``h`` and ``w`` must be divisible by ``m_size``.
        num_grid_per_side: G, side length of the learnable table.
        m_size: spatial-merge side length (2 for Qwen3-VL).
        dtype: output dtype, defaults to ``embed_weight.dtype``.

    Returns:
        ``[t * h * w, hidden_dim]`` in spatial-merge order.
    """
    if embed_weight.ndim != 2:
        raise ValueError("embed_weight must be 2D [num_grid_per_side**2, hidden_dim]")
    if h % m_size != 0 or w % m_size != 0:
        raise ValueError(f"h={h} and w={w} must be divisible by m_size={m_size}")
    if dtype is None:
        dtype = embed_weight.dtype
    _require_inference("vision_pos_embed_interp", embed_weight)
    forward, _ = load_impl("vision_pos_embed_interp", implementation)
    return forward(embed_weight, t, h, w, num_grid_per_side, m_size, dtype)


def vision_pos_embed_interp_multi(
    embed_weight,
    grid_thw,
    num_grid_per_side,
    m_size,
    dtype=None,
    *,
    implementation=None,
):
    """Single-launch multi-grid variant; ``grid_thw`` is a sequence of (t, h, w)."""
    if embed_weight.ndim != 2:
        raise ValueError("embed_weight must be 2D [num_grid_per_side**2, hidden_dim]")
    grid_thw = [tuple(int(x) for x in g) for g in grid_thw]
    if not grid_thw:
        raise ValueError("grid_thw must be non-empty")
    if dtype is None:
        dtype = embed_weight.dtype
    _require_inference("vision_pos_embed_interp_multi", embed_weight)
    forward, _ = load_impl("vision_pos_embed_interp_multi", implementation)
    return forward(embed_weight, grid_thw, num_grid_per_side, m_size, dtype)
