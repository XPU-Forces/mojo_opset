"""Inference RoPE contracts from Mojo operators (distinct from training RoPE)."""

from ._checks import _require_inference
from ._dispatch import load_impl


def apply_rope_infer(q, k, cos, sin, *, head_first=True, keep_cos_sin_dtype=False, implementation=None):
    """Rotate the trailing cos.shape[-1] channels; preserve leading non-RoPE channels."""
    if q.ndim != k.ndim or q.ndim not in (3, 4) or cos.shape != sin.shape:
        raise ValueError("q/k must both be 3D or 4D and cos/sin shapes must match")
    if cos.shape[-1] > q.shape[-1] or cos.shape[-1] % 2:
        raise ValueError("rotary dimensions must be even and no larger than head_dim")
    _require_inference("apply_rope_infer", q, k, cos, sin)
    forward, _ = load_impl("apply_rope_infer", implementation)
    return forward(q, k, cos, sin, head_first, keep_cos_sin_dtype)


def apply_vision_rope2d_infer(q, k, cos, sin, *, implementation=None):
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
    _require_inference("apply_vision_rope2d_infer", q, k, cos, sin)
    forward, _ = load_impl("apply_vision_rope2d_infer", implementation)
    return forward(q, k, cos, sin)
