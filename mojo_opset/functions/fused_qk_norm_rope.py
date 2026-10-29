"""Public Q/K RMSNorm plus packed RoPE function."""

from typing import Optional

import torch
import torch.nn.functional as F

from mojo_opset.functions._dispatch import load_impl

_HEAD_DIM = 128
_ROTARY_DIM = 96


def _apply_rope_table(
    x: torch.Tensor,
    rope_table: torch.Tensor,
    rotary_dim: int,
    *,
    interleaved: bool,
) -> torch.Tensor:
    half = rotary_dim // 2
    compute_dtype = torch.float32 if interleaved else x.dtype
    x_compute = x.to(compute_dtype)
    cos = rope_table[..., :half].to(compute_dtype).unsqueeze(1)
    sin = rope_table[..., half:].to(compute_dtype).unsqueeze(1)
    if interleaved:
        first = x_compute[..., 0:rotary_dim:2]
        second = x_compute[..., 1:rotary_dim:2]
    else:
        first = x_compute[..., :half]
        second = x_compute[..., half:rotary_dim]
    return torch.cat(
        (
            first * cos - second * sin,
            second * cos + first * sin,
            x_compute[..., rotary_dim:],
        ),
        dim=-1,
    ).to(x.dtype)


def _eager_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_norm = F.rms_norm(q, (head_dim,), q_weight, eps)
    k_norm = F.rms_norm(k, (head_dim,), k_weight, eps)
    return (
        _apply_rope_table(q_norm, rope_table, rotary_dim, interleaved=interleaved),
        _apply_rope_table(k_norm, rope_table, rotary_dim, interleaved=interleaved),
    )


def _triton_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    interleaved: bool,
) -> bool:
    return (
        not interleaved
        and q.device.type == "npu"
        and k.device.type == "npu"
        and q.device == k.device
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and head_dim == _HEAD_DIM
        and rotary_dim == _ROTARY_DIM
    )


def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    *,
    head_dim: Optional[int] = None,
    rotary_dim: Optional[int] = None,
    interleaved: bool = False,
    cuda_rms_norm: bool = False,
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head RMSNorm followed by packed half-split or interleaved RoPE."""
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(f"q and k must be [tokens, heads, head_dim], got {q.shape} and {k.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"q and k shapes are incompatible: {q.shape} and {k.shape}")
    if q.dtype != k.dtype or q.device != k.device:
        raise ValueError("q and k must have the same dtype and device")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"fused QK RMSNorm/RoPE requires floating inputs, got {q.dtype}")

    head_dim = q.shape[-1] if head_dim is None else head_dim
    rotary_dim = rope_table.shape[-1] if rotary_dim is None else rotary_dim
    if q.shape[-1] != head_dim:
        raise ValueError(f"Expected q/k head_dim={head_dim}, got {q.shape[-1]}")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even and in [2, {head_dim}], got {rotary_dim}")
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError(f"Expected norm weights [{head_dim}], got {tuple(q_weight.shape)} and {tuple(k_weight.shape)}")
    if q_weight.device != q.device or k_weight.device != q.device:
        raise ValueError("Q/K norm weights must be on the activation device")
    if rope_table.device != q.device or rope_table.dtype not in (q.dtype, torch.float32):
        raise ValueError("rope_table must be on q/k's device with their dtype or float32")
    if rope_table.shape != (q.shape[0], rotary_dim):
        raise ValueError(f"Expected rope_table [{q.shape[0]}, {rotary_dim}], got {tuple(rope_table.shape)}")

    support_args = (q, k, head_dim, rotary_dim, interleaved)
    if implementation not in (None, "torch") and _triton_supported(*support_args):
        forward, _ = load_impl("fused_qk_norm_rope", implementation)
    elif implementation is None and _triton_supported(*support_args):
        forward, _ = load_impl("fused_qk_norm_rope", implementation)
    else:
        forward = None

    if forward is None:
        return _eager_qk_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            head_dim,
            rotary_dim,
            interleaved,
        )
    return forward(
        q,
        k,
        q_weight.contiguous(),
        k_weight.contiguous(),
        rope_table.contiguous(),
        float(eps),
    )
