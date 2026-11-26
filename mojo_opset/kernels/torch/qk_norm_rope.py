"""Eager QK RMSNorm + RoPE reference (upstream _eager_qk_norm_rope, verbatim)."""

import torch
import torch.nn.functional as F


def _apply_rope_table(x, rope_table, rotary_dim, *, interleaved=False, dtype=None):
    compute_dtype = x.dtype if dtype is None else dtype
    x_c = x.to(compute_dtype)
    half = rotary_dim // 2
    cos = rope_table[..., :half].to(compute_dtype).unsqueeze(1)
    sin = rope_table[..., half:].to(compute_dtype).unsqueeze(1)
    first = x_c[..., 0:rotary_dim:2] if interleaved else x_c[..., :half]
    second = x_c[..., 1:rotary_dim:2] if interleaved else x_c[..., half:rotary_dim]
    rotated_first = first * cos - second * sin
    rotated_second = second * cos + first * sin
    if interleaved:
        rotated = torch.stack((rotated_first, rotated_second), dim=-1).flatten(-2)
        return torch.cat((rotated, x_c[..., rotary_dim:]), dim=-1).to(x.dtype)
    return torch.cat((rotated_first, rotated_second, x_c[..., rotary_dim:]), dim=-1).to(x.dtype)


def qk_norm_rope_fwd(q, k, q_weight, k_weight, rope_table, eps, interleaved):
    head_dim, rotary_dim = q.shape[-1], rope_table.shape[-1]
    q_norm = F.rms_norm(q, (head_dim,), q_weight, eps)
    k_norm = F.rms_norm(k, (head_dim,), k_weight, eps)
    rope_dtype = torch.float32 if interleaved else None
    return (
        _apply_rope_table(q_norm, rope_table, rotary_dim, interleaved=interleaved, dtype=rope_dtype),
        _apply_rope_table(k_norm, rope_table, rotary_dim, interleaved=interleaved, dtype=rope_dtype),
    )
