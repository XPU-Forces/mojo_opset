"""Eager reference for inverse RoPE + block FP8 quantisation.

Mirrors the vllm kernel arithmetic: FP32 inverse rotation on the last ``rope_dim``
dims (GPT-J pairing, positive signs), then one power-of-two scale per
``quant_group_size`` block and an RNE cast to E4M3. The output and scale buffers
use the kernel's layouts, including the TMA-aligned zero padding rows.
"""

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
EPS = 1e-10


def _tma_aligned(num_tokens, element_size=4):
    multiple = 16 // element_size
    return (num_tokens + multiple - 1) // multiple * multiple


def _rotate(x, positions, cache, nope_dim, rope_dim):
    head_dim = x.shape[-1]
    offsets = torch.arange(head_dim, device=x.device)
    is_rope = offsets >= nope_dim
    rope_local = offsets - nope_dim
    half = rope_dim // 2
    index = (rope_local >> 1).clamp(min=0)
    cos = cache[positions.long()][:, index]
    sin = cache[positions.long()][:, half + index]
    cos = torch.where(is_rope, cos, torch.ones_like(cos))
    sin = torch.where(is_rope, sin, torch.zeros_like(sin))
    partner = torch.where(is_rope, x[..., offsets ^ 1], torch.zeros_like(x))
    add = x * cos[:, None, :] + partner * sin[:, None, :]
    sub = x * cos[:, None, :] - partner * sin[:, None, :]
    rotated = torch.where(((rope_local & 1) == 0), add, sub)
    return torch.where(is_rope, rotated, x)


def _quantize_blocks(x, n_groups, heads_per_group, head_dim, quant_group_size, tma_aligned_scales):
    num_tokens = x.shape[0]
    d = heads_per_group * head_dim
    blocks = d // quant_group_size
    grouped = x.reshape(num_tokens, n_groups, d).reshape(
        num_tokens, n_groups, blocks, quant_group_size
    )
    absmax = torch.maximum(grouped.abs().amax(dim=3), torch.tensor(EPS, dtype=torch.float32, device=x.device))
    scale = torch.exp2(torch.ceil(torch.log2(absmax * (1.0 / FP8_MAX))))
    values = (grouped / scale[..., None]).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).reshape(
        num_tokens, n_groups, d
    )
    if tma_aligned_scales:
        bits = scale.view(torch.int32)
        ue8 = (bits >> 23) & 0xFF
        packed = (
            ue8[..., 0::4].to(torch.int64)
            | (ue8[..., 1::4].to(torch.int64) << 8)
            | (ue8[..., 2::4].to(torch.int64) << 16)
            | (ue8[..., 3::4].to(torch.int64) << 24)
        ).to(torch.int32)
        return values, packed
    return values, scale


def fused_inv_rope_fp8_quant_fwd(
    o,
    positions,
    cos_sin_cache,
    n_groups,
    heads_per_group,
    nope_dim,
    rope_dim,
    quant_group_size,
    tma_aligned_scales,
    quantize,
):
    num_tokens, _, head_dim = o.shape
    d = heads_per_group * head_dim
    x = _rotate(o.to(torch.float32), positions, cos_sin_cache, nope_dim, rope_dim)

    if not quantize:
        values = x.reshape(num_tokens, n_groups, d).to(o.dtype)
        out_buf = torch.empty((n_groups, num_tokens, d), dtype=o.dtype, device=o.device)
        out_buf.copy_(values.permute(1, 0, 2))
        return out_buf.transpose(0, 1), torch.empty(0, dtype=torch.float32, device=o.device)

    values, scale = _quantize_blocks(
        x, n_groups, heads_per_group, head_dim, quant_group_size, tma_aligned_scales
    )
    out_buf = torch.empty((n_groups, num_tokens, d), dtype=FP8_DTYPE, device=o.device)
    out_buf.copy_(values.permute(1, 0, 2))

    blocks = d // quant_group_size
    inner = (blocks + 3) // 4 if tma_aligned_scales else blocks
    aligned = _tma_aligned(num_tokens)
    scale_dtype = torch.int32 if tma_aligned_scales else torch.float32
    # Zeroed backing buffer: the kernel writes the alignment padding rows as 0 and
    # the strided view never covers them.
    scale_buf = torch.zeros(
        n_groups * inner * aligned, dtype=scale_dtype, device=o.device
    ).as_strided(
        (n_groups, num_tokens, inner),
        (inner * aligned, 1, aligned),
    )
    scale_buf.copy_(scale.permute(1, 0, 2))
    return out_buf.transpose(0, 1), scale_buf.transpose(0, 1)
