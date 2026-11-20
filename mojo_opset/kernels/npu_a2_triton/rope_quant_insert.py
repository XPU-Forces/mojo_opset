"""Ascend A2 (910B) Triton latent RoPE + plain bf16 paged-cache insert.

Only the FP8-free plain ``bf16`` record is implemented here. A2 has no FP8 (the
DAV_2201 platform ini carries no ``DT_FLOAT8_*`` and Triton's ``f32 -> fp8`` cast
aborts in ``arith::TruncFOp`` lowering), so the 584/528-byte quantised records and
the ``float8_e4m3fn`` plain record fail fast on the host.

Tiling: ``RPP=4`` tokens unrolled per program. The 950 row-assembly optimisations
(``insert_slice``/``extract_slice``) are not available on this arch.
"""

import torch
import triton
import triton.language as tl

_RPP = 4


@triton.jit
def _rotated_tail(cos_sin, base, position, COS_STRIDE: tl.constexpr, COMPRESS_RATIO: tl.constexpr):
    """GPT-J half-rotation of the last 64 dims (32 pairs, contiguous cos/sin)."""
    j = tl.arange(0, 32)
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + j).to(tl.float32)
    s = tl.load(cs + 32 + j).to(tl.float32)
    even = tl.load(base + 448 + 2 * j).to(tl.float32)
    odd = tl.load(base + 449 + 2 * j).to(tl.float32)
    return tl.interleave(even * c - odd * s, odd * c + even * s)


@triton.jit
def _insert_plain_tok(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    t,
    num_tokens,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    """Plain bf16 record [448 NoPE | 64 RoPE]: NoPE as loaded, tail rotated."""
    if t < num_tokens:
        slot = tl.load(cache_slots + t)
        if slot >= 0:
            position = tl.load(positions + t)
            if (position + 1) % COMPRESS_RATIO == 0:
                d = tl.arange(0, 512)
                base = latent + t.to(tl.int64) * 512
                row = tl.load(base + d)
                dst = (
                    cache
                    + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
                    + (slot % CACHE_BLOCK) * ROW_STRIDE
                )
                rot = _rotated_tail(cos_sin, base, position, COS_STRIDE, COMPRESS_RATIO)
                tl.store(dst + d, row, d < 448)
                rope_dst = (dst + 448).to(tl.pointer_type(tl.bfloat16))
                tl.store(rope_dst + tl.arange(0, 64), rot.to(tl.bfloat16))


@triton.jit
def _rope_plain_insert(
    latent, positions, cos_sin, cache, cache_slots, num_tokens,
    COS_STRIDE: tl.constexpr, CACHE_STRIDE: tl.constexpr, ROW_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr, COMPRESS_RATIO: tl.constexpr, RPP: tl.constexpr,
):
    pid = tl.program_id(0)
    for r in tl.static_range(RPP):
        _insert_plain_tok(
            latent, positions, cos_sin, cache, cache_slots, pid * RPP + r, num_tokens,
            COS_STRIDE, CACHE_STRIDE, ROW_STRIDE, CACHE_BLOCK, COMPRESS_RATIO,
        )


def _rope_quant_insert_impl(latent, positions, cos_sin_cache, kv_cache, slot_mapping, compress_ratio, fp8_scale, slot_contig=None):
    num_tokens = slot_mapping.numel()
    if num_tokens == 0:
        return
    if kv_cache.dtype == torch.uint8:
        raise NotImplementedError(
            "rope_quant_insert: 910B (A2/DAV_2201) has no FP8, so the 584/528-byte "
            "quantised records cannot be written on this arch"
        )
    if kv_cache.dtype == torch.float8_e4m3fn:
        raise NotImplementedError(
            "rope_quant_insert: the float8_e4m3fn plain record needs FP8 casts, "
            "which 910B (A2/DAV_2201) does not support"
        )
    _rope_plain_insert[(triton.cdiv(num_tokens, _RPP),)](
        latent,
        positions,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        num_tokens,
        COS_STRIDE=cos_sin_cache.stride(0),
        CACHE_STRIDE=kv_cache.stride(0),
        ROW_STRIDE=kv_cache.stride(1),
        CACHE_BLOCK=kv_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        RPP=_RPP,
    )


@torch.library.custom_op(
    "mojo_npu_triton_a2::rope_quant_insert_fwd",
    mutates_args=("kv_cache",),
)
def rope_quant_insert_fwd(
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    fp8_scale: torch.Tensor | None,
    slot_contig: torch.Tensor | None,
) -> None:
    _rope_quant_insert_impl(
        latent.contiguous(),
        positions.contiguous(),
        cos_sin_cache,
        kv_cache,
        slot_mapping.contiguous(),
        compress_ratio,
        fp8_scale,
        slot_contig,
    )


@rope_quant_insert_fwd.register_fake
def _rope_quant_insert_fake(
    latent, positions, cos_sin_cache, kv_cache, slot_mapping, compress_ratio, fp8_scale, slot_contig
) -> None:
    return None
