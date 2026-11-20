"""Latent RoPE + quantised insert into the paged compressed-KV cache."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def rope_quant_insert(
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    fp8_scale: Optional[torch.Tensor] = None,
    slot_contig: Optional[torch.Tensor] = None,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Rotate the last 64 latent dims and write the group-tail token into ``kv_cache``.

    Only tokens with ``(positions[t] + 1) % compress_ratio == 0`` and
    ``slot_mapping[t] >= 0`` write; every other token leaves the cache untouched.
    The rotation is a GPT-J half-rotation at the group start position
    (``position // compress_ratio * compress_ratio``).

    ``latent`` is bf16 ``[tokens, 512]``; ``cos_sin_cache`` is FP32
    ``[max_pos, 64]`` holding 32 cos then 32 sin values; ``slot_mapping`` is int64.
    ``slot_contig`` (optional, ``[tokens]`` uint8/int32/int64, same device) marks
    which tokens START a contiguous run of group-tail slots: a non-negative
    ``slot_contig[t]`` means token ``t`` is a valid group tail whose slot continues
    the run of the previous token (``-1``/``0`` semantics per producer contract:
    any value ``> 0`` marks a run start, ``0`` marks "not a run start"). When
    supplied by the caller (e.g. by ``get_compressed_slot_mapping``), the insert
    uses 2D merged writes for contiguous runs instead of per-token writes; when
    omitted the kernel detects runs internally where cheap, else writes per token.
    ``kv_cache`` selects the record layout:

    * ``uint8 [pages, block, 584]`` - 448 UE8M0 FP8 dims + 128-byte BF16 RoPE tail,
      with the 7 scale bytes in the page's trailing scale area;
    * ``uint8 [pages, block, 528]`` - MXFP8 over all 512 rotated dims, 16 scales
      inside the record;
    * ``bf16`` / ``float8_e4m3fn`` ``[pages, block, 512]`` - plain ``[448 | 64]`` row;
      the FP8 variant needs ``fp8_scale`` (a one-element FP32 tensor).

    The cache is written in place and returned.
    """
    _require_inference("rope_quant_insert", latent, cos_sin_cache, kv_cache)
    if latent.ndim != 2 or latent.shape[1] != 512 or latent.dtype != torch.bfloat16:
        raise ValueError("latent must be a bf16 [tokens, 512] tensor")
    if not latent.is_contiguous():
        raise ValueError("latent must be contiguous")
    if positions.ndim != 1 or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be a 1D int32/int64 tensor")
    if slot_mapping.ndim != 1 or slot_mapping.dtype != torch.int64:
        raise ValueError("slot_mapping must be a 1D int64 tensor")
    if slot_mapping.numel() > min(latent.shape[0], positions.numel()):
        raise ValueError("slot_mapping cannot be longer than latent/positions")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[-1] != 64 or cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache must be an FP32 [max_pos, 64] tensor")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")
    for name, tensor in (
        ("positions", positions),
        ("cos_sin_cache", cos_sin_cache),
        ("kv_cache", kv_cache),
        ("slot_mapping", slot_mapping),
    ):
        if tensor.device != latent.device:
            raise ValueError(f"{name} must be on the latent device")
    if kv_cache.ndim != 3:
        raise ValueError("kv_cache must be [pages, block, record]")
    if kv_cache.dtype == torch.uint8:
        if kv_cache.shape[-1] not in (584, 528):
            raise ValueError(
                "uint8 records must be 584 (FP8 + BF16 tail) or 528 (MXFP8); "
                "288-byte NVFP4 records have no Ascend equivalent"
            )
        if kv_cache.stride(-1) != 1:
            raise ValueError("kv_cache must be contiguous in the record dimension")
    elif kv_cache.dtype in (torch.bfloat16, torch.float8_e4m3fn):
        if kv_cache.shape[-1] != 512 or kv_cache.stride(-1) != 1:
            raise ValueError("plain records must be [pages, block, 512] with unit last stride")
        if kv_cache.dtype == torch.float8_e4m3fn:
            if fp8_scale is None or fp8_scale.numel() != 1 or fp8_scale.dtype != torch.float32:
                raise ValueError("float8_e4m3fn caches need a one-element FP32 fp8_scale")
            if fp8_scale.device != latent.device:
                raise ValueError("fp8_scale must be on the latent device")
    else:
        raise ValueError("kv_cache must be uint8, bfloat16 or float8_e4m3fn")
    if slot_contig is not None:
        if slot_contig.dtype not in (torch.uint8, torch.int32, torch.int64):
            raise ValueError("slot_contig must be uint8/int32/int64")
        if slot_contig.numel() != slot_mapping.numel() or slot_contig.device != latent.device:
            raise ValueError("slot_contig must be [tokens] on the latent device")
    forward, _ = load_impl("rope_quant_insert", implementation)
    forward(
        latent,
        positions,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        int(compress_ratio),
        fp8_scale,
        slot_contig,
    )
    return kv_cache
