"""Compressed-KV slot mapping for the DeepSeek V4 sparse-MLA indexer."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def get_compressed_slot_mapping(
    num_tokens: int,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: Optional[torch.Tensor] = None,
    dcp_world_size: int = 1,
    cp_interleave: int = 1,
    max_query_len: Optional[int] = None,
    positions: Optional[torch.Tensor] = None,
    slot_contig: Optional[torch.Tensor] = None,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Map every token of a ratio-N compressed KV cache to its physical slot.

    Only the last token of each ``compress_ratio`` group writes a state; the rest
    map to ``PAD_ID = -1``. ``slot_mapping`` is the token's own int64 slot (``-1``
    marks a replayed / other-rank token), ``query_start_loc`` is the int32/int64
    cu_seqlens of length ``num_reqs + 1``, ``seq_lens`` int32/int64 ``[num_reqs]``
    and ``block_table`` int32/int64 ``[num_reqs, width]``. ``block_size`` counts
    compressed states per page.

    ``out`` may reuse a persistent int64 buffer of at least ``num_tokens``
    elements; the tail beyond ``num_tokens`` is reset to ``-1``. Returns a
    ``[num_tokens]`` int64 view (or a fresh allocation). ``max_query_len`` is an
    optional host-side scalar: without it the conservative per-request structure
    is used. ``positions`` (int32/int64 ``[num_tokens]``) is the token's absolute
    position and is consumed together with ``slot_contig`` (uint8/int32/int64
    ``[num_tokens]``), which marks the group-tail tokens that continue the
    previous token's slot at a consecutive position.
    """
    _require_inference("get_compressed_slot_mapping", slot_mapping, query_start_loc, seq_lens, block_table)
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    if block_size <= 0 or compress_ratio <= 0 or dcp_world_size <= 0 or cp_interleave <= 0:
        raise ValueError("block_size, compress_ratio, dcp_world_size and cp_interleave must be positive")
    if slot_mapping.ndim != 1 or slot_mapping.dtype != torch.int64 or slot_mapping.numel() != num_tokens:
        raise ValueError("slot_mapping must be an int64 [num_tokens] tensor")
    if block_table.ndim != 2:
        raise ValueError("block_table must be 2D")
    if block_table.stride(1) != 1:
        raise ValueError("block_table must be contiguous in the page dimension")
    num_reqs = block_table.shape[0]
    for name, tensor, length in (
        ("query_start_loc", query_start_loc, num_reqs + 1),
        ("seq_lens", seq_lens, num_reqs),
    ):
        if tensor.ndim != 1 or tensor.numel() != length or tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be a 1D int32/int64 tensor of length {length}")
    for name, tensor in (
        ("slot_mapping", slot_mapping),
        ("query_start_loc", query_start_loc),
        ("seq_lens", seq_lens),
        ("block_table", block_table),
    ):
        if tensor.device != slot_mapping.device:
            raise ValueError(f"{name} must be on the slot_mapping device")
    if out is not None:
        if out.ndim != 1 or out.dtype != torch.int64 or out.numel() < num_tokens:
            raise ValueError("out must be a 1D int64 buffer of at least num_tokens elements")
        if out.device != slot_mapping.device or not out.is_contiguous():
            raise ValueError("out must be a contiguous buffer on the slot_mapping device")
    if max_query_len is not None and max_query_len < 0:
        raise ValueError("max_query_len must be non-negative")
    if slot_contig is not None and positions is None:
        raise ValueError("slot_contig output needs the token positions tensor")
    if positions is not None:
        if (
            positions.ndim != 1
            or positions.numel() != num_tokens
            or positions.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("positions must be a 1D int32/int64 tensor of length num_tokens")
        if positions.device != slot_mapping.device:
            raise ValueError("positions must be on the slot_mapping device")
    if slot_contig is not None:
        if (
            slot_contig.ndim != 1
            or slot_contig.numel() != num_tokens
            or slot_contig.dtype not in (torch.uint8, torch.int32, torch.int64)
        ):
            raise ValueError("slot_contig must be a 1D uint8/int32/int64 tensor of length num_tokens")
        if slot_contig.device != slot_mapping.device:
            raise ValueError("slot_contig must be on the slot_mapping device")
    forward, _ = load_impl("get_compressed_slot_mapping", implementation)
    if out is None:
        # The kernel overwrites [0, num_tokens) on every dispatch path (invalid
        # lanes store PAD explicitly), so a fresh allocation needs no prefill.
        buffer = torch.empty(num_tokens, dtype=torch.int64, device=slot_mapping.device)
    else:
        buffer = out
    forward(
        int(num_tokens),
        slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        int(block_size),
        int(compress_ratio),
        buffer,
        int(dcp_world_size),
        int(cp_interleave),
        None if max_query_len is None else int(max_query_len),
        slot_contig,
        positions,
    )
    return buffer[:num_tokens]
