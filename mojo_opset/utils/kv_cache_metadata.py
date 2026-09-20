"""Provider-independent metadata planning for in-place paged cache writes."""

from typing import Optional

import torch


def assert_paged_kv_store_contract(chunk_metadata: torch.Tensor) -> None:
    assert chunk_metadata.dtype == torch.int32
    assert chunk_metadata.dim() == 2
    assert chunk_metadata.shape[1] == 4


def assert_paged_kv_layout_contract(
    block_table: torch.Tensor, cu_q_lens: Optional[torch.Tensor], context_kv_lens: Optional[torch.Tensor]
) -> None:
    assert block_table.dtype == torch.int32
    assert block_table.dim() == 2
    if cu_q_lens is not None:
        assert cu_q_lens.dtype == torch.int32
        assert cu_q_lens.dim() == 1
    if context_kv_lens is not None:
        assert context_kv_lens.dtype == torch.int32
        assert context_kv_lens.dim() == 1
        assert block_table.shape[0] == context_kv_lens.shape[0]


def build_paged_kv_chunk_metadata(
    block_table: torch.Tensor, cu_q_lens: Optional[torch.Tensor], context_kv_lens: torch.Tensor, block_size: int
) -> torch.Tensor:
    assert_paged_kv_layout_contract(block_table, cu_q_lens, context_kv_lens)
    if cu_q_lens is not None:
        assert cu_q_lens.shape[0] == context_kv_lens.shape[0] + 1
    batch_size = context_kv_lens.shape[0]
    device = block_table.device
    if cu_q_lens is None:
        q_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
    else:
        q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    max_blocks_per_seq = block_table.shape[1]
    if batch_size == 0 or max_blocks_per_seq == 0:
        return torch.empty((0, 4), dtype=torch.int32, device=device)
    q_lens_i32 = q_lens.to(torch.int32)
    context_i32 = context_kv_lens.to(torch.int32)
    if cu_q_lens is None:
        src_token_bases = torch.arange(batch_size, dtype=torch.int32, device=device)
        safe_context = torch.clamp_min(context_i32, 0)
        logical_block = torch.div(safe_context, block_size, rounding_mode="floor")
        valid_rows = (context_i32 >= 0) & (logical_block < max_blocks_per_seq)
        safe_logical_block = torch.clamp(logical_block, 0, max_blocks_per_seq - 1)
        row_index = torch.arange(batch_size, dtype=torch.int32, device=device)
        physical_block = block_table[row_index, safe_logical_block]
        valid_rows = valid_rows & (physical_block >= 0)
        chunk_metadata = torch.stack(
            (
                src_token_bases,
                physical_block,
                torch.remainder(safe_context, block_size),
                torch.ones_like(src_token_bases),
            ),
            dim=-1,
        )
        return chunk_metadata[valid_rows]
    else:
        src_token_bases = cu_q_lens[:-1].to(torch.int32)
    logical_blocks = torch.arange(max_blocks_per_seq, dtype=torch.int32, device=device).unsqueeze(0)
    block_start = logical_blocks * block_size
    block_end = block_start + block_size
    seq_start = context_i32.unsqueeze(1)
    seq_end = (context_i32 + q_lens_i32).unsqueeze(1)
    overlap_start = torch.maximum(seq_start, block_start)
    overlap_end = torch.minimum(seq_end, block_end)
    chunk_lens = torch.clamp_min(overlap_end - overlap_start, 0)
    valid_rows = (q_lens_i32 > 0).unsqueeze(1) & (context_i32 >= 0).unsqueeze(1) & (chunk_lens > 0) & (block_table >= 0)
    src_token_starts = src_token_bases.unsqueeze(1) + (overlap_start - seq_start)
    dst_block_offsets = overlap_start - block_start
    chunk_metadata = torch.stack((src_token_starts, block_table, dst_block_offsets, chunk_lens), dim=-1)
    return chunk_metadata[valid_rows]
