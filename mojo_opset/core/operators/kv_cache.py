from typing import Optional
from typing import Tuple

import torch

from ..operator import MojoOperator


def assert_paged_kv_store_contract(
    chunk_metadata: torch.Tensor,
) -> None:
    assert chunk_metadata.dtype == torch.int32
    assert chunk_metadata.dim() == 2
    assert chunk_metadata.shape[1] == 4


def assert_paged_kv_layout_contract(
    block_table: torch.Tensor,
    cu_q_lens: Optional[torch.Tensor],
    context_kv_lens: Optional[torch.Tensor],
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
    block_table: torch.Tensor,
    cu_q_lens: Optional[torch.Tensor],
    context_kv_lens: torch.Tensor,
    block_size: int,
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
    chunk_metadata = torch.stack(
        (
            src_token_starts,
            block_table,
            dst_block_offsets,
            chunk_lens,
        ),
        dim=-1,
    )
    return chunk_metadata[valid_rows]


class MojoStoreKVCache(MojoOperator):
    pass


class MojoStorePagedKVCache(MojoOperator):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        chunk_metadata: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Copy new K/V tokens into a paged KV cache using precomputed chunk metadata.

        Args:
            key_states (torch.Tensor): Shape (token_num, kv_head_num, head_dim) — new key tokens.
            value_states (torch.Tensor): Shape (token_num, kv_head_num, head_dim) — new value tokens.
            key_cache (torch.Tensor): Shape (total_phys_blocks, kv_heads, block_size, head_dim) — key cache.
            value_cache (torch.Tensor): Shape (total_phys_blocks, kv_heads, block_size, head_dim) — value cache.
            chunk_metadata (torch.Tensor): Shape ``(num_chunks, 4)`` with per-row
                ``(src_token_start, dst_block_id, dst_block_offset, chunk_len)``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated `(key_cahce, value_cahce)` after in-place writes.
        """
        assert len(key_states.shape) == 3 and len(value_states.shape) == 3 and key_states.shape == value_states.shape, (
            "key/value states must be (token_num, kv_head_num, head_dim), please check."
        )
        assert_paged_kv_store_contract(chunk_metadata)

        if chunk_metadata.shape[0] == 0:
            return key_cache, value_cache

        for src_token_start, dst_block_id, dst_block_offset, chunk_len in chunk_metadata.tolist():
            src_end = src_token_start + chunk_len
            dst_end = dst_block_offset + chunk_len
            key_cache[dst_block_id, :, dst_block_offset:dst_end, :] = key_states[src_token_start:src_end].permute(
                1, 0, 2
            )
            value_cache[dst_block_id, :, dst_block_offset:dst_end, :] = value_states[src_token_start:src_end].permute(
                1, 0, 2
            )

        return key_cache, value_cache


class MojoStoreMLAKVCache(MojoOperator):
    pass


class MojoStorePagedMLAKVCache(MojoOperator):
    """Append new MLA compressed-KV and positional-key tokens into paged caches.

    MLA (Multi-head Latent Attention) stores a low-rank compressed latent
    ``compressed_kv`` and a positional key ``k_pe`` instead of full K/V per
    head.  This operator writes incoming tokens into the block-based paged
    caches following the same block-table scheme as
    :class:`MojoStorePagedKVCache`.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        compressed_kv_states: torch.Tensor,
        k_pe_states: torch.Tensor,
        compressed_kv_cache: torch.Tensor,
        k_pe_cache: torch.Tensor,
        block_table: torch.Tensor,
        cu_q_lens: torch.Tensor,
        context_kv_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            compressed_kv_states: ``(token_num, kv_lora_rank)`` new compressed
                KV latent tokens.
            k_pe_states: ``(token_num, qk_rope_head_dim)`` new positional key
                tokens.
            compressed_kv_cache: ``(N_blocks, 1, block_size, kv_lora_rank)``
                paged compressed-KV cache (modified in-place).
            k_pe_cache: ``(N_blocks, 1, block_size, qk_rope_head_dim)``
                paged positional-key cache (modified in-place).
            block_table: ``(B, max_blocks_per_seq)`` logical-to-physical block
                mapping.
            cu_q_lens: ``(B+1,)`` cumulative query lengths for prefill.
                ``None`` indicates decode mode (1 token per batch).
            context_kv_lens: ``(B,)`` history sequence lengths before
                storing the current tokens. Padding entries use -1.

        Returns:
            ``(compressed_kv_cache, k_pe_cache)`` after in-place writes.
        """
        assert_paged_kv_layout_contract(block_table, cu_q_lens, context_kv_lens)
        block_size = compressed_kv_cache.shape[2]
        num_batches = len(context_kv_lens) if context_kv_lens is not None else 0
        is_decode = cu_q_lens is None

        for batch_id in range(num_batches):
            if not is_decode:
                t_start = cu_q_lens[batch_id].item()
                t_end = cu_q_lens[batch_id + 1].item()
                seq_len = t_end - t_start
            else:
                t_start = batch_id
                t_end = batch_id + 1
                seq_len = 1

            if seq_len <= 0:
                continue

            ckv_slice = compressed_kv_states[t_start:t_end]  # (seq_len, kv_lora_rank)
            kpe_slice = k_pe_states[t_start:t_end]  # (seq_len, qk_rope_head_dim)

            write_start = context_kv_lens[batch_id].item()
            if write_start < 0:
                continue
            bt = block_table[batch_id]
            if bt.numel() == 0 or bt[0].item() < 0:
                continue

            blk_idx = write_start // block_size
            blk_off = write_start % block_size
            src = 0
            remain = seq_len

            while remain > 0:
                if blk_idx >= bt.shape[0]:
                    break
                phys_id = bt[blk_idx].item()
                if phys_id < 0:
                    break

                cap = block_size - blk_off
                n = min(remain, cap)

                compressed_kv_cache[phys_id, 0, blk_off : blk_off + n, :] = ckv_slice[src : src + n]
                k_pe_cache[phys_id, 0, blk_off : blk_off + n, :] = kpe_slice[src : src + n]

                src += n
                remain -= n
                blk_idx += 1
                blk_off = 0

        return compressed_kv_cache, k_pe_cache
