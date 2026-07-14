from typing import Tuple, Optional

import torch

from mojo_opset.core.operators.kv_cache import (
    assert_paged_kv_layout_contract,
    assert_paged_kv_store_contract,
    build_paged_kv_chunk_metadata
)
from mojo_opset.core.operator import MojoOperator


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

class MojoStorePagedKVCacheC8(MojoOperator):
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
        key_scale: torch.Tensor,
        value_scale: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        cu_q_lens: Optional[torch.Tensor] = None,
        context_kv_lens: Optional[torch.Tensor] = None,
        *,
        chunk_metadata: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Copy new K/V tokens into a paged KV cache with int8 per_channal quant.

        Args:
            key_states (torch.Tensor): Shape (token_num, kv_head_num, head_dim) — new key tokens.
            value_states (torch.Tensor): Shape (token_num, kv_head_num, head_dim) — new value tokens.
            key_cache (torch.Tensor): Shape (total_phys_blocks, kv_heads, block_size, head_dim) — key cache.
            value_cache (torch.Tensor): Shape (total_phys_blocks, kv_heads, block_size, head_dim) — value cache.
            key_scale (torch.Tensor): Shape (kv_head_num, head_dim) — key scale.
            value_scale (torch.Tensor): Shape (kv_head_num, head_dim) — value scale.
            block_table (torch.Tensor | None): Legacy logical-to-physical block mapping.
            cu_q_lens (torch.Tensor | None): Legacy cumulative query lengths. ``None`` indicates decode mode.
            context_kv_lens (torch.Tensor | None): Legacy KV lengths before storing current tokens.
            chunk_metadata (torch.Tensor | None): Optimized precomputed store plan with shape ``(num_chunks, 4)``
                and per-row ``(src_token_start, dst_block_id, dst_block_offset, chunk_len)``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated `(key_cahce, value_cahce)` after in-place writes.
        """
        assert len(key_states.shape) == 3 and len(value_states.shape) == 3 and key_states.shape == value_states.shape, (
            "key/value states must be (token_num, kv_head_num, head_dim), please check."
        )
        if chunk_metadata is None:
            assert block_table is not None, "block_table is required when chunk_metadata is not provided."
            assert context_kv_lens is not None, "context_kv_lens is required when chunk_metadata is not provided."
            chunk_metadata = build_paged_kv_chunk_metadata(
                block_table,
                cu_q_lens,
                context_kv_lens,
                key_cache.shape[2],
            )
        else:
            assert block_table is None and cu_q_lens is None and context_kv_lens is None, (
                "chunk_metadata path should not be mixed with block_table/cu_q_lens/context_kv_lens."
            )

        assert key_scale is not None and value_scale is not None
        assert_paged_kv_store_contract(chunk_metadata)

        if chunk_metadata.shape[0] == 0:
            return key_cache, value_cache

        key_q = torch.round(key_states / key_scale).clamp(-128, 127).to(torch.int8)
        value_q = torch.round(value_states / value_scale).clamp(-128, 127).to(torch.int8)

        for src_token_start, dst_block_id, dst_block_offset, chunk_len in chunk_metadata.tolist():
            src_end = src_token_start + chunk_len
            dst_end = dst_block_offset + chunk_len
            key_cache[dst_block_id, :, dst_block_offset:dst_end, :] = key_q[src_token_start:src_end].permute(
                1, 0, 2
            )
            value_cache[dst_block_id, :, dst_block_offset:dst_end, :] = value_q[src_token_start:src_end].permute(
                1, 0, 2
            )

        return key_cache, value_cache

class MojoDequantFromPagedKVCache(MojoOperator):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        *,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        key_cache: torch.Tensor,
        key_cache_scale: torch.Tensor,
        value_cache: Optional[torch.Tensor] = None,
        value_cache_scale: Optional[torch.Tensor] = None,
        context_lengths: Optional[torch.Tensor] = None,
        max_context_len: int,
        context_seq_offset: Optional[torch.Tensor] = None,
        block_tables: torch.Tensor,
    ):
        r"""
        Copy and dequantize from Transformer int8 paged K/V cache to linear K/V states.

        Args:
            key (torch.Tensor): Shape (total_seq_len, head_num, head_size) — key states.
            value (torch.Tensor | None): Shape (total_seq_len, head_num, head_size) — value states.
            key_cache (torch.Tensor): Shape (block_num, head_num, block_size, head_size) — key cache.
            value_cache (torch.Tensor | None): Shape (block_num, head_num, block_size, head_size) — value cache.
            key_cache_scale (torch.Tensor): Shape (head_num, head_size) — key cache scale.
            value_cache_scale (torch.Tensor | None): Shape (head_num, head_size) — value scale.
            context_lengths (torch.Tensor): Shape (batch_size,) — Valid sequence length for each batch sample.
            max_context_len (int): Scalar int value — Maximum valid sequence length across all batches during context prefill phase.
            context_seq_offset (torch.Tensor | None): Shape (batch_size,) — Cumulative sequence offset for each batch to guarantee non-overlapping sequence storage.
            block_tables (torch.Tensor | None): Shape (batch_size, max_block_num) — Logical-to-physical block mapping table for paged cache.

        Returns:
            None: All writes are performed in-place on key and value tensors.
        """
        def dequant_from_cache(quant_data: torch.Tensor, scale_data: torch.Tensor):
            quant_data_fp32 = quant_data.clone().to(torch.float)
            scale_data_fp32 = scale_data.clone().to(torch.float)
            scale_data_fp32 = scale_data[..., None, :]
            dequant_data_fp32 = quant_data_fp32 * scale_data_fp32
            return dequant_data_fp32

        batch_size = context_lengths.size(0)
        if context_seq_offset is None:
            cu_seq_offset = torch.cumsum(context_lengths, dim=-1)
            context_seq_offset = torch.zeros_like(cu_seq_offset)
            context_seq_offset[1:] = cu_seq_offset[:-1]

        total_seqlen = 0
        block_size = key_cache.size(2)
        for i in range(batch_size):
            context_len = context_lengths[i].item()
            seq_begin = context_seq_offset[i].item()
            seq_end = seq_begin + context_len
            total_seqlen += context_len
            full_block_num = context_len // block_size
            rem_token_num = context_len % block_size

            # dequant key from cache
            key_i = key[seq_begin:seq_end].transpose(1, 0)
            key_cache_i = torch.concat(
                [key_cache[block_tables[i, j], ...] for j in range(full_block_num)]
                + ([key_cache[block_tables[i, full_block_num], :, :rem_token_num, :]] if rem_token_num > 0 else []),
                dim=-2,
            )
            dequant_key_i = dequant_from_cache(key_cache_i, key_cache_scale)
            key_i[...] = dequant_key_i.to(key_i.dtype)

            # dequant value from cache
            if not (value_cache is None or value is None or value_cache_scale is None):
                value_i = value[seq_begin:seq_end].transpose(1, 0)
                value_cache_i = torch.concat(
                    [value_cache[block_tables[i, j], ...] for j in range(full_block_num)]
                    + ([value_cache[block_tables[i, full_block_num], :, :rem_token_num, :]] if rem_token_num > 0 else []),
                    dim=-2,
                )
                dequant_value_i = dequant_from_cache(value_cache_i, value_cache_scale)
                value_i[...] = dequant_value_i.to(value_i.dtype)
        return key, value

__all__ = [
    "MojoStorePagedMLAKVCache",
    "MojoStorePagedKVCacheC8",
    "MojoDequantFromPagedKVCache",
]
