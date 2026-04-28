from typing import Optional, Tuple

import torch

from ..operator import MojoOperator


def assert_paged_kv_store_contract(
    block_table: torch.Tensor,
    cu_q_lens: Optional[torch.Tensor],
    context_kv_lens: Optional[torch.Tensor],
) -> None:
    assert block_table.dtype == torch.int32
    if cu_q_lens is not None:
        assert cu_q_lens.dtype == torch.int32
    if context_kv_lens is not None:
        assert context_kv_lens.dtype == torch.int32
        assert block_table.shape[0] == context_kv_lens.shape[0]
    assert block_table.dim() == 2


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
        block_table: torch.Tensor,
        cu_q_lens: torch.Tensor,
        context_kv_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new K/V tokens into a block-based KV cache.
        Supports both prefill (via cu_q_lens) and decode (batch of 1 token) scenarios.

        Args:
            key_states (torch.Tensor): Shape (token_num, kv_head_num, head_dim) — new key tokens.
            value_states (torch.Tensor): Shape (token_num, kv_head_num, head_dim) — new value tokens.
            key_cache (torch.Tensor): Shape (total_phys_blocks, kv_heads, block_size, head_dim) — key cache.
            value_cache (torch.Tensor): Shape (total_phys_blocks, kv_heads, block_size, head_dim) — value cache.
            block_table (torch.Tensor): Shape (bsz, max_blocks_per_seq) mapping logical blocks to physical IDs.
            cu_q_lens (Optional[torch.Tensor]): Shape (bsz + 1,) cumulative query lengths for prefill.
                                                 If None, assumes decode phase (1 token per sequence).
            context_kv_lens (torch.Tensor): Shape (bsz,) history sequence
                lengths before storing the current tokens. Padding entries use -1.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated `(key_cahce, value_cahce)` after in-place writes.
        """
        assert len(key_states.shape) == 3 and len(value_states.shape) == 3 and key_states.shape == value_states.shape, (
            "key/value states must be (token_num, kv_head_num, head_dim), please check."
        )
        assert_paged_kv_store_contract(block_table, cu_q_lens, context_kv_lens)

        block_size = key_cache.shape[2]

        num_batches = len(context_kv_lens) if context_kv_lens is not None else 0

        is_decode_mode = cu_q_lens is None

        for batch_id in range(num_batches):
            if not is_decode_mode:
                k_start = cu_q_lens[batch_id].item()
                k_end = cu_q_lens[batch_id + 1].item()
                now_seq_len = k_end - k_start
            else:
                k_start = batch_id
                k_end = batch_id + 1
                now_seq_len = 1

            if now_seq_len <= 0:
                continue

            now_key = key_states[k_start:k_end]
            now_value = value_states[k_start:k_end]

            now_key = now_key.permute(1, 0, 2)
            now_value = now_value.permute(1, 0, 2)

            now_kv_len_start = context_kv_lens[batch_id].item()
            if now_kv_len_start < 0:
                continue
            now_block_table = block_table[batch_id]
            if now_block_table.numel() == 0 or now_block_table[0].item() < 0:
                continue

            start_block_table_idx = now_kv_len_start // block_size
            block_offset_in_first_block = now_kv_len_start % block_size

            remain_to_store = now_seq_len
            source_ptr = 0

            current_block_table_idx = start_block_table_idx
            current_block_offset = block_offset_in_first_block

            while remain_to_store > 0:
                if current_block_table_idx >= len(now_block_table):
                    break

                block_id = now_block_table[current_block_table_idx].item()
                if block_id < 0:
                    break

                capacity = block_size - current_block_offset
                store_len = min(remain_to_store, capacity)

                key_cache[block_id, :, current_block_offset : current_block_offset + store_len, :] = now_key[
                    :, source_ptr : source_ptr + store_len, :
                ]

                value_cache[block_id, :, current_block_offset : current_block_offset + store_len, :] = now_value[
                    :, source_ptr : source_ptr + store_len, :
                ]

                source_ptr += store_len
                remain_to_store -= store_len

                current_block_table_idx += 1
                current_block_offset = 0

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
        assert_paged_kv_store_contract(block_table, cu_q_lens, context_kv_lens)
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

            ckv_slice = compressed_kv_states[t_start:t_end]   # (seq_len, kv_lora_rank)
            kpe_slice = k_pe_states[t_start:t_end]             # (seq_len, qk_rope_head_dim)

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

                compressed_kv_cache[phys_id, 0, blk_off:blk_off + n, :] = ckv_slice[src:src + n]
                k_pe_cache[phys_id, 0, blk_off:blk_off + n, :] = kpe_slice[src:src + n]

                src += n
                remain -= n
                blk_idx += 1
                blk_off = 0

        return compressed_kv_cache, k_pe_cache
