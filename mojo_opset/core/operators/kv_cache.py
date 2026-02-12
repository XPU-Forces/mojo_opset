from typing import Tuple

import torch

from ..operator import MojoOperator


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
        cu_seq_lens: torch.Tensor,
        kv_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new K/V tokens into a block-based KV cache.
        Supports both prefill (via cu_seq_lens) and decode (batch of 1 token) scenarios.

        Args:
            key_states (torch.Tensor): Shape (token_num, kv_head_num, head_dim) — new key tokens.
            value_states (torch.Tensor): Shape (token_num, kv_head_num, head_dim) — new value tokens.
            key_cache (torch.Tensor): Shape (total_phys_blocks, kv_heads, block_size, head_dim) — key cache.
            value_cache (torch.Tensor): Shape (total_phys_blocks, kv_heads, block_size, head_dim) — value cache.
            block_table (torch.Tensor): Shape (bsz, max_blocks_per_seq) mapping logical blocks to physical IDs.
            cu_seq_lens (Optional[torch.Tensor]): Shape (bsz + 1,) cumulative sequence lengths for prefill.
                                                 If None, assumes decode phase (1 token per sequence).
            kv_lens (torch.Tensor): Shape (bsz,) current history sequence lengths per batch (start position for write).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated `(key_cahce, value_cahce)` after in-place writes.
        """
        assert len(key_states.shape) == 3 and len(value_states.shape) == 3 and key_states.shape == value_states.shape, (
            "key/value states must be (token_num, kv_head_num, head_dim), please check."
        )

        block_size = key_cache.shape[2]

        num_batches = len(kv_lens) if kv_lens is not None else 0

        is_decode_mode = cu_seq_lens is None

        for batch_id in range(num_batches):
            if not is_decode_mode:
                k_start = cu_seq_lens[batch_id].item()
                k_end = cu_seq_lens[batch_id + 1].item()
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

            now_kv_len_start = kv_lens[batch_id].item()
            now_block_table = block_table[batch_id]

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
    pass
