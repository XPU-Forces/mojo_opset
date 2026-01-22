from typing import Tuple

import torch

from ..operator import MojoOperator


class MojoStoreKVCache(MojoOperator):
    pass


class MojoStorePagedKVCache(MojoOperator):
    def __init__(
        self,
        block_size: int = 16,
    ):
        """
        Initialize block-based KV cache configuration.

        Args:
            block_size (int, default=16): Size of each cache block; must be > 0.

        Notes:
            This only stores configuration; forward paths use `block_size` to map logical
            positions to (block_id, offset) within the cache.
        """
        super().__init__()
        self.block_size = block_size

    def forward(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        key_cahce: torch.Tensor,
        value_cahce: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new K/V tokens into a block-based KV cache.

        Args:
            key_states (torch.Tensor): Shape (B, Hkv, T_new, Dkv) — new key tokens.
            value_states (torch.Tensor): Shape (B, Hkv, T_new, Dkv) — new value tokens.
            key_cahce (torch.Tensor): Shape (N_blocks, Hkv, block_size, Dkv) — key cache.
            value_cahce (torch.Tensor): Shape (N_blocks, Hkv, block_size, Dkv) — value cache.
            block_tables (torch.Tensor): Shape (B, num_blocks) mapping logical blocks to physical IDs.
            context_lens (torch.Tensor): Shape (B,) current sequence lengths per batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated `(key_cahce, value_cahce)` after in-place writes.

        Notes:
            - Logical position = `context_len + j`; block index = `pos // self.block_size`;
              position within block = `pos % self.block_size`.
            - Writes are performed in-place without bounds checking; caller must ensure capacity.
        """
        batch_size, _, new_seq_len, _ = key_states.shape

        for i in range(batch_size):
            context_len = context_lens[i].item()

            for j in range(new_seq_len):
                logical_pos = context_len + j
                block_idx_in_table = logical_pos // self.block_size
                pos_in_block = logical_pos % self.block_size

                physical_block_id = block_tables[i, block_idx_in_table].item()

                key_cahce[physical_block_id, :, pos_in_block, :] = key_states[i, :, j, :]
                value_cahce[physical_block_id, :, pos_in_block, :] = value_states[i, :, j, :]

        return key_cahce, value_cahce


class MojoStoreMLAKVCache(MojoOperator):
    pass


class MojoStorePagedMLAKVCache(MojoOperator):
    pass


class MojoKVCacheCast(MojoOperator):
    pass
