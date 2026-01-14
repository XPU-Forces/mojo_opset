from typing import Tuple

import torch

from .. import VALID_KV_LAYOUTS
from ..operator import MojoOperator


class MojoStoreKVCache(MojoOperator):
    pass


class MojoStorePagedKVCache(MojoOperator):
    def __init__(
        self,
        kv_layout: str = VALID_KV_LAYOUTS[0],
        block_size: int = 16,
        op_name: str = "",
    ):
        """
        Initialize KV cache operator configuration.

        Args:
            kv_layout (str, default=VALID_KV_LAYOUTS[0]): Layout identifier; must be one of
                `VALID_KV_LAYOUTS`.
            block_size (int, default=16): Block length used when the layout is block-based.
            op_name (str, default=""): Operator name metadata.

        Raises:
            ValueError: If `kv_layout` is not in `VALID_KV_LAYOUTS`.

        Notes:
            Stores configuration only; actual read/write behavior depends on `kv_layout`
            and is implemented in forward.
        """
        super().__init__(op_name)
        if kv_layout not in VALID_KV_LAYOUTS:
            raise ValueError(f"kv_layout must be one of {VALID_KV_LAYOUTS}, got {kv_layout}")

        self.kv_layout = kv_layout
        self.block_size = block_size

    def forward(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new K/V tokens into a block-based KV cache.

        Args:
            key_states (torch.Tensor): Shape (B, Hkv, T_new, Dkv) — new key tokens.
            value_states (torch.Tensor): Shape (B, Hkv, T_new, Dkv) — new value tokens.
            k_cache (torch.Tensor): Shape (N_blocks, Hkv, block_size, Dkv) — key cache.
            v_cache (torch.Tensor): Shape (N_blocks, Hkv, block_size, Dkv) — value cache.
            block_tables (torch.Tensor): Shape (B, num_blocks) mapping logical blocks to physical IDs.
            context_lens (torch.Tensor): Shape (B,) current sequence lengths per batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated `(k_cache, v_cache)` after in-place writes.

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

                k_cache[physical_block_id, :, pos_in_block, :] = key_states[i, :, j, :]
                v_cache[physical_block_id, :, pos_in_block, :] = value_states[i, :, j, :]

        return k_cache, v_cache


class MojoStoreMLAKVCache(MojoOperator):
    pass


class MojoStorePagedMLAKVCache(MojoOperator):
    pass


class MojoKVCacheCast(MojoOperator):
    pass
