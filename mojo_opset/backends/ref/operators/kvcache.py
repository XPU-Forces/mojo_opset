from typing import Tuple

import torch

from mojo_opset.core import MojoStorePagedKVCache


class RefStorePagedKVCache(MojoStorePagedKVCache):
    def forward(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
