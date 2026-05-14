from typing import Optional, Tuple

import torch
from ixformer import functions as ixf_f

from mojo_opset.core import MojoStorePagedKVCache

class IxformerStorePagedKVCache(MojoStorePagedKVCache):
    supported_platforms_list = ["ilu"]

    def forward(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        cu_q_lens: Optional[torch.Tensor] = None,
        context_kv_lens: Optional[torch.Tensor] = None,
        *,
        chunk_metadata: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Store new K/V tokens into ixformer's block-based paged KV cache.

        Args:
            key_states (torch.Tensor): New key tokens with shape
                (token_num, kv_head_num, head_dim).
            value_states (torch.Tensor): New value tokens with shape
                (token_num, kv_head_num, head_dim).
            key_cache (torch.Tensor): Paged key cache with shape
                (num_blocks, kv_head_num, block_size, head_dim), updated in-place.
            value_cache (torch.Tensor): Paged value cache with shape
                (num_blocks, kv_head_num, block_size, head_dim), updated in-place.
            block_table (torch.Tensor | None): Logical-to-physical block mapping with
                shape (batch_size, max_blocks_per_sequence).
            cu_q_lens (torch.Tensor | None): Cumulative query lengths for
                prefill with shape (batch_size + 1,). None indicates decode mode.
            context_kv_lens (torch.Tensor | None): Existing KV lengths before storing
                the current tokens, shape (batch_size,). Padding entries use -1.
            chunk_metadata (torch.Tensor | None): Optional optimized precomputed metadata.
                Ixformer does not provide a dedicated kernel for this path and
                falls back to the base implementation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated key_cache and value_cache.
        """
        if key_states.shape != value_states.shape or key_states.dim() != 3:
            raise ValueError("key/value states must be (token_num, kv_head_num, head_dim).")
        if key_cache.shape != value_cache.shape or key_cache.dim() != 4:
            raise ValueError("key/value cache must be (num_blocks, kv_head_num, block_size, head_dim).")
        if key_states.dtype != value_states.dtype or key_cache.dtype != key_states.dtype or value_cache.dtype != key_states.dtype:
            raise ValueError("IxformerStorePagedKVCache requires all key/value tensors to have the same dtype.")

        if chunk_metadata is not None:
            return super().forward(
                key_states,
                value_states,
                key_cache,
                value_cache,
                chunk_metadata=chunk_metadata,
            )

        if block_table is None or context_kv_lens is None:
            raise ValueError("block_table and context_kv_lens are required when chunk_metadata is not provided.")

        ixf_f.paged_store_kv_cache_with_block_table(
            key_states,
            value_states,
            key_cache,
            value_cache,
            block_table,
            cu_q_lens,
            context_kv_lens,
        )
        return key_cache, value_cache
