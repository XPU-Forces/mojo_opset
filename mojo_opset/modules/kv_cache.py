"""Paged cache mutation without ownership of the caller's cache storage."""

import torch

from mojo_opset import functions


class StorePagedKVCache(torch.nn.Module):
    def __init__(self, *, implementation=None):
        super().__init__()
        self.implementation = implementation

    def forward(
        self,
        key_states,
        value_states,
        key_cache,
        value_cache,
        block_table=None,
        cu_q_lens=None,
        context_kv_lens=None,
        *,
        chunk_metadata=None,
    ):
        return functions.store_paged_kv_cache(
            key_states,
            value_states,
            key_cache,
            value_cache,
            block_table,
            cu_q_lens,
            context_kv_lens,
            chunk_metadata=chunk_metadata,
            implementation=self.implementation,
        )
