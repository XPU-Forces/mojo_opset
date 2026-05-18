from typing import Optional, Tuple

import torch

from mojo_opset.backends.ttx.kernels import store_paged_kv
from mojo_opset.core import MojoStorePagedKVCache
from mojo_opset.core.operators.kv_cache import assert_paged_kv_store_contract
from mojo_opset.core.operators.kv_cache import build_paged_kv_chunk_metadata


class TTXStorePagedKVCache(MojoStorePagedKVCache):
    supported_platforms_list = ["npu", "ilu", "mlu"]

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
        assert_paged_kv_store_contract(chunk_metadata)
        return store_paged_kv(
            key_states,
            value_states,
            key_cache,
            value_cache,
            chunk_metadata,
        )
