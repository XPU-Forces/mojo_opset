from typing import Tuple

import torch

from mojo_opset.backends.ttx.kernels import store_paged_kv
from mojo_opset.backends.ttx.kernels import store_paged_kv_mla
from mojo_opset.core import MojoStorePagedKVCache
from mojo_opset.core import MojoStorePagedMLAKVCache
from mojo_opset.core.operators.kv_cache import assert_paged_kv_store_contract


class TTXStorePagedKVCache(MojoStorePagedKVCache):
    supported_platforms_list = ["npu", "ilu"]

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
        assert_paged_kv_store_contract(block_table, cu_q_lens, context_kv_lens)
        return store_paged_kv(
            key_states,
            value_states,
            key_cache,
            value_cache,
            block_table,
            cu_q_lens,
            context_kv_lens,
        )

class TTXStorePagedMLAKVCache(MojoStorePagedMLAKVCache):
    supported_platforms_list = ["ilu"]

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
        return store_paged_kv_mla(compressed_kv_states,
        k_pe_states,
        compressed_kv_cache,
        k_pe_cache,
        block_table,
        cu_q_lens,
        context_kv_lens)
