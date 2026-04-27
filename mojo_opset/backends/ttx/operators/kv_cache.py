from typing import Tuple

import torch

from mojo_opset.backends.ttx.kernels import store_paged_kv
from mojo_opset.core import MojoStorePagedKVCache
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
        cu_seq_lens: torch.Tensor,
        kv_lens_before_store: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert_paged_kv_store_contract(block_table, cu_seq_lens, kv_lens_before_store)
        return store_paged_kv(
            key_states,
            value_states,
            key_cache,
            value_cache,
            block_table,
            cu_seq_lens,
            kv_lens_before_store,
        )
