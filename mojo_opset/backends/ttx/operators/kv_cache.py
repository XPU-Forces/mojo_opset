from typing import Tuple

import torch

from mojo_opset.backends.ttx.kernels import store_paged_kv
from mojo_opset.core import MojoStorePagedKVCache


class TTXStorePagedKVCache(MojoStorePagedKVCache):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.kv_layout == "NPU_ND", "Only NPU_ND layout is supported."

        store_paged_kv(
            key_states,
            value_states,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            self.block_size,
        )
