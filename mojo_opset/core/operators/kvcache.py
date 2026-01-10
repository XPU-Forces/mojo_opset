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
        pass


class MojoStoreMLAKVCache(MojoOperator):
    pass


class MojoStorePagedMLAKVCache(MojoOperator):
    pass


class MojoKVCacheCast(MojoOperator):
    pass
