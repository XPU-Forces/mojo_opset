"""
Copyright (c) 2026 Bytedance. All Rights Reserved.
"""

import logging
from typing import Tuple

import torch
import torch_npu

from mojo_opset.core import MojoStorePagedKVCache

logger = logging.getLogger(__name__)


def _build_slot_mapping(
    key_states: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_q_lens: torch.Tensor,
    context_kv_lens: torch.Tensor,
) -> torch.Tensor:
    block_size = key_cache.shape[2]
    num_batches = len(context_kv_lens) if context_kv_lens is not None else 0
    is_decode_mode = cu_q_lens is None
    slots = []

    for batch_id in range(num_batches):
        if not is_decode_mode:
            k_start = int(cu_q_lens[batch_id].item())
            k_end = int(cu_q_lens[batch_id + 1].item())
        else:
            k_start = batch_id
            k_end = batch_id + 1

        write_start = int(context_kv_lens[batch_id].item())
        if write_start < 0:
            continue

        now_block_table = block_table[batch_id]
        for token_idx in range(k_start, k_end):
            if token_idx >= key_states.shape[0]:
                break
            write_pos = write_start + token_idx - k_start
            block_table_idx = write_pos // block_size
            block_offset = write_pos % block_size
            if block_table_idx >= now_block_table.numel():
                break
            block_id = int(now_block_table[block_table_idx].item())
            if block_id < 0:
                break
            slots.append(block_id * block_size + block_offset)

    return torch.tensor(slots, dtype=torch.int32, device=key_states.device)


class TorchNpuStorePagedKVCache(MojoStorePagedKVCache, default_priority=0):
    """Store paged KV cache through torch_npu when layout constraints are met."""

    supported_platforms_list = ["npu"]

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
        if key_states.device.type == "npu":
            try:
                slot_mapping = _build_slot_mapping(key_states, key_cache, block_table, cu_q_lens, context_kv_lens)
                if slot_mapping.numel() != key_states.shape[0]:
                    raise ValueError(f"slot_mapping length {slot_mapping.numel()} != token count {key_states.shape[0]}")

                # PTA Norm mode expects cache as [num_blocks, block_size, num_heads, head_dim],
                # while Mojo stores [num_blocks, num_heads, block_size, head_dim].
                key_cache_norm = key_cache.permute(0, 2, 1, 3).contiguous()
                value_cache_norm = value_cache.permute(0, 2, 1, 3).contiguous()
                torch_npu.npu_scatter_pa_kv_cache(
                    key_states.contiguous(),
                    value_states.contiguous(),
                    key_cache_norm,
                    value_cache_norm,
                    slot_mapping,
                    cache_mode="Norm",
                )
                key_cache.copy_(key_cache_norm.permute(0, 2, 1, 3))
                value_cache.copy_(value_cache_norm.permute(0, 2, 1, 3))
                return key_cache, value_cache
            except Exception as exc:
                logger.warning(
                    "TorchNpuStorePagedKVCache: npu_scatter_pa_kv_cache failed (%s), fallback core block-table writes",
                    exc,
                )

        return super().forward(
            key_states,
            value_states,
            key_cache,
            value_cache,
            block_table,
            cu_q_lens,
            context_kv_lens,
        )
