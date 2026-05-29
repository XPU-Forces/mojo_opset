from typing import Tuple

import torch
import torch_npu

from mojo_opset.core import MojoStorePagedKVCache
from mojo_opset.core.operators.kv_cache import assert_paged_kv_store_contract


def _build_slot_mapping(block_table, cu_q_lens, context_kv_lens, block_size, total_tokens):
    batch_size = context_kv_lens.shape[0]
    is_decode = cu_q_lens is None
    slot_mapping = torch.full((total_tokens,), -1, dtype=torch.int32)
    for batch_idx in range(batch_size):
        kv_len_start = context_kv_lens[batch_idx].item()
        if kv_len_start < 0:
            continue
        if is_decode:
            token_start = batch_idx
            seq_len = 1
        else:
            token_start = cu_q_lens[batch_idx].item()
            seq_len = cu_q_lens[batch_idx + 1].item() - token_start
        for t in range(seq_len):
            logical_pos = kv_len_start + t
            bt_idx = logical_pos // block_size
            bt_off = logical_pos % block_size
            phys_block = block_table[batch_idx, bt_idx].item()
            if phys_block < 0:
                continue
            slot_mapping[token_start + t] = phys_block * block_size + bt_off
    return slot_mapping


def _nhsd_to_nz(cache, last_dim_k):
    B, H, BS, HD = cache.shape
    return (
        cache.reshape(B, H, BS, HD // last_dim_k, last_dim_k)
        .permute(0, 1, 3, 2, 4)
        .reshape(B, H * HD // last_dim_k, BS, last_dim_k)
        .contiguous()
    )


def _nz_to_nhsd(cache, num_heads, head_dim, last_dim_k):
    B, BS = cache.shape[0], cache.shape[2]
    return (
        cache.reshape(B, num_heads, head_dim // last_dim_k, BS, last_dim_k)
        .permute(0, 1, 3, 2, 4)
        .reshape(B, num_heads, BS, head_dim)
        .contiguous()
    )


def _nhsd_to_nshd(cache):
    return cache.permute(0, 2, 1, 3).contiguous()


def _nshd_to_nhsd(cache):
    return cache.permute(0, 2, 1, 3).contiguous()


class TorchNpuStorePagedKVCache(MojoStorePagedKVCache, default_priority=0):
    _USE_NZ_MODE = None

    def __init__(self):
        super().__init__()

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
        total_tokens = key_states.shape[0]
        kv_heads = key_states.shape[1]
        head_dim = key_states.shape[2]
        block_size = key_cache.shape[2]

        assert_paged_kv_store_contract(block_table, cu_q_lens, context_kv_lens)

        slot_mapping = _build_slot_mapping(
            block_table, cu_q_lens, context_kv_lens, block_size, total_tokens,
        )

        key_npu = key_states.npu()
        value_npu = value_states.npu()
        slot_mapping_npu = slot_mapping.npu()

        if TorchNpuStorePagedKVCache._USE_NZ_MODE is True:
            return self._forward_nz(
                key_npu, value_npu, key_cache, value_cache, slot_mapping_npu, kv_heads, head_dim,
            )
        elif TorchNpuStorePagedKVCache._USE_NZ_MODE is False:
            return self._forward_normal(
                key_npu, value_npu, key_cache, value_cache, slot_mapping_npu, kv_heads, head_dim,
            )
        else:
            try:
                result = self._forward_nz(
                    key_npu, value_npu, key_cache, value_cache, slot_mapping_npu, kv_heads, head_dim,
                )
                TorchNpuStorePagedKVCache._USE_NZ_MODE = True
                return result
            except RuntimeError:
                TorchNpuStorePagedKVCache._USE_NZ_MODE = False
                return self._forward_normal(
                    key_npu, value_npu, key_cache, value_cache, slot_mapping_npu, kv_heads, head_dim,
                )

    def _forward_nz(self, key_npu, value_npu, key_cache, value_cache, slot_mapping_npu, kv_heads, head_dim):
        last_dim_k = 32 // key_npu.element_size()
        assert head_dim % last_dim_k == 0

        key_cache_nz = _nhsd_to_nz(key_cache, last_dim_k)
        value_cache_nz = _nhsd_to_nz(value_cache, last_dim_k)
        key_cache_npu = key_cache_nz.npu()
        value_cache_npu = value_cache_nz.npu()

        torch_npu.npu_scatter_pa_kv_cache(
            key_npu, value_npu, key_cache_npu, value_cache_npu, slot_mapping_npu,
        )
        torch.npu.synchronize()

        key_cache_out = _nz_to_nhsd(key_cache_npu, kv_heads, head_dim, last_dim_k)
        value_cache_out = _nz_to_nhsd(value_cache_npu, kv_heads, head_dim, last_dim_k)
        return key_cache_out, value_cache_out

    def _forward_normal(self, key_npu, value_npu, key_cache, value_cache, slot_mapping_npu, kv_heads, head_dim):
        key_cache_nshd = _nhsd_to_nshd(key_cache)
        value_cache_nshd = _nhsd_to_nshd(value_cache)
        key_cache_npu = key_cache_nshd.npu()
        value_cache_npu = value_cache_nshd.npu()

        torch_npu.npu_scatter_pa_kv_cache(
            key_npu, value_npu, key_cache_npu, value_cache_npu, slot_mapping_npu,
        )
        torch.npu.synchronize()

        key_cache_out = _nshd_to_nhsd(key_cache_npu)
        value_cache_out = _nshd_to_nhsd(value_cache_npu)
        return key_cache_out, value_cache_out
