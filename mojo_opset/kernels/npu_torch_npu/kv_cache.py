"""Paged KV scatter, using an explicit architecture-specific cache layout."""

from typing import Optional

import torch
import torch_npu

from mojo_opset.utils.kv_cache import build_paged_kv_chunk_metadata


def _nz(cache, last_dim):
    pages, heads, size, dim = cache.shape
    return (
        cache.reshape(pages, heads, size, dim // last_dim, last_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(pages, heads * dim // last_dim, size, last_dim)
        .contiguous()
    )


def _from_nz(cache, heads, dim, last_dim):
    pages, size = cache.shape[0], cache.shape[2]
    return (
        cache.reshape(pages, heads, dim // last_dim, size, last_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(pages, heads, size, dim)
        .contiguous()
    )


@torch.library.custom_op("mojo_npu_torch_npu::kv_format_cast", mutates_args=())
def _format_cast(x: torch.Tensor, acl_format: int) -> torch.Tensor:
    return torch_npu.npu_format_cast(x, acl_format)


@_format_cast.register_fake
def _format_cast_fake(x, acl_format):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("mojo_npu_torch_npu::scatter_paged_kv", mutates_args=("key_cache", "value_cache"))
def _scatter(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    torch_npu.npu_scatter_pa_kv_cache(key_states, value_states, key_cache, value_cache, slots)


@_scatter.register_fake
def _scatter_fake(key_states, value_states, key_cache, value_cache, slots):
    return None


def _store(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cu_q_lens: Optional[torch.Tensor],
    context_kv_lens: Optional[torch.Tensor],
    chunk_metadata: Optional[torch.Tensor],
    use_nz: bool,
) -> None:
    if chunk_metadata is None:
        chunk_metadata = build_paged_kv_chunk_metadata(
            block_table,
            cu_q_lens,
            context_kv_lens,
            key_cache.shape[2],
        )
    slots = torch.full((key_states.shape[0],), -1, dtype=torch.int32)
    for start, block, offset, length in chunk_metadata.tolist():
        slots[start : start + length] = torch.arange(
            block * key_cache.shape[2] + offset,
            block * key_cache.shape[2] + offset + length,
            dtype=torch.int32,
        )
    slots = slots.to(key_states.device)
    if use_nz:
        last_dim = 32 // key_states.element_size()
        if key_states.shape[-1] % last_dim:
            raise NotImplementedError("NZ paged-cache scatter requires an aligned head dimension")
        key = _format_cast(_nz(key_cache, last_dim), 29)
        value = _format_cast(_nz(value_cache, last_dim), 29)
    else:
        key = key_cache.permute(0, 2, 1, 3).contiguous()
        value = value_cache.permute(0, 2, 1, 3).contiguous()
    _scatter(key_states, value_states, key, value, slots)
    if use_nz:
        heads, dim = key_states.shape[1:]
        key = _from_nz(_format_cast(key, 2), heads, dim, last_dim)
        value = _from_nz(_format_cast(value, 2), heads, dim, last_dim)
    else:
        key, value = key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)
    key_cache.copy_(key)
    value_cache.copy_(value)


def store_paged_kv_cache_fwd(
    key_states, value_states, key_cache, value_cache, block_table, cu_q_lens, context_kv_lens, chunk_metadata
):
    # Mojo master test_kv_cache.py skipped all torch_npu variants (including
    # metadata/no-metadata and padded buckets) because of CI coredumps.
    # Keep the original helper for future porting; Triton/reference stay usable.
    raise NotImplementedError(
        "torch_npu store_paged_kv_cache is disabled: the original master skipped this provider due to CI coredumps."
    )
