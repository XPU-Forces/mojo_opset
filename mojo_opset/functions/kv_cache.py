"""Paged-cache writes and reusable chunk metadata, independent of provider."""

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def store_paged_kv_cache(
    key_states,
    value_states,
    key_cache,
    value_cache,
    block_table=None,
    cu_q_lens=None,
    context_kv_lens=None,
    *,
    chunk_metadata=None,
    implementation=None,
):
    """Write new [T,H,D] tokens into [pages,H,page_size,D] caches in place.

    Use either block_table/context lengths or precomputed [chunks,4] metadata.
    Returns the original cache tensors after updating their contents.
    """
    if key_states.ndim != 3 or key_states.shape != value_states.shape:
        raise ValueError("key/value states must have matching [T,H,D] shapes")
    if chunk_metadata is not None and any(t is not None for t in (block_table, cu_q_lens, context_kv_lens)):
        raise ValueError("chunk_metadata must not be mixed with block-table metadata")
    _require_inference("store_paged_kv_cache", key_states, value_states, key_cache, value_cache)
    forward, _ = load_impl("store_paged_kv_cache", implementation)
    return forward(
        key_states, value_states, key_cache, value_cache, block_table, cu_q_lens, context_kv_lens, chunk_metadata
    )


def kpool_seed_tail_cache(tail_kv_cache, key, gate_score, tslot, *, implementation=None):
    """Seed the tail cache in place for KPool=4 and head dimension 128.

    K and gate are contiguous BF16 [N, 128] tensors. The cache is
    [blocks, 2, ring, 128] with ring >= 4 and contiguous rows. The signed
    int32/int64 [N] slot tensor uses negative entries to skip input rows.
    A valid row is skipped if the row four positions later is valid and
    maps to the same cache block. Valid slots must be in range, selected
    destinations must be unique, and the cache must not alias any input.
    Returns None. Triton implementations are available for Ascend A2 and A5.
    """
    if key.dtype != torch.bfloat16 or gate_score.dtype != key.dtype or tail_kv_cache.dtype != key.dtype:
        raise ValueError("K, gate and cache must use BF16")
    if not (key.device == gate_score.device == tail_kv_cache.device == tslot.device):
        raise ValueError("All tensors must be on the same device")
    if key.ndim != 2 or key.shape[1] != 128 or gate_score.shape != key.shape:
        raise ValueError("K and gate must have shape [N, 128]")
    if tail_kv_cache.ndim != 4 or tail_kv_cache.shape[1] != 2 or tail_kv_cache.shape[3] != 128:
        raise ValueError("Cache must have shape [num_blocks, 2, RING, 128]")
    ring = tail_kv_cache.shape[2]
    if ring < 4 or tail_kv_cache.stride(3) != 1 or tail_kv_cache.stride(2) != 128:
        raise ValueError("RING >= 4 and contiguous cache rows are required")
    if tail_kv_cache.stride(1) < ring * 128 or tail_kv_cache.stride(0) < tail_kv_cache.stride(1) + ring * 128:
        raise ValueError("Cache blocks and channels must not overlap")
    if tail_kv_cache.shape[0] * ring >= 2**31:
        raise ValueError("The addressable slot range must fit signed int32")
    if tslot.ndim != 1 or tslot.shape[0] != key.shape[0] or tslot.dtype not in (torch.int32, torch.int64):
        raise ValueError("tslot must be signed int32/int64 with shape [N]")
    if not key.is_contiguous() or not gate_score.is_contiguous() or not tslot.is_contiguous():
        raise ValueError("K, gate and tslot must be contiguous")
    _require_inference("kpool_seed_tail_cache", tail_kv_cache, key, gate_score)
    forward, _ = load_impl("kpool_seed_tail_cache", implementation)
    return forward(tail_kv_cache, key, gate_score, tslot)
