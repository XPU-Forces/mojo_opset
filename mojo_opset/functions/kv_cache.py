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
    _require_inference("kpool_seed_tail_cache", tail_kv_cache, key, gate_score)
    forward, _ = load_impl("kpool_seed_tail_cache", implementation)
    return forward(tail_kv_cache, key, gate_score, tslot)
