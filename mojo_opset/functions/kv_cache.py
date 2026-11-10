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


def kpool_compress_and_write_cache(
    kv_cache,
    slot_k,
    slot_score,
    ape,
    loc,
    pool_size,
    head_dim=128,
    *,
    write_mask=None,
    round_scale=True,
    return_compressed=False,
    write_cache=True,
    implementation=None,
):
    """Pool BF16 keys with softmax weights, rotate and quantize to E4M3FN.

    The key and score tensors have shape [N, pool_size, 128], with pool_size
    equal to 4 or 16. The position bias is FP32 [pool_size, 128]. The uint8
    cache has shape [pages, page_size, 132]. Each page stores all 128-byte
    keys first, followed by all FP32 scales. Page payloads must be contiguous,
    and their starts must be 4-byte aligned. The cache must not alias inputs.

    When writing, the int32/int64 [N] loc maps rows to flat cache slots.
    Selected destinations must be unique and in range. A bool [N] write_mask
    skips rows and can be used only in write-only mode. round_scale rounds
    scales up to powers of two. Returns None or (FP8 keys, FP32 scales) when
    return_compressed is set. At least one of write_cache and return_compressed
    must be set.
    """
    if head_dim != 128 or pool_size not in (4, 16):
        raise ValueError("Expected head_dim=128 and pool_size=4 or 16")
    if slot_k.ndim != 3 or slot_k.shape[1:] != (pool_size, head_dim) or slot_score.shape != slot_k.shape:
        raise ValueError("K and score must have matching [N, pool_size, 128] shapes")
    if slot_k.dtype != torch.bfloat16 or slot_score.dtype != torch.bfloat16:
        raise ValueError("K and score must use BF16")
    if ape.dtype != torch.float32 or ape.shape != (pool_size, head_dim):
        raise ValueError("Expected FP32 position bias with shape [pool_size, 128]")
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.ndim != 3
        or kv_cache.shape[2] != head_dim + 4
        or kv_cache.shape[1] == 0
        or kv_cache.stride(2) != 1
        or kv_cache.stride(1) != head_dim + 4
        or kv_cache.stride(0) < kv_cache.shape[1] * (head_dim + 4)
        or kv_cache.stride(0) % 4
        or kv_cache.storage_offset() % 4
    ):
        raise ValueError("Expected a uint8 [pages, page_size, 132] cache with aligned contiguous page payloads")
    if loc.dtype not in (torch.int32, torch.int64) or loc.shape != (slot_k.shape[0],):
        raise ValueError("Expected int32 or int64 loc with shape [N]")
    if not (write_cache or return_compressed):
        raise ValueError("Enable write_cache, return_compressed, or both")
    if write_mask is not None and (
        return_compressed or write_mask.shape != loc.shape or write_mask.dtype != torch.bool
    ):
        raise ValueError("Expected a bool [N] write_mask in write-only mode")
    _require_inference("kpool_compress_and_write_cache", slot_k, slot_score, ape)
    forward, _ = load_impl("kpool_compress_and_write_cache", implementation)
    return forward(
        kv_cache,
        slot_k,
        slot_score,
        ape,
        loc,
        pool_size,
        head_dim=head_dim,
        write_mask=write_mask,
        round_scale=round_scale,
        return_compressed=return_compressed,
        write_cache=write_cache,
    )
