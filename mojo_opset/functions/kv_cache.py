"""Paged-cache writes and reusable chunk metadata, independent of provider."""

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
