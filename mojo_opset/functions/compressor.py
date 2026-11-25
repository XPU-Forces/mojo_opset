"""Token-aligned pooling of projected C2 activations and ring-state upkeep."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl

STATE_RING_ROWS = 32


def _validate_projected(
    kv,
    scores,
    state_cache,
    metadata,
    out,
    max_query_len,
    num_cores,
    prev_rows,
    prev_row_indices,
):
    if kv.dtype != torch.float32 or scores.dtype != torch.float32 or state_cache.dtype != torch.float32:
        raise ValueError("kv, scores and state_cache must be FP32")
    if kv.ndim != 2 or scores.shape != kv.shape:
        raise ValueError("kv and scores must share one [tokens, width] shape")
    tokens, width = kv.shape
    if width < 1 or width & (width - 1):
        raise ValueError("width must be a positive power of two")
    if state_cache.ndim != 3 or state_cache.shape[1:] != (STATE_RING_ROWS, 2 * width):
        raise ValueError(f"state_cache must have shape [blocks, {STATE_RING_ROWS}, 2*width]")
    if out.dtype != torch.bfloat16 or out.shape != kv.shape:
        raise ValueError("out must be BF16 with the kv shape")
    if metadata.dtype != torch.int32 or metadata.ndim != 2 or metadata.shape[0] != 5:
        raise ValueError("metadata must be INT32 [5, requests]")
    tensors = [kv, scores, state_cache, metadata, out]
    if (prev_rows is None) != (prev_row_indices is None):
        raise ValueError("prev_rows and prev_row_indices must be provided together")
    if prev_rows is not None:
        if prev_rows.dtype != torch.float32 or prev_rows.ndim != 2 or prev_rows.shape[1] != 2 * width:
            raise ValueError("prev_rows must be FP32 [gathered_rows, 2*width]")
        if prev_row_indices.dtype != torch.int64 or prev_row_indices.shape != (metadata.shape[1],):
            raise ValueError("prev_row_indices must be INT64 [requests]")
        tensors.extend((prev_rows, prev_row_indices))
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("all compressor tensors must be contiguous")
    if not all(t.device == kv.device for t in tensors):
        raise ValueError("all compressor tensors must share one device")
    if num_cores <= 0 or not 0 <= max_query_len <= tokens:
        raise ValueError("num_cores must be positive and max_query_len must lie in [0, tokens]")
    _require_inference("compressor_from_projected", *tensors)


def compressor_from_projected(
    kv: torch.Tensor,
    scores: torch.Tensor,
    state_cache: torch.Tensor,
    metadata: torch.Tensor,
    out: torch.Tensor,
    *,
    max_query_len: int,
    num_cores: int,
    prev_rows: Optional[torch.Tensor] = None,
    prev_row_indices: Optional[torch.Tensor] = None,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Zero `out`, pool completed global pairs and update the ring in place.

    Every request in `metadata` contributes pairs of consecutive global positions;
    the completed (odd) position of a pair is written as its softmax-weighted BF16
    row. Without `prev_rows`, the tail `min(used, 32)` rows of each valid page are
    overwritten with `concat(kv, scores)`; with `prev_rows`, the ring is read-only
    and the caller writes the gathered tails. Returns `out` so graph capture keeps
    the preallocated storage.
    """
    _validate_projected(
        kv,
        scores,
        state_cache,
        metadata,
        out,
        max_query_len,
        num_cores,
        prev_rows,
        prev_row_indices,
    )
    forward, _ = load_impl("compressor_from_projected", implementation)
    return forward(
        kv,
        scores,
        state_cache,
        metadata,
        out,
        int(max_query_len),
        int(num_cores),
        prev_rows,
        prev_row_indices,
    )
