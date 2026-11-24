"""Inference GQA/SWA APIs: packed prefill and single-token decode are distinct contracts."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def paged_decode_gqa_infer(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    total_seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
    max_total_seq_len: Optional[int] = None,
    *,
    is_causal=True,
    gqa_layout="AABB",
    implementation=None,
) -> torch.Tensor:
    if gqa_layout not in ("ABAB", "AABB"):
        raise ValueError("gqa_layout must be ABAB or AABB")
    _require_inference("paged_decode_gqa_infer", query, key_cache, value_cache)
    forward, _ = load_impl("paged_decode_gqa_infer", implementation)
    return forward(
        query,
        key_cache,
        value_cache,
        total_seq_lens,
        block_tables,
        softmax_scale,
        mask,
        max_total_seq_len,
        is_causal,
        gqa_layout,
    )


def prefill_gqa_infer(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    softmax_scale: Optional[float] = None,
    *,
    is_causal=True,
    gqa_layout="ABAB",
    implementation=None,
) -> torch.Tensor:
    if gqa_layout not in ("ABAB", "AABB"):
        raise ValueError("gqa_layout must be ABAB or AABB")
    _require_inference("prefill_gqa_infer", query, k_cache, v_cache)
    forward, _ = load_impl("prefill_gqa_infer", implementation)
    return forward(query, k_cache, v_cache, cu_q_lens, softmax_scale, is_causal, gqa_layout)


def paged_prefill_gqa_infer(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: Optional[float] = None,
    cu_total_seq_lens: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    max_total_seq_len: Optional[int] = None,
    *,
    is_causal=True,
    gqa_layout="AABB",
    scheduler_metadata=None,
    implementation=None,
) -> torch.Tensor:
    if gqa_layout not in ("ABAB", "AABB"):
        raise ValueError("gqa_layout must be ABAB or AABB")
    _require_inference("paged_prefill_gqa_infer", query, key_cache, value_cache)
    forward, _ = load_impl("paged_prefill_gqa_infer", implementation)
    return forward(
        query,
        key_cache,
        value_cache,
        cu_q_lens,
        block_tables,
        softmax_scale,
        cu_total_seq_lens,
        mask,
        max_q_len,
        max_total_seq_len,
        is_causal,
        gqa_layout,
        scheduler_metadata,
    )


def _prepare_paged_prefill_metadata(
    cu_q_lens, cu_total_seq_lens, num_q_heads, num_kv_heads, page_size, gqa_layout, implementation
):
    forward, _ = load_impl("paged_prefill_gqa_infer", implementation)
    prepare = getattr(forward, "prepare_metadata", None)
    if prepare is None:
        return None
    return prepare(cu_q_lens, cu_total_seq_lens, num_q_heads, num_kv_heads, page_size, gqa_layout)


def paged_prefill_swa_infer(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: Optional[float] = None,
    cu_total_seq_lens: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    max_total_seq_len: Optional[int] = None,
    *,
    is_causal=True,
    gqa_layout="AABB",
    global_window_size=None,
    local_window_size=None,
    implementation=None,
) -> torch.Tensor:
    if gqa_layout not in ("ABAB", "AABB"):
        raise ValueError("gqa_layout must be ABAB or AABB")
    _require_inference("paged_prefill_swa_infer", query, key_cache, value_cache)
    forward, _ = load_impl("paged_prefill_swa_infer", implementation)
    return forward(
        query,
        key_cache,
        value_cache,
        cu_q_lens,
        block_table,
        softmax_scale,
        cu_total_seq_lens,
        max_q_len,
        max_total_seq_len,
        is_causal,
        gqa_layout,
        global_window_size,
        local_window_size,
    )


def paged_decode_swa_infer(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    total_seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: Optional[float] = None,
    max_total_seq_len: Optional[int] = None,
    *,
    is_causal=True,
    gqa_layout="AABB",
    global_window_size=None,
    local_window_size=None,
    implementation=None,
) -> torch.Tensor:
    if gqa_layout not in ("ABAB", "AABB"):
        raise ValueError("gqa_layout must be ABAB or AABB")
    _require_inference("paged_decode_swa_infer", query, key_cache, value_cache)
    forward, _ = load_impl("paged_decode_swa_infer", implementation)
    return forward(
        query,
        key_cache,
        value_cache,
        total_seq_lens,
        block_table,
        softmax_scale,
        max_total_seq_len,
        is_causal,
        gqa_layout,
        global_window_size,
        local_window_size,
    )
