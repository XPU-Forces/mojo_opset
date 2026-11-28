"""Forward-only request index conversion for paged caches."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
    num_topk_tokens: int,
    has_prefill: bool = False,
    prefill_workspace_request_ids: Optional[torch.Tensor] = None,
    prefill_workspace_starts: Optional[torch.Tensor] = None,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Map per-request token indices to paged cache or prefill workspace indices.

    All tensors are int32 on one device. Token indices are -1 or nonnegative;
    request IDs and nonnegative prefill IDs must index their respective tables.
    Decode indices beyond the block table map to -1. Prefill indices are
    workspace-relative and are not limited by the decode block table.
    Both block_size and num_topk_tokens must be positive powers of two.
    block_size is at most 2**30. For nonempty inputs, T must be below 2**30
    and both T*K and R*M must be below 2**31 for signed int32 addressing.
    A2 Triton supports K <= 2048 and M <= 4096. A5 Triton supports
    K <= 8192 and M <= 4096. Computed indices must fit int32.
    """
    tensors = (req_id, block_table, token_indices)
    if any(t.dtype != torch.int32 for t in tensors):
        raise ValueError("req_id, block_table and token_indices must be int32")
    if req_id.ndim != 1 or block_table.ndim != 2 or token_indices.ndim != 2:
        raise ValueError("expected req_id [T], block_table [R, M], token_indices [T, K]")
    if not isinstance(num_topk_tokens, int) or num_topk_tokens <= 0 or num_topk_tokens & (num_topk_tokens - 1):
        raise ValueError("num_topk_tokens must be a positive power of two")
    # The Torch reference uses a positive signed int32 scalar divisor.
    if not isinstance(block_size, int) or not 0 < block_size <= 2**30 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two no greater than 2**30")
    if (
        token_indices.shape != (req_id.numel(), num_topk_tokens)
        or block_table.shape[0] == 0
        or block_table.shape[1] == 0
    ):
        raise ValueError("incompatible or empty input shapes")
    if has_prefill:
        if prefill_workspace_request_ids is None or prefill_workspace_starts is None:
            raise ValueError("prefill requires workspace request IDs and starts")
        if prefill_workspace_request_ids.dtype != torch.int32 or prefill_workspace_starts.dtype != torch.int32:
            raise ValueError("prefill tensors must be int32")
        if (
            prefill_workspace_request_ids.shape != req_id.shape
            or prefill_workspace_starts.ndim != 1
            or prefill_workspace_starts.numel() == 0
        ):
            raise ValueError("invalid prefill tensor shapes")
        tensors += (prefill_workspace_request_ids, prefill_workspace_starts)
    if any(t.device != req_id.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    _require_inference("triton_convert_req_index_to_global_index", *tensors)
    tokens = req_id.numel()
    if tokens == 0:
        return torch.empty_like(token_indices)
    # The A2 request scan can form c0 + T before clipping to the segment end.
    if tokens >= 2**30 or token_indices.numel() >= 2**31 or block_table.numel() >= 2**31:
        raise ValueError("require T < 2**30 and T*K, R*M < 2**31 for signed int32 addressing")
    forward, _ = load_impl("triton_convert_req_index_to_global_index", implementation)
    return forward(
        req_id,
        block_table,
        token_indices,
        block_size,
        num_topk_tokens,
        has_prefill,
        prefill_workspace_request_ids,
        prefill_workspace_starts,
    )
