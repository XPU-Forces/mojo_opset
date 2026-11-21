"""Eager reference for the short-context top-k fill (vllm CPU indexer semantics)."""

import torch


def fill_short_context_topk_fwd(topk_indices_buffer, positions, top_k, compress_ratio):
    """Write ``i`` for ``i < (positions[r] + 1) // cr`` and ``-1`` elsewhere, in place."""
    device = topk_indices_buffer.device
    columns = torch.arange(top_k, device=device, dtype=torch.int64)
    num_compressed = (positions.to(torch.int64) + 1) // compress_ratio
    valid = columns.unsqueeze(0) < num_compressed.unsqueeze(-1)
    filled = torch.where(valid, columns.unsqueeze(0), torch.full_like(columns.unsqueeze(0), -1))
    topk_indices_buffer[:] = filled.to(topk_indices_buffer.dtype)
    return topk_indices_buffer
