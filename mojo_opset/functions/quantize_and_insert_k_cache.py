"""Quantize bf16 K rows to UE8M0 FP8 and insert them into a paged K cache."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl

V4_BYTES_PER_TOKEN = 584
V41_BYTES_PER_TOKEN = 528


def quantize_and_insert_k_cache(
    k: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
    is_ue8m0: bool = True,
    use_fnuz: bool = False,
    bytes_per_token: int = V4_BYTES_PER_TOKEN,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Quantize ``[num_tokens, 512]`` bf16 K and write it into ``k_cache`` in place.

    ``slot_mapping[i]`` is the global slot of token ``i``; ``-1`` skips it.
    ``k_cache`` is a flattened ``uint8`` view of ``[num_blocks, block_bytes]``
    pages; its ``stride(0)`` is the byte stride used for addressing.

    ``bytes_per_token`` selects the page record:

    * ``584`` (V4): 448 fp8 dims + 64 bf16 dims, 7 UE8M0 scales of 64 dims.
    * ``528`` (V4.1): all 512 dims fp8, 16 UE8M0 scales of 32 dims.

    Returns ``k_cache`` after updating its contents.
    """
    if k.dim() != 2 or k.shape[1] != 512:
        raise ValueError(f"k must be [num_tokens, 512], got {tuple(k.shape)}")
    if k.dtype != torch.bfloat16:
        raise ValueError(f"k must be bfloat16, got {k.dtype}")
    if k_cache.dtype != torch.uint8 or k_cache.dim() != 2:
        raise ValueError("k_cache must be a 2-D uint8 tensor")
    if not k_cache.is_contiguous():
        raise ValueError("k_cache must be contiguous (in-place scatter)")
    if slot_mapping.dim() != 1 or slot_mapping.dtype != torch.int64:
        raise ValueError("slot_mapping must be a 1-D int64 tensor")
    if not is_ue8m0:
        raise ValueError("Only UE8M0 quantization is supported")
    if bytes_per_token not in (V4_BYTES_PER_TOKEN, V41_BYTES_PER_TOKEN):
        raise ValueError(f"Unsupported bytes_per_token={bytes_per_token}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    _require_inference("quantize_and_insert_k_cache", k, k_cache, slot_mapping)

    forward, _ = load_impl("quantize_and_insert_k_cache", implementation)
    return forward(
        k.contiguous(),
        k_cache,
        slot_mapping.contiguous(),
        block_size,
        is_ue8m0,
        use_fnuz,
        bytes_per_token,
    )
