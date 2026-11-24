"""Reference in-place paged KV cache writes."""

from typing import Optional
from typing import Tuple

import torch

from mojo_opset.utils.kv_cache_metadata import assert_paged_kv_store_contract
from mojo_opset.utils.kv_cache_metadata import build_paged_kv_chunk_metadata


def store_paged_kv_cache_fwd(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    cu_q_lens: Optional[torch.Tensor] = None,
    context_kv_lens: Optional[torch.Tensor] = None,
    chunk_metadata: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(key_states.shape) == 3 and len(value_states.shape) == 3 and (key_states.shape == value_states.shape), (
        "key/value states must be (token_num, kv_head_num, head_dim), please check."
    )
    if chunk_metadata is None:
        assert block_table is not None, "block_table is required when chunk_metadata is not provided."
        assert context_kv_lens is not None, "context_kv_lens is required when chunk_metadata is not provided."
        chunk_metadata = build_paged_kv_chunk_metadata(block_table, cu_q_lens, context_kv_lens, key_cache.shape[2])
    else:
        assert block_table is None and cu_q_lens is None and (context_kv_lens is None), (
            "chunk_metadata path should not be mixed with block_table/cu_q_lens/context_kv_lens."
        )
    assert_paged_kv_store_contract(chunk_metadata)
    if chunk_metadata.shape[0] == 0:
        return (key_cache, value_cache)
    for src_token_start, dst_block_id, dst_block_offset, chunk_len in chunk_metadata.tolist():
        src_end = src_token_start + chunk_len
        dst_end = dst_block_offset + chunk_len
        key_cache[dst_block_id, :, dst_block_offset:dst_end, :] = key_states[src_token_start:src_end].permute(1, 0, 2)
        value_cache[dst_block_id, :, dst_block_offset:dst_end, :] = value_states[src_token_start:src_end].permute(
            1, 0, 2
        )
    return (key_cache, value_cache)
