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


def kpool_compress_and_write_cache_fwd(
    kv_cache,
    slot_k,
    slot_score,
    ape,
    loc,
    pool_size,
    head_dim=128,
    write_mask=None,
    round_scale=True,
    return_compressed=False,
    write_cache=True,
):
    if write_mask is not None:
        slot_k, slot_score, loc = slot_k[write_mask], slot_score[write_mask], loc[write_mask]

    logits = slot_score.float() + ape
    weights = (logits - logits.amax(dim=1, keepdim=True)).exp()
    numerator = torch.zeros_like(logits[:, 0])
    denominator = torch.zeros_like(numerator)
    for slot in range(pool_size):
        numerator = torch.addcmul(numerator, slot_k[:, slot].float(), weights[:, slot])
        denominator = denominator + weights[:, slot]
    rotated = (numerator / denominator).bfloat16().float()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        a, b = rotated.reshape(rotated.shape[0], head_dim // (2 * stride), 2, stride).unbind(2)
        rotated = torch.stack((a + b, a - b), dim=2).reshape(rotated.shape[0], head_dim)
    rotated = (rotated * (head_dim**-0.5)).bfloat16().float()
    scales = rotated.abs().amax(dim=1).clamp_min(1e-4) * (1.0 / 448.0)
    if round_scale:
        scales = torch.exp2(torch.ceil(torch.log2(scales)))
    keys = (rotated / scales[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    if write_cache and loc.numel():
        page_size = kv_cache.shape[1]
        page_stride = kv_cache.stride(0)
        pages, rows = loc.long() // page_size, loc.long() % page_size
        page_offsets = pages * page_stride
        columns = torch.arange(head_dim, device=loc.device)
        scale_bytes = torch.arange(4, device=loc.device)
        key_offsets = (page_offsets + rows * head_dim)[:, None] + columns
        scale_offsets = (page_offsets + page_size * head_dim + rows * 4)[:, None] + scale_bytes
        storage_length = (kv_cache.shape[0] - 1) * page_stride + page_size * (head_dim + 4)
        flat = kv_cache.as_strided((storage_length,), (1,))
        flat[key_offsets] = keys.view(torch.uint8)
        flat[scale_offsets] = scales.view(torch.uint8).reshape(-1, 4)
    if return_compressed:
        return keys, scales
    return None
