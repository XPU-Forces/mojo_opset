import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores
from mojo_opset.backends.ttx.kernels.utils import prepare_lens


def prepare_kv_chunk_indices(cu_seqlens: torch.Tensor, kv_lens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Generates metadata for each chunk to support arbitrary KV start positions.

    Output Tensor Shape: [Total_Chunks, 3]
    Columns:
        0: batch_idx (Sequence ID)
        1: token_offset_in_seq (Offset of this chunk in the current K/V sequence)
        2: logical_kv_start_index (Logical KV position = kv_lens[batch] + token_offset)
    """
    seqlens = prepare_lens(cu_seqlens)

    chunks_per_seq = triton.cdiv(seqlens, chunk_size)
    seq_ids = torch.repeat_interleave(torch.arange(len(seqlens), device=cu_seqlens.device), chunks_per_seq)

    cumulative_chunks = torch.cumsum(chunks_per_seq, 0)
    chunk_starts = torch.cat([torch.tensor([0], device=cu_seqlens.device), cumulative_chunks[:-1]])

    flat_indices = torch.arange(chunks_per_seq.sum(), device=cu_seqlens.device)
    chunk_idx_in_seq = flat_indices - chunk_starts[seq_ids]

    token_offset_in_qkv = (chunk_idx_in_seq * chunk_size).to(torch.int32)

    if kv_lens is not None:
        batch_kv_lens = kv_lens[seq_ids]
        logical_kv_start = batch_kv_lens.to(torch.int32) + token_offset_in_qkv
    else:
        logical_kv_start = token_offset_in_qkv

    indices = torch.stack([seq_ids.to(torch.int32), token_offset_in_qkv, logical_kv_start], dim=1).to(torch.int32)
    return indices


@triton.jit
def _store_paged_kv_cache_kernel(
    k_ptr,
    v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    cu_seqlens_ptr,
    chunk_indices_ptr,
    stride_k_tok,
    stride_k_head,
    stride_k_dim,
    stride_v_tok,
    stride_v_head,
    stride_v_dim,
    stride_kc_blk,
    stride_kc_head,
    stride_kc_tok,
    stride_kc_dim,
    stride_vc_blk,
    stride_vc_head,
    stride_vc_tok,
    stride_vc_dim,
    stride_bt_batch,
    stride_bt_blk,
    num_kv_heads,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    total_chunks,
    CHUNK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for chunk_id_linear in range(pid, total_chunks, num_programs):
        meta_ptr = chunk_indices_ptr + chunk_id_linear * 3
        batch_idx = tl.load(meta_ptr)
        token_offset_in_seq = tl.load(meta_ptr + 1)
        logical_kv_start = tl.load(meta_ptr + 2)

        seq_start_tok = tl.load(cu_seqlens_ptr + batch_idx)
        global_token_idx = seq_start_tok + token_offset_in_seq

        seq_len_curr = tl.load(cu_seqlens_ptr + batch_idx + 1) - seq_start_tok
        valid_len = seq_len_curr - token_offset_in_seq

        curr_log_pos = logical_kv_start
        curr_kv_pos = global_token_idx

        remain_chunk_len = CHUNK_SIZE
        remain_chunk_len = tl.minimum(remain_chunk_len, valid_len)

        processed = 0
        while processed < remain_chunk_len:
            block_table_idx = curr_log_pos // block_size
            block_inner_off = curr_log_pos % block_size

            physical_block_id = tl.load(block_table_ptr + batch_idx * stride_bt_batch + block_table_idx * stride_bt_blk)

            space_in_block = block_size - block_inner_off
            sub_len = tl.minimum(remain_chunk_len - processed, space_in_block).to(tl.int32)

            offs_sub = tl.arange(0, CHUNK_SIZE)
            mask_sub = offs_sub < sub_len

            offs_d = tl.arange(0, head_dim)

            for h in range(num_kv_heads):
                src_k_ptr = (
                    k_ptr
                    + (curr_kv_pos + offs_sub[:, None]) * stride_k_tok
                    + h * stride_k_head
                    + offs_d[None, :] * stride_k_dim
                )

                k_val = tl.load(src_k_ptr, mask=mask_sub[:, None], other=0.0)

                dst_k_ptr = (
                    key_cache_ptr
                    + physical_block_id * stride_kc_blk
                    + h * stride_kc_head
                    + (block_inner_off + offs_sub[:, None]) * stride_kc_tok
                    + offs_d[None, :] * stride_kc_dim
                )

                tl.store(dst_k_ptr, k_val, mask=mask_sub[:, None])

                src_v_ptr = (
                    v_ptr
                    + (curr_kv_pos + offs_sub[:, None]) * stride_v_tok
                    + h * stride_v_head
                    + offs_d[None, :] * stride_v_dim
                )

                v_val = tl.load(src_v_ptr, mask=mask_sub[:, None], other=0.0)

                dst_v_ptr = (
                    value_cache_ptr
                    + physical_block_id * stride_vc_blk
                    + h * stride_vc_head
                    + (block_inner_off + offs_sub[:, None]) * stride_vc_tok
                    + offs_d[None, :] * stride_vc_dim
                )

                tl.store(dst_v_ptr, v_val, mask=mask_sub[:, None])

            processed += sub_len
            curr_log_pos += sub_len
            curr_kv_pos += sub_len


def store_paged_kv_impl(
    k_states: torch.Tensor,
    v_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kv_lens: torch.Tensor,
):
    assert k_states.is_contiguous() and v_states.is_contiguous()

    num_kv_heads = k_states.shape[1]
    head_dim = k_states.shape[2]

    block_size = key_cache.shape[2]

    chunk_indices = prepare_kv_chunk_indices(cu_seqlens, kv_lens, block_size)
    total_chunks = chunk_indices.shape[0]

    num_programs = get_num_cores("vector")
    grid = (num_programs,)

    _store_paged_kv_cache_kernel[grid](
        k_states,
        v_states,
        key_cache,
        value_cache,
        block_table,
        cu_seqlens,
        chunk_indices,
        k_states.stride(0),
        k_states.stride(1),
        k_states.stride(2),
        v_states.stride(0),
        v_states.stride(1),
        v_states.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        num_kv_heads,
        head_dim,
        block_size,
        total_chunks,
        CHUNK_SIZE=block_size,
    )

    return key_cache, value_cache
