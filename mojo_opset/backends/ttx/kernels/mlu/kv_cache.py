import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.mlu.utils import get_mlu_total_cores
from mojo_opset.backends.ttx.kernels.utils import prepare_lens


def prepare_kv_chunk_indices(
    cu_seqlens: torch.Tensor, kv_lens: torch.Tensor, chunk_size: int, is_decode: bool = False
) -> torch.Tensor:
    """
    Generates metadata for each chunk to support arbitrary KV start positions.

    Output Tensor Shape: [Total_Chunks, 3]
    Columns:
        0: batch_idx (Sequence ID)
        1: token_offset_in_seq (Offset of this chunk in the current K/V sequence)
        2: logical_kv_start_index (Logical KV position = kv_lens[batch] + token_offset)
    """
    if is_decode:
        batch_size = kv_lens.shape[0]
        seq_ids = torch.arange(batch_size, device=cu_seqlens.device, dtype=torch.int32)
        token_offset_in_qkv = torch.zeros(batch_size, device=cu_seqlens.device, dtype=torch.int32)
        logical_kv_start = kv_lens.to(torch.int32)
        indices = torch.stack([seq_ids, token_offset_in_qkv, logical_kv_start], dim=1)
        return indices

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


@triton.heuristics({
    'BLOCK_HEADS': lambda args: min(8, args['num_kv_heads']),
    'BLOCK_CHUNK': lambda args: args['block_size'],
    'BLOCK_D': lambda args: args['head_dim'],
    'num_warps': lambda args: 1,
})
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
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    total_chunks,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Store key/value states into paged KV cache for MLU.
    Optimized with chunk-level parallelism and vectorized IO.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Process heads in chunks to exploit memory parallelism
    num_head_chunks = tl.cdiv(num_kv_heads, BLOCK_HEADS)

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

        remain_chunk_len = BLOCK_CHUNK
        remain_chunk_len = tl.minimum(remain_chunk_len, valid_len)

        processed = 0
        while processed < remain_chunk_len:
            block_table_idx = curr_log_pos // block_size
            block_inner_off = curr_log_pos % block_size

            bt_base = batch_idx * stride_bt_batch
            physical_block_id = tl.load(block_table_ptr + bt_base + block_table_idx * stride_bt_blk)

            space_in_block = block_size - block_inner_off
            sub_len = tl.minimum(remain_chunk_len - processed, space_in_block).to(tl.int32)

            if physical_block_id >= 0:
                offs_chunk = tl.arange(0, BLOCK_CHUNK)
                mask_chunk = offs_chunk < sub_len

                offs_d = tl.arange(0, BLOCK_D)

                # Process heads in parallel chunks using vectorized IO
                for head_chunk in range(num_head_chunks):
                    head_start = head_chunk * BLOCK_HEADS
                    head_end = tl.minimum(head_start + BLOCK_HEADS, num_kv_heads)
                    num_valid_heads = head_end - head_start

                    offs_h = tl.arange(0, BLOCK_HEADS)
                    mask_h = offs_h < num_valid_heads

                    # Vectorized load of K using NPU-style pointer broadcasting
                    # Load [sub_len, BLOCK_HEADS, BLOCK_D] in one operation
                    # This is more flexible than make_block_ptr for dynamic sizes
                    src_k_base = k_ptr + curr_kv_pos * stride_k_tok + head_start * stride_k_head
                    offs_chunk_3d = offs_chunk[:, None, None]
                    offs_h_3d = offs_h[None, :, None]
                    offs_d_3d = offs_d[None, None, :]

                    k_ptrs = (
                        src_k_base
                        + offs_chunk_3d * stride_k_tok
                        + offs_h_3d * stride_k_head
                        + offs_d_3d * stride_k_dim
                    )
                    k_vals = tl.load(k_ptrs, mask=(mask_chunk[:, None, None] & mask_h[None, :, None]), other=0.0)
                    k_vals = k_vals.to(tl.float32)

                    # Vectorized store of K
                    dst_k_base = (
                        key_cache_ptr
                        + physical_block_id * stride_kc_blk
                        + head_start * stride_kc_head
                        + block_inner_off * stride_kc_tok
                    )
                    dst_k_ptrs = (
                        dst_k_base
                        + offs_chunk_3d * stride_kc_tok
                        + offs_h_3d * stride_kc_head
                        + offs_d_3d * stride_kc_dim
                    )
                    tl.store(dst_k_ptrs, k_vals, mask=(mask_chunk[:, None, None] & mask_h[None, :, None]))

                    # Vectorized load of V
                    src_v_base = v_ptr + curr_kv_pos * stride_v_tok + head_start * stride_v_head
                    v_ptrs = (
                        src_v_base
                        + offs_chunk_3d * stride_v_tok
                        + offs_h_3d * stride_v_head
                        + offs_d_3d * stride_v_dim
                    )
                    v_vals = tl.load(v_ptrs, mask=(mask_chunk[:, None, None] & mask_h[None, :, None]), other=0.0)
                    v_vals = v_vals.to(tl.float32)

                    # Vectorized store of V
                    dst_v_base = (
                        value_cache_ptr
                        + physical_block_id * stride_vc_blk
                        + head_start * stride_vc_head
                        + block_inner_off * stride_vc_tok
                    )
                    dst_v_ptrs = (
                        dst_v_base
                        + offs_chunk_3d * stride_vc_tok
                        + offs_h_3d * stride_vc_head
                        + offs_d_3d * stride_vc_dim
                    )
                    tl.store(dst_v_ptrs, v_vals, mask=(mask_chunk[:, None, None] & mask_h[None, :, None]))

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
    """
    Store key/value states into paged KV cache for MLU.
    Optimized with chunk-level parallelism and vectorized IO.
    """
    assert k_states.is_contiguous() and v_states.is_contiguous()
    assert block_table.is_contiguous()

    is_decode = cu_seqlens is None
    if cu_seqlens is None:
        cu_seqlens = torch.arange(k_states.shape[0] + 1, device=k_states.device)

    num_kv_heads = k_states.shape[1]
    head_dim = k_states.shape[2]
    block_size = key_cache.shape[2]

    chunk_indices = prepare_kv_chunk_indices(cu_seqlens, kv_lens, block_size, is_decode=is_decode)
    total_chunks = chunk_indices.shape[0]

    # Use all cores
    grid = (get_mlu_total_cores(),)

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
        opt_level="O3",
    )

    return key_cache, value_cache