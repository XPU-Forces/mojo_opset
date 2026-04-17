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


# Heuristics to set BLOCK_D dynamically based on head_dim
@triton.heuristics({
    'BLOCK_D': lambda args: args['head_dim'],
})
@triton.autotune(
    configs=[
        # Use a single welltested config to avoid autotuner issues
        triton.Config({'BLOCK_HEADS': 2, 'BLOCK_CHUNK': 128}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_HEADS': 4, 'BLOCK_CHUNK': 128}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_HEADS': 2, 'BLOCK_CHUNK': 256}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_HEADS': 4, 'BLOCK_CHUNK': 256}, num_stages=1, num_warps=1),
    ],
    key=['num_kv_heads', 'head_dim', 'block_size', 'total_chunks'],
)
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
    total_head_chunks,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Store key/value states into paged KV cache for MLU.
    Optimized with 2D grid parallelism and aggressive autotuning for maximum IO throughput.
    """
    pid_chunk = tl.program_id(0)
    pid_head_chunk = tl.program_id(1)

    total_programs_chunk = tl.num_programs(0)

    # Loop over chunks with stride for load balancing
    for chunk_idx in range(pid_chunk, total_chunks, total_programs_chunk):
        # Load chunk metadata once
        meta_ptr = chunk_indices_ptr + chunk_idx * 3
        batch_idx = tl.load(meta_ptr)
        token_offset_in_seq = tl.load(meta_ptr + 1)
        logical_kv_start = tl.load(meta_ptr + 2)

        seq_start_tok = tl.load(cu_seqlens_ptr + batch_idx)
        global_token_idx = seq_start_tok + token_offset_in_seq

        seq_len_curr = tl.load(cu_seqlens_ptr + batch_idx + 1) - seq_start_tok
        valid_len = seq_len_curr - token_offset_in_seq

        curr_log_pos = logical_kv_start
        curr_kv_pos = global_token_idx

        # Limit processing to valid_len to avoid overshooting
        max_process = tl.minimum(BLOCK_CHUNK, valid_len)

        processed = 0
        while processed < max_process:
            block_table_idx = curr_log_pos // block_size
            block_inner_off = curr_log_pos % block_size

            bt_base = batch_idx * stride_bt_batch
            physical_block_id = tl.load(block_table_ptr + bt_base + block_table_idx * stride_bt_blk)

            space_in_block = block_size - block_inner_off
            sub_len = tl.minimum(max_process - processed, space_in_block).to(tl.int32)

            if physical_block_id >= 0:
                head_start = pid_head_chunk * BLOCK_HEADS

                if head_start < num_kv_heads:
                    head_end = tl.minimum(head_start + BLOCK_HEADS, num_kv_heads)
                    num_valid_heads = head_end - head_start

                    # Pre-compute offsets for vectorized memory access
                    offs_d = tl.arange(0, BLOCK_D)
                    offs_h = tl.arange(0, BLOCK_HEADS)
                    mask_h = offs_h < num_valid_heads
                    offs_chunk = tl.arange(0, BLOCK_CHUNK)
                    mask_chunk = offs_chunk < sub_len

                    # Vectorized load of K - use input strides
                    src_k_base = k_ptr + curr_kv_pos * stride_k_tok + head_start * stride_k_head
                    offs_chunk_h_2d_src = offs_chunk[:, None] * stride_k_tok + offs_h[None, :] * stride_k_head
                    k_ptrs = src_k_base + offs_chunk_h_2d_src[:, :, None] + offs_d[None, None, :] * stride_k_dim
                    k_vals = tl.load(k_ptrs, mask=(mask_chunk[:, None, None] & mask_h[None, :, None]), other=0.0)

                    # Vectorized store of K - use cache strides!
                    dst_k_base = (
                        key_cache_ptr
                        + physical_block_id * stride_kc_blk
                        + head_start * stride_kc_head
                        + block_inner_off * stride_kc_tok
                    )
                    offs_chunk_h_2d_dst = offs_chunk[:, None] * stride_kc_tok + offs_h[None, :] * stride_kc_head
                    dst_k_ptrs = dst_k_base + offs_chunk_h_2d_dst[:, :, None] + offs_d[None, None, :] * stride_kc_dim
                    tl.store(dst_k_ptrs, k_vals, mask=(mask_chunk[:, None, None] & mask_h[None, :, None]))

                    # Vectorized load and store of V - use cache strides for dst!
                    src_v_base = v_ptr + curr_kv_pos * stride_v_tok + head_start * stride_v_head
                    offs_chunk_h_2d_v_src = offs_chunk[:, None] * stride_v_tok + offs_h[None, :] * stride_v_head
                    v_ptrs = src_v_base + offs_chunk_h_2d_v_src[:, :, None] + offs_d[None, None, :] * stride_v_dim
                    v_vals = tl.load(v_ptrs, mask=(mask_chunk[:, None, None] & mask_h[None, :, None]), other=0.0)

                    dst_v_base = (
                        value_cache_ptr
                        + physical_block_id * stride_vc_blk
                        + head_start * stride_vc_head
                        + block_inner_off * stride_vc_tok
                    )
                    offs_chunk_h_2d_v_dst = offs_chunk[:, None] * stride_vc_tok + offs_h[None, :] * stride_vc_head
                    dst_v_ptrs = dst_v_base + offs_chunk_h_2d_v_dst[:, :, None] + offs_d[None, None, :] * stride_vc_dim
                    tl.store(dst_v_ptrs, v_vals, mask=(mask_chunk[:, None, None] & mask_h[None, :, None]))

            advance = sub_len
            processed += advance
            curr_log_pos += advance
            curr_kv_pos += advance


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
    Highly optimized with 2D grid parallelism and aggressive autotuning for maximum IO throughput.
    Grid[0] processes chunks, Grid[1] processes head chunks.
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

    # Use large BLOCK_HEADS for better head dimension parallelism
    default_block_heads = 4
    total_head_chunks = (num_kv_heads + default_block_heads - 1) // default_block_heads

    # Calculate optimal grid size for 2D parallelism
    total_cores = get_mlu_total_cores()

    # For MLU, we want a more balanced 2D grid to maximize bandwidth
    # Split cores more evenly between chunk and head dimensions
    # Use a minimum of 2 programs in each dimension for better load balancing
    if total_cores >= 64:
        # High core count - use both dimensions efficiently
        grid_chunk = min(total_chunks, total_cores // 2)
        grid_head = min(total_head_chunks, total_cores // max(1, grid_chunk)) if grid_chunk > 0 else total_cores
    else:
        # Lower core count - prioritize balanced distribution
        grid_chunk = min(total_chunks, max(4, total_cores // 2))
        grid_head = min(total_head_chunks, total_cores // max(1, grid_chunk)) if grid_chunk > 0 else total_cores

    # Ensure at least some programs in each dimension with minimum thresholds
    grid_chunk = max(1, grid_chunk)
    grid_head = max(1, grid_head)

    # For small workloads, ensure we have enough programs to utilize bandwidth
    if grid_chunk < 4 and total_chunks > 1:
        grid_chunk = min(total_chunks, 4)

    grid = (grid_chunk, grid_head)

    grid = (grid_chunk, grid_head)

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
        total_head_chunks,
        opt_level="O3",
    )

    return key_cache, value_cache