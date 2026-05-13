import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.mlu.utils import get_mlu_total_cores

_MAX_CHUNK_TILE_ELEMS = 16384


def _floor_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x.bit_length() - 1)


def _get_chunk_size(block_size: int, head_dim: int) -> int:
    if head_dim <= 0:
        return 1

    chunk_cap = max(1, _MAX_CHUNK_TILE_ELEMS // head_dim)
    return min(block_size, _floor_power_of_two(chunk_cap))


def _get_num_subchunks(block_size: int, chunk_size: int) -> int:
    return max(1, triton.cdiv(block_size, chunk_size))


@triton.jit
def _store_kv_chunk(
    k_ptr,
    v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    batch_idx,
    head_start,
    num_valid_heads,
    seq_start_tok,
    seq_len_curr,
    token_offset_in_seq,
    write_start,
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
    batch_size: tl.constexpr,
    max_chunks_per_seq: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Helper function to store a single chunk of KV data."""
    global_token_idx = seq_start_tok + token_offset_in_seq
    curr_log_pos = write_start + token_offset_in_seq
    valid_len = seq_len_curr - token_offset_in_seq

    max_process = tl.minimum(BLOCK_CHUNK, valid_len)

    processed = 0
    while processed < max_process:
        block_table_idx = curr_log_pos // block_size
        block_inner_off = curr_log_pos % block_size

        bt_base = batch_idx * stride_bt_batch
        physical_block_id = tl.load(block_table_ptr + bt_base + block_table_idx * stride_bt_blk)

        valid_block = physical_block_id >= 0
        physical_block_id = tl.maximum(physical_block_id, 0)

        space_in_block = block_size - block_inner_off
        sub_len = tl.minimum(max_process - processed, space_in_block).to(tl.int32)

        offs_d = tl.arange(0, BLOCK_D)
        offs_h = tl.arange(0, BLOCK_HEADS)
        mask_h = offs_h < num_valid_heads
        offs_chunk = tl.arange(0, BLOCK_CHUNK)
        mask_chunk = offs_chunk < sub_len

        src_k_base = k_ptr + global_token_idx * stride_k_tok + head_start * stride_k_head
        offs_chunk_h_2d_src = offs_chunk[:, None] * stride_k_tok + offs_h[None, :] * stride_k_head
        k_ptrs = src_k_base + offs_chunk_h_2d_src[:, :, None] + offs_d[None, None, :] * stride_k_dim
        k_vals = tl.load(k_ptrs, mask=(mask_chunk[:, None, None] & mask_h[None, :, None]), other=0.0)

        dst_k_base = (
            key_cache_ptr
            + physical_block_id * stride_kc_blk
            + head_start * stride_kc_head
            + block_inner_off * stride_kc_tok
        )
        offs_chunk_h_2d_dst = offs_chunk[:, None] * stride_kc_tok + offs_h[None, :] * stride_kc_head
        dst_k_ptrs = dst_k_base + offs_chunk_h_2d_dst[:, :, None] + offs_d[None, None, :] * stride_kc_dim
        tl.store(dst_k_ptrs, k_vals, mask=(valid_block & mask_chunk[:, None, None] & mask_h[None, :, None]))

        src_v_base = v_ptr + global_token_idx * stride_v_tok + head_start * stride_v_head
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
        tl.store(dst_v_ptrs, v_vals, mask=(valid_block & mask_chunk[:, None, None] & mask_h[None, :, None]))

        advance = sub_len
        processed += advance
        curr_log_pos += advance
        global_token_idx += advance


@triton.heuristics({
    'BLOCK_D': lambda args: args['head_dim'],
})
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HEADS': 2, 'BLOCK_CHUNK': 128}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_HEADS': 4, 'BLOCK_CHUNK': 128}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_HEADS': 2, 'BLOCK_CHUNK': 256}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_HEADS': 4, 'BLOCK_CHUNK': 256}, num_stages=1, num_warps=1),
    ],
    key=['num_kv_heads', 'head_dim', 'block_size', 'batch_size', 'max_chunks_per_seq', 'total_head_blocks'],
)
@triton.jit
def _store_paged_kv_cache_kernel_legacy(
    k_ptr,
    v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    cu_seqlens_ptr,
    kv_lens_ptr,
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
    batch_size,
    max_chunks_per_seq,
    total_head_blocks,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_KV_LENS: tl.constexpr,
):
    """
    Store key/value states into paged KV cache for MLU.
    Supports padding logic and cuda graph.
    """
    pid_chunk = tl.program_id(0)
    pid_head_block = tl.program_id(1)

    total_programs_chunk = tl.num_programs(0)
    total_programs_head = tl.num_programs(1)

    for head_block_idx in range(pid_head_block, total_head_blocks, total_programs_head):
        head_start = head_block_idx * BLOCK_HEADS

        if head_start < num_kv_heads:
            head_end = tl.minimum(head_start + BLOCK_HEADS, num_kv_heads)
            num_valid_heads = head_end - head_start

            for chunk_idx in range(pid_chunk, batch_size * max_chunks_per_seq, total_programs_chunk):
                batch_idx = chunk_idx // max_chunks_per_seq
                chunk_in_seq = chunk_idx % max_chunks_per_seq

                # Process only if batch_idx < batch_size
                if batch_idx < batch_size:
                    seq_start_tok = tl.load(cu_seqlens_ptr + batch_idx)
                    seq_end_tok = tl.load(cu_seqlens_ptr + batch_idx + 1)
                    seq_len_curr = seq_end_tok - seq_start_tok

                    token_offset_in_seq = chunk_in_seq * BLOCK_CHUNK
                    # Process only if token_offset_in_seq < seq_len_curr
                    if token_offset_in_seq < seq_len_curr:
                        write_start = 0
                        if HAS_KV_LENS:
                            write_start = tl.load(kv_lens_ptr + batch_idx)
                            # Process only if write_start >= 0
                            if write_start >= 0:
                                write_start = write_start.to(tl.int32)
                                _store_kv_chunk(
                                    k_ptr, v_ptr,
                                    key_cache_ptr, value_cache_ptr,
                                    block_table_ptr,
                                    batch_idx, head_start, num_valid_heads,
                                    seq_start_tok, seq_len_curr, token_offset_in_seq,
                                    write_start,
                                    stride_k_tok, stride_k_head, stride_k_dim,
                                    stride_v_tok, stride_v_head, stride_v_dim,
                                    stride_kc_blk, stride_kc_head, stride_kc_tok, stride_kc_dim,
                                    stride_vc_blk, stride_vc_head, stride_vc_tok, stride_vc_dim,
                                    stride_bt_batch, stride_bt_blk,
                                    num_kv_heads, batch_size, max_chunks_per_seq,
                                    head_dim, block_size,
                                    BLOCK_HEADS, BLOCK_CHUNK, BLOCK_D,
                                )
                        else:
                            _store_kv_chunk(
                                k_ptr, v_ptr,
                                key_cache_ptr, value_cache_ptr,
                                block_table_ptr,
                                batch_idx, head_start, num_valid_heads,
                                seq_start_tok, seq_len_curr, token_offset_in_seq,
                                write_start,
                                stride_k_tok, stride_k_head, stride_k_dim,
                                stride_v_tok, stride_v_head, stride_v_dim,
                                stride_kc_blk, stride_kc_head, stride_kc_tok, stride_kc_dim,
                                stride_vc_blk, stride_vc_head, stride_vc_tok, stride_vc_dim,
                                stride_bt_batch, stride_bt_blk,
                                num_kv_heads, batch_size, max_chunks_per_seq,
                                head_dim, block_size,
                                BLOCK_HEADS, BLOCK_CHUNK, BLOCK_D,
                            )


def store_paged_kv_impl_legacy(
    k_states: torch.Tensor,
    v_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kv_lens_before_store: torch.Tensor,
):
    """
    Store key/value states into paged KV cache for MLU.
    Supports padding logic and cuda graph.
    """
    assert k_states.is_contiguous() and v_states.is_contiguous()
    assert block_table.is_contiguous()

    is_decode = cu_seqlens is None
    if cu_seqlens is None:
        cu_seqlens = torch.arange(k_states.shape[0] + 1, device=k_states.device, dtype=torch.int32)

    num_kv_heads = k_states.shape[1]
    head_dim = k_states.shape[2]
    block_size = key_cache.shape[2]

    batch_size: int = block_table.shape[0]
    max_chunks_per_seq = block_table.shape[1]

    total_cores = get_mlu_total_cores()

    # Grid for chunks: batch_size * max_chunks_per_seq
    total_chunks = batch_size * max_chunks_per_seq

    if total_cores >= 64:
        grid_chunk = min(total_chunks, total_cores // 2)
    else:
        grid_chunk = min(total_chunks, max(4, total_cores // 2))

    grid_chunk = max(1, grid_chunk)

    if grid_chunk < 4 and total_chunks > 1:
        grid_chunk = min(total_chunks, 4)

    # Calculate total head blocks needed (each block handles BLOCK_HEADS heads)
    # Use BLOCK_HEADS=2 (minimum in autotuner) for grid calculation
    total_head_blocks = (num_kv_heads + 1) // 2

    # Calculate grid_head - limit based on available cores
    grid_head_limit = max(1, total_cores // max(1, grid_chunk))
    grid_head = min(total_head_blocks, grid_head_limit)
    grid = (grid_chunk, grid_head)

    _store_paged_kv_cache_kernel_legacy[grid](
        k_states,
        v_states,
        key_cache,
        value_cache,
        block_table,
        cu_seqlens,
        kv_lens_before_store,
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
        batch_size,
        max_chunks_per_seq,
        total_head_blocks,
        head_dim,
        block_size,
        opt_level="O3",
        HAS_KV_LENS=kv_lens_before_store is not None,
    )

    return key_cache, value_cache


@triton.jit
def _store_paged_kv_cache_chunk_kernel(
    k_ptr,
    v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    chunk_meta_ptr,
    num_chunks,
    stride_cm_row,
    stride_cm_col,
    num_kv_heads,
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
    head_dim: tl.constexpr,
    num_subchunks: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offs_sub = tl.arange(0, CHUNK_SIZE)
    offs_d = tl.arange(0, head_dim)

    for chunk_idx in range(pid, num_chunks, num_programs):
        chunk_base = chunk_meta_ptr + chunk_idx * stride_cm_row
        src_token_start = tl.load(chunk_base + 0 * stride_cm_col)
        dst_block_id = tl.load(chunk_base + 1 * stride_cm_col)
        dst_block_offset = tl.load(chunk_base + 2 * stride_cm_col)
        chunk_len = tl.load(chunk_base + 3 * stride_cm_col)

        for subchunk_idx in range(num_subchunks):
            processed = subchunk_idx * CHUNK_SIZE
            token_offsets = processed + offs_sub
            mask_sub = token_offsets < chunk_len

            for h in range(num_kv_heads):
                src_k_ptr = (
                    k_ptr
                    + (src_token_start + token_offsets[:, None]) * stride_k_tok
                    + h * stride_k_head
                    + offs_d[None, :] * stride_k_dim
                )
                k_val = tl.load(src_k_ptr, mask=mask_sub[:, None], other=0.0)

                dst_k_ptr = (
                    key_cache_ptr
                    + dst_block_id * stride_kc_blk
                    + h * stride_kc_head
                    + (dst_block_offset + token_offsets[:, None]) * stride_kc_tok
                    + offs_d[None, :] * stride_kc_dim
                )
                tl.store(dst_k_ptr, k_val, mask=mask_sub[:, None])

                src_v_ptr = (
                    v_ptr
                    + (src_token_start + token_offsets[:, None]) * stride_v_tok
                    + h * stride_v_head
                    + offs_d[None, :] * stride_v_dim
                )
                v_val = tl.load(src_v_ptr, mask=mask_sub[:, None], other=0.0)

                dst_v_ptr = (
                    value_cache_ptr
                    + dst_block_id * stride_vc_blk
                    + h * stride_vc_head
                    + (dst_block_offset + token_offsets[:, None]) * stride_vc_tok
                    + offs_d[None, :] * stride_vc_dim
                )
                tl.store(dst_v_ptr, v_val, mask=mask_sub[:, None])


def _get_chunk_num_programs(num_chunks: int) -> int:
    return max(1, min(get_mlu_total_cores(), num_chunks))


def store_paged_kv_impl(
    k_states: torch.Tensor,
    v_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    chunk_metadata: torch.Tensor,
):
    assert k_states.is_contiguous() and v_states.is_contiguous()
    assert chunk_metadata.dim() == 2
    assert chunk_metadata.shape[1] == 4

    num_chunks = chunk_metadata.shape[0]
    if num_chunks == 0:
        return key_cache, value_cache

    num_kv_heads = k_states.shape[1]
    head_dim = k_states.shape[2]
    block_size = key_cache.shape[2]
    chunk_size = _get_chunk_size(block_size, head_dim)
    num_subchunks = _get_num_subchunks(block_size, chunk_size)
    grid = (_get_chunk_num_programs(num_chunks),)

    _store_paged_kv_cache_chunk_kernel[grid](
        k_states,
        v_states,
        key_cache,
        value_cache,
        chunk_metadata,
        num_chunks,
        chunk_metadata.stride(0),
        chunk_metadata.stride(1),
        num_kv_heads,
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
        head_dim=head_dim,
        num_subchunks=num_subchunks,
        CHUNK_SIZE=chunk_size,
    )

    return key_cache, value_cache
