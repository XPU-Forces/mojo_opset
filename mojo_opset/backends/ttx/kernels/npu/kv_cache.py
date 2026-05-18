import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores

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
def _store_paged_kv_cache_kernel_legacy(
    k_ptr,
    v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    cu_seqlens_ptr,
    kv_lens_ptr,
    batch_size,
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
    CHUNK_SIZE: tl.constexpr,
    IS_DECODE: tl.constexpr,
    HAS_KV_LENS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    prev_chunks = 0
    for batch_idx in range(batch_size):
        if IS_DECODE:
            seq_start_tok = batch_idx
            seq_len_curr = 1
        else:
            seq_start_tok = tl.load(cu_seqlens_ptr + batch_idx)
            seq_end_tok = tl.load(cu_seqlens_ptr + batch_idx + 1)
            seq_len_curr = seq_end_tok - seq_start_tok

        write_start = 0
        if HAS_KV_LENS:
            write_start = tl.load(kv_lens_ptr + batch_idx)
        valid_write = write_start >= 0
        cur_chunks = tl.where(valid_write, tl.cdiv(seq_len_curr, CHUNK_SIZE), 0)
        start_chunk = (pid + num_programs - prev_chunks % num_programs) % num_programs
        prev_chunks += cur_chunks

        for chunk_idx in range(start_chunk, cur_chunks, num_programs):
            token_offset_in_seq = chunk_idx * CHUNK_SIZE
            valid_len = seq_len_curr - token_offset_in_seq
            curr_log_pos = write_start + token_offset_in_seq
            curr_kv_pos = seq_start_tok + token_offset_in_seq

            remain_chunk_len = tl.minimum(CHUNK_SIZE, valid_len)

            processed = 0
            while processed < remain_chunk_len:
                block_table_idx = curr_log_pos // block_size
                block_inner_off = curr_log_pos % block_size

                physical_block_id = tl.load(
                    block_table_ptr + batch_idx * stride_bt_batch + block_table_idx * stride_bt_blk
                )
                valid_block = physical_block_id >= 0
                physical_block_id = tl.maximum(physical_block_id, 0)

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
                    tl.store(dst_k_ptr, k_val, mask=valid_block & mask_sub[:, None])

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
                    tl.store(dst_v_ptr, v_val, mask=valid_block & mask_sub[:, None])

                processed += sub_len
                curr_log_pos += sub_len
                curr_kv_pos += sub_len


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

            base_src_k_ptr = (
                k_ptr
                + (src_token_start + token_offsets[:, None]) * stride_k_tok
                + offs_d[None, :] * stride_k_dim
            )
            base_dst_k_ptr = (
                key_cache_ptr
                + dst_block_id * stride_kc_blk
                + (dst_block_offset + token_offsets[:, None]) * stride_kc_tok
                + offs_d[None, :] * stride_kc_dim
            )
            base_src_v_ptr = (
                v_ptr
                + (src_token_start + token_offsets[:, None]) * stride_v_tok
                + offs_d[None, :] * stride_v_dim
            )
            base_dst_v_ptr = (
                value_cache_ptr
                + dst_block_id * stride_vc_blk
                + (dst_block_offset + token_offsets[:, None]) * stride_vc_tok
                + offs_d[None, :] * stride_vc_dim
            )

            for h in range(num_kv_heads):
                k_val = tl.load(base_src_k_ptr + h * stride_k_head, mask=mask_sub[:, None], other=0.0)
                tl.store(base_dst_k_ptr + h * stride_kc_head, k_val, mask=mask_sub[:, None])

                v_val = tl.load(base_src_v_ptr + h * stride_v_head, mask=mask_sub[:, None], other=0.0)
                tl.store(base_dst_v_ptr + h * stride_vc_head, v_val, mask=mask_sub[:, None])

def _get_chunk_num_programs(num_chunks: int) -> int:
    return max(1, min(get_num_cores("vector"), num_chunks))


def store_paged_kv_impl_legacy(
    k_states: torch.Tensor,
    v_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kv_lens_before_store: torch.Tensor,
):
    assert k_states.is_contiguous() and v_states.is_contiguous()

    is_decode = cu_seqlens is None
    if cu_seqlens is None:
        cu_seqlens = torch.arange(k_states.shape[0] + 1, device=k_states.device, dtype=torch.int32)
    batch_size = (
        kv_lens_before_store.shape[0] if is_decode and kv_lens_before_store is not None else cu_seqlens.shape[0] - 1
    )

    num_kv_heads = k_states.shape[1]
    head_dim = k_states.shape[2]
    block_size = key_cache.shape[2]

    num_programs = get_num_cores("vector")
    grid = (num_programs,)

    _store_paged_kv_cache_kernel_legacy[grid](
        k_states,
        v_states,
        key_cache,
        value_cache,
        block_table,
        cu_seqlens,
        kv_lens_before_store,
        batch_size,
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
        CHUNK_SIZE=block_size,
        IS_DECODE=is_decode,
        HAS_KV_LENS=kv_lens_before_store is not None,
    )

    return key_cache, value_cache


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
