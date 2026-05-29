import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores


@triton.jit
def _store_paged_kv_cache_kernel(
    k_ptr,
    v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    cu_seqlens_ptr,
    kv_lens_ptr,
    batch_size,
    stride_k_tok,
    stride_v_tok,
    stride_kc_blk,
    stride_kc_tok,
    stride_vc_blk,
    stride_vc_tok,
    stride_bt_batch,
    stride_bt_blk,
    kv_dim: tl.constexpr,
    block_size: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    IS_DECODE: tl.constexpr,
    HAS_KV_LENS: tl.constexpr,
    TOKEN_STEP: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    offs_kv = tl.arange(0, kv_dim)
    offs_step = tl.arange(0, TOKEN_STEP)

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

        bt_base = batch_idx * stride_bt_batch

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
                    block_table_ptr + bt_base + block_table_idx * stride_bt_blk
                )
                valid_block = physical_block_id >= 0
                physical_block_id = tl.maximum(physical_block_id, 0)

                space_in_block = block_size - block_inner_off
                sub_len = tl.minimum(remain_chunk_len - processed, space_in_block).to(tl.int32)

                kc_blk = key_cache_ptr + physical_block_id * stride_kc_blk
                vc_blk = value_cache_ptr + physical_block_id * stride_vc_blk

                step = 0
                while step < sub_len:
                    actual = tl.minimum(TOKEN_STEP, sub_len - step).to(tl.int32)
                    mask_step = offs_step < actual
                    mask_2d = valid_block & mask_step[:, None]
                    step_off = block_inner_off + step

                    src_k_ptr = k_ptr + (curr_kv_pos + step + offs_step[:, None]) * stride_k_tok + offs_kv[None, :]
                    src_v_ptr = v_ptr + (curr_kv_pos + step + offs_step[:, None]) * stride_v_tok + offs_kv[None, :]
                    k_val = tl.load(src_k_ptr, mask=mask_2d, other=0.0)
                    v_val = tl.load(src_v_ptr, mask=mask_2d, other=0.0)

                    dst_k_ptr = kc_blk + (step_off + offs_step[:, None]) * stride_kc_tok + offs_kv[None, :]
                    dst_v_ptr = vc_blk + (step_off + offs_step[:, None]) * stride_vc_tok + offs_kv[None, :]
                    tl.store(dst_k_ptr, k_val, mask=mask_2d)
                    tl.store(dst_v_ptr, v_val, mask=mask_2d)

                    step += actual

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
    kv_lens_before_store: torch.Tensor,
):
    assert k_states.is_contiguous() and v_states.is_contiguous()

    is_decode = cu_seqlens is None
    if cu_seqlens is None:
        cu_seqlens = torch.arange(k_states.shape[0] + 1, device=k_states.device, dtype=torch.int32)
    batch_size = kv_lens_before_store.shape[0] if is_decode and kv_lens_before_store is not None else cu_seqlens.shape[0] - 1

    num_kv_heads = k_states.shape[1]
    head_dim = k_states.shape[2]
    kv_dim = num_kv_heads * head_dim

    block_size = key_cache.shape[2]

    # Source: [tokens, num_kv_heads, head_dim] → [tokens, kv_dim] (view, zero-copy)
    k_flat = k_states.view(k_states.shape[0], kv_dim)
    v_flat = v_states.view(v_states.shape[0], kv_dim)

    # Cache: NHSD [blocks, heads, block_size, dim] → NSHD [blocks, block_size, kv_dim]
    kc_work = key_cache.permute(0, 2, 1, 3).contiguous()
    vc_work = value_cache.permute(0, 2, 1, 3).contiguous()

    # TOKEN_STEP: fit within UB budget (each step holds K+V buffers of [TOKEN_STEP, kv_dim])
    UB_AVAILABLE = 224 * 1024
    bytes_per_step_kv = 2 * kv_dim * k_states.element_size()
    max_step = UB_AVAILABLE // bytes_per_step_kv
    TOKEN_STEP = max(1, min(max_step, block_size))

    num_programs = get_num_cores("vector")
    grid = (num_programs,)

    total_tokens = int(cu_seqlens[-1])
    units_per_chunk = num_kv_heads
    target_chunks = max(num_programs // units_per_chunk, 1)

    raw = total_tokens / target_chunks
    import math
    chunk = 1 << int(math.floor(math.log2(max(raw,1 ))))
    CHUNK_SIZE = max(32,min(chunk, block_size))
    
    _store_paged_kv_cache_kernel[grid](
        k_flat,
        v_flat,
        kc_work,
        vc_work,
        block_table,
        cu_seqlens,
        kv_lens_before_store,
        batch_size,
        k_flat.stride(0),
        v_flat.stride(0),
        kc_work.stride(0),
        kc_work.stride(1),
        vc_work.stride(0),
        vc_work.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        kv_dim,
        block_size,
        CHUNK_SIZE,
        IS_DECODE=is_decode,
        HAS_KV_LENS=kv_lens_before_store is not None,
        TOKEN_STEP=TOKEN_STEP,
    )

    # NSHD [blocks, block_size, heads, dim] → NHSD [blocks, heads, block_size, dim]
    key_cache.copy_(kc_work.permute(0, 2, 1, 3))
    value_cache.copy_(vc_work.permute(0, 2, 1, 3))

    return key_cache, value_cache
