import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores


@triton.jit
def _decode_store_kv_fast_path(
    k_ptr,
    v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
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
    HAS_KV_LENS: tl.constexpr,
    TOKEN_STEP: tl.constexpr,
):
    """
    Decode fast path: each batch has exactly 1 token, head_pid/num_heads split
    across program_id(0). Core-level parallelism is not needed (too little work),
    so all tokens for a head are handled by a single program.
    """
    head_pid = tl.program_id(0)
    batch_pid = tl.program_id(1) if tl.num_programs(1) > 0 else 0
    batch_step = tl.num_programs(1) if tl.num_programs(1) > 0 else 1

    offs_d = tl.arange(0, head_dim)

    for batch_idx in range(batch_pid, batch_size, batch_step):
        write_start = 0
        if HAS_KV_LENS:
            write_start = tl.load(kv_lens_ptr + batch_idx)
        if write_start < 0:
            continue

        block_table_idx = write_start // block_size
        block_inner_off = write_start % block_size

        physical_block_id = tl.load(
            block_table_ptr + batch_idx * stride_bt_batch + block_table_idx * stride_bt_blk
        )
        valid_block = physical_block_id >= 0
        physical_block_id = tl.maximum(physical_block_id, 0)

        src_k_pos = batch_idx
        src_k_ptr = (
            k_ptr
            + src_k_pos * stride_k_tok
            + head_pid * stride_k_head
            + offs_d * stride_k_dim
        )
        k_val = tl.load(src_k_ptr, mask=valid_block, other=0.0)

        dst_k_ptr = (
            key_cache_ptr
            + physical_block_id * stride_kc_blk
            + head_pid * stride_kc_head
            + block_inner_off * stride_kc_tok
            + offs_d * stride_kc_dim
        )
        tl.store(dst_k_ptr, k_val, mask=valid_block)

        src_v_ptr = (
            v_ptr
            + src_k_pos * stride_v_tok
            + head_pid * stride_v_head
            + offs_d * stride_v_dim
        )
        v_val = tl.load(src_v_ptr, mask=valid_block, other=0.0)

        dst_v_ptr = (
            value_cache_ptr
            + physical_block_id * stride_vc_blk
            + head_pid * stride_vc_head
            + block_inner_off * stride_vc_tok
            + offs_d * stride_vc_dim
        )
        tl.store(dst_v_ptr, v_val, mask=valid_block)


@triton.jit
def _prefill_store_kv_kernel(
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
    HAS_KV_LENS: tl.constexpr,
    TOKEN_STEP: tl.constexpr,
):
    """
    Prefill path: uses round-robin chunk scheduling across cores for load balance.
    2D grid: (num_kv_heads, num_cores_per_head)
    """
    head_pid = tl.program_id(0)
    core_pid = tl.program_id(1)
    num_cores = tl.num_programs(1)

    offs_d = tl.arange(0, head_dim)
    offs_step = tl.arange(0, TOKEN_STEP)

    prev_chunks = 0
    for batch_idx in range(batch_size):
        seq_start_tok = tl.load(cu_seqlens_ptr + batch_idx)
        seq_end_tok = tl.load(cu_seqlens_ptr + batch_idx + 1)
        seq_len_curr = seq_end_tok - seq_start_tok

        write_start = 0
        if HAS_KV_LENS:
            write_start = tl.load(kv_lens_ptr + batch_idx)
        valid_write = write_start >= 0
        cur_chunks = tl.where(valid_write, tl.cdiv(seq_len_curr, CHUNK_SIZE), 0)
        start_chunk = (core_pid + num_cores - prev_chunks % num_cores) % num_cores
        prev_chunks += cur_chunks

        bt_base = batch_idx * stride_bt_batch

        for chunk_idx in range(start_chunk, cur_chunks, num_cores):
            token_offset_in_seq = chunk_idx * CHUNK_SIZE
            valid_len = seq_len_curr - token_offset_in_seq
            curr_log_pos = write_start + token_offset_in_seq
            curr_kv_pos = seq_start_tok + token_offset_in_seq

            remain_chunk_len = tl.minimum(CHUNK_SIZE, valid_len)
            block_end = block_size - curr_log_pos % block_size

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

                kc_step = 0
                while kc_step < sub_len:
                    actual = tl.minimum(TOKEN_STEP, sub_len - kc_step).to(tl.int32)
                    mask_step = offs_step < actual
                    mask_2d = mask_step[:, None]
                    step_off = block_inner_off + kc_step

                    src_k_ptr = (
                        k_ptr
                        + (curr_kv_pos + kc_step + offs_step[:, None]) * stride_k_tok
                        + head_pid * stride_k_head
                        + offs_d[None, :] * stride_k_dim
                    )
                    src_v_ptr = (
                        v_ptr
                        + (curr_kv_pos + kc_step + offs_step[:, None]) * stride_v_tok
                        + head_pid * stride_v_head
                        + offs_d[None, :] * stride_v_dim
                    )

                    k_val = tl.load(src_k_ptr, mask=mask_2d, other=0.0)
                    v_val = tl.load(src_v_ptr, mask=mask_2d, other=0.0)

                    dst_k_ptr = (
                        key_cache_ptr
                        + physical_block_id * stride_kc_blk
                        + head_pid * stride_kc_head
                        + (step_off + offs_step[:, None]) * stride_kc_tok
                        + offs_d[None, :] * stride_kc_dim
                    )
                    dst_v_ptr = (
                        value_cache_ptr
                        + physical_block_id * stride_vc_blk
                        + head_pid * stride_vc_head
                        + (step_off + offs_step[:, None]) * stride_vc_tok
                        + offs_d[None, :] * stride_vc_dim
                    )

                    tl.store(dst_k_ptr, k_val, mask=valid_block & mask_2d)
                    tl.store(dst_v_ptr, v_val, mask=valid_block & mask_2d)

                    kc_step += actual

                processed += sub_len
                curr_log_pos += sub_len
                curr_kv_pos += sub_len


def _compute_token_step(kv_heads, head_dim, block_size, element_size=2):
    """
    Compute TOKEN_STEP to stay within the UB (Unified Buffer) budget.

    Each step holds buffers for [TOKEN_STEP, head_dim] for both K and V,
    with intermediate buffers for load and store operations.
    """
    UB_AVAILABLE = 224 * 1024
    bytes_per_step_kv = 2 * head_dim * element_size
    max_step = max(1, UB_AVAILABLE // bytes_per_step_kv)
    return max(1, min(block_size, max_step))


def store_paged_kv_opt_impl(
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
    block_size = key_cache.shape[2]

    num_total_cores = get_num_cores("vector")
    TOKEN_STEP = _compute_token_step(num_kv_heads, head_dim, block_size)
    CHUNK_SIZE = block_size

    if is_decode:
        num_head_programs = num_kv_heads
        num_batch_programs = min(batch_size, num_total_cores // max(1, num_kv_heads))
        num_batch_programs = max(1, num_batch_programs)
        grid = (num_head_programs, num_batch_programs)

        _decode_store_kv_fast_path[grid](
            k_states,
            v_states,
            key_cache,
            value_cache,
            block_table,
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
            head_dim=head_dim,
            block_size=block_size,
            HAS_KV_LENS=kv_lens_before_store is not None,
            TOKEN_STEP=TOKEN_STEP,
        )
    else:
        num_cores_per_head = max(1, num_total_cores // max(1, num_kv_heads))
        grid = (num_kv_heads, num_cores_per_head)

        _prefill_store_kv_kernel[grid](
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
            head_dim=head_dim,
            block_size=block_size,
            CHUNK_SIZE=CHUNK_SIZE,
            HAS_KV_LENS=kv_lens_before_store is not None,
            TOKEN_STEP=TOKEN_STEP,
        )

    return key_cache, value_cache