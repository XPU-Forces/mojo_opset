import triton
import triton.language as tl


@triton.jit
def _store_paged_kv_kernel(
    K_States,
    V_States,
    K_Cache,
    V_Cache,
    Block_Tables,
    Seq_Lens,
    stride_ks_b,
    stride_ks_h,
    stride_ks_l,
    stride_ks_d,
    stride_vs_b,
    stride_vs_h,
    stride_vs_l,
    stride_vs_d,
    stride_kc_blk,
    stride_kc_h,
    stride_kc_s,
    stride_kc_d,
    stride_vc_blk,
    stride_vc_h,
    stride_vc_s,
    stride_vc_d,
    stride_bt_b,
    stride_bt_s,
    NEW_SEQ_LEN,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_chunk = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)

    start_token_idx = pid_chunk * BLOCK_SIZE
    if start_token_idx >= NEW_SEQ_LEN:
        return

    offs_token_in_chunk = tl.arange(0, BLOCK_SIZE)
    offs_token_global = start_token_idx + offs_token_in_chunk
    mask_token = offs_token_global < NEW_SEQ_LEN

    context_len = tl.load(Seq_Lens + pid_batch)
    logical_pos_vector = context_len + offs_token_global

    logical_block_idx_vector = logical_pos_vector // BLOCK_SIZE
    offset_in_block_vector = logical_pos_vector % BLOCK_SIZE

    ptr_block_table = Block_Tables + pid_batch * stride_bt_b + logical_block_idx_vector * stride_bt_s
    physical_block_ids = tl.load(ptr_block_table, mask=mask_token, other=0)

    offs_d = tl.arange(0, HEAD_DIM)

    ptr_k_in = (
        K_States
        + pid_batch * stride_ks_b
        + pid_head * stride_ks_h
        + offs_token_global[:, None] * stride_ks_l
        + offs_d[None, :] * stride_ks_d
    )
    ptr_v_in = (
        V_States
        + pid_batch * stride_vs_b
        + pid_head * stride_vs_h
        + offs_token_global[:, None] * stride_vs_l
        + offs_d[None, :] * stride_vs_d
    )

    k_data = tl.load(ptr_k_in, mask=mask_token[:, None], other=0.0)
    v_data = tl.load(ptr_v_in, mask=mask_token[:, None], other=0.0)

    ptr_k_out = (
        K_Cache
        + physical_block_ids[:, None] * stride_kc_blk
        + pid_head * stride_kc_h
        + offset_in_block_vector[:, None] * stride_kc_s
        + offs_d[None, :] * stride_kc_d
    )
    ptr_v_out = (
        V_Cache
        + physical_block_ids[:, None] * stride_vc_blk
        + pid_head * stride_vc_h
        + offset_in_block_vector[:, None] * stride_vc_s
        + offs_d[None, :] * stride_vc_d
    )

    tl.store(ptr_k_out, k_data, mask=mask_token[:, None])
    tl.store(ptr_v_out, v_data, mask=mask_token[:, None])


def store_paged_kv_impl(key_states, value_states, k_cache, v_cache, block_tables, context_lens, block_size):
    batch_size, num_heads, new_seq_len, head_dim = key_states.shape
    grid = ((new_seq_len + block_size - 1) // block_size, batch_size, num_heads)

    _store_paged_kv_kernel[grid](
        key_states,
        value_states,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        key_states.stride(0),
        key_states.stride(1),
        key_states.stride(2),
        key_states.stride(3),
        value_states.stride(0),
        value_states.stride(1),
        value_states.stride(2),
        value_states.stride(3),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        new_seq_len,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
    )
