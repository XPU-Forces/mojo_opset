import torch
import triton
import triton.language as tl


@triton.jit
def _store_label_cache_triton_kernel(
    label_cache_ptr,
    key_lr_ptr,
    block_idx_list_ptr,
    token_idx_list_ptr,
    head_num: tl.constexpr,
    head_dim: tl.constexpr,
    token_num: tl.constexpr,
    l_stride_b: tl.constexpr,
    l_stride_h: tl.constexpr,
    l_stride_t: tl.constexpr,
    l_stride_d: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    BATCH_BLOCK_NUM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    b_start = pid_b * BATCH_BLOCK_NUM
    b_end = tl.minimum(b_start + BATCH_BLOCK_NUM, token_num)
    b = tl.arange(0, BATCH_BLOCK_NUM) + b_start
    b_3d = b[:, None, None]
    h = tl.arange(0, head_num)
    h_3d = h[None, :, None]
    d = tl.arange(0, head_dim)
    d_3d = d[None, None, :]
    block_idx = tl.load(block_idx_list_ptr + b_3d, mask=(b_3d < b_end), other=0)
    token_idx = tl.load(token_idx_list_ptr + b_3d, mask=(b_3d < b_end), other=0)

    label_cache_addr = block_idx * l_stride_b + h_3d * l_stride_h + token_idx * l_stride_t + d_3d * l_stride_d
    key_lr_offset = b_3d * k_stride_s + h_3d * k_stride_h + d_3d * k_stride_d

    valid_mask = (b_3d < b_end) & (h_3d < head_num) & (d_3d < head_dim)

    key_lr_data = tl.load(key_lr_ptr + key_lr_offset, mask=valid_mask, other=0.0)
    tl.store(label_cache_ptr + label_cache_addr, key_lr_data, mask=valid_mask)


def store_label_cache_fwd_impl(
    label_cache: torch.Tensor,
    key_lr: torch.Tensor,
    block_idxs: torch.Tensor,
    token_idxs: torch.Tensor,
    token_num: torch.Tensor,
):
    """
    store label cache for each token

    Args:
        label_cache: A tensor with the shape of [block_num, head_num, block_size, head_dim], which is used
        to store the low-rank key cache.
        key_lr: A tensor with the shape of [seqlen, head_num, head_dim], which represents the low rank key
        cache to be stored.
        block_idxs: A tensor with the shape of [token_num,], which represents the block index of each token
        to be stored.
        token_idxs: A tensor with the shape of [token_num,], which represents the token index of each token
        to be stored.
        token_num: A scalar tensor, which represents the number of tokens to be stored.

    Returns: label cache with shape of [block_num, head_num, block_size, head_dim]
    Notes:
        - This function will store the low-rank key cache for each token in the label_cache.
    """
    block_num, head_num, block_size, head_dim = label_cache.shape
    assert label_cache.dtype == key_lr.dtype, "label_cache and key_lr must have the same dtype"

    ub_buffer = 192

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    BATCH_BLOCK_NUM = (token_num - 1 + num_programs) // num_programs

    batch_data_size = BATCH_BLOCK_NUM * head_num * head_dim * key_lr.element_size()
    if batch_data_size > ub_buffer:
        BATCH_BLOCK_NUM = 16
        num_programs = (token_num - 1 + BATCH_BLOCK_NUM) // BATCH_BLOCK_NUM
    BATCH_BLOCK_NUM = max(16, BATCH_BLOCK_NUM)
    grid = (num_programs,)

    _store_label_cache_triton_kernel[grid](
        label_cache_ptr=label_cache,
        key_lr_ptr=key_lr,
        block_idx_list_ptr=block_idxs,
        token_idx_list_ptr=token_idxs,
        head_num=head_num,
        head_dim=head_dim,
        token_num=token_num,
        l_stride_b=label_cache.stride(0),
        l_stride_h=label_cache.stride(1),
        l_stride_t=label_cache.stride(2),
        l_stride_d=label_cache.stride(3),
        k_stride_s=key_lr.stride(0),
        k_stride_h=key_lr.stride(1),
        k_stride_d=key_lr.stride(2),
        BATCH_BLOCK_NUM=BATCH_BLOCK_NUM,
    )
    return label_cache
