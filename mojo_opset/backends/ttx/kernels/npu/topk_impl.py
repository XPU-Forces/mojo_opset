import torch
import triton
import triton.language as tl
import triton.language.core as core
from triton.runtime.libentry import libentry
from triton.language.standard import _log2, zeros_like


# ============================================================================
# Triton JIT helpers (bitonic argsort with packed keys)
# ============================================================================

@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]

    y = core.reshape(x, shape)
    y_idx = core.reshape(ids, shape)

    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(x.dtype)
    right = core.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(x.dtype)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)

    left_idx = core.broadcast_to(tl.max(y_idx * (1 - mask), 1)[:, None, :], shape).to(
        ids.dtype
    )
    right_idx = core.broadcast_to(tl.max(y_idx * mask, 1)[:, None, :], shape).to(
        ids.dtype
    )
    left_idx = core.reshape(left_idx, ids.shape)
    right_idx = core.reshape(right_idx, ids.shape)

    if core.constexpr(x.dtype.primitive_bitwidth) == 8:
        idtype = core.int8
    elif core.constexpr(x.dtype.primitive_bitwidth) == 16:
        idtype = core.int16
    elif core.constexpr(x.dtype.primitive_bitwidth) == 32:
        idtype = core.int32
    elif core.constexpr(x.dtype.primitive_bitwidth) == 64:
        idtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip
    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))

    if core.constexpr(ids.dtype.primitive_bitwidth) == 8:
        idx_dtype = core.int8
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 16:
        idx_dtype = core.int16
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 32:
        idx_dtype = core.int32
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 64:
        idx_dtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft_idx = left_idx.to(idx_dtype, bitcast=True)
    iright_idx = right_idx.to(idx_dtype, bitcast=True)
    ix_idx = ids.to(idx_dtype, bitcast=True)
    ret_idx = ix_idx ^ core.where(cond, ileft_idx ^ iright_idx, zeros_like(ix_idx))

    return ret.to(x.dtype, bitcast=True), ret_idx.to(ids.dtype, bitcast=True)


# We do not use triton_patch.language.standard sort_impl / topk here: that stack only supports signed-number
# sorting and only descending order; this file keeps a custom bitonic argsort with packed keys instead.
@triton.jit
def _bitonic_merge(
    x, ids, stage: core.constexpr, order: core.constexpr, n_dims: core.constexpr
):
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    if order == 2:
        shape: core.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x, ids, dim: tl.constexpr, descending: core.constexpr):
    _dim: core.constexpr = dim
    n_dims: core.constexpr = _log2(x.shape[_dim])
    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids


@triton.jit
def _float_to_sortable_i32(x):
    bits = x.to(tl.int32, bitcast=True)
    sign_bit = tl.full(bits.shape, -2147483648, tl.int32)
    return tl.where(bits < 0, (~bits) ^ sign_bit, bits)


@triton.jit
def _sortable_i32_to_float(sortable):
    sign_bit = tl.full(sortable.shape, -2147483648, tl.int32)
    bits = tl.where(sortable >= 0, sortable, ~(sortable ^ sign_bit))
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _pack_sort_keys(values, indices):
    max_u32 = tl.full(values.shape, 4294967295, tl.int64)
    sortable = _float_to_sortable_i32(values).to(tl.int64)
    tie = max_u32 - indices.to(tl.int64)
    return (sortable << 32) | tie


@triton.jit
def _unpack_sort_values(keys):
    sortable = (keys >> 32).to(tl.int32)
    return _sortable_i32_to_float(sortable)


@triton.jit
def _unpack_sort_indices(keys):
    max_u32 = tl.full(keys.shape, 4294967295, tl.int64)
    return (max_u32 - (keys & max_u32)).to(tl.int32)


# ============================================================================
# Triton kernels
# ============================================================================

@triton.jit
def _precompute_sortable_kernel(
    x_ptr, sortable_ptr, scratch_ptr,
    n_rows, n_cols, stride_xm, stride_sort,
    K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Convert float→sortable int32 + determine sign bit for radix select."""
    pid = tl.program_id(0)
    offs_bn = tl.arange(0, BLOCK_N)
    for row_id in range(pid, n_rows, tl.num_programs(0)):
        row_base = row_id * stride_xm
        sort_base = row_id * stride_sort
        count_nonneg = 0
        for block_start in range(0, n_cols, BLOCK_N):
            offs = block_start + offs_bn
            valid = offs < n_cols
            x = tl.load(x_ptr + row_base + offs, mask=valid, other=0.0)
            x_int = _float_to_sortable_i32(x)
            tl.store(sortable_ptr + sort_base + offs, x_int, mask=valid)
            count_nonneg = count_nonneg + tl.sum(
                (valid & (x_int >= 0)).to(tl.int32))
        sign_bit_val = tl.where(count_nonneg >= K, 0, -2147483648)
        tl.store(scratch_ptr + row_id, sign_bit_val)


@triton.jit
def _radix_select_kernel(
    sortable_ptr, scratch_ptr,
    n_rows, n_cols, stride_sort,
    K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """1-bit radix select (bits 30→0) to find the K-th largest pivot."""
    pid = tl.program_id(0)
    offs_bn = tl.arange(0, BLOCK_N)
    for row_id in range(pid, n_rows, tl.num_programs(0)):
        sort_base = row_id * stride_sort
        for bit in range(30, -1, -1):
            pivot = tl.load(scratch_ptr + row_id)
            trial = pivot | (1 << bit)
            count = 0
            for block_start in range(0, n_cols, BLOCK_N):
                offs = block_start + offs_bn
                valid = offs < n_cols
                s = tl.load(sortable_ptr + sort_base + offs,
                            mask=valid, other=-2147483648)
                count = count + tl.sum((s >= trial).to(tl.int32))
            new_pivot = tl.where(count >= K, trial, pivot)
            tl.store(scratch_ptr + row_id, new_pivot)


@triton.jit
def _gather_topk_kernel(
    sortable_ptr, out_val_ptr, out_idx_ptr, scratch_ptr,
    n_rows, n_cols, stride_sort, stride_out,
    K: tl.constexpr, K_PAD: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Gather first K elements >= pivot, reconstruct float, then fused bitonic sort."""
    pid = tl.program_id(0)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, K_PAD)
    k_mask = offs_k < K
    for row_id in range(pid, n_rows, tl.num_programs(0)):
        sort_base = row_id * stride_sort
        out_base = row_id * stride_out
        pivot = tl.load(scratch_ptr + row_id)
        write_pos = 0
        for block_start in range(0, n_cols, BLOCK_N):
            offs = block_start + offs_bn
            valid = offs < n_cols
            s = tl.load(sortable_ptr + sort_base + offs,
                        mask=valid, other=-2147483648)
            keep = s >= pivot
            local_n = tl.sum(keep.to(tl.int32))
            if local_n > 0:
                prefix = tl.cumsum(keep.to(tl.int32), axis=0)
                dst = write_pos + prefix - 1
                ok = keep & (dst >= 0) & (dst < K)
                x = _sortable_i32_to_float(s)
                tl.store(out_val_ptr + out_base + dst, x, mask=ok)
                tl.store(out_idx_ptr + out_base + dst, offs.to(tl.int32),
                         mask=ok)
            write_pos = write_pos + local_n
        vals = tl.load(out_val_ptr + out_base + offs_k,
                       mask=k_mask, other=float("-inf"))
        idxs = tl.load(out_idx_ptr + out_base + offs_k,
                       mask=k_mask, other=0).to(tl.int32)
        keys = _pack_sort_keys(vals, idxs)
        sorted_keys, _ = argsort(keys, tl.zeros_like(keys), 0, descending=1)
        sorted_vals = _unpack_sort_values(sorted_keys)
        sorted_idxs = _unpack_sort_indices(sorted_keys)
        tl.store(out_val_ptr + out_base + offs_k, sorted_vals, mask=k_mask)
        tl.store(out_idx_ptr + out_base + offs_k,
                 sorted_idxs.to(tl.int32), mask=k_mask)


@libentry()
@triton.jit
def _topk_softmax_kernel(
    input_ptr, output_ptr,
    TOTAL_TASKS,
    K: tl.constexpr, STRIDE_B: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offs = tl.arange(0, K)
    for task_id in range(pid, TOTAL_TASKS, num_programs):
        base = task_id * STRIDE_B
        x = tl.load(input_ptr + base + offs)
        x_max = tl.max(x, 0)
        exp_x = tl.exp(x - x_max)
        probs = exp_x / tl.sum(exp_x, 0)
        tl.store(output_ptr + base + offs, probs)


def _get_npu_vectorcore_count() -> int:
    return triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]


def top_k_impl(
    logits: torch.FloatTensor,
    top_k: int = 50,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
    largest: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Radix-select top-k with fused softmax on NPU.

    Algorithm: O(n*31) radix select + O(k log²k) bitonic sort + softmax.

    Returns:
        (softmax_probs, topk_indices) each of shape (batch, K).
    """
    del filter_value
    device = logits.device
    logits = logits.to(torch.float32)

    vocab_size = logits.size(-1)
    batch_size = logits.numel() // vocab_size
    logits_2d = logits.reshape(batch_size, vocab_size).contiguous()

    top_k = min(top_k, vocab_size)
    top_k = max(top_k, min_tokens_to_keep)

    working_logits = logits_2d if largest else (-logits_2d)

    k_pad = triton.next_power_of_2(top_k)
    precomp_block_n = min(4096, triton.next_power_of_2(vocab_size))
    radix_block_n = min(8192, triton.next_power_of_2(vocab_size))
    gather_block_n = min(256, triton.next_power_of_2(vocab_size))

    topk_vals = torch.full(
        (batch_size, top_k), float("-inf"), device=device, dtype=torch.float32,
    )
    topk_idxs = torch.zeros(
        (batch_size, top_k), device=device, dtype=torch.int32,
    )
    sortable_buf = torch.empty(
        (batch_size, vocab_size), device=device, dtype=torch.int32,
    )
    scratch = torch.zeros(batch_size, device=device, dtype=torch.int32)

    num_programs = _get_npu_vectorcore_count()
    grid = (min(batch_size, num_programs),)

    _precompute_sortable_kernel[grid](
        working_logits, sortable_buf, scratch,
        batch_size, vocab_size,
        working_logits.stride(0), sortable_buf.stride(0),
        K=top_k, BLOCK_N=precomp_block_n,
    )

    _radix_select_kernel[grid](
        sortable_buf, scratch,
        batch_size, vocab_size,
        sortable_buf.stride(0),
        K=top_k, BLOCK_N=radix_block_n,
    )

    _gather_topk_kernel[grid](
        sortable_buf,
        topk_vals, topk_idxs, scratch,
        batch_size, vocab_size,
        sortable_buf.stride(0), topk_vals.stride(0),
        K=top_k, K_PAD=k_pad, BLOCK_N=gather_block_n,
    )

    if not largest:
        topk_vals = -topk_vals

    _topk_softmax_kernel[(min(batch_size, num_programs),)](
        topk_vals, topk_vals,
        batch_size,
        K=top_k, STRIDE_B=topk_vals.stride(0),
    )

    return topk_vals, topk_idxs