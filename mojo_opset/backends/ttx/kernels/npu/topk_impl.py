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

@libentry()
@triton.jit
def _topk_stage1_kernel(
    y_ptr,
    y_index_ptr,
    x_ptr,
    filter_value: tl.constexpr,
    k: tl.constexpr,
    TOTAL_TASKS,
    VOCAB_SIZE: tl.constexpr,
    CHUNK_NUM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)

    for task_id in range(pid, TOTAL_TASKS, grid_size):
        cur_batch = task_id // CHUNK_NUM
        cur_chunk_idx = task_id % CHUNK_NUM
        chunk_offset = cur_chunk_idx * CHUNK_SIZE
        off_col = chunk_offset + tl.arange(0, CHUNK_SIZE)
        row_start = cur_batch * VOCAB_SIZE
        safe_off_x = tl.where(off_col < VOCAB_SIZE, row_start + off_col, row_start)
        batch_y_ptr = y_ptr + cur_batch * ROW_STRIDE + cur_chunk_idx * CHUNK_SIZE
        batch_y_index_ptr = y_index_ptr + cur_batch * ROW_STRIDE + cur_chunk_idx * CHUNK_SIZE

        mask_x = off_col < VOCAB_SIZE
        pad_value = filter_value if DESCENDING else -filter_value
        x = tl.load(x_ptr + safe_off_x, mask=mask_x, other=pad_value)
        x = tl.where(mask_x, x, pad_value)
        x_index = tl.where(mask_x, off_col, 0).to(tl.int32)
        sort_keys = _pack_sort_keys(x, x_index)
        sorted_keys, _ = argsort(sort_keys, tl.zeros_like(sort_keys), 0, descending=DESCENDING)
        sorted_x = _unpack_sort_values(sorted_keys)
        sorted_index = _unpack_sort_indices(sorted_keys)

        cols = tl.arange(0, CHUNK_SIZE)
        tl.store(batch_y_ptr + cols, sorted_x)
        tl.store(batch_y_index_ptr + cols, sorted_index.to(tl.int32))


@libentry()
@triton.jit
def _topk_merge_kernel(
    y_ptr,
    y_index_ptr,
    x_ptr,
    x_index_ptr,
    filter_value: tl.constexpr,
    k: tl.constexpr,
    INPUT_ELEMS: tl.constexpr,
    INPUT_ROW_STRIDE: tl.constexpr,
    OUTPUT_ROW_STRIDE: tl.constexpr,
    NEXT_GROUPS: tl.constexpr,
    GROUP_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_batch = pid // NEXT_GROUPS
    cur_group = pid % NEXT_GROUPS

    group_start = cur_group * GROUP_CHUNKS * k
    off_col = tl.arange(0, BLOCK_SIZE)
    valid = (off_col < GROUP_CHUNKS * k) & ((group_start + off_col) < INPUT_ELEMS)

    in_row_start = cur_batch * INPUT_ROW_STRIDE
    safe_off_x = tl.where(valid, in_row_start + group_start + off_col, in_row_start)

    pad_value = filter_value if DESCENDING else -filter_value
    chunk_x = tl.load(x_ptr + safe_off_x, mask=valid, other=pad_value)
    chunk_x = tl.where(valid, chunk_x, pad_value)
    chunk_index = tl.load(x_index_ptr + safe_off_x, mask=valid, other=0).to(tl.int32)
    chunk_index = tl.where(valid, chunk_index, 0)

    sort_keys = _pack_sort_keys(chunk_x, chunk_index)
    sorted_keys, _ = argsort(sort_keys, tl.zeros_like(sort_keys), 0, descending=DESCENDING)
    sorted_logits = _unpack_sort_values(sorted_keys)
    sorted_index = _unpack_sort_indices(sorted_keys)

    out_row_start = cur_batch * OUTPUT_ROW_STRIDE + cur_group * BLOCK_SIZE
    tl.store(y_ptr + out_row_start + off_col, sorted_logits)
    tl.store(y_index_ptr + out_row_start + off_col, sorted_index.to(tl.int32))


# TODO: triton softmax kernel hangs on NPU, needs investigation.
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


# ============================================================================
# Torch helper
# ============================================================================

def _compact_sorted_blocks(
    sorted_vals: torch.Tensor,
    sorted_idx: torch.Tensor,
    batch_size: int,
    block_count: int,
    block_size: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    compact_vals = (
        sorted_vals.view(batch_size, block_count, block_size)[:, :, :top_k]
        .reshape(batch_size, block_count * top_k)
        .contiguous()
    )
    compact_idx = (
        sorted_idx.view(batch_size, block_count, block_size)[:, :, :top_k]
        .reshape(batch_size, block_count * top_k)
        .contiguous()
    )
    return compact_vals, compact_idx


# ============================================================================
# Public entry point
# ============================================================================

def top_k_impl(
    logits: torch.FloatTensor,
    top_k: int = 50,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
    largest: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k selection with fused softmax on NPU.

    Returns:
        (softmax_probs, topk_indices) each of shape (batch, K).
    """
    device = logits.device
    logits = logits.to(torch.float32)

    vocab_size = logits.size(-1)
    batch_size = logits.numel() // vocab_size

    logits_2d = logits.reshape(batch_size, vocab_size).contiguous()

    top_k = min(top_k, vocab_size)
    top_k = max(top_k, min_tokens_to_keep)

    descending = 1 if largest else 0

    chunk_size = 128
    if chunk_size < top_k:
        chunk_size = triton.next_power_of_2(top_k)
    chunk_num = triton.cdiv(vocab_size, chunk_size)

    stage1_elem_cnt = chunk_num * top_k
    row_stride = stage1_elem_cnt
    stage1_sorted_row_stride = chunk_num * chunk_size

    pad_val = filter_value if descending else -filter_value

    stage1_sorted = torch.full((batch_size, stage1_sorted_row_stride), pad_val, device=device, dtype=logits.dtype).contiguous()
    stage1_sorted_index = torch.zeros((batch_size, stage1_sorted_row_stride), device=device, dtype=torch.int32).contiguous()

    stage1_total_tasks = batch_size * chunk_num
    _topk_stage1_kernel[(min(stage1_total_tasks, 65535),)](
        stage1_sorted,
        stage1_sorted_index,
        logits_2d,
        filter_value,
        top_k,
        stage1_total_tasks,
        vocab_size,
        chunk_num,
        chunk_size,
        stage1_sorted_row_stride,
        descending,
    )

    stage1_out, stage1_out_index = _compact_sorted_blocks(
        stage1_sorted,
        stage1_sorted_index,
        batch_size,
        chunk_num,
        chunk_size,
        top_k,
    )

    candidate_vals = stage1_out
    candidate_idx = stage1_out_index
    current_groups = chunk_num
    current_row_stride = row_stride

    max_merge_candidates = 128
    merge_group_chunks = max(1, max_merge_candidates // top_k)
    while merge_group_chunks > 1 and (merge_group_chunks * top_k) > max_merge_candidates:
        merge_group_chunks //= 2
    if merge_group_chunks < 2:
        merge_group_chunks = 2

    while current_groups > 1:
        group_chunks = min(merge_group_chunks, current_groups)
        next_groups = triton.cdiv(current_groups, group_chunks)
        next_row_stride = next_groups * top_k
        block_size = triton.next_power_of_2(group_chunks * top_k)
        next_sorted_row_stride = next_groups * block_size

        next_sorted_vals = torch.full((batch_size, next_sorted_row_stride), pad_val, device=device, dtype=logits.dtype).contiguous()
        next_sorted_idx = torch.zeros((batch_size, next_sorted_row_stride), device=device, dtype=torch.int32).contiguous()
        merge_total_tasks = batch_size * next_groups

        _topk_merge_kernel[(merge_total_tasks,)](
            next_sorted_vals,
            next_sorted_idx,
            candidate_vals,
            candidate_idx,
            filter_value,
            top_k,
            current_groups * top_k,
            current_row_stride,
            next_sorted_row_stride,
            next_groups,
            group_chunks,
            block_size,
            descending,
        )

        next_vals, next_idx = _compact_sorted_blocks(
            next_sorted_vals,
            next_sorted_idx,
            batch_size,
            next_groups,
            block_size,
            top_k,
        )

        candidate_vals = next_vals
        candidate_idx = next_idx
        current_groups = next_groups
        current_row_stride = next_row_stride

    final_candidate_vals = candidate_vals[:, :top_k].contiguous()
    final_candidate_idx = candidate_idx[:, :top_k].contiguous()
    
    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    _topk_softmax_kernel[(min(batch_size, num_programs),)](
        final_candidate_vals, final_candidate_vals,
        batch_size,
        K=top_k, STRIDE_B=final_candidate_vals.stride(0),
    )

    # final_candidate_vals = torch.nn.functional.softmax(final_candidate_vals, dim=-1)

    return final_candidate_vals, final_candidate_idx


# ============================================================================
# Streaming implementation (commented out, pending fixes)
# ============================================================================

# TODO: Streaming version still has some issues, need to be fixed and re-enabled.

# triton_patch_topk = importlib.import_module("triton.triton_patch.language.standard").topk
# triton_patch_sort = importlib.import_module("triton.triton_patch.language.standard").sort_impl

# @triton.jit
# def _float_to_sortable_int(x):
#     if x.dtype == tl.float32:
#         bits = x.to(tl.int32, bitcast=True)
#         sign_bit = tl.full(bits.shape, -0x80000000, tl.int32)
#         return tl.where(bits < 0, (~bits) ^ sign_bit, bits)
#     else:  # float16
#         bits = x.to(tl.int16, bitcast=True)
#         sign_bit = tl.full(bits.shape, -0x8000, tl.int16)
#         return tl.where(bits < 0, (~bits) ^ sign_bit, bits)


# @triton.jit
# def _sortable_int_to_float(sortable, float_dtype):
#     if float_dtype == tl.float32:
#         sign_bit = tl.full(sortable.shape, -0x80000000, tl.int32)
#         bits = tl.where(sortable >= 0, sortable, ~(sortable ^ sign_bit))
#         return bits.to(tl.float32, bitcast=True)
#     else:  # float16
#         sign_bit = tl.full(sortable.shape, -0x8000, tl.int16)
#         bits = tl.where(sortable >= 0, sortable, ~(sortable ^ sign_bit))
#         return bits.to(tl.float16, bitcast=True)

# @triton.jit
# def _pack_sort_keys(values, indices):
#     if values.dtype == tl.float16:
#         sortable = _float_to_sortable_int(values).to(tl.int32)
#         if indices.dtype == tl.int16:
#             tie = tl.full(indices.shape, 0xFFFF, tl.int32) - indices.to(tl.int32)
#             return (sortable << 16) | tie
#         elif indices.dtype == tl.int8:
#             tie = tl.full(indices.shape, 0xFF, tl.int32) - indices.to(tl.int32)
#             return (sortable << 8) | tie
#         else:
#             sortable = sortable.to(tl.int64)
#             tie = tl.full(indices.shape, 0xFFFFFFFF, tl.int64) - indices.to(tl.int64)
#             return (sortable << 32) | tie
#     else:
#         sortable = _float_to_sortable_int(values).to(tl.int64)
#         if indices.dtype == tl.int32:
#             tie = tl.full(indices.shape, 0xFFFFFFFF, tl.int64) - indices.to(tl.int64)
#             return (sortable << 32) | tie
#         elif indices.dtype == tl.int16:
#             tie = tl.full(indices.shape, 0xFFFF, tl.int64) - indices.to(tl.int64)
#             return (sortable << 32) | tie
#         else:
#             tie = tl.full(indices.shape, 0xFF, tl.int64) - indices.to(tl.int64)
#             return (sortable << 32) | tie

# @triton.jit
# def _unpack_sort_values(keys, float_dtype, idx_dtype):
#     if float_dtype == tl.float16:
#         if idx_dtype == tl.int16:
#             sortable = (keys >> 16).to(tl.int16)
#         elif idx_dtype == tl.int8:
#             sortable = (keys >> 8).to(tl.int16)
#         else:
#             shift = 32
#             sortable = (keys >> shift).to(tl.int16)
#     else:
#         shift = 32
#         sortable = (keys >> shift).to(tl.int32)
#
#     return _sortable_int_to_float(sortable, float_dtype)

# @triton.jit
# def _unpack_sort_indices(keys, idx_dtype):
#     if keys.dtype == tl.int32:
#         if idx_dtype == tl.int16:
#             max_tensor = tl.full(keys.shape, 0xFFFF, tl.int32)
#         else:
#             max_tensor = tl.full(keys.shape, 0xFF, tl.int32)
#         return (max_tensor - (keys & max_tensor)).to(idx_dtype)
#     else:
#         if idx_dtype == tl.int32:
#             max_tensor = tl.full(keys.shape, 0xFFFFFFFF, tl.int64)
#         elif idx_dtype == tl.int16:
#             max_tensor = tl.full(keys.shape, 0xFFFF, tl.int64)
#         else:
#             max_tensor = tl.full(keys.shape, 0xFF, tl.int64)
#         return (max_tensor - (keys & max_tensor)).to(idx_dtype)

# @triton.jit
# def _streaming_topk_kernel(
#     x_ptr,
#     y_val_ptr,
#     y_idx_ptr,
#     n_rows,
#     n_cols,
#     stride_xm,
#     stride_ym,
#     K_PAD: tl.constexpr,
#     N_COLS_PAD: tl.constexpr,
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
# ):
#     pid = tl.program_id(0)
#     num_programs = tl.num_programs(0)

#     offs_bn = tl.arange(0, BLOCK_N)
#     offs_k = tl.arange(0, K_PAD)

#     total_tasks = tl.cdiv(n_rows, BLOCK_M)

#     for task_id in range(pid, total_tasks, num_programs):
#         offs_m = task_id * BLOCK_M + tl.arange(0, BLOCK_M)
#         mask_m = offs_m[:, None] < n_rows

#         loop_iterations: tl.constexpr = N_COLS_PAD // BLOCK_N - 1
#         offs_n = loop_iterations * BLOCK_N + offs_bn
#         mask_n = offs_n[None, :] < n_cols

#         x = tl.load(
#             x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :],
#             mask=mask_m & mask_n,
#             other=float("-inf"),
#         )
#         idx = tl.broadcast_to(offs_n[None, :], x.shape).to(y_idx_ptr.type.element_ty)
#         keys = _pack_sort_keys(x, idx)
#         acc = triton_patch_topk(keys, K_PAD, dim=1)
#         for _ in (tl.static_range if loop_iterations <= 4 else range)(loop_iterations):
#             pass
#             acc = triton_patch_sort(acc, dim=1, descending=False)
#             offs_n -= BLOCK_N
#             mask_n = offs_n[None, :] < n_cols
#             x = tl.load(
#                 x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :],
#                 mask=mask_m & mask_n,
#                 other=float("-inf"),
#             )
#             idx = tl.broadcast_to(offs_n[None, :], x.shape).to(y_idx_ptr.type.element_ty)
#             keys = _pack_sort_keys(x, idx)
#             cand = triton_patch_topk(keys, K_PAD, dim=1)
#             acc = tl.maximum(acc, cand)

#         acc = triton_patch_topk(acc, K_PAD, dim=1)

#         y_vals = _unpack_sort_values(acc, x_ptr.type.element_ty, y_idx_ptr.type.element_ty)
#         y_indices = _unpack_sort_indices(acc, y_idx_ptr.type.element_ty)
#
#         tl.store(
#             y_val_ptr + offs_m[:, None] * stride_ym + offs_k[None, :],
#             y_vals,
#             mask=(offs_m[:, None] < n_rows) & (offs_k[None, :] < K_PAD),
#         )
#         tl.store(
#             y_idx_ptr + offs_m[:, None] * stride_ym + offs_k[None, :],
#             y_indices,
#             mask=(offs_m[:, None] < n_rows) & (offs_k[None, :] < K_PAD),
#         )

# def top_k_impl_streaming(
#     logits: torch.Tensor,
#     top_k: int = 50,
#     filter_value: float = -float("Inf"),
#     min_tokens_to_keep: int = 1,
#     largest = True
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     device = logits.device
#
#     vocab_size = logits.size(-1)
#     batch_size = logits.numel() // vocab_size
#     logits_2d = logits.reshape(batch_size, vocab_size).contiguous()
#
#     top_k = min(top_k, vocab_size)
#     top_k = max(top_k, min_tokens_to_keep)
#
#     working_logits = logits_2d if largest else (-logits_2d)
#
#     k_pad = triton.next_power_of_2(top_k)
#     n_cols_pad = triton.next_power_of_2(vocab_size)
#
#     block_m = 1
#     block_n = min(256, n_cols_pad)
#     while n_cols_pad % block_n != 0:
#         block_n //= 2
#
#     if vocab_size <= 128:
#         idx_dtype = torch.int8
#     elif vocab_size <= 32768:
#         idx_dtype = torch.int16
#     elif vocab_size <= 2147483648:
#         idx_dtype = torch.int32
#     else:
#         idx_dtype = torch.int64
#
#     topk_vals_pad = torch.empty((batch_size, k_pad), device=device, dtype=torch.float32)
#     topk_idx_pad = torch.empty((batch_size, k_pad), device=device, dtype=idx_dtype)
#
#     num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
#
#     _streaming_topk_kernel[(num_programs,)](
#         working_logits,
#         topk_vals_pad,
#         topk_idx_pad,
#         batch_size,
#         vocab_size,
#         working_logits.stride(0),
#         topk_vals_pad.stride(0),
#         K_PAD=k_pad,
#         N_COLS_PAD=n_cols_pad,
#         BLOCK_M=block_m,
#         BLOCK_N=block_n,
#     )
#
#     final_candidate_vals = topk_vals_pad[:, :top_k].contiguous()
#     final_candidate_idx = topk_idx_pad[:, :top_k].contiguous()
#     if not largest:
#         final_candidate_vals = -final_candidate_vals
#
#     # Triton fused softmax (replaces torch.softmax)
#     _topk_softmax_kernel[(min(batch_size, num_programs),)](
#         final_candidate_vals, final_candidate_vals,
#         batch_size,
#         K=top_k, STRIDE_B=final_candidate_vals.stride(0),
#     )
#     final_probs_dist = final_candidate_vals
#
#     return final_probs_dist, final_candidate_idx

# def top_k_sampling_impl_streaming(
#     logits: torch.Tensor,
#     top_k: int = 50,
#     filter_value: float = -float("Inf"),
#     min_tokens_to_keep: int = 1,
#     largest = True
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     probs, indices = top_k_impl_streaming(
#         logits=logits,
#         top_k=top_k,
#         filter_value=filter_value,
#         min_tokens_to_keep=min_tokens_to_keep,
#         largest=largest
#     )
#     select_index = torch.multinomial(probs, num_samples=1)
#     next_token = torch.gather(indices, dim=-1, index=select_index)
#     next_prob = torch.gather(probs, dim=-1, index=select_index)
#     return next_prob, next_token
