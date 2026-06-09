import torch
import triton
import triton.language as tl

from triton.runtime.libentry import libentry

from mojo_opset.backends.ttx.kernels.npu.utils import VEC_ALIGN_BYTES
from mojo_opset.backends.ttx.kernels.utils import align
from mojo_opset.backends.ttx.kernels.npu.rmsnorm import COL_BLOCKING_THRESHOLD
from mojo_opset.backends.ttx.kernels.npu.rmsnorm import rms_norm_fwd_heuristics
from mojo_opset.backends.ttx.kernels.npu.rmsnorm import rmsnorm_infer_impl


@libentry()
@triton.jit
def _group_rmsnorm_interleaved_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    head_to_group_ptr,
    stride_x_row,
    stride_y_row,
    stride_w_group,
    n_rows,
    n_cols: tl.constexpr,
    total_heads: tl.constexpr,
    eps: tl.constexpr,
    G: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offsets < n_cols

    w0 = tl.load(W_ptr + 0 * stride_w_group + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    w1 = tl.load(W_ptr + 1 * stride_w_group + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    if G > 2:
        w2 = tl.load(W_ptr + 2 * stride_w_group + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    if G > 3:
        w3 = tl.load(W_ptr + 3 * stride_w_group + col_offsets, mask=col_mask, other=0.0).to(tl.float32)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    # first_row_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # first_row_mask = first_row_offsets < n_rows
    # first_x_ptrs = X_ptr + (first_row_offsets[:, None] * stride_x_row + col_offsets[None, :])
    # x_cur = tl.load(first_x_ptrs, mask=first_row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start = row_task_id * BLOCK_SIZE_M

        current_row_offsets = block_start + tl.arange(0, BLOCK_SIZE_M)
        row_mask = current_row_offsets < n_rows
        x_ptrs = X_ptr + (current_row_offsets[:, None] * stride_x_row + col_offsets[None, :])
        x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)

        head_ids = current_row_offsets % total_heads
        group_ids = tl.load(head_to_group_ptr + head_ids)

        if G == 2:
            w_selected = tl.where((group_ids == 0)[:, None], w0[None, :], w1[None, :])
        elif G == 3:
            w_selected = tl.where((group_ids == 0)[:, None], w0[None, :],
                         tl.where((group_ids == 1)[:, None], w1[None, :], w2[None, :]))
        else:
            w_selected = tl.where((group_ids == 0)[:, None], w0[None, :],
                         tl.where((group_ids == 1)[:, None], w1[None, :],
                         tl.where((group_ids == 2)[:, None], w2[None, :], w3[None, :])))

        ss_acc = tl.sum(x * x, axis=1)
        ss_acc = tl.where(row_mask, ss_acc, 0)
        mean_square = ss_acc / n_cols
        rrms = tl.rsqrt(mean_square + eps)
        rrms = tl.where(row_mask, rrms, 0.0)

        y_ptrs = Y_ptr + (current_row_offsets[:, None] * stride_y_row + col_offsets[None, :])
        y = x * rrms[:, None] * w_selected

        tl.store(
            y_ptrs,
            y.to(Y_ptr.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )


_group_rmsnorm_cache = {}


def _try_get_original_tensor(input_groups):
    G = len(input_groups)
    if G == 0:
        return None

    first = input_groups[0]
    if first.ndim != 3:
        return None

    bsz = first.shape[0]
    N = first.shape[-1]
    total_heads = sum(xg.shape[1] for xg in input_groups)

    if first.stride(2) != 1:
        return None
    if first.stride(1) != N:
        return None
    if first.stride(0) != total_heads * N:
        return None

    storage = first.untyped_storage()
    storage_offset = first.storage_offset()
    if storage_offset != 0:
        return None

    expected_offset = 0
    for xg in input_groups:
        if xg.untyped_storage().data_ptr() != storage.data_ptr():
            return None
        if xg.storage_offset() != expected_offset:
            return None
        if xg.shape[0] != bsz or xg.shape[-1] != N:
            return None
        if xg.stride(0) != total_heads * N or xg.stride(1) != N or xg.stride(2) != 1:
            return None
        expected_offset += xg.shape[1] * N

    return torch.empty(0, dtype=first.dtype, device=first.device).set_(
        storage,
        storage_offset=0,
        size=(bsz, total_heads, N),
        stride=(total_heads * N, N, 1),
    )


def _get_head_to_group(group_dims, device):
    key = tuple(group_dims)
    if key in _group_rmsnorm_cache:
        cached_device, cached_tensor = _group_rmsnorm_cache[key]
        if cached_device == device:
            return cached_tensor

    total_heads = sum(group_dims)
    head_to_group = torch.zeros(total_heads, dtype=torch.int32, device="cpu")
    offset = 0
    for g, gd in enumerate(group_dims):
        head_to_group[offset:offset + gd] = g
        offset += gd

    head_to_group = head_to_group.to(device)
    _group_rmsnorm_cache[key] = (device, head_to_group)
    return head_to_group


def group_rmsnorm_impl(
    input_groups,
    weight=None,
    eps=1e-6,
    output_like_input_stride=True,
) -> list[torch.Tensor]:
    assert isinstance(input_groups, (list, tuple))
    assert len(input_groups) > 0

    G = len(input_groups)
    N = input_groups[0].shape[-1]

    if weight is not None:
        assert weight.shape == (G, N), (
            f"weight shape must be ({G}, {N}), got {weight.shape}"
        )
        assert weight.is_contiguous(), (
            f"weight must be contiguous, got stride={weight.stride()}"
        )

    if N == 0:
        output_groups = []
        for g in range(G):
            xg = input_groups[g]
            if output_like_input_stride:
                yg = torch.empty_strided(
                    size=xg.shape,
                    stride=xg.stride(),
                    dtype=xg.dtype,
                    device=xg.device,
                )
            else:
                yg = torch.empty_like(xg, memory_format=torch.contiguous_format)
            output_groups.append(yg)
        return output_groups

    for g in range(G):
        xg = input_groups[g]
        assert xg.ndim == 3, f"group {g} input must be [token, num_head, norm_size]"
        assert xg.shape[-1] == N, (
            f"group {g} last dim mismatch: {xg.shape[-1]} vs {N}"
        )

    bsz = input_groups[0].shape[0]
    group_dims = [xg.shape[1] for xg in input_groups]
    total_heads = sum(group_dims)

    BLOCK_SIZE_N = align(input_groups[0], N, VEC_ALIGN_BYTES) if N <= COL_BLOCKING_THRESHOLD else COL_BLOCKING_THRESHOLD
    BLOCK_SIZE_M = rms_norm_fwd_heuristics({"n_cols": N})

    use_fused = N <= COL_BLOCKING_THRESHOLD and BLOCK_SIZE_N >= N and G <= 4

    if use_fused:
        x_full = _try_get_original_tensor(input_groups)
        if x_full is None:
            x_full = torch.cat(input_groups, dim=1)

        X_2d = x_full.reshape(-1, N)
        n_total = X_2d.shape[0]
        Y_2d = torch.empty_like(X_2d)

        head_to_group = _get_head_to_group(group_dims, X_2d.device)

        num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
        grid = (num_programs,)
        _group_rmsnorm_interleaved_kernel[grid](
            X_2d,
            Y_2d,
            weight,
            head_to_group,
            X_2d.stride(0),
            Y_2d.stride(0),
            weight.stride(0),
            n_rows=n_total,
            n_cols=N,
            total_heads=total_heads,
            eps=eps,
            G=G,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        Y_full = Y_2d.reshape(bsz, total_heads, N)
        output_groups = list(torch.split(Y_full, group_dims, dim=1))
    else:
        output_groups = []
        for g in range(G):
            xg = input_groups[g]
            wg = None if weight is None else weight[g]
            if N != 0:
                xg = xg.contiguous()
                yg = rmsnorm_infer_impl(xg, wg, eps)
            else:
                if output_like_input_stride:
                    yg = torch.empty_strided(
                        size=xg.shape,
                        stride=xg.stride(),
                        dtype=xg.dtype,
                        device=xg.device,
                    )
                else:
                    yg = torch.empty_like(xg, memory_format=torch.contiguous_format)
            output_groups.append(yg)

    return output_groups
