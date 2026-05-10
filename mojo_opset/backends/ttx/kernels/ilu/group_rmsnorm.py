import torch

import triton
import triton.language as tl

from .utils import _block_size_n_pow2
from .utils import COL_BLOCKING_THRESHOLD
from .utils import ilu_grid_dim_from_row_tasks
from .utils import rms_norm_fwd_heuristics


def _group_rmsnorm_grid_n_programs(n_rows: int, n_cols: int) -> int:
    block_m = rms_norm_fwd_heuristics({"n_cols": n_cols})
    n_tasks = triton.cdiv(n_rows, block_m)
    return ilu_grid_dim_from_row_tasks(n_tasks)


@triton.heuristics({"BLOCK_SIZE_M": rms_norm_fwd_heuristics})
@triton.jit
def _group_rmsnorm_fwd_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    eps,
    N_COLS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    task_mask = pid < num_row_tasks

    block_start_row = pid * BLOCK_SIZE_M
    current_row_offsets = block_start_row + tl.arange(0, BLOCK_SIZE_M)
    row_mask = task_mask & (current_row_offsets < n_rows)

    ss_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for col_offset in range(0, N_COLS, BLOCK_SIZE_N):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_offsets < N_COLS

        x_ptrs = X_ptr + (current_row_offsets[:, None] * stride_x_row + col_offsets[None, :])
        x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
        ss_acc += tl.sum(x * x, axis=1)

    ss_acc = tl.where(row_mask, ss_acc, 0)
    mean_square = ss_acc / N_COLS
    rrms = tl.rsqrt(mean_square + eps)
    rrms = tl.where(row_mask, rrms, 0.0)

    for col_offset in range(0, N_COLS, BLOCK_SIZE_N):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_offsets < N_COLS

        x_ptrs = X_ptr + (current_row_offsets[:, None] * stride_x_row + col_offsets[None, :])
        y_ptrs = Y_ptr + (current_row_offsets[:, None] * stride_y_row + col_offsets[None, :])

        x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        x_f32 = x.to(tl.float32)
        x_normalized = x_f32 * rrms[:, None]

        if HAS_WEIGHT:
            w = tl.load(W_ptr + col_offsets, mask=col_mask, other=0.0)
            w_f32 = w.to(tl.float32)
            y = x_normalized * w_f32[None, :]
        else:
            y = x_normalized

        tl.store(y_ptrs, y, mask=row_mask[:, None] & col_mask[None, :])


def _rmsnorm_fwd_single(
    x: torch.Tensor,
    weight: torch.Tensor = None,
    eps: float = 1e-6,
    output_like_input_stride: bool = False,
) -> torch.Tensor:
    assert x.ndim == 3, f"x must be 3D [token, num_head, norm_size], got {x.shape}"
    T, H, N = x.shape
    assert x.stride(2) == 1, f"x last dim must be contiguous, got stride={x.stride()}"

    if weight is not None:
        assert weight.shape == (N,)
        assert weight.is_contiguous()

    x_contig = x.contiguous()
    x_2d = x_contig.reshape(-1, N)
    n_rows = x_2d.shape[0]
    y_2d = torch.empty_like(x_2d)

    if N > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = COL_BLOCKING_THRESHOLD
    else:
        BLOCK_SIZE_N = _block_size_n_pow2(N)

    grid = (_group_rmsnorm_grid_n_programs(n_rows, N),)

    _group_rmsnorm_fwd_kernel[grid](
        x_2d,
        y_2d,
        weight,
        x_2d.stride(0),
        y_2d.stride(0),
        n_rows=n_rows,
        eps=eps,
        N_COLS=N,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        HAS_WEIGHT=weight is not None,
    )

    y_3d = y_2d.reshape(T, H, N)
    if output_like_input_stride and not x.is_contiguous():
        y = torch.empty_strided(
            size=x.shape,
            stride=x.stride(),
            dtype=x.dtype,
            device=x.device,
        )
        y.copy_(y_3d)
        return y
    return y_3d


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
        assert weight.shape == (G, N)
        assert weight.is_contiguous(), f"weight must be contiguous, got stride={weight.stride()}"

    output_groups = []
    for g in range(G):
        xg = input_groups[g]
        assert xg.ndim == 3, f"group {g} input must be [token, num_head, norm_size]"
        assert xg.shape[-1] == N, f"group {g} last dim mismatch: {xg.shape[-1]} vs {N}"
        wg = None if weight is None else weight[g]
        if N != 0:
            yg = _rmsnorm_fwd_single(
                x=xg,
                weight=wg,
                eps=eps,
                output_like_input_stride=output_like_input_stride,
            )
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
