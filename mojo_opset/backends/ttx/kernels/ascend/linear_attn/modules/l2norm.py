# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2025, Jianqiao Lu, and the Triton-X team.

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl

from triton.language.math import rsqrt
from triton.runtime.libentry import libentry

from mojo_opset.backends.ttx.kernels.ascend.utils import VEC_ALIGN_BYTES
from mojo_opset.backends.ttx.kernels.ascend.utils import align
from mojo_opset.backends.ttx.kernels.ascend.utils import get_num_cores
from mojo_opset.backends.ttx.kernels.ascend.utils import input_guard
from mojo_opset.backends.ttx.kernels.ascend.utils import torch_to_triton_dtype


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1}),
        triton.Config({"BLOCK_SIZE_M": 2}),
        triton.Config({"BLOCK_SIZE_M": 4}),
        triton.Config({"BLOCK_SIZE_M": 8}),
        triton.Config({"BLOCK_SIZE_M": 16}),
        triton.Config({"BLOCK_SIZE_M": 32}),
    ],
    key=["n_rows", "n_cols"],
)
@libentry()
@triton.jit
def _l2norm_fwd_kernel(
    X_ptr,
    Y_ptr,
    RSTD_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Unified L2Norm forward kernel using NPU persistent programming paradigm.
    Applies row-wise parallelization with grid-stride loop and column-wise splitting.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, num_programs):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        sum_sq_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            x_chunk = tl.load(
                X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0
            ).to(tl.float32)

            sum_sq_acc += tl.sum(x_chunk * x_chunk, axis=1)

        rstd = rsqrt(sum_sq_acc + eps)
        tl.store(RSTD_ptr + rows_off, rstd, mask=rows_mask)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            x_chunk = tl.load(
                X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0
            ).to(tl.float32)

            y_chunk = x_chunk * rstd[:, None]

            tl.store(
                Y_ptr + rows_off[:, None] * stride_y_row + cols_off[None, :],
                y_chunk.to(Y_ptr.dtype.element_ty),
                mask=block_mask,
            )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1}),
        triton.Config({"BLOCK_SIZE_M": 2}),
        triton.Config({"BLOCK_SIZE_M": 4}),
        triton.Config({"BLOCK_SIZE_M": 8}),
        triton.Config({"BLOCK_SIZE_M": 16}),
        triton.Config({"BLOCK_SIZE_M": 32}),
    ],
    key=["n_rows", "n_cols"],
)
@libentry()
@triton.jit
def _l2norm_bwd_kernel(
    Y_ptr,
    RSTD_ptr,
    DY_ptr,
    DX_ptr,
    stride_y_row,
    stride_dy_row,
    stride_dx_row,
    n_rows,
    n_cols,
    OUTPUT_DTYPE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Unified L2Norm backward kernel.
    The backward formula `dx = rstd * (dy - y * sum(y * dy))` requires a full row statistic `sum(y * dy)`.
    Therefore, we use a two-pass approach similar to the large-cols version of layer_norm backward.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, num_programs):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        rstd = tl.load(RSTD_ptr + rows_off, mask=rows_mask, other=0.0).to(tl.float32)

        c_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            y_chunk = tl.load(
                Y_ptr + rows_off[:, None] * stride_y_row + cols_off[None, :], mask=block_mask, other=0.0
            ).to(tl.float32)

            dy_chunk = tl.load(
                DY_ptr + rows_off[:, None] * stride_dy_row + cols_off[None, :], mask=block_mask, other=0.0
            ).to(tl.float32)

            c_acc += tl.sum(y_chunk * dy_chunk, axis=1)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            y_chunk = tl.load(
                Y_ptr + rows_off[:, None] * stride_y_row + cols_off[None, :], mask=block_mask, other=0.0
            ).to(tl.float32)

            dy_chunk = tl.load(
                DY_ptr + rows_off[:, None] * stride_dy_row + cols_off[None, :], mask=block_mask, other=0.0
            ).to(tl.float32)

            dx_chunk = rstd[:, None] * (dy_chunk - y_chunk * c_acc[:, None])

            tl.store(
                DX_ptr + rows_off[:, None] * stride_dx_row + cols_off[None, :],
                dx_chunk.to(OUTPUT_DTYPE),
                mask=block_mask,
            )


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    n_rows, n_cols = x.shape

    if output_dtype is None:
        output_dtype = x.dtype

    y = torch.empty((n_rows, n_cols), dtype=output_dtype, device=x.device)
    rstd = torch.empty((n_rows,), dtype=torch.float32, device=x.device)

    COL_BLOCKING_THRESHOLD = 4096
    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = 2048
    else:
        BLOCK_SIZE_N = align(x, n_cols, VEC_ALIGN_BYTES)

    num_cores = get_num_cores()
    grid = (num_cores,)

    _l2norm_fwd_kernel[grid](
        x,
        y,
        rstd,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


def l2norm_bwd(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor):
    y_shape_og = y.shape
    y = y.view(-1, y.shape[-1])
    dy = dy.view(-1, dy.shape[-1])
    rstd = rstd.view(-1)
    n_rows, n_cols = y.shape

    dx = torch.empty_like(y)

    COL_BLOCKING_THRESHOLD = 4096
    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = 2048
    else:
        BLOCK_SIZE_N = align(dx, n_cols, VEC_ALIGN_BYTES)

    num_cores = get_num_cores()
    grid = (num_cores,)

    _l2norm_bwd_kernel[grid](
        y,
        rstd,
        dy,
        dx,
        y.stride(0),
        dy.stride(0),
        dx.stride(0),
        n_rows,
        n_cols,
        torch_to_triton_dtype[dx.dtype],
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return dx.view(y_shape_og)


class L2NormFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(ctx, x, eps=1e-6, output_dtype=None):
        y, rstd = l2norm_fwd(x, eps, output_dtype)
        ctx.eps = eps

        ctx.save_for_backward(y, rstd)
        return y

    @staticmethod
    @input_guard
    def backward(ctx, dy):
        y, rstd = ctx.saved_tensors

        dx = l2norm_bwd(y, rstd, dy)
        return dx, None, None


def l2norm(x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Performs L2 normalization on the last dimension of a tensor.
    This implementation is optimized for NPU using Triton.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        eps (float): A small value to add to the denominator for numerical stability.
        output_dtype (Optional[torch.dtype]): The desired data type of the output.
                                              If None, the output has the same dtype as the input.

    Returns:
        torch.Tensor: The L2-normalized tensor.
    """
    return L2NormFunction.apply(x, eps, output_dtype)


l2_norm = l2norm


class L2Norm(nn.Module):
    """
    A Layer that performs L2 normalization on the last dimension of the input.

    Args:
        eps (float): A small value added to the denominator for numerical stability. Default: 1e-6.
        output_dtype (Optional[torch.dtype]): The data type of the output tensor. If None, it matches the input.
    """

    def __init__(self, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.eps = eps
        self.output_dtype = output_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm(x, self.eps, self.output_dtype)
