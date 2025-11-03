import torch
import triton
import triton.language as tl

from triton.runtime.libentry import libentry

from .utils import VEC_ALIGN_BYTES
from .utils import align

"""
This file contains the implementation of SiLU (Sigmoid Linear Unit) for NPU.

SiLU formula: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

Based on SwiGLU implementation pattern and Liger Kernel style.

Modifications for NPU architecture by triton-x team, 2025.
"""


COL_BLOCKING_THRESHOLD = 4096


@triton.jit
def silu_activation(x):
    """SiLU activation function: x * sigmoid(x)"""
    return x * tl.sigmoid(x)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 2, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 4, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 8, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 12, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 16, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 20, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 24, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 32, "multibuffer": True}),
    ],
    key=["n_rows", "n_cols"],
)
@libentry()
@triton.jit
def _silu_fwd_kernel(
    X_ptr,
    Y_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            x_ptrs = X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :]
            y_ptrs = Y_ptr + rows_off[:, None] * stride_y_row + cols_off[None, :]

            x_chunk = tl.load(x_ptrs, mask=block_mask, other=0.0)

            x_f32 = x_chunk.to(tl.float32)
            y_f32 = silu_activation(x_f32)

            y_chunk = y_f32.to(x_chunk.dtype)

            tl.store(y_ptrs, y_chunk, mask=block_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 2, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 4, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 8, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 12, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 16, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 20, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 24, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 32, "multibuffer": True}),
    ],
    key=["n_rows", "n_cols"],
    restore_value=["dY_ptr", "dX_ptr"],
)
@libentry()
@triton.jit
def _silu_bwd_kernel(
    dY_ptr,
    X_ptr,
    dX_ptr,
    stride_dy_row,
    stride_x_row,
    stride_dx_row,
    n_rows,
    n_cols,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            dy_ptrs = dY_ptr + rows_off[:, None] * stride_dy_row + cols_off[None, :]
            x_ptrs = X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :]
            dx_ptrs = dX_ptr + rows_off[:, None] * stride_dx_row + cols_off[None, :]

            dy_chunk = tl.load(dy_ptrs, mask=block_mask, other=0.0)
            x_chunk = tl.load(x_ptrs, mask=block_mask, other=0.0)

            x_f32 = x_chunk.to(tl.float32)
            sigmoid_x = tl.sigmoid(x_f32)

            dsilu_dx = sigmoid_x * (1 + x_f32 * (1 - sigmoid_x))

            dx_chunk = dy_chunk * dsilu_dx.to(dy_chunk.dtype)

            tl.store(dx_ptrs, dx_chunk, mask=block_mask)


def silu_fwd(x):
    """
    Forward pass for SiLU.

    Args:
        x: Input tensor

    Returns:
        y: Output tensor y = silu(x) = x * sigmoid(x)
    """
    ori_shape = x.shape
    n_cols = ori_shape[-1]

    x_2d = x.view(-1, n_cols)
    n_rows = x_2d.shape[0]

    y = torch.empty_like(x_2d)

    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = 2048
    else:
        BLOCK_SIZE_N = align(x, n_cols, VEC_ALIGN_BYTES)

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    _silu_fwd_kernel[grid](
        x_2d,
        y,
        x_2d.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return y.view(*ori_shape)


def silu_bwd(dy, x):
    """
    Backward pass for SiLU.

    Args:
        dy: Gradient w.r.t. output
        x: Input tensor (from forward pass)

    Returns:
        dx: Gradient w.r.t. input
    """
    ori_shape = dy.shape
    n_cols = ori_shape[-1]

    dy_2d = dy.view(-1, n_cols)
    x_2d = x.view(-1, n_cols)
    n_rows = dy_2d.shape[0]

    dx = torch.empty_like(x_2d)

    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = 2048
    else:
        BLOCK_SIZE_N = align(dy, n_cols, VEC_ALIGN_BYTES)

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    _silu_bwd_kernel[grid](
        dy_2d,
        x_2d,
        dx,
        dy_2d.stride(0),
        x_2d.stride(0),
        dx.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return dx.view(*ori_shape)


class TTXSiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of SiLU function.

        Args:
            x: Input tensor

        Returns:
            y: Output tensor y = silu(x) = x * sigmoid(x)
        """
        y = silu_fwd(x)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass of SiLU function.

        Args:
            dy: Gradient w.r.t. output

        Returns:
            dx: Gradient w.r.t. input
        """
        (x,) = ctx.saved_tensors
        dx = silu_bwd(dy, x)
        return dx


def ttx_silu(x):
    """
    TTX SiLU activation function for inference.

    Implements: output = silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

    SiLU is also known as Swish activation function. It provides smooth,
    non-monotonic activation that can improve model performance.

    Args:
        x: Input tensor of any shape

    Returns:
        Output tensor with same shape as input

    Example:
        >>> x = torch.randn(32, 128, 1024)
        >>> output = ttx_silu(x)  # Shape: (32, 128, 1024)
    """
    return TTXSiLUFunction.apply(x)
