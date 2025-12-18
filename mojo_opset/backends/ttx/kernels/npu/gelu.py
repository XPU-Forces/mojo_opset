import torch
import triton
import triton.language as tl

from triton.runtime.libentry import libentry

from mojo_opset.backends.ttx.kernels.npu.utils import VEC_ALIGN_BYTES
from mojo_opset.backends.ttx.kernels.utils import align

"""
This file contains the implementation of GELU (Gaussian Error Linear Unit) for NPU.

GELU formula: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

Based on Liger Kernel implementation:
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/geglu.py

Modifications for NPU architecture by triton-x team, 2025.
"""


COL_BLOCKING_THRESHOLD = 4096


@triton.jit
def gelu_tanh_approx(x):
    """GELU activation using tanh approximation"""
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / π)
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    return 0.5 * x * (1 + tl.tanh(tanh_arg))


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
def _gelu_fwd_kernel(
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
            y_f32 = gelu_tanh_approx(x_f32)

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
def _gelu_bwd_kernel(
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
            sqrt_2_over_pi = 0.7978845608028654
            x_cubed = x_f32 * x_f32 * x_f32
            tanh_arg = sqrt_2_over_pi * (x_f32 + 0.044715 * x_cubed)
            tanh_result = tl.tanh(tanh_arg)

            term1 = 0.5 * (1 + tanh_result)
            tanh_sq = tanh_result * tanh_result
            term2 = 0.5 * x_f32 * (1 - tanh_sq) * (sqrt_2_over_pi * (1 + 3 * 0.044715 * x_f32 * x_f32))
            dgelu_dx = term1 + term2

            dx_chunk = dy_chunk * dgelu_dx.to(dy_chunk.dtype)

            tl.store(dx_ptrs, dx_chunk, mask=block_mask)


def gelu_fwd_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for GELU.

    Args:
        x: Input tensor

    Returns:
        y: Output tensor y = gelu(x)
    """
    ori_shape = x.shape
    n_cols = ori_shape[-1]

    x_2d = x.reshape(-1, n_cols)
    n_rows = x_2d.shape[0]

    y = torch.empty_like(x_2d)

    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = 2048
    else:
        BLOCK_SIZE_N = align(x, n_cols, VEC_ALIGN_BYTES)

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    _gelu_fwd_kernel[grid](
        x_2d,
        y,
        x_2d.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return y.reshape(*ori_shape)


def gelu_bwd_impl(
    dy: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Backward pass for GELU.

    Args:
        dy: Gradient w.r.t. output
        x: Input tensor (from forward pass)

    Returns:
        dx: Gradient w.r.t. input
    """
    ori_shape = dy.shape
    n_cols = ori_shape[-1]

    dy_2d = dy.reshape(-1, n_cols)
    x_2d = x.reshape(-1, n_cols)
    n_rows = dy_2d.shape[0]

    dx = torch.empty_like(x_2d)

    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = 2048
    else:
        BLOCK_SIZE_N = align(dy, n_cols, VEC_ALIGN_BYTES)

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    _gelu_bwd_kernel[grid](
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

    return dx.reshape(*ori_shape)
