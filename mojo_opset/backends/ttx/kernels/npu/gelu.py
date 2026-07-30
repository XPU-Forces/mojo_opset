import torch
import triton
import triton.language as tl

from .utils import libentry

from mojo_opset.backends.ttx.kernels.npu.utils import VEC_ALIGN_BYTES
from mojo_opset.backends.ttx.kernels.utils import align

"""
This file contains the implementation of GELU (Gaussian Error Linear Unit) for NPU.

GELU formula: gelu(x) = x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 * x^3))
                    = x / (1 + exp(-x * (c + k * x^2)))
  where c = 2 * sqrt(2/pi), k = c * 0.044715

Based on Liger Kernel implementation:
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/geglu.py

Modifications for NPU architecture by triton-x team, 2025.
"""


COL_BLOCKING_THRESHOLD = 4096
FWD_2D_THRESHOLD = 10240
MAX_BLOCK_SIZE_N = 1024


@triton.jit
def gelu_tanh_approx(x):
    """GELU activation using sigmoid (exp) approximation."""
    c = 1.5957691216057308   # 2 * sqrt(2/pi)
    k = 0.07135481627260025  # c * 0.044715
    x_sq = x * x
    inner = x * (c + k * x_sq)
    return x / (1 + tl.math.exp(-inner))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
        triton.Config({"BLOCK_SIZE": 8192}),
        triton.Config({"BLOCK_SIZE": 16384}),
    ],
    key=["n_elements"],
)
@libentry()
@triton.jit
def _gelu_fwd_kernel(
    x,
    y,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    1D flattened GELU kernel.
    """
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    for block_id in range(pid, num_blocks, grid_size):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_chunk = tl.load(x + offsets, mask=mask, other=0.0)
        x_f32 = x_chunk.to(tl.float32)
        y_f32 = gelu_tanh_approx(x_f32)
        y_chunk = y_f32.to(x_chunk.dtype)
        tl.store(y + offsets, y_chunk, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1}),
        triton.Config({"BLOCK_SIZE_M": 2}),
        triton.Config({"BLOCK_SIZE_M": 4}),
        triton.Config({"BLOCK_SIZE_M": 8}),
        triton.Config({"BLOCK_SIZE_M": 12}),
        triton.Config({"BLOCK_SIZE_M": 16}),
        triton.Config({"BLOCK_SIZE_M": 20}),
        triton.Config({"BLOCK_SIZE_M": 24}),
        triton.Config({"BLOCK_SIZE_M": 32}),
    ],
    key=["n_rows", "n_cols"],
)
@libentry()
@triton.jit
def _gelu_fwd_2d_kernel(
    x,
    y,
    stride_row,
    n_rows,
    n_cols,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    COL_ALIGNED: tl.constexpr,
):
    """
    2D GELU kernel for large column counts (n_cols > FWD_2D_THRESHOLD).
    Iterates over column blocks per row block to avoid loading entire rows at once.
    """
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)

            x_ptrs = x + rows_off[:, None] * stride_row + cols_off[None, :]
            y_ptrs = y + rows_off[:, None] * stride_row + cols_off[None, :]

            if COL_ALIGNED:
                rows_only_mask = rows_mask[:, None]
                x_chunk = tl.load(x_ptrs, mask=rows_only_mask, other=0.0)
            else:
                cols_mask = cols_off < n_cols
                block_mask = rows_mask[:, None] & cols_mask[None, :]
                x_chunk = tl.load(x_ptrs, mask=block_mask, other=0.0)

            x_f32 = x_chunk.to(tl.float32)
            y_f32 = gelu_tanh_approx(x_f32)
            y_chunk = y_f32.to(x_chunk.dtype)

            if COL_ALIGNED:
                tl.store(y_ptrs, y_chunk, mask=rows_mask[:, None])
            else:
                cols_mask = cols_off < n_cols
                block_mask = rows_mask[:, None] & cols_mask[None, :]
                tl.store(y_ptrs, y_chunk, mask=block_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1}),
        triton.Config({"BLOCK_SIZE_M": 2}),
        triton.Config({"BLOCK_SIZE_M": 4}),
        triton.Config({"BLOCK_SIZE_M": 8}),
        triton.Config({"BLOCK_SIZE_M": 12}),
        triton.Config({"BLOCK_SIZE_M": 16}),
        triton.Config({"BLOCK_SIZE_M": 20}),
        triton.Config({"BLOCK_SIZE_M": 24}),
        triton.Config({"BLOCK_SIZE_M": 32}),
    ],
    key=["n_rows", "n_cols"],
    restore_value=["dy", "dx"],
)
@libentry()
@triton.jit
def _gelu_bwd_kernel(
    dy,
    x,
    dx,
    stride_row,
    n_rows,
    n_cols,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    COL_ALIGNED: tl.constexpr,
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

            dy_ptrs = dy + rows_off[:, None] * stride_row + cols_off[None, :]
            x_ptrs = x + rows_off[:, None] * stride_row + cols_off[None, :]
            dx_ptrs = dx + rows_off[:, None] * stride_row + cols_off[None, :]

            if COL_ALIGNED:
                rows_only_mask = rows_mask[:, None]
                dy_chunk = tl.load(dy_ptrs, mask=rows_only_mask, other=0.0)
                x_chunk = tl.load(x_ptrs, mask=rows_only_mask, other=0.0)
            else:
                cols_mask = cols_off < n_cols
                block_mask = rows_mask[:, None] & cols_mask[None, :]
                dy_chunk = tl.load(dy_ptrs, mask=block_mask, other=0.0)
                x_chunk = tl.load(x_ptrs, mask=block_mask, other=0.0)

            x_f32 = x_chunk.to(tl.float32)
            c_fwd = 1.5957691216057308       # 2 * sqrt(2/pi)
            k_fwd = 0.07135481627260025      # c_fwd * 0.044715
            c_orig = 0.7978845608028654      # sqrt(2/pi)
            kkk = 0.10703222440890038        # 3 * c_orig * 0.044715

            x_sq = x_f32 * x_f32
            inner = x_f32 * (c_fwd + k_fwd * x_sq)
            s = 1.0 / (1.0 + tl.math.exp(-inner))
            dy_dx = c_orig + kkk * x_sq
            s_mul_1_minus_s = s * (1.0 - s)
            dgelu_dx = s + 2.0 * x_f32 * s_mul_1_minus_s * dy_dx

            dx_chunk = dy_chunk * dgelu_dx.to(dy_chunk.dtype)

            if COL_ALIGNED:
                tl.store(dx_ptrs, dx_chunk, mask=rows_mask[:, None])
            else:
                cols_mask = cols_off < n_cols
                block_mask = rows_mask[:, None] & cols_mask[None, :]
                tl.store(dx_ptrs, dx_chunk, mask=block_mask)


def gelu_fwd_impl(x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for GELU.

    Uses 1D flattened kernel by default; switches to 2D row-blocked kernel
    when the last dimension exceeds FWD_2D_THRESHOLD to avoid excessive memory.
    """
    ori_shape = x.shape
    n_cols = ori_shape[-1]

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    if n_cols > FWD_2D_THRESHOLD:
        x_2d = x.reshape(-1, n_cols)
        n_rows = x_2d.shape[0]
        y = torch.empty_like(x_2d)

        BLOCK_SIZE_N = min(triton.next_power_of_2(n_cols), MAX_BLOCK_SIZE_N)
        col_aligned = (n_cols % BLOCK_SIZE_N) == 0

        _gelu_fwd_2d_kernel[grid](
            x_2d,
            y,
            x_2d.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            COL_ALIGNED=col_aligned,
        )
    else:
        x_1d = x.reshape(-1).contiguous()
        n_elements = x_1d.numel()
        y = torch.empty_like(x_1d)

        _gelu_fwd_kernel[grid](
            x_1d,
            y,
            n_elements,
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
    col_aligned = (n_cols % BLOCK_SIZE_N) == 0
    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    _gelu_bwd_kernel[grid](
        dy_2d,
        x_2d,
        dx,
        dy_2d.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        COL_ALIGNED=col_aligned,
    )

    return dx.reshape(*ori_shape)
