from typing import Tuple

import torch
import triton
import triton.language as tl

from triton.runtime.libentry import libentry

from mojo_opset.backends.ttx.kernels.npu.utils import VEC_ALIGN_BYTES
from mojo_opset.backends.ttx.kernels.utils import align

"""
This file contains the implementation of SwiGLU (Swish-Gated Linear Unit) for NPU.

SwiGLU formula: c = silu(a) * b, where silu(x) = x * sigmoid(x)

Based on Liger Kernel implementation:
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py

Modifications for NPU architecture by triton-x team, 2025.
"""


COL_BLOCKING_THRESHOLD = 2048


@triton.jit
def silu(x):
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
def _swiglu_fwd_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    stride_a_row,
    stride_b_row,
    stride_c_row,
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

            a_ptrs = A_ptr + rows_off[:, None] * stride_a_row + cols_off[None, :]
            b_ptrs = B_ptr + rows_off[:, None] * stride_b_row + cols_off[None, :]
            c_ptrs = C_ptr + rows_off[:, None] * stride_c_row + cols_off[None, :]

            a_chunk = tl.load(a_ptrs, mask=block_mask, other=0.0)
            b_chunk = tl.load(b_ptrs, mask=block_mask, other=0.0)

            a_f32 = a_chunk.to(tl.float32)
            silu_a = silu(a_f32)

            c_chunk = silu_a.to(a_chunk.dtype) * b_chunk

            tl.store(c_ptrs, c_chunk, mask=block_mask)


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
    restore_value=["dC_ptr", "dA_ptr", "dB_ptr"],
)
@libentry()
@triton.jit
def _swiglu_bwd_kernel(
    dC_ptr,
    A_ptr,
    B_ptr,
    dA_ptr,
    dB_ptr,
    stride_dc_row,
    stride_a_row,
    stride_b_row,
    stride_da_row,
    stride_db_row,
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

            dc_ptrs = dC_ptr + rows_off[:, None] * stride_dc_row + cols_off[None, :]
            a_ptrs = A_ptr + rows_off[:, None] * stride_a_row + cols_off[None, :]
            b_ptrs = B_ptr + rows_off[:, None] * stride_b_row + cols_off[None, :]
            da_ptrs = dA_ptr + rows_off[:, None] * stride_da_row + cols_off[None, :]
            db_ptrs = dB_ptr + rows_off[:, None] * stride_db_row + cols_off[None, :]

            dc_chunk = tl.load(dc_ptrs, mask=block_mask, other=0.0)
            a_chunk = tl.load(a_ptrs, mask=block_mask, other=0.0)
            b_chunk = tl.load(b_ptrs, mask=block_mask, other=0.0)

            a_f32 = a_chunk.to(tl.float32)
            sigmoid_a = tl.sigmoid(a_f32)
            silu_a = a_f32 * sigmoid_a

            db_chunk = dc_chunk * silu_a.to(dc_chunk.dtype)

            da_factor = silu_a * (1 - sigmoid_a) + sigmoid_a
            da_chunk = dc_chunk * b_chunk * da_factor.to(dc_chunk.dtype)

            tl.store(da_ptrs, da_chunk, mask=block_mask)
            tl.store(db_ptrs, db_chunk, mask=block_mask)


def swiglu_fwd_impl(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass for SwiGLU.

    Args:
        a: Input tensor A
        b: Input tensor B

    Returns:
        c: Output tensor C = silu(a) * b
    """
    ori_shape = a.shape
    n_cols = ori_shape[-1]

    a_2d = a.reshape(-1, n_cols)
    b_2d = b.reshape(-1, n_cols)
    n_rows = a_2d.shape[0]

    c = torch.empty_like(a_2d)

    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = 2048
    else:
        BLOCK_SIZE_N = align(a, n_cols, VEC_ALIGN_BYTES)

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    _swiglu_fwd_kernel[grid](
        a_2d,
        b_2d,
        c,
        a_2d.stride(0),
        b_2d.stride(0),
        c.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return c.reshape(*ori_shape)


def swiglu_bwd_impl(
    dc: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for SwiGLU.

    Args:
        dc: Gradient w.r.t. output
        a: Input tensor A (from forward pass)
        b: Input tensor B (from forward pass)

    Returns:
        da: Gradient w.r.t. input A
        db: Gradient w.r.t. input B
    """
    ori_shape = dc.shape
    n_cols = ori_shape[-1]

    dc_2d = dc.reshape(-1, n_cols)
    a_2d = a.reshape(-1, n_cols)
    b_2d = b.reshape(-1, n_cols)
    n_rows = dc_2d.shape[0]

    da = torch.empty_like(a_2d)
    db = torch.empty_like(b_2d)

    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = 2048
    else:
        BLOCK_SIZE_N = align(dc, n_cols, VEC_ALIGN_BYTES)

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    _swiglu_bwd_kernel[grid](
        dc_2d,
        a_2d,
        b_2d,
        da,
        db,
        dc_2d.stride(0),
        a_2d.stride(0),
        b_2d.stride(0),
        da.stride(0),
        db.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return da.reshape(*ori_shape), db.reshape(*ori_shape)
