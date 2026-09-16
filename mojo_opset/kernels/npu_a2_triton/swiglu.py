from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import libentry

from mojo_opset.kernels._npu_triton_utils import VEC_ALIGN_BYTES
from mojo_opset.kernels._triton_utils import align

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
    key=["n_cols"],
)
@libentry()
@triton.jit(do_not_specialize=["n_rows"])
def _swiglu_fwd_kernel(
    a,
    b,
    scales,
    c,
    stride_row,
    n_rows,
    n_cols,
    HAS_SCALES: tl.constexpr,
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
        if HAS_SCALES:
            scale = tl.load(scales + rows_off, mask=rows_mask, other=0.0).to(tl.float32)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            a_ptrs = a + rows_off[:, None] * stride_row + cols_off[None, :]
            b_ptrs = b + rows_off[:, None] * stride_row + cols_off[None, :]
            c_ptrs = c + rows_off[:, None] * stride_row + cols_off[None, :]

            a_chunk = tl.load(a_ptrs, mask=block_mask, other=0.0)
            b_chunk = tl.load(b_ptrs, mask=block_mask, other=0.0)

            a_f32 = a_chunk.to(tl.float32)
            silu_a = silu(a_f32)

            c_chunk = silu_a * b_chunk.to(tl.float32)
            if HAS_SCALES:
                c_chunk *= scale[:, None]

            tl.store(c_ptrs, c_chunk, mask=block_mask)


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
    key=["n_cols"],
    restore_value=["dc", "da", "db", "dscales"],
)
@libentry()
@triton.jit(do_not_specialize=["n_rows"])
def _swiglu_bwd_kernel(
    dc,
    a,
    b,
    scales,
    da,
    db,
    dscales,
    stride_row,
    n_rows,
    n_cols,
    HAS_SCALES: tl.constexpr,
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
        dscale_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        if HAS_SCALES:
            scale = tl.load(scales + rows_off, mask=rows_mask, other=0.0).to(tl.float32)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            dc_ptrs = dc + rows_off[:, None] * stride_row + cols_off[None, :]
            a_ptrs = a + rows_off[:, None] * stride_row + cols_off[None, :]
            b_ptrs = b + rows_off[:, None] * stride_row + cols_off[None, :]
            da_ptrs = da + rows_off[:, None] * stride_row + cols_off[None, :]
            db_ptrs = db + rows_off[:, None] * stride_row + cols_off[None, :]

            dc_chunk = tl.load(dc_ptrs, mask=block_mask, other=0.0).to(tl.float32)
            a_chunk = tl.load(a_ptrs, mask=block_mask, other=0.0).to(tl.float32)
            b_chunk = tl.load(b_ptrs, mask=block_mask, other=0.0).to(tl.float32)

            sigmoid_a = tl.sigmoid(a_chunk)
            silu_a = a_chunk * sigmoid_a
            grad_base = dc_chunk
            if HAS_SCALES:
                grad_base *= scale[:, None]
                dscale_acc += tl.sum(dc_chunk * silu_a * b_chunk, axis=1)

            db_chunk = grad_base * silu_a

            da_factor = silu_a * (1 - sigmoid_a) + sigmoid_a
            da_chunk = grad_base * b_chunk * da_factor

            tl.store(da_ptrs, da_chunk, mask=block_mask)
            tl.store(db_ptrs, db_chunk, mask=block_mask)

        if HAS_SCALES:
            tl.store(dscales + rows_off, dscale_acc, mask=rows_mask)


def swiglu_train_fwd_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward pass for SwiGLU.

    Args:
        a: Input tensor A
        b: Input tensor B

    Returns:
        c: Output tensor C = silu(a) * b
    """
    if a.shape != b.shape:
        raise ValueError(f"SwiGLU inputs must match shape; got {a.shape} and {b.shape}.")
    if a.dtype != b.dtype or a.device != b.device:
        raise ValueError("SwiGLU inputs must share dtype and device.")
    if a.dim() < 1:
        raise ValueError("SwiGLU inputs must have at least one dimension.")
    if scales is not None and scales.device != a.device:
        raise ValueError("SwiGLU scales must be on the same device as the inputs.")

    ori_shape = a.shape
    n_cols = ori_shape[-1]

    a_2d = a.reshape(-1, n_cols).contiguous()
    b_2d = b.reshape(-1, n_cols).contiguous()
    n_rows = a_2d.shape[0]
    scales_1d = None if scales is None else scales.reshape(-1).contiguous()
    if scales_1d is not None and scales_1d.numel() != n_rows:
        raise ValueError(f"SwiGLU scales must have {n_rows} elements, got {scales_1d.numel()}.")

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
        scales_1d if scales_1d is not None else a_2d,
        c,
        a_2d.stride(0),
        n_rows,
        n_cols,
        HAS_SCALES=scales_1d is not None,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return c.reshape(*ori_shape)


def swiglu_train_bwd_impl(
    dc: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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
    if dc.shape != a.shape or a.shape != b.shape:
        raise ValueError("SwiGLU grad_output and inputs must have identical shapes.")
    if dc.dtype != a.dtype or a.dtype != b.dtype:
        raise ValueError("SwiGLU grad_output and inputs must share dtype.")
    if dc.device != a.device or a.device != b.device:
        raise ValueError("SwiGLU grad_output and inputs must share device.")
    if scales is not None and scales.device != a.device:
        raise ValueError("SwiGLU scales must be on the same device as the inputs.")

    ori_shape = a.shape
    n_cols = ori_shape[-1]

    dc_2d = dc.reshape(-1, n_cols).contiguous()
    a_2d = a.reshape(-1, n_cols).contiguous()
    b_2d = b.reshape(-1, n_cols).contiguous()
    n_rows = dc_2d.shape[0]
    scales_1d = None if scales is None else scales.reshape(-1).contiguous()
    if scales_1d is not None and scales_1d.numel() != n_rows:
        raise ValueError(f"SwiGLU scales must have {n_rows} elements, got {scales_1d.numel()}.")

    da = torch.empty_like(a_2d)
    db = torch.empty_like(b_2d)
    dscales = None if scales_1d is None else torch.empty_like(scales_1d)

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
        scales_1d if scales_1d is not None else dc_2d,
        da,
        db,
        dscales if dscales is not None else dc_2d,
        dc_2d.stride(0),
        n_rows,
        n_cols,
        HAS_SCALES=scales_1d is not None,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    grad_scales = None if dscales is None else dscales.reshape(scales.shape)
    return da.reshape(*ori_shape), db.reshape(*ori_shape), grad_scales


def swiglu_fwd_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return swiglu_train_fwd_impl(a, b)


def swiglu_bwd_impl(
    dc: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    da, db, _ = swiglu_train_bwd_impl(dc, a, b)
    return da, db


@torch.library.custom_op("mojo_npu_triton_a2::swiglu_fwd", mutates_args=())
def swiglu_fwd(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return swiglu_train_fwd_impl(x1, x2, scales)


@swiglu_fwd.register_fake
def _swiglu_fwd_fake(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del x2, scales
    return torch.empty_like(x1, memory_format=torch.contiguous_format)


@torch.library.custom_op("mojo_npu_triton_a2::swiglu_bwd", mutates_args=())
def swiglu_bwd(
    grad_output: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_x1, grad_x2, grad_scales = swiglu_train_bwd_impl(grad_output, x1, x2, scales)
    if grad_scales is None:
        grad_scales = torch.empty(0, dtype=x1.dtype, device=x1.device)
    return grad_x1, grad_x2, grad_scales


@swiglu_bwd.register_fake
def _swiglu_bwd_fake(
    grad_output: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del grad_output
    grad_scales = (torch.empty(0, dtype=x1.dtype, device=x1.device) if scales is None
                   else torch.empty_like(scales, memory_format=torch.contiguous_format))
    return (torch.empty_like(x1, memory_format=torch.contiguous_format),
            torch.empty_like(x2, memory_format=torch.contiguous_format), grad_scales)
