from typing import Tuple

import torch
import triton
import triton.language as tl

from triton.runtime.libentry import libentry

from mojo_opset.backends.ttx.kernels.ascend.utils import VEC_ALIGN_BYTES
from mojo_opset.backends.ttx.kernels.utils import align
from mojo_opset.backends.ttx.kernels.utils import ceil_div
from mojo_opset.backends.ttx.kernels.utils import torch_to_triton_dtype

"""
This file incorporates code from Unsloth licensed under the Apache License, Version 2.0.
See the original Unsloth repository at https://github.com/unslothai/unsloth.

Portions of this file are adapted from:
https://github.com/linkedin/Liger-Kernel/blob/7382a8761f9af679482b968f9348013d933947c7/src/liger_kernel/ops/rmsnorm.py#L30
which in turn is based on Unsloth code at:
https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22

Modifications in this repository by triton-x team, 2025.
"""


COL_BLOCKING_THRESHOLD = 2048

_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)


def rms_norm_fwd_heuristics(args):
    # Empirical value; hard-coded.
    if args["n_cols"] <= COL_BLOCKING_THRESHOLD:
        return ceil_div(8192, args["n_cols"])
    else:
        return 4


@triton.heuristics({"BLOCK_SIZE_M": rms_norm_fwd_heuristics})
@libentry()
@triton.jit
def _rmsnorm_infer_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M

        current_row_offsets = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        row_mask = current_row_offsets < n_rows

        ss_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            col_offsets = col_offset + tl.arange(0, BLOCK_SIZE_N)
            col_mask = col_offsets < n_cols

            x_ptrs = X_ptr + (current_row_offsets[:, None] * stride_x_row + col_offsets[None, :])

            x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)

            ss_acc += tl.sum(x * x, axis=1)

        ss_acc = tl.where(row_mask, ss_acc, 0)

        mean_square = ss_acc / n_cols
        rrms = tl.rsqrt(mean_square + eps)

        rrms = tl.where(row_mask, rrms, 0.0)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            col_offsets = col_offset + tl.arange(0, BLOCK_SIZE_N)
            col_mask = col_offsets < n_cols

            x_ptrs = X_ptr + (current_row_offsets[:, None] * stride_x_row + col_offsets[None, :])
            w_ptrs = W_ptr + col_offsets
            y_ptrs = Y_ptr + (current_row_offsets[:, None] * stride_y_row + col_offsets[None, :])

            x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=col_mask, other=0.0)

            x_f32 = x.to(tl.float32)
            w_f32 = w.to(tl.float32)

            x_normalized = x_f32 * rrms[:, None]

            y = x_normalized * w_f32[None, :]

            tl.store(
                y_ptrs,
                y.to(Y_ptr.dtype.element_ty),
                mask=row_mask[:, None] & col_mask[None, :],
            )


def rmsnorm_infer_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    shape = x.shape
    dim = shape[-1]
    X_2d = x.reshape(-1, dim)
    n_rows, n_cols = X_2d.shape

    y = torch.empty_like(X_2d)

    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = COL_BLOCKING_THRESHOLD
    else:
        BLOCK_SIZE_N = align(x, n_cols, VEC_ALIGN_BYTES)

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]

    grid = (num_programs,)

    _rmsnorm_infer_kernel[grid](
        x,
        y,
        w,
        X_2d.stride(0),
        y.stride(0),
        n_rows=n_rows,
        n_cols=n_cols,
        eps=eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return y.reshape(*shape)


@triton.heuristics({"BLOCK_SIZE_M": rms_norm_fwd_heuristics})
@libentry()
@triton.jit
def _rmsnorm_fwd_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows,
    n_cols,
    eps,
    offset,
    casting_mode_int: tl.constexpr,
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

        X_ptr_row_block = X_ptr + rows_off[:, None] * X_row_stride
        X_dtype = X_ptr.dtype.element_ty

        var_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            X_chunk = tl.load(X_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0).to(tl.float32)
            var_acc += tl.sum(X_chunk * X_chunk, axis=1)

        var = var_acc / n_cols
        rstd_vec = tl.rsqrt(var + eps)
        tl.store(RSTD_ptr + rows_off * RSTD_row_stride, rstd_vec, mask=rows_mask)

        Y_ptr_row_block = Y_ptr + rows_off[:, None] * Y_row_stride
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            X_chunk = tl.load(X_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)

            if casting_mode_int == _CASTING_MODE_GEMMA:
                X_chunk = X_chunk.to(tl.float32)
                W_chunk = W_chunk.to(tl.float32)
            elif casting_mode_int == _CASTING_MODE_LLAMA:
                X_chunk = X_chunk.to(tl.float32)

            if casting_mode_int == _CASTING_MODE_LLAMA:
                normed_X_chunk = (X_chunk * rstd_vec[:, None]).to(X_dtype)
            else:
                normed_X_chunk = X_chunk * rstd_vec[:, None]

            Y_chunk = normed_X_chunk * (W_chunk[None, :] + offset)
            if casting_mode_int == _CASTING_MODE_GEMMA:
                Y_chunk = Y_chunk.to(X_dtype)

            tl.store(Y_ptr_row_block + cols_off[None, :], Y_chunk, mask=block_mask)


@triton.heuristics({"BLOCK_SIZE_M": lambda args: ceil_div(4096, args["n_cols"])})
@libentry()
@triton.jit
def _rmsnorm_bwd_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    casting_mode_int: tl.constexpr,
    X_dtype: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    dW_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    cols_off = tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols_off < n_cols
    W_row = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)
    W_row_offset = W_row + offset

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M

        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows
        block_mask = rows_mask[:, None] & cols_mask[None, :]

        dY_block = tl.load(dY_ptr + rows_off[:, None] * dY_row_stride + cols_off[None, :], mask=block_mask, other=0.0)
        X_block = tl.load(X_ptr + rows_off[:, None] * X_row_stride + cols_off[None, :], mask=block_mask, other=0.0)
        rstd_vec = tl.load(RSTD_ptr + rows_off * RSTD_row_stride, mask=rows_mask, other=0.0)

        X_block_f32 = X_block.to(tl.float32)
        normed_X_block = X_block_f32 * rstd_vec[:, None]

        if casting_mode_int == _CASTING_MODE_LLAMA:
            m_block = (dY_block * W_row_offset[None, :]).to(tl.float32)
            dW_acc += tl.sum(dY_block * normed_X_block.to(X_dtype), axis=0)
        elif casting_mode_int == _CASTING_MODE_GEMMA:
            dY_block_f32 = dY_block.to(tl.float32)
            W_row_offset = W_row_offset.to(tl.float32)

            m_block = dY_block_f32 * W_row_offset[None, :]
            dW_acc += tl.sum(dY_block_f32 * normed_X_block, axis=0)
        else:
            m_block = dY_block * W_row_offset[None, :]
            dW_acc += tl.sum(dY_block * normed_X_block, axis=0)

        dot_product_vec = tl.sum(m_block * X_block_f32, axis=1)
        rstd_vec_sq = rstd_vec * rstd_vec

        term1 = rstd_vec[:, None] * m_block
        term2 = -(1 / n_cols) * rstd_vec_sq[:, None] * rstd_vec[:, None] * dot_product_vec[:, None] * X_block_f32

        dX_block = term1 + term2

        tl.store(dX_ptr + rows_off[:, None] * dX_row_stride + cols_off[None, :], dX_block.to(X_dtype), mask=block_mask)

    dW_ptr_prog = dW_ptr + pid * dW_row_stride + cols_off
    tl.store(dW_ptr_prog, dW_acc, mask=cols_mask)


@libentry()
@triton.jit
def _rmsnorm_bwd_large_cols_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    casting_mode_int: tl.constexpr,
    X_dtype: tl.constexpr,
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

        rstd_vec = tl.load(RSTD_ptr + rows_off * RSTD_row_stride, mask=rows_mask, other=0.0)

        dot_product_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            dY_chunk = tl.load(
                dY_ptr + rows_off[:, None] * dY_row_stride + cols_off[None, :], mask=block_mask, other=0.0
            )
            X_chunk = tl.load(
                X_ptr + rows_off[:, None] * X_row_stride + cols_off[None, :], mask=block_mask, other=0.0
            ).to(tl.float32)
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)

            W_chunk_offset = W_chunk + offset
            m_chunk = dY_chunk * W_chunk_offset[None, :]
            if casting_mode_int != _CASTING_MODE_NONE:
                m_chunk = m_chunk.to(tl.float32)

            dot_product_acc += tl.sum(m_chunk * X_chunk, axis=1)

        rstd_vec_sq = rstd_vec * rstd_vec
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            dY_chunk = tl.load(
                dY_ptr + rows_off[:, None] * dY_row_stride + cols_off[None, :], mask=block_mask, other=0.0
            )
            X_chunk = tl.load(X_ptr + rows_off[:, None] * X_row_stride + cols_off[None, :], mask=block_mask, other=0.0)
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)

            W_chunk_offset = W_chunk + offset
            X_chunk_f32 = X_chunk.to(tl.float32)
            normed_X_chunk = X_chunk_f32 * rstd_vec[:, None]

            if casting_mode_int == _CASTING_MODE_LLAMA:
                m_chunk = (dY_chunk * W_chunk_offset[None, :]).to(tl.float32)
                dW_chunk_sum = tl.sum(dY_chunk * normed_X_chunk.to(X_dtype), axis=0)
            elif casting_mode_int == _CASTING_MODE_GEMMA:
                dY_chunk_f32 = dY_chunk.to(tl.float32)
                W_chunk_offset = W_chunk_offset.to(tl.float32)
                m_chunk = dY_chunk_f32 * W_chunk_offset[None, :]
                dW_chunk_sum = tl.sum(dY_chunk_f32 * normed_X_chunk, axis=0)
            else:
                m_chunk = dY_chunk * W_chunk_offset[None, :]
                dW_chunk_sum = tl.sum(dY_chunk * normed_X_chunk, axis=0)

            term1 = rstd_vec[:, None] * m_chunk
            term2 = -(1 / n_cols) * rstd_vec_sq[:, None] * rstd_vec[:, None] * dot_product_acc[:, None] * X_chunk_f32
            dX_chunk = term1 + term2

            tl.store(
                dX_ptr + rows_off[:, None] * dX_row_stride + cols_off[None, :], dX_chunk.to(X_dtype), mask=block_mask
            )

            tl.atomic_add(dW_ptr + cols_off, dW_chunk_sum, mask=cols_mask)


def rmsnorm_fwd_impl(
    X: torch.Tensor,
    W: torch.Tensor,
    eps: float,
    offset: float,
    casting_mode_int: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = X.shape
    dim = shape[-1]
    X_2d = X.reshape(-1, dim)
    n_rows, n_cols = X_2d.shape

    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = COL_BLOCKING_THRESHOLD
    else:
        BLOCK_SIZE_N = align(X, n_cols, VEC_ALIGN_BYTES)

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]

    grid = (num_programs,)
    Y = torch.empty_like(X_2d)

    rstd_dtype = torch.float32 if casting_mode_int in (0, 1) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    _rmsnorm_fwd_kernel[grid](
        Y,
        Y.stride(0),
        X_2d,
        X_2d.stride(0),
        W,
        RSTD,
        RSTD.stride(0),
        n_rows,
        n_cols,
        eps,
        offset,
        casting_mode_int=casting_mode_int,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        multibuffer=True,
    )

    Y = Y.reshape(*shape)

    return Y, RSTD


def rmsnorm_bwd_impl(
    dY: torch.Tensor,
    X: torch.Tensor,
    W: torch.Tensor,
    RSTD: torch.Tensor,
    offset: float,
    casting_mode_int: int,
    X_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = dY.shape
    dim = shape[-1]
    dY_2d = dY.reshape(-1, dim)
    X_2d = X.reshape(-1, dim)
    n_rows, n_cols = dY_2d.shape

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    X_dtype_triton = torch_to_triton_dtype[X_dtype]

    grid = (num_programs,)

    if n_cols <= COL_BLOCKING_THRESHOLD:
        _dW = torch.zeros((num_programs, n_cols), dtype=torch.float32, device=W.device)
    else:
        _dW = torch.zeros((1, n_cols), dtype=torch.float32, device=W.device)

    # if in_place:
    #     dX_2d = dY_2d
    # else:
    #     dX_2d = torch.empty_like(dY_2d)
    dX_2d = torch.empty_like(dY_2d)

    if n_cols <= COL_BLOCKING_THRESHOLD:
        _rmsnorm_bwd_kernel[grid](
            dY_2d,
            dY_2d.stride(0),
            dX_2d,
            dX_2d.stride(0),
            X_2d,
            X_2d.stride(0),
            W,
            RSTD,
            RSTD.stride(0),
            _dW,
            _dW.stride(0),
            n_rows,
            n_cols,
            offset,
            casting_mode_int,
            X_dtype_triton,
            BLOCK_SIZE_N=align(X_2d, n_cols, VEC_ALIGN_BYTES),
            multibuffer=True,
        )
        dW = _dW.sum(dim=0).to(W.dtype)
    else:
        _dW.zero_()
        _rmsnorm_bwd_large_cols_kernel[grid](
            dY_2d,
            dY_2d.stride(0),
            dX_2d,
            dX_2d.stride(0),
            X_2d,
            X_2d.stride(0),
            W,
            RSTD,
            RSTD.stride(0),
            _dW,
            _dW.stride(0),
            n_rows,
            n_cols,
            offset,
            casting_mode_int,
            X_dtype_triton,
            BLOCK_SIZE_N=COL_BLOCKING_THRESHOLD,
            BLOCK_SIZE_M=2,  # Empirical value
            multibuffer=True,
        )
        dW = _dW.squeeze(0).to(W.dtype)

    dX = dX_2d.reshape(*shape)

    return dX, dW
