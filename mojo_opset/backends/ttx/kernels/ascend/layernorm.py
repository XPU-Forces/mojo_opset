import torch
import triton
import triton.language as tl

from triton.language.math import rsqrt
from triton.runtime.libentry import libentry

from mojo_opset.backends.ttx.kernels.ascend.utils import VEC_ALIGN_BYTES
from mojo_opset.backends.ttx.kernels.utils import align
from mojo_opset.backends.ttx.kernels.utils import torch_to_triton_dtype

"""
This file incorporates code from Liger Kernel licensed under the Apache License, Version 2.0.
See the original Liger Kernel repository at https://github.com/linkedin/Liger-Kernel.

Portions of this file are adapted from:
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/layer_norm.py

Modifications in this repository by triton-x team, 2025.
"""

COL_BLOCKING_THRESHOLD = 4096


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
def _layer_norm_fwd_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    B_ptr,
    Mean_ptr,
    RSTD_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, num_programs):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        # Pass 1: Compute mean
        sum_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            x_chunk = tl.load(
                X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0
            ).to(tl.float32)

            sum_acc += tl.sum(x_chunk, axis=1)

        mean = sum_acc / n_cols

        # Store mean
        tl.store(Mean_ptr + rows_off, mean, mask=rows_mask)

        # Pass 2: Compute variance and output
        var_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            # Load input data
            x_chunk = tl.load(
                X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0
            ).to(tl.float32)

            # Load weight and bias
            w_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)
            b_chunk = tl.load(B_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

            # Compute centered values
            x_centered = x_chunk - mean[:, None]

            # Accumulate variance
            var_acc += tl.sum(x_centered * x_centered, axis=1)

        # Compute rstd
        var = var_acc / n_cols
        rstd = rsqrt(var + eps)

        # Store rstd
        tl.store(RSTD_ptr + rows_off, rstd, mask=rows_mask)

        # Pass 3: Compute and store output
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            # Load input data
            x_chunk = tl.load(
                X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0
            ).to(tl.float32)

            # Load weight and bias
            w_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)
            b_chunk = tl.load(B_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

            # Compute output: (x - mean) * rstd * w + b
            x_centered = x_chunk - mean[:, None]
            y_chunk = x_centered * rstd[:, None] * w_chunk[None, :] + b_chunk[None, :]

            # Store output
            tl.store(
                Y_ptr + rows_off[:, None] * stride_y_row + cols_off[None, :],
                y_chunk.to(Y_ptr.dtype.element_ty),
                mask=block_mask,
            )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1, "multibuffer": True}),
        # triton.Config({"BLOCK_SIZE_M": 2, "multibuffer": True}),
        # triton.Config({"BLOCK_SIZE_M": 4, "multibuffer": True}),
        # triton.Config({"BLOCK_SIZE_M": 8, "multibuffer": True}),
        # triton.Config({"BLOCK_SIZE_M": 12, "multibuffer": True}),
        # triton.Config({"BLOCK_SIZE_M": 16, "multibuffer": True}),
        # triton.Config({"BLOCK_SIZE_M": 20, "multibuffer": True}),
        # triton.Config({"BLOCK_SIZE_M": 24, "multibuffer": True}),
        # triton.Config({"BLOCK_SIZE_M": 32, "multibuffer": True}),
    ],
    key=["n_rows", "n_cols"],
    # restore_value=["DY_ptr", "DX_ptr", "DW_ptr", "DB_ptr"],
)
@libentry()
@triton.jit
def _layer_norm_bwd_kernel(
    DY_ptr,
    DX_ptr,
    DW_ptr,
    DB_ptr,
    X_ptr,
    W_ptr,
    Mean_ptr,
    RSTD_ptr,
    stride_dy_row,
    stride_dx_row,
    stride_x_row,
    n_rows,
    n_cols,
    X_dtype: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    # Initialize local accumulators for weight and bias gradients
    cols_off = tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols_off < n_cols

    dW_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    dB_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    # Load weights
    w = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

    for row_task_id in range(pid, num_row_tasks, num_programs):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows
        block_mask = rows_mask[:, None] & cols_mask[None, :]

        # Load row-wise statistics
        mean = tl.load(Mean_ptr + rows_off, mask=rows_mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD_ptr + rows_off, mask=rows_mask, other=0.0).to(tl.float32)

        # Load input and output gradients
        dy = tl.load(DY_ptr + rows_off[:, None] * stride_dy_row + cols_off[None, :], mask=block_mask, other=0.0).to(
            tl.float32
        )

        x = tl.load(X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0).to(
            tl.float32
        )

        # Compute normalized input
        x_hat = (x - mean[:, None]) * rstd[:, None]

        # Accumulate weight and bias gradients
        dW_acc += tl.sum(dy * x_hat, axis=0)
        dB_acc += tl.sum(dy, axis=0)

        # Compute input gradients
        # dx = (w * dy - (x_hat * sum(w * dy * x_hat) + sum(w * dy)) / n_cols) * rstd
        wdy = w[None, :] * dy
        c1 = tl.sum(x_hat * wdy, axis=1) / n_cols
        c2 = tl.sum(wdy, axis=1) / n_cols
        dx = (wdy - (x_hat * c1[:, None] + c2[:, None])) * rstd[:, None]

        # Store input gradients
        tl.store(DX_ptr + rows_off[:, None] * stride_dx_row + cols_off[None, :], dx.to(X_dtype), mask=block_mask)

    # Store weight and bias gradients using atomic operations
    tl.atomic_add(DW_ptr + cols_off, dW_acc, mask=cols_mask)
    tl.atomic_add(DB_ptr + cols_off, dB_acc, mask=cols_mask)


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
    restore_value=["DY_ptr", "DX_ptr", "DW_ptr", "DB_ptr"],
)
@libentry()
@triton.jit
def _layer_norm_bwd_large_cols_kernel(
    DY_ptr,
    DX_ptr,
    DW_ptr,
    DB_ptr,
    X_ptr,
    W_ptr,
    Mean_ptr,
    RSTD_ptr,
    stride_dy_row,
    stride_dx_row,
    stride_x_row,
    n_rows,
    n_cols,
    X_dtype: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, num_programs):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        # Load row-wise statistics
        mean = tl.load(Mean_ptr + rows_off, mask=rows_mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD_ptr + rows_off, mask=rows_mask, other=0.0).to(tl.float32)

        # Compute coefficients for input gradient (requires full row)
        c1_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        c2_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        # First pass: accumulate coefficients
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            # Load data
            dy = tl.load(DY_ptr + rows_off[:, None] * stride_dy_row + cols_off[None, :], mask=block_mask, other=0.0).to(
                tl.float32
            )

            x = tl.load(X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0).to(
                tl.float32
            )

            w = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

            # Compute normalized input
            x_hat = (x - mean[:, None]) * rstd[:, None]
            wdy = w[None, :] * dy

            # Accumulate coefficients
            c1_acc += tl.sum(x_hat * wdy, axis=1)
            c2_acc += tl.sum(wdy, axis=1)

        c1 = c1_acc / n_cols
        c2 = c2_acc / n_cols

        # Second pass: compute gradients
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            # Load data
            dy = tl.load(DY_ptr + rows_off[:, None] * stride_dy_row + cols_off[None, :], mask=block_mask, other=0.0).to(
                tl.float32
            )

            x = tl.load(X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0).to(
                tl.float32
            )

            w = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

            # Compute gradients
            x_hat = (x - mean[:, None]) * rstd[:, None]
            wdy = w[None, :] * dy

            # Weight and bias gradients
            dW_chunk = tl.sum(dy * x_hat, axis=0)
            dB_chunk = tl.sum(dy, axis=0)

            # Input gradients
            dx = (wdy - (x_hat * c1[:, None] + c2[:, None])) * rstd[:, None]

            # Store gradients
            tl.store(DX_ptr + rows_off[:, None] * stride_dx_row + cols_off[None, :], dx.to(X_dtype), mask=block_mask)

            tl.atomic_add(DW_ptr + cols_off, dW_chunk, mask=cols_mask)
            tl.atomic_add(DB_ptr + cols_off, dB_chunk, mask=cols_mask)


def layer_norm_fwd(x, w, b, eps):
    shape = x.shape
    dim = shape[-1]
    x_2d = x.reshape(-1, dim)
    n_rows, n_cols = x_2d.shape

    # Determine block size using the standard pattern
    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = 2048
    else:
        BLOCK_SIZE_N = align(x, n_cols, VEC_ALIGN_BYTES)

    # Get number of programs
    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    # Create output tensors
    y = torch.empty_like(x_2d)
    mean = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    rstd = torch.empty(n_rows, dtype=x.dtype, device=x.device)

    # Launch kernel
    _layer_norm_fwd_kernel[grid](
        x_2d,
        y,
        w,
        b,
        mean,
        rstd,
        x_2d.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return y.reshape(*shape), x_2d, mean, rstd


def layer_norm_bwd(dy, x_2d, w, b, mean, rstd):
    shape = dy.shape
    dim = shape[-1]
    dy_2d = dy.reshape(-1, dim)
    n_rows, n_cols = dy_2d.shape

    # Determine block size using the standard pattern
    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = 2048
    else:
        BLOCK_SIZE_N = align(x_2d, n_cols, VEC_ALIGN_BYTES)

    # Get number of programs
    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    # Create output tensors
    dx = torch.empty_like(dy_2d)

    if n_cols <= COL_BLOCKING_THRESHOLD:
        # For small columns, use per-program accumulation
        dw = torch.zeros(n_cols, dtype=torch.float32, device=w.device)
        db = torch.zeros(n_cols, dtype=torch.float32, device=b.device)

        # Launch kernel
        _layer_norm_bwd_kernel[grid](
            dy_2d,
            dx,
            dw,
            db,
            x_2d,
            w,
            mean,
            rstd,
            dy_2d.stride(0),
            dx.stride(0),
            x_2d.stride(0),
            n_rows,
            n_cols,
            torch_to_triton_dtype[x_2d.dtype],
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
    else:
        # For large columns, use atomic operations
        dw = torch.zeros(n_cols, dtype=torch.float32, device=w.device)
        db = torch.zeros(n_cols, dtype=torch.float32, device=b.device)

        # Launch kernel
        _layer_norm_bwd_large_cols_kernel[grid](
            dy_2d,
            dx,
            dw,
            db,
            x_2d,
            w,
            mean,
            rstd,
            dy_2d.stride(0),
            dx.stride(0),
            x_2d.stride(0),
            n_rows,
            n_cols,
            torch_to_triton_dtype[x_2d.dtype],
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

    # Convert gradients back to original dtype
    dw = dw.to(w.dtype)
    db = db.to(b.dtype)

    return dx.reshape(*shape), dw, db


class TTXLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, eps):
        """
        x: input tensor of any shape
        w: weight tensor of shape (hidden_size,)
        b: bias tensor of shape (hidden_size,)
        eps: small constant for numerical stability
        """
        y, x_2d, mean, rstd = layer_norm_fwd(x, w, b, eps)
        ctx.save_for_backward(x_2d, w, b, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        """
        dy: gradient of output
        """
        x_2d, w, b, mean, rstd = ctx.saved_tensors
        dx, dw, db = layer_norm_bwd(dy, x_2d, w, b, mean, rstd)
        return dx, dw, db, None


def ttx_layer_norm(x, weight, bias, eps=1e-6):
    """
    TTX LayerNorm activation function for inference.

    Implements: y = (x - mean) / sqrt(var + eps) * weight + bias

    Args:
        x: Input tensor of any shape
        weight: Weight tensor of shape (hidden_size,)
        bias: Bias tensor of shape (hidden_size,)
        eps: Small constant for numerical stability. Default: 1e-6

    Returns:
        Output tensor with same shape as input
    """
    return TTXLayerNormFunction.apply(x, weight, bias, eps)
