from typing import Tuple

import torch
import triton
import triton.language as tl

from triton.runtime.libentry import libentry

from mojo_opset.backends.ttx.kernels.npu.utils import VEC_ALIGN_BYTES
from mojo_opset.backends.ttx.kernels.utils import align

"""
This file contains the implementation of Fused Add Layer Norm for NPU.

This op supports two modes for residual addition based on the user's definition:
1. 'pre':
    - S = X + R
    - Y = layernorm(S)
    - Returns (Y, S). Y is the input to the next sublayer, S is the new residual.

2. 'post':
    - S = X + R
    - Y = layernorm(S)
    - Returns (Y, Y). Y is used as both the new hidden state and the new residual.

The core computation kernel is identical for both modes; the difference lies in the
return values and gradient flow handled by the autograd.Function.

Modifications for NPU architecture and updated 'post' mode by triton-x team, 2025.
"""

COL_BLOCKING_THRESHOLD = 4096


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1}),
        triton.Config({"BLOCK_SIZE_M": 2}),
        triton.Config({"BLOCK_SIZE_M": 4}),
        triton.Config({"BLOCK_SIZE_M": 8}),
    ],
    key=["n_cols"],
)
@libentry()
@triton.jit
def _fused_add_layernorm_fwd_kernel(
    Y_ptr,
    Y_row_stride,
    S_ptr,
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    W_ptr,
    B_ptr,
    Mean_ptr,
    Mean_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows,
    n_cols,
    eps,
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
        R_ptr_row_block = R_ptr + rows_off[:, None] * R_row_stride
        S_ptr_row_block = S_ptr + rows_off[:, None] * S_row_stride
        Y_ptr_row_block = Y_ptr + rows_off[:, None] * Y_row_stride

        mean_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        var_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            block_mask = rows_mask[:, None] & (cols_off[None, :] < n_cols)

            X_chunk = tl.load(X_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            R_chunk = tl.load(R_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            S_chunk = X_chunk + R_chunk
            tl.store(S_ptr_row_block + cols_off[None, :], S_chunk, mask=block_mask)

            S_chunk_f32 = S_chunk.to(tl.float32)
            mean_acc += tl.sum(S_chunk_f32, axis=1)
            var_acc += tl.sum(S_chunk_f32 * S_chunk_f32, axis=1)

        mean_vec = mean_acc / n_cols
        var_vec = (var_acc / n_cols) - (mean_vec * mean_vec)
        rstd_vec = tl.rsqrt(var_vec + eps)
        tl.store(Mean_ptr + rows_off * Mean_row_stride, mean_vec, mask=rows_mask)
        tl.store(RSTD_ptr + rows_off * RSTD_row_stride, rstd_vec, mask=rows_mask)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            S_chunk = tl.load(S_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0).to(tl.float32)
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)
            B_chunk = tl.load(B_ptr + cols_off, mask=cols_mask, other=0.0)

            normed_S_chunk = (S_chunk - mean_vec[:, None]) * rstd_vec[:, None]
            Y_chunk = normed_S_chunk * W_chunk[None, :] + B_chunk[None, :]
            tl.store(Y_ptr_row_block + cols_off[None, :], Y_chunk.to(Y_ptr.dtype.element_ty), mask=block_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1}),
        triton.Config({"BLOCK_SIZE_M": 4}),
        triton.Config({"BLOCK_SIZE_M": 8}),
    ],
    key=["n_cols"],
    restore_value=["dY_ptr", "dX_ptr", "dW_ptr", "dB_ptr"],
)
@libentry()
@triton.jit
def _fused_add_layernorm_bwd_kernel(
    dY_ptr,
    dY_row_stride,
    dS_out_ptr,
    dS_out_row_stride,
    dX_ptr,
    dX_row_stride,
    S_ptr,
    S_row_stride,
    W_ptr,
    Mean_ptr,
    RSTD_ptr,
    dW_ptr,
    dW_row_stride,
    dB_ptr,
    dB_row_stride,
    n_rows,
    n_cols,
    has_dS_out: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    dW_acc_ptr = dW_ptr + pid * dW_row_stride
    dB_acc_ptr = dB_ptr + pid * dB_row_stride

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        mean_vec = tl.load(Mean_ptr + rows_off, mask=rows_mask, other=0.0).to(tl.float32)
        rstd_vec = tl.load(RSTD_ptr + rows_off, mask=rows_mask, other=0.0).to(tl.float32)

        S_ptr_row_block = S_ptr + rows_off[:, None] * S_row_stride
        dY_ptr_row_block = dY_ptr + rows_off[:, None] * dY_row_stride
        dS_out_ptr_row_block = dS_out_ptr + rows_off[:, None] * dS_out_row_stride if has_dS_out else None

        ds_dx_hat_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        ds_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            S_chunk = tl.load(S_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0).to(tl.float32)
            dY_chunk = tl.load(dY_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0).to(tl.float32)
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

            S_hat_chunk = (S_chunk - mean_vec[:, None]) * rstd_vec[:, None]
            dY_W_chunk = dY_chunk * W_chunk[None, :]

            ds_dx_hat_acc += tl.sum(dY_W_chunk * S_hat_chunk, axis=1)
            ds_acc += tl.sum(dY_W_chunk, axis=1)

            dW_chunk_acc = tl.sum(dY_chunk * S_hat_chunk, axis=0)
            dB_chunk_acc = tl.sum(dY_chunk, axis=0)
            tl.atomic_add(dW_acc_ptr + cols_off, dW_chunk_acc, mask=cols_mask)
            tl.atomic_add(dB_acc_ptr + cols_off, dB_chunk_acc, mask=cols_mask)

        dX_ptr_row_block = dX_ptr + rows_off[:, None] * dX_row_stride
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            S_chunk = tl.load(S_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0).to(tl.float32)
            dY_chunk = tl.load(dY_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0).to(tl.float32)
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

            S_hat_chunk = (S_chunk - mean_vec[:, None]) * rstd_vec[:, None]
            dY_W_chunk = dY_chunk * W_chunk[None, :]

            grad_of_norm_input = (
                dY_W_chunk - (S_hat_chunk * ds_dx_hat_acc[:, None] + ds_acc[:, None]) / n_cols
            ) * rstd_vec[:, None]

            dS_block = grad_of_norm_input
            if has_dS_out:
                dS_out_chunk = tl.load(dS_out_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
                dS_block += dS_out_chunk.to(dS_block.dtype)

            tl.store(dX_ptr_row_block + cols_off[None, :], dS_block.to(dX_ptr.dtype.element_ty), mask=block_mask)


class TTXFusedAddLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, R, W, B, add_mode, eps, in_place):
        shape = X.shape
        dim = shape[-1]
        X_2d = X.reshape(-1, dim)
        R_2d = R.reshape(-1, dim)
        n_rows, n_cols = X_2d.shape

        BLOCK_SIZE_N = align(X_2d, n_cols, VEC_ALIGN_BYTES)
        num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
        grid = (num_programs,)

        ctx.add_mode = add_mode
        ctx.in_place = in_place

        Mean = torch.empty(n_rows, dtype=torch.float32, device=X.device)
        RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)

        Y = torch.empty_like(X_2d)
        S = torch.empty_like(X_2d)

        _fused_add_layernorm_fwd_kernel[grid](
            Y,
            Y.stride(0),
            S,
            S.stride(0),
            X_2d,
            X_2d.stride(0),
            R_2d,
            R_2d.stride(0),
            W,
            B,
            Mean,
            Mean.stride(0),
            RSTD,
            RSTD.stride(0),
            n_rows,
            n_cols,
            eps,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        ctx.save_for_backward(S.reshape(-1, dim), W, B, Mean, RSTD)

        if add_mode == "pre":
            return Y.reshape(*shape), S.reshape(*shape)
        elif add_mode == "post":
            return Y.reshape(*shape), Y.reshape(*shape)
        else:
            raise ValueError(f"Invalid add_mode: '{add_mode}'. Must be 'pre' or 'post'.")

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output1, grad_output2 = grad_outputs

        if ctx.add_mode == "pre":
            dY, dS_out = grad_output1, grad_output2
        else:
            dY = grad_output1 + grad_output2
            dS_out = None

        S_2d, W, B, Mean, RSTD = ctx.saved_tensors

        shape = dY.shape
        dim = shape[-1]
        dY_2d = dY.reshape(-1, dim)
        n_rows, n_cols = dY_2d.shape

        has_dS_out = dS_out is not None
        dS_out_2d = dS_out.reshape(-1, dim) if has_dS_out else torch.empty((0, 0), device=dY.device)

        num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
        grid = (num_programs,)

        _dW = torch.zeros((num_programs, n_cols), dtype=torch.float32, device=W.device)
        _dB = torch.zeros((num_programs, n_cols), dtype=torch.float32, device=B.device)
        dX_2d = torch.empty_like(dY_2d)

        BLOCK_SIZE_N = align(S_2d, n_cols, VEC_ALIGN_BYTES)

        _fused_add_layernorm_bwd_kernel[grid](
            dY_2d,
            dY_2d.stride(0),
            dS_out_2d,
            dS_out_2d.stride(0),
            dX_2d,
            dX_2d.stride(0),
            S_2d,
            S_2d.stride(0),
            W,
            Mean,
            RSTD,
            _dW,
            _dW.stride(0),
            _dB,
            _dB.stride(0),
            n_rows,
            n_cols,
            has_dS_out=has_dS_out,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        dW = _dW.sum(0).to(W.dtype)
        dB = _dB.sum(0).to(B.dtype)

        dX = dX_2d.reshape(*shape)
        dR = dX.clone()

        return dX, dR, dW, dB, None, None, None


def ttx_fused_add_layer_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    add_mode: str = "pre",
    eps: float = 1e-5,
    in_place: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TTX Fused Add Layer Norm function.

    Performs fused residual addition and Layer Normalization.
    Supports two modes via the `add_mode` parameter:

    1. `add_mode="pre"` (default):
        - `S = hidden_states + residual`
        - `Y = layernorm(S, weight, bias)`
        - Returns a tuple: `(Y, S)`.

    2. `add_mode="post"`:
        - `S = hidden_states + residual`
        - `Y = layernorm(S, weight, bias)`
        - Returns a tuple: `(Y, Y)`.

    Args:
        hidden_states: Input tensor.
        residual: Residual tensor of the same shape.
        weight: Gamma weights for LayerNorm of shape (H,).
        bias: Beta biases for LayerNorm of shape (H,).
        add_mode: The mode of residual addition, "pre" or "post". Default: "pre".
        eps: Small value for numerical stability. Default: 1e-5.
        in_place: Whether to use in-place operations. Not fully supported. Default: False.

    Returns:
        A tuple `(output, new_residual)`.
    """
    return TTXFusedAddLayerNormFunction.apply(hidden_states, residual, weight, bias, add_mode, eps, in_place)
