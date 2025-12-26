from typing import Tuple

import torch
import triton
import triton.language as tl

from triton.runtime.libentry import libentry

from mojo_opset.backends.ttx.kernels.npu.utils import VEC_ALIGN_BYTES
from mojo_opset.backends.ttx.kernels.utils import align
from mojo_opset.backends.ttx.kernels.utils import torch_to_triton_dtype

"""
This file contains the implementation of Fused Add RMS Norm for NPU.

This op supports two modes for residual addition based on the user's definition:
1. 'pre':
    - S = X + R
    - Y = rmsnorm(S)
    - Returns (Y, S). Y is the input to the next sublayer, S is the new residual.

2. 'post':
    - S = X + R
    - Y = rmsnorm(S)
    - Returns (Y, Y). Y is both the input to the next sublayer and the new residual.

The core computation kernel is identical for both modes; the difference lies in the
return values and gradient flow handled by the autograd.Function.

Original 'pre' mode implementation based on Liger Kernel:
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_add_rms_norm.py

Modifications for NPU architecture and updated 'post' mode by triton-x team, 2025.
"""

COL_BLOCKING_THRESHOLD = 4096

_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 2, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 4, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 8, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 12, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 16, "multibuffer": True}),
    ],
    key=["n_cols"],
)
@libentry()
@triton.jit
def _fused_add_rms_norm_fwd_kernel(
    Y_ptr,
    Y_row_stride,
    S_ptr,
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
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

        var_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            block_mask = rows_mask[:, None] & (cols_off[None, :] < n_cols)

            X_chunk = tl.load(X_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            R_chunk = tl.load(R_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            S_chunk = X_chunk + R_chunk
            tl.store(S_ptr_row_block + cols_off[None, :], S_chunk, mask=block_mask)

            S_chunk_f32 = S_chunk.to(tl.float32)
            var_acc += tl.sum(S_chunk_f32 * S_chunk_f32, axis=1)

        var = var_acc / n_cols
        rstd_vec = tl.rsqrt(var + eps)
        tl.store(RSTD_ptr + rows_off * RSTD_row_stride, rstd_vec, mask=rows_mask)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            S_chunk = tl.load(S_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)

            if casting_mode == _CASTING_MODE_GEMMA:
                S_chunk = S_chunk.to(tl.float32)
                W_chunk = W_chunk.to(tl.float32)
            elif casting_mode == _CASTING_MODE_LLAMA:
                S_chunk = S_chunk.to(tl.float32)

            if casting_mode == _CASTING_MODE_LLAMA:
                normed_S_chunk = (S_chunk * rstd_vec[:, None]).to(S_ptr.dtype.element_ty)
            else:
                normed_S_chunk = S_chunk * rstd_vec[:, None]

            Y_chunk = normed_S_chunk * (W_chunk[None, :] + offset)

            if casting_mode == _CASTING_MODE_GEMMA:
                Y_chunk = Y_chunk.to(S_ptr.dtype.element_ty)

            tl.store(Y_ptr_row_block + cols_off[None, :], Y_chunk, mask=block_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 4, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 8, "multibuffer": True}),
    ],
    key=["n_cols"],
    restore_value=["dY_ptr", "dS_out_ptr", "dX_ptr", "dW_ptr"],
)
@libentry()
@triton.jit
def _fused_add_rms_norm_bwd_kernel(
    dY_ptr,
    dY_row_stride,
    dS_out_ptr,
    dS_out_row_stride,
    dX_ptr,
    dX_row_stride,
    S_ptr,
    S_row_stride,
    W_ptr,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    casting_mode: tl.constexpr,
    S_dtype: tl.constexpr,
    has_dS_out: tl.constexpr,
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
        S_block = tl.load(S_ptr + rows_off[:, None] * S_row_stride + cols_off[None, :], mask=block_mask, other=0.0)
        rstd_vec = tl.load(RSTD_ptr + rows_off * RSTD_row_stride, mask=rows_mask, other=0.0)

        S_block_f32 = S_block.to(tl.float32)
        normed_S_block = S_block_f32 * rstd_vec[:, None]

        if casting_mode == _CASTING_MODE_LLAMA:
            m_block = (dY_block * W_row_offset[None, :]).to(tl.float32)
            dW_acc += tl.sum(dY_block * normed_S_block.to(S_dtype), axis=0)
        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_block_f32 = dY_block.to(tl.float32)
            W_row_offset_f32 = W_row_offset.to(tl.float32)
            m_block = dY_block_f32 * W_row_offset_f32[None, :]
            dW_acc += tl.sum(dY_block_f32 * normed_S_block, axis=0)
        else:
            m_block = dY_block * W_row_offset[None, :]
            dW_acc += tl.sum(dY_block * normed_S_block, axis=0)

        dot_product_vec = tl.sum(m_block * S_block_f32, axis=1)
        rstd_vec_sq = rstd_vec * rstd_vec

        term1 = rstd_vec[:, None] * m_block
        term2 = -(1 / n_cols) * rstd_vec_sq[:, None] * rstd_vec[:, None] * dot_product_vec[:, None] * S_block_f32

        grad_after_norm = term1 + term2

        dS_block = grad_after_norm
        if has_dS_out:
            dS_out_block = tl.load(
                dS_out_ptr + rows_off[:, None] * dS_out_row_stride + cols_off[None, :], mask=block_mask, other=0.0
            )
            dS_block += dS_out_block.to(dS_block.dtype)

        tl.store(dX_ptr + rows_off[:, None] * dX_row_stride + cols_off[None, :], dS_block.to(S_dtype), mask=block_mask)

    dW_ptr_prog = dW_ptr + pid * dW_row_stride + cols_off
    tl.store(dW_ptr_prog, dW_acc, mask=cols_mask)


class TTXFusedAddRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, R, W, add_mode, eps, offset, casting_mode, in_place):
        shape = X.shape
        dim = shape[-1]
        X_2d = X.reshape(-1, dim)
        R_2d = R.reshape(-1, dim)
        n_rows, n_cols = X_2d.shape

        if n_cols > COL_BLOCKING_THRESHOLD:
            BLOCK_SIZE_N = 2048
        else:
            BLOCK_SIZE_N = align(X, n_cols, VEC_ALIGN_BYTES)

        num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
        grid = (num_programs,)

        str_to_casting_mode = {"llama": 0, "gemma": 1, "none": -1}
        _casting_mode = str_to_casting_mode[casting_mode]

        ctx.add_mode = add_mode
        ctx.offset = offset
        ctx.in_place = in_place
        ctx.casting_mode_int = _casting_mode
        ctx.S_dtype_triton = torch_to_triton_dtype.get(X.dtype)

        rstd_dtype = torch.float32 if _casting_mode in (0, 1) else X.dtype
        RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

        Y = torch.empty_like(X_2d)
        S = torch.empty_like(X_2d)

        _fused_add_rms_norm_fwd_kernel[grid](
            Y,
            Y.stride(0),
            S,
            S.stride(0),
            X_2d,
            X_2d.stride(0),
            R_2d,
            R_2d.stride(0),
            W,
            RSTD,
            RSTD.stride(0),
            n_rows,
            n_cols,
            eps,
            offset,
            casting_mode=_casting_mode,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        ctx.save_for_backward(S.reshape(-1, dim), W, RSTD)

        if add_mode == "pre":
            return Y.reshape(*shape), S.reshape(*shape)
        elif add_mode == "post":
            return Y.reshape(*shape), Y.reshape(*shape)
        else:
            raise ValueError(f"Invalid add_mode: {add_mode}. Must be 'pre' or 'post'.")

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output1, grad_output2 = grad_outputs

        if ctx.add_mode == "pre":
            dY = grad_output1
            dS_out = grad_output2
        else:
            dY = grad_output1 + grad_output2
            dS_out = None

        S_2d, W, RSTD = ctx.saved_tensors

        shape = dY.shape
        dim = shape[-1]
        dY_2d = dY.reshape(-1, dim)
        n_rows, n_cols = dY_2d.shape

        has_dS_out = dS_out is not None
        dS_out_2d = dS_out.reshape(-1, dim) if has_dS_out else torch.empty((0, 0), device=dY.device, dtype=dY.dtype)

        num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
        grid = (num_programs,)

        if n_cols <= COL_BLOCKING_THRESHOLD:
            _dW = torch.zeros((num_programs, n_cols), dtype=torch.float32, device=W.device)
            BLOCK_SIZE_N = align(S_2d, n_cols, VEC_ALIGN_BYTES)
        else:
            _dW = torch.zeros((1, n_cols), dtype=torch.float32, device=W.device)
            BLOCK_SIZE_N = 2048

        dX_2d = torch.empty_like(dY_2d)

        _fused_add_rms_norm_bwd_kernel[grid](
            dY_2d,
            dY_2d.stride(0),
            dS_out_2d,
            dS_out_2d.stride(0),
            dX_2d,
            dX_2d.stride(0),
            S_2d,
            S_2d.stride(0),
            W,
            RSTD,
            RSTD.stride(0),
            _dW,
            _dW.stride(0),
            n_rows,
            n_cols,
            ctx.offset,
            casting_mode=ctx.casting_mode_int,
            S_dtype=ctx.S_dtype_triton,
            has_dS_out=has_dS_out,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        dW = _dW.sum(dim=0).to(W.dtype)

        dX = dX_2d.reshape(*shape)
        dR = dX.clone()

        return dX, dR, dW, None, None, None, None, None


def ttx_fused_add_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    add_mode: str = "pre",
    eps: float = 1e-6,
    offset: float = 0.0,
    casting_mode: str = "llama",
    in_place: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TTX Fused Add RMS Norm function.

    This function performs a fused operation combining residual addition with RMS normalization.
    It supports two modes via the `add_mode` parameter, based on common transformer patterns:

    1. `add_mode="pre"` (default, e.g., Llama):
        - `S = hidden_states + residual`
        - `Y = rmsnorm(S)`
        - Returns a tuple: `(Y, S)`. `Y` is the normalized output, `S` is the new residual.

    2. `add_mode="post"` (e.g., as defined by user):
        - `S = hidden_states + residual`
        - `Y = rmsnorm(S)`
        - Returns a tuple: `(Y, Y)`. `Y` is used as both the new hidden state and the new residual.

    Args:
        hidden_states: Input tensor of shape (B, T, H) or (BT, H).
        residual: Residual tensor of the same shape as hidden_states.
        weight: Weight tensor for RMS norm of shape (H,).
        add_mode: The mode of residual addition, either "pre" or "post". Default: "pre".
        eps: Small value for numerical stability. Default: 1e-6.
        offset: Offset value for the weight tensor. Default: 0.0.
        casting_mode: Precision casting mode ("llama", "gemma", "none"). Default: "llama".
        in_place: Whether to use in-place operations. Default: False.

    Returns:
        A tuple `(output, new_residual)`. The content depends on `add_mode`.
    """
    return TTXFusedAddRMSNormFunction.apply(
        hidden_states, residual, weight, add_mode, eps, offset, casting_mode, in_place
    )
