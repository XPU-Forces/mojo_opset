from typing import Tuple

import torch
import triton
import triton.language as tl

from .utils import COL_BLOCKING_THRESHOLD
from .utils import VEC_ALIGN_BYTES
from .utils import ilu_grid_dim_from_row_tasks
from .utils import libentry
from .utils import rms_norm_fwd_heuristics
from mojo_opset.backends.ttx.kernels.utils import align
from mojo_opset.backends.ttx.kernels.utils import ceil_div
from mojo_opset.backends.ttx.kernels.utils import torch_to_triton_dtype
from mojo_opset.utils.misc import get_bool_env

IS_DETERMINISTIC = get_bool_env("MOJO_DETERMINISTIC", default=False)


def _rmsnorm_fwd_grid_n_programs(n_rows: int, n_cols: int) -> int:
    block_m = rms_norm_fwd_heuristics({"n_cols": n_cols})
    n_tasks = triton.cdiv(n_rows, block_m)
    return ilu_grid_dim_from_row_tasks(n_tasks)


@triton.heuristics({"BLOCK_SIZE_M": rms_norm_fwd_heuristics})
# @libentry()
@triton.jit
def _rmsnorm_infer_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    eps,
    N_COLS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    task_mask = pid < num_row_tasks

    block_start_row = pid * BLOCK_SIZE_M

    current_row_offsets = block_start_row + tl.arange(0, BLOCK_SIZE_M)
    row_mask = task_mask & (current_row_offsets < n_rows)

    ss_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for col_offset in range(0, N_COLS, BLOCK_SIZE_N):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_offsets < N_COLS

        x_ptrs = X_ptr + (current_row_offsets[:, None] * stride_x_row + col_offsets[None, :])

        x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)

        ss_acc += tl.sum(x * x, axis=1)

    ss_acc = tl.where(row_mask, ss_acc, 0)

    mean_square = ss_acc / N_COLS
    rrms = tl.rsqrt(mean_square + eps)

    rrms = tl.where(row_mask, rrms, 0.0)

    for col_offset in range(0, N_COLS, BLOCK_SIZE_N):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_offsets < N_COLS

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
            y,
            mask=row_mask[:, None] & col_mask[None, :],
        )


def rmsnorm_infer_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    assert x.size(-1) == w.size(-1)
    shape = x.shape
    dim = shape[-1]
    X_2d = x.reshape(-1, dim)
    n_rows, n_cols = X_2d.shape

    y = torch.empty_like(X_2d)

    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = COL_BLOCKING_THRESHOLD
    else:
        BLOCK_SIZE_N = align(x, n_cols, VEC_ALIGN_BYTES)

    grid = (_rmsnorm_fwd_grid_n_programs(n_rows, n_cols),)

    _rmsnorm_infer_kernel[grid](
        x,
        y,
        w,
        X_2d.stride(0),
        y.stride(0),
        n_rows=n_rows,
        eps=eps,
        N_COLS=n_cols,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return y.reshape(*shape)


@triton.heuristics({"BLOCK_SIZE_M": rms_norm_fwd_heuristics})
# @libentry()
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
    eps,
    offset,
    casting_mode_int: tl.constexpr,
    X_dtype: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    task_mask = pid < num_row_tasks

    block_start_row = pid * BLOCK_SIZE_M
    rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
    rows_mask = task_mask & (rows_off < n_rows)

    X_ptr_row_block = X_ptr + rows_off[:, None] * X_row_stride

    var_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for col_offset in range(0, N_COLS, BLOCK_SIZE_N):
        cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
        cols_mask = cols_off < N_COLS
        block_mask = rows_mask[:, None] & cols_mask[None, :]

        X_chunk = tl.load(X_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0).to(tl.float32)
        var_acc += tl.sum(X_chunk * X_chunk, axis=1)

    var = var_acc / N_COLS
    rstd_vec = tl.rsqrt(var + eps)
    tl.store(RSTD_ptr + rows_off * RSTD_row_stride, rstd_vec, mask=rows_mask)

    Y_ptr_row_block = Y_ptr + rows_off[:, None] * Y_row_stride
    for col_offset in range(0, N_COLS, BLOCK_SIZE_N):
        cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
        cols_mask = cols_off < N_COLS
        block_mask = rows_mask[:, None] & cols_mask[None, :]

        X_chunk = tl.load(X_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
        W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)

        if casting_mode_int == 1:  # GEMMA
            X_chunk = X_chunk.to(tl.float32)
            W_chunk = W_chunk.to(tl.float32)
        elif casting_mode_int == 0:  # LLAMA
            X_chunk = X_chunk.to(tl.float32)

        if casting_mode_int == 0:  # LLAMA
            normed_X_chunk = (X_chunk * rstd_vec[:, None]).to(X_dtype)
        else:
            normed_X_chunk = X_chunk * rstd_vec[:, None]

        Y_chunk = normed_X_chunk * (W_chunk[None, :] + offset)
        if casting_mode_int == 1:  # GEMMA
            Y_chunk = Y_chunk.to(X_dtype)

        tl.store(Y_ptr_row_block + cols_off[None, :], Y_chunk, mask=block_mask)


@triton.heuristics({"BLOCK_SIZE_M": lambda args: ceil_div(4096, args.get("n_cols", args.get("N_COLS")))})
# @libentry()
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
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    task_mask = pid < num_row_tasks

    dW_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    cols_off = tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols_off < n_cols
    W_row = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)
    W_row_offset = W_row + offset

    block_start_row = pid * BLOCK_SIZE_M

    rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
    rows_mask = task_mask & (rows_off < n_rows)
    block_mask = rows_mask[:, None] & cols_mask[None, :]

    dY_block = tl.load(dY_ptr + rows_off[:, None] * dY_row_stride + cols_off[None, :], mask=block_mask, other=0.0)
    X_block = tl.load(X_ptr + rows_off[:, None] * X_row_stride + cols_off[None, :], mask=block_mask, other=0.0)
    rstd_vec = tl.load(RSTD_ptr + rows_off * RSTD_row_stride, mask=rows_mask, other=0.0)

    X_block_f32 = X_block.to(tl.float32)
    normed_X_block = X_block_f32 * rstd_vec[:, None]

    if casting_mode_int == 0:  # LLAMA
        m_block = (dY_block * W_row_offset[None, :]).to(tl.float32)
        dW_acc += tl.sum(dY_block * normed_X_block.to(X_dtype), axis=0)
    elif casting_mode_int == 1:  # GEMMA
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


# @libentry()
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
    offset,
    casting_mode_int: tl.constexpr,
    X_dtype: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    IS_DETERMINISTIC: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    task_mask = pid < num_row_tasks

    block_start_row = pid * BLOCK_SIZE_M
    rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
    rows_mask = task_mask & (rows_off < n_rows)

    rstd_vec = tl.load(RSTD_ptr + rows_off * RSTD_row_stride, mask=rows_mask, other=0.0)

    dot_product_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for col_offset in range(0, N_COLS, BLOCK_SIZE_N):
        cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
        cols_mask = cols_off < N_COLS
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
        if casting_mode_int != -1:  # NONE
            m_chunk = m_chunk.to(tl.float32)

        dot_product_acc += tl.sum(m_chunk * X_chunk, axis=1)

    rstd_vec_sq = rstd_vec * rstd_vec
    for col_offset in range(0, N_COLS, BLOCK_SIZE_N):
        cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
        cols_mask = cols_off < N_COLS
        block_mask = rows_mask[:, None] & cols_mask[None, :]

        dY_chunk = tl.load(
            dY_ptr + rows_off[:, None] * dY_row_stride + cols_off[None, :], mask=block_mask, other=0.0
        )
        X_chunk = tl.load(X_ptr + rows_off[:, None] * X_row_stride + cols_off[None, :], mask=block_mask, other=0.0)
        W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)

        W_chunk_offset = W_chunk + offset
        X_chunk_f32 = X_chunk.to(tl.float32)
        normed_X_chunk = X_chunk_f32 * rstd_vec[:, None]

        if casting_mode_int == 0:  # LLAMA
            m_chunk = (dY_chunk * W_chunk_offset[None, :]).to(tl.float32)
            dW_chunk_sum = tl.sum(dY_chunk * normed_X_chunk.to(X_dtype), axis=0)
        elif casting_mode_int == 1:  # GEMMA
            dY_chunk_f32 = dY_chunk.to(tl.float32)
            W_chunk_offset = W_chunk_offset.to(tl.float32)
            m_chunk = dY_chunk_f32 * W_chunk_offset[None, :]
            dW_chunk_sum = tl.sum(dY_chunk_f32 * normed_X_chunk, axis=0)
        else:
            m_chunk = dY_chunk * W_chunk_offset[None, :]
            dW_chunk_sum = tl.sum(dY_chunk * normed_X_chunk, axis=0)

        term1 = rstd_vec[:, None] * m_chunk
        term2 = -(1 / N_COLS) * rstd_vec_sq[:, None] * rstd_vec[:, None] * dot_product_acc[:, None] * X_chunk_f32
        dX_chunk = term1 + term2

        tl.store(
            dX_ptr + rows_off[:, None] * dX_row_stride + cols_off[None, :], dX_chunk.to(X_dtype), mask=block_mask
        )

        if IS_DETERMINISTIC:
            dW_existing = tl.load(dW_ptr + pid * dW_row_stride + cols_off, mask=cols_mask, other=0.0)
            tl.store(dW_ptr + pid * dW_row_stride + cols_off, dW_existing + dW_chunk_sum, mask=cols_mask)
        else:
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

    grid = (_rmsnorm_fwd_grid_n_programs(n_rows, n_cols),)
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
        eps,
        offset,
        casting_mode_int=casting_mode_int,
        X_dtype=torch_to_triton_dtype[X.dtype],
        N_COLS=n_cols,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
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

    X_dtype_triton = torch_to_triton_dtype[X_dtype]

    dX_2d = torch.empty_like(dY_2d)

    if n_cols <= COL_BLOCKING_THRESHOLD:
        n_programs = ilu_grid_dim_from_row_tasks(triton.cdiv(n_rows, ceil_div(4096, n_cols)))
        grid = (n_programs,)
        _dW = torch.zeros((n_programs, n_cols), dtype=torch.float32, device=W.device)
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
        )
        dW = _dW.sum(dim=0).to(W.dtype)
    else:
        n_programs = ilu_grid_dim_from_row_tasks(triton.cdiv(n_rows, 2))
        grid = (n_programs,)
        if IS_DETERMINISTIC:
            _dW = torch.zeros((n_programs, n_cols), dtype=torch.float32, device=W.device)
        else:
            _dW = torch.zeros((1, n_cols), dtype=torch.float32, device=W.device)

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
            offset,
            casting_mode_int,
            X_dtype_triton,
            N_COLS=n_cols,
            BLOCK_SIZE_N=COL_BLOCKING_THRESHOLD,
            BLOCK_SIZE_M=2,  # Empirical value
            IS_DETERMINISTIC=IS_DETERMINISTIC,
        )

        if IS_DETERMINISTIC:
            dW = _dW.sum(dim=0).to(W.dtype)
        else:
            dW = _dW.squeeze(0).to(W.dtype)

    dX = dX_2d.reshape(*shape)

    return dX, dW
