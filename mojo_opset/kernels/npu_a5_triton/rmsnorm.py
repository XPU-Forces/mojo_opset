from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import libentry

from mojo_opset.kernels._npu_triton_utils import VEC_ALIGN_BYTES
from mojo_opset.kernels._triton_utils import align
from mojo_opset.kernels._triton_utils import ceil_div
from mojo_opset.kernels._triton_utils import torch_to_triton_dtype
COL_BLOCKING_THRESHOLD = 10240
COL_BLOCKING_BWD_THRESHOLD = 2048

_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)

TOKEN_BLOCK_SIZE_TABLE = {
    10240: 2,
    8192: 2,
    4096: 6,
    2048: 8,
    1024: 10,
    512: 18,
    256: 24,
    128: 48,
}


def rms_norm_fwd_heuristics(args):
    hidden_dim = args["n_cols"]
    if hidden_dim <= COL_BLOCKING_THRESHOLD:
        if hidden_dim in TOKEN_BLOCK_SIZE_TABLE:
            return TOKEN_BLOCK_SIZE_TABLE[hidden_dim]

        for dim_thresh, block_size in sorted(TOKEN_BLOCK_SIZE_TABLE.items()):
            if hidden_dim <= dim_thresh:
                return block_size
        return 1
    else:
        return 1


def rms_norm_bwd_heuristics(args):
    rows_per_task = ceil_div(4096, args["n_cols"])
    active_rows = triton.next_power_of_2(max(1, args["n_rows"]))
    return min(rows_per_task, active_rows)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 2048}), 
        triton.Config({"BLOCK_SIZE_N": 4096}), 
        triton.Config({"BLOCK_SIZE_N": 8192}), 
        triton.Config({"BLOCK_SIZE_N": 6144}),
        triton.Config({"BLOCK_SIZE_N": 10240}),  
    ],
    key=["n_rows", "n_cols"],
)
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

def _prune_oversized_tiles(configs, nargs, **kwargs):
    N = kwargs["BLOCK_SIZE_N"]
    return [
        cfg
        for cfg in configs
        if cfg.kwargs.get("BLOCK_SIZE_M", 0) * N <= 1048576
    ]

@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": BM, "multibuffer": MF, "enable_vf_fusion": EF, })
        for BM in [1, 2, 4, 8, 16, 32, 64, 128, 256]
        for MF in [True, False]
        for EF in [True, False]
    ],
    key=["n_rows", "n_cols", "X_ptr.dtype"],
    prune_configs_by={"early_config_prune": _prune_oversized_tiles},
)
@triton.jit
def _rmsnorm_infer_kernel_single(
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
    
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    col_mask = col_offsets < n_cols
    
    w_ptrs = W_ptr + col_offsets
    w = tl.load(w_ptrs, mask=col_mask, other=0.0)
    w_f32 = w.to(tl.float32)


    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M

        current_row_offsets = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        row_mask = current_row_offsets < n_rows

        x_ptrs = X_ptr + (current_row_offsets[:, None] * stride_x_row + col_offsets[None, :])

        x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)

        ss_acc = tl.sum(x * x, axis=1)

        ss_acc = tl.where(row_mask, ss_acc, 0)

        mean_square = ss_acc / n_cols
        rrms = tl.rsqrt(mean_square + eps)

        rrms = tl.where(row_mask, rrms, 0.0)

        y_ptrs = Y_ptr + (current_row_offsets[:, None] * stride_y_row + col_offsets[None, :])

        x_normalized = x * rrms[:, None]

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
    assert x.size(-1) == w.size(-1)
    shape = x.shape
    dim = shape[-1]
    X_2d = x.reshape(-1, dim)
    n_rows, n_cols = X_2d.shape

    y = torch.empty_like(X_2d)

    if n_cols > COL_BLOCKING_THRESHOLD:
        if n_cols * 8 <= 200000:
            BLOCK_SIZE_N = n_cols # only load X for once
        else:
            BLOCK_SIZE_N = COL_BLOCKING_THRESHOLD # load X for twice
    else:
        BLOCK_SIZE_N = align(x, n_cols, VEC_ALIGN_BYTES)

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]

    grid = (num_programs,)

    if BLOCK_SIZE_N < n_cols:
        _rmsnorm_infer_kernel[grid](
            x,
            y,
            w,
            X_2d.stride(0),
            y.stride(0),
            n_rows=n_rows,
            n_cols=n_cols,
            eps=eps,
        )
    else:
        _rmsnorm_infer_kernel_single[grid](
            x,
            y,
            w,
            X_2d.stride(0),
            y.stride(0),
            n_rows=n_rows,
            n_cols=n_cols,
            eps=eps,
            BLOCK_SIZE_N=n_cols,
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


@triton.heuristics({"BLOCK_SIZE_M": rms_norm_bwd_heuristics})
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
            dW_acc += tl.sum(dY_block * normed_X_block.to(tl.float32), axis=0)
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
            if casting_mode_int == _CASTING_MODE_GEMMA:
                m_chunk = dY_chunk.to(tl.float32) * W_chunk_offset.to(tl.float32)[None, :]
            else:
                m_chunk = dY_chunk * W_chunk_offset[None, :]
            if casting_mode_int == _CASTING_MODE_LLAMA:
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
                dW_chunk_sum = tl.sum(dY_chunk.to(tl.float32) * normed_X_chunk, axis=0)
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

            # Each program owns one partial row. The first iteration starts at
            # zero; subsequent iterations accumulate in a fixed row-task order.
            partial_ptr = dW_ptr + pid * dW_row_stride + cols_off
            dW_existing = tl.load(partial_ptr, mask=cols_mask & (row_task_id != pid), other=0.0)
            tl.store(partial_ptr, dW_existing + dW_chunk_sum, mask=cols_mask)


@libentry()
@triton.jit
def _rmsnorm_reduce_dw_kernel(
    partial_ptr,
    dW_ptr,
    n_cols,
    num_partials: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    partials = tl.arange(0, BLOCK_PARTIALS)
    values = tl.load(
        partial_ptr + partials[:, None] * n_cols + cols[None, :],
        mask=(partials[:, None] < num_partials) & (cols[None, :] < n_cols),
        other=0.0,
    )
    # One fixed reduction tree per column; no cross-program floating atomics.
    tl.store(dW_ptr + cols, tl.sum(values, axis=0), mask=cols < n_cols)


def rmsnorm_fwd_impl(
    X: torch.Tensor,
    W: torch.Tensor,
    eps: float,
    offset: float,
    casting_mode_int: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = X.shape
    dim = shape[-1]
    X_2d = X.reshape(-1, dim).contiguous()
    W = W.contiguous()
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
    if dY.shape != X.shape:
        raise ValueError(f"RMSNorm grad_output shape {dY.shape} must match input shape {X.shape}.")
    shape = X.shape
    dim = shape[-1]
    dY_2d = dY.reshape(-1, dim).contiguous()
    X_2d = X.reshape(-1, dim).contiguous()
    W = W.contiguous()
    RSTD = RSTD.reshape(-1).contiguous()
    n_rows, n_cols = dY_2d.shape

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    X_dtype_triton = torch_to_triton_dtype[X_dtype]

    grid = (num_programs,)

    dX_2d = torch.empty_like(X_2d)

    if n_cols <= COL_BLOCKING_BWD_THRESHOLD:
        _dW = torch.zeros((num_programs, n_cols), dtype=torch.float32, device=W.device)
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
        if n_rows == 0:
            return dX_2d.reshape(*shape), torch.zeros_like(W)
        # Bound scratch to one FP32 partial row per active vector-core program,
        # independent of token count; every active program writes every column.
        num_partials = min(num_programs, ceil_div(n_rows, 2))
        _dW = torch.empty((num_partials, n_cols), dtype=torch.float32, device=W.device)

        _rmsnorm_bwd_large_cols_kernel[(num_partials,)](
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
            BLOCK_SIZE_N=COL_BLOCKING_BWD_THRESHOLD,
            BLOCK_SIZE_M=2,
        )

        dW = torch.empty_like(W)
        _rmsnorm_reduce_dw_kernel[(triton.cdiv(n_cols, 128),)](
            _dW,
            dW,
            n_cols,
            num_partials,
            BLOCK_PARTIALS=triton.next_power_of_2(num_partials),
            BLOCK_SIZE_N=128,
        )

    dX = dX_2d.reshape(*shape)

    return dX, dW


def rmsnorm_train_fwd_impl(
    X: torch.Tensor,
    W: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return rmsnorm_fwd_impl(X, W, eps, offset=0.0, casting_mode_int=1)


def rmsnorm_train_bwd_impl(
    dY: torch.Tensor,
    X: torch.Tensor,
    W: torch.Tensor,
    RSTD: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return rmsnorm_bwd_impl(
        dY,
        X,
        W,
        RSTD,
        offset=0.0,
        casting_mode_int=1,
        X_dtype=X.dtype,
    )


@torch.library.custom_op("mojo_npu_triton_a5::rms_norm_fwd", mutates_args=())
def rms_norm_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    return rmsnorm_train_fwd_impl(x, weight, eps)


@rms_norm_fwd.register_fake
def _rms_norm_fwd_fake(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    del weight, eps
    rows = x.numel() // x.shape[-1]
    return (torch.empty_like(x, memory_format=torch.contiguous_format),
            torch.empty(rows, dtype=torch.float32, device=x.device))


@torch._dynamo.assume_constant_result
def _rmsnorm_vectorcores():
    # Hardware metadata is a capture-time constant, never probed by fake kernels.
    return int(triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"])


@torch.library.custom_op("mojo_npu_triton_a5::rms_norm_bwd_partial", mutates_args=())
def rms_norm_bwd_partial(
    grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor,
    rstd: torch.Tensor, num_programs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows, cols = x.shape
    grad_x = torch.empty_like(x)
    # Both kernels write every column of every partial row, including idle
    # small-kernel programs. No zeroing or output-buffer copy is needed.
    partial = torch.empty((num_programs, cols), dtype=torch.float32, device=weight.device)
    args = (
        grad_output, grad_output.stride(0), grad_x, grad_x.stride(0),
        x, x.stride(0), weight, rstd, rstd.stride(0), partial, partial.stride(0),
        rows, cols, 0.0, 1, torch_to_triton_dtype[x.dtype],
    )
    if cols <= COL_BLOCKING_BWD_THRESHOLD:
        _rmsnorm_bwd_kernel[(num_programs,)](
            *args, BLOCK_SIZE_N=align(x, cols, VEC_ALIGN_BYTES),
        )
    else:
        _rmsnorm_bwd_large_cols_kernel[(num_programs,)](
            *args, BLOCK_SIZE_N=COL_BLOCKING_BWD_THRESHOLD, BLOCK_SIZE_M=2,
        )
    return grad_x, partial


@rms_norm_bwd_partial.register_fake
def _rms_norm_bwd_partial_fake(grad_output, x, weight, rstd, num_programs):
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty((num_programs, x.shape[1]), dtype=torch.float32, device=weight.device),
    )


@torch.library.custom_op("mojo_npu_triton_a5::rms_norm_dw_reduce", mutates_args=())
def rms_norm_dw_reduce(partial: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    num_partials, cols = partial.shape
    grad_weight = torch.empty(cols, dtype=output_dtype, device=partial.device)
    _rmsnorm_reduce_dw_kernel[(triton.cdiv(cols, 128),)](
        partial, grad_weight, cols, num_partials,
        BLOCK_PARTIALS=triton.next_power_of_2(num_partials), BLOCK_SIZE_N=128,
    )
    return grad_weight


@rms_norm_dw_reduce.register_fake
def _rms_norm_dw_reduce_fake(partial, output_dtype):
    return torch.empty(partial.shape[1], dtype=output_dtype, device=partial.device)


def rms_norm_bwd(
    grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, rstd: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if grad_output.shape != x.shape:
        raise ValueError(f"RMSNorm grad_output shape {grad_output.shape} must match input shape {x.shape}.")
    cols = x.shape[-1]
    x_2d = x.reshape(-1, cols).contiguous()
    grad_2d = grad_output.reshape(-1, cols).contiguous()
    weight = weight.contiguous()
    rstd = rstd.reshape(-1).contiguous()
    rows = x_2d.shape[0]
    if rows == 0:
        return torch.empty_like(x_2d).reshape(x.shape), torch.zeros_like(weight)
    num_programs = _rmsnorm_vectorcores()
    large = cols > COL_BLOCKING_BWD_THRESHOLD
    if large:
        num_programs = min(num_programs, (rows + 1) // 2)
    grad_x, partial = rms_norm_bwd_partial(grad_2d, x_2d, weight, rstd, num_programs)
    grad_weight = rms_norm_dw_reduce(partial, weight.dtype) if large else partial.sum(dim=0).to(weight.dtype)
    return grad_x.reshape(x.shape), grad_weight


@torch.library.custom_op("mojo_npu_triton_a5::rms_norm_infer", mutates_args=())
def rms_norm_infer_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return rmsnorm_infer_impl(x.contiguous(), weight.contiguous(), eps)


@rms_norm_infer_fwd.register_fake
def _rms_norm_infer_fake(x, weight, eps):
    return torch.empty_like(x, memory_format=torch.contiguous_format)
