import torch
import triton
import triton.language as tl

from triton.language.math import rsqrt
from triton.runtime.libentry import libentry

from mojo_opset.backends.ttx.kernels.npu.utils import VEC_ALIGN_BYTES
from mojo_opset.backends.ttx.kernels.utils import align
from mojo_opset.backends.ttx.kernels.utils import ceil_div
from mojo_opset.backends.ttx.kernels.utils import torch_to_triton_dtype

COL_BLOCKING_THRESHOLD = 2048

TOKEN_BLOCK_SIZE_TABLE = {
    2048: 4,
    1024: 8,
    512: 10,
    256: 18,
    128: 24,
}


def layer_norm_fwd_heuristics(args):
    hidden_dim = args["n_cols"]
    if hidden_dim <= COL_BLOCKING_THRESHOLD:
        if hidden_dim in TOKEN_BLOCK_SIZE_TABLE:
            return TOKEN_BLOCK_SIZE_TABLE[hidden_dim]

        for dim_thresh, block_size in sorted(TOKEN_BLOCK_SIZE_TABLE.items()):
            if hidden_dim <= dim_thresh:
                return block_size
        return 1
    else:
        return 4


@triton.heuristics({"BLOCK_SIZE_M": layer_norm_fwd_heuristics})
@libentry()
@triton.jit
def _layernorm_fwd_kernel(
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

        tl.store(Mean_ptr + rows_off, mean, mask=rows_mask)

        var_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            x_chunk = tl.load(
                X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0
            ).to(tl.float32)

            w_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)
            b_chunk = tl.load(B_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

            x_centered = x_chunk - mean[:, None]

            var_acc += tl.sum(x_centered * x_centered, axis=1)

        var = var_acc / n_cols
        rstd = rsqrt(var + eps)

        tl.store(RSTD_ptr + rows_off, rstd, mask=rows_mask)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            x_chunk = tl.load(
                X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0
            ).to(tl.float32)

            w_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)
            b_chunk = tl.load(B_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

            x_centered = x_chunk - mean[:, None]
            y_chunk = x_centered * rstd[:, None] * w_chunk[None, :] + b_chunk[None, :]

            tl.store(
                Y_ptr + rows_off[:, None] * stride_y_row + cols_off[None, :],
                y_chunk.to(Y_ptr.dtype.element_ty),
                mask=block_mask,
            )


@triton.heuristics({"BLOCK_SIZE_M": lambda args: ceil_div(4096, args["n_cols"])})
@libentry()
@triton.jit
def _layernorm_bwd_kernel(
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

    cols_off = tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols_off < n_cols

    dW_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    dB_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    w = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

    for row_task_id in range(pid, num_row_tasks, num_programs):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows
        block_mask = rows_mask[:, None] & cols_mask[None, :]

        mean = tl.load(Mean_ptr + rows_off, mask=rows_mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD_ptr + rows_off, mask=rows_mask, other=0.0).to(tl.float32)

        dy = tl.load(DY_ptr + rows_off[:, None] * stride_dy_row + cols_off[None, :], mask=block_mask, other=0.0).to(
            tl.float32
        )

        x = tl.load(X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0).to(
            tl.float32
        )

        x_hat = (x - mean[:, None]) * rstd[:, None]

        dW_acc += tl.sum(dy * x_hat, axis=0)
        dB_acc += tl.sum(dy, axis=0)

        wdy = w[None, :] * dy
        c1 = tl.sum(x_hat * wdy, axis=1) / n_cols
        c2 = tl.sum(wdy, axis=1) / n_cols
        dx = (wdy - (x_hat * c1[:, None] + c2[:, None])) * rstd[:, None]

        tl.store(DX_ptr + rows_off[:, None] * stride_dx_row + cols_off[None, :], dx.to(X_dtype), mask=block_mask)

    tl.atomic_add(DW_ptr + cols_off, dW_acc, mask=cols_mask)
    tl.atomic_add(DB_ptr + cols_off, dB_acc, mask=cols_mask)


@libentry()
@triton.jit
def _layernorm_bwd_large_cols_kernel(
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

        mean = tl.load(Mean_ptr + rows_off, mask=rows_mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD_ptr + rows_off, mask=rows_mask, other=0.0).to(tl.float32)

        c1_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        c2_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            dy = tl.load(DY_ptr + rows_off[:, None] * stride_dy_row + cols_off[None, :], mask=block_mask, other=0.0).to(
                tl.float32
            )

            x = tl.load(X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0).to(
                tl.float32
            )

            w = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

            x_hat = (x - mean[:, None]) * rstd[:, None]
            wdy = w[None, :] * dy

            c1_acc += tl.sum(x_hat * wdy, axis=1)
            c2_acc += tl.sum(wdy, axis=1)

        c1 = c1_acc / n_cols
        c2 = c2_acc / n_cols

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            dy = tl.load(DY_ptr + rows_off[:, None] * stride_dy_row + cols_off[None, :], mask=block_mask, other=0.0).to(
                tl.float32
            )

            x = tl.load(X_ptr + rows_off[:, None] * stride_x_row + cols_off[None, :], mask=block_mask, other=0.0).to(
                tl.float32
            )

            w = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0).to(tl.float32)

            x_hat = (x - mean[:, None]) * rstd[:, None]
            wdy = w[None, :] * dy

            dW_chunk = tl.sum(dy * x_hat, axis=0)
            dB_chunk = tl.sum(dy, axis=0)

            dx = (wdy - (x_hat * c1[:, None] + c2[:, None])) * rstd[:, None]

            tl.store(DX_ptr + rows_off[:, None] * stride_dx_row + cols_off[None, :], dx.to(X_dtype), mask=block_mask)

            tl.atomic_add(DW_ptr + cols_off, dW_chunk, mask=cols_mask)
            tl.atomic_add(DB_ptr + cols_off, dB_chunk, mask=cols_mask)


def layernorm_infer_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    shape = hidden_states.shape
    dim = shape[-1]
    x_2d = hidden_states.reshape(-1, dim)
    n_rows, n_cols = x_2d.shape

    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = 2048
    else:
        BLOCK_SIZE_N = align(hidden_states, n_cols, VEC_ALIGN_BYTES)

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    y = torch.empty_like(x_2d)
    mean = torch.empty(n_rows, dtype=hidden_states.dtype, device=hidden_states.device)
    rstd = torch.empty(n_rows, dtype=hidden_states.dtype, device=hidden_states.device)

    _layernorm_fwd_kernel[grid](
        x_2d,
        y,
        weight,
        bias,
        mean,
        rstd,
        x_2d.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return y.reshape(*shape)


def layernorm_fwd_impl(x, w, b, eps):
    shape = x.shape
    dim = shape[-1]
    x_2d = x.reshape(-1, dim)
    n_rows, n_cols = x_2d.shape

    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = 2048
    else:
        BLOCK_SIZE_N = align(x, n_cols, VEC_ALIGN_BYTES)

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    y = torch.empty_like(x_2d)
    mean = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    rstd = torch.empty(n_rows, dtype=x.dtype, device=x.device)

    _layernorm_fwd_kernel[grid](
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


def layernorm_bwd_impl(dy, x_2d, w, b, mean, rstd):
    shape = dy.shape
    dim = shape[-1]
    dy_2d = dy.reshape(-1, dim)
    n_rows, n_cols = dy_2d.shape

    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = 2048
    else:
        BLOCK_SIZE_N = align(x_2d, n_cols, VEC_ALIGN_BYTES)

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = (num_programs,)

    dx = torch.empty_like(dy_2d)

    if n_cols <= COL_BLOCKING_THRESHOLD:
        dw = torch.zeros(n_cols, dtype=torch.float32, device=w.device)
        db = torch.zeros(n_cols, dtype=torch.float32, device=b.device)

        _layernorm_bwd_kernel[grid](
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
        dw = torch.zeros(n_cols, dtype=torch.float32, device=w.device)
        db = torch.zeros(n_cols, dtype=torch.float32, device=b.device)

        _layernorm_bwd_large_cols_kernel[grid](
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
            BLOCK_SIZE_M=2,
        )

    dw = dw.to(w.dtype)
    db = db.to(b.dtype)

    return dx.reshape(*shape), dw, db
