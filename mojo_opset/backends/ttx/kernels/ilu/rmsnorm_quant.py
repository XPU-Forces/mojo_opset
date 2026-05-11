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

def _rmsnorm_fwd_grid_n_programs(n_rows: int, n_cols: int) -> int:
    block_m = rms_norm_fwd_heuristics({"n_cols": n_cols})
    n_tasks = triton.cdiv(n_rows, block_m)
    return ilu_grid_dim_from_row_tasks(n_tasks)

@triton.heuristics({"BLOCK_SIZE_M": rms_norm_fwd_heuristics})
# @libentry()
@triton.jit
def _rmsnorm_quant_infer_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    scale_ptr,
    stride_x_row,
    stride_y_row,
    n_rows,
    eps,
    q_min, 
    q_max,
    IS_INT8: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    dtype = tl.int8 if IS_INT8 else tl.float8e4nv

    pid = tl.program_id(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    task_mask = pid < num_row_tasks

    block_start_row = pid * BLOCK_SIZE_M

    row_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
    row_mask = task_mask & (row_off < n_rows)

    ss_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for col_start in range(0, N_COLS, BLOCK_SIZE_N):
        col_off = col_start + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_off < N_COLS

        x_ptrs = X_ptr + (row_off[:, None] * stride_x_row + col_off[None, :])
        x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)

        ss_acc += tl.sum(x * x, axis=1)

    ss_acc = tl.where(row_mask, ss_acc, 0)

    mean_square = ss_acc / N_COLS
    rrms = tl.rsqrt(mean_square + eps)

    rrms = tl.where(row_mask, rrms, 0.0)

    max_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for col_start in range(0, N_COLS, BLOCK_SIZE_N):
        col_off = col_start + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_off < N_COLS

        x_ptrs = X_ptr + (row_off[:, None] * stride_x_row + col_off[None, :])
        w_ptrs = W_ptr + col_off

        x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=col_mask, other=0.0)

        x_f32 = x.to(tl.float32)
        w_f32 = w.to(tl.float32)

        x_normalized = x_f32 * rrms[:, None]
        y = x_normalized * w_f32[None, :]

        y_max = tl.max(y.abs(), axis=1)
        max_acc = tl.maximum(max_acc, y_max)

    quant_scale = max_acc / q_max
    tl.store(scale_ptr + row_off, quant_scale, mask=row_mask)

    for col_start in range(0, N_COLS, BLOCK_SIZE_N):
        col_off = col_start + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_off < N_COLS
        rc_mask = row_mask[:, None] & col_mask[None, :]

        x_ptrs = X_ptr + (row_off[:, None] * stride_x_row + col_off[None, :])
        w_ptrs = W_ptr + col_off

        x = tl.load(x_ptrs, mask=rc_mask, other=0.0)
        w = tl.load(w_ptrs, mask=col_mask, other=0.0)

        x_f32 = x.to(tl.float32)
        w_f32 = w.to(tl.float32)

        x_normalized = x_f32 * rrms[:, None]
        y = x_normalized * w_f32[None, :]
        y = y / quant_scale[:, None]
        y = tl.floor(tl.where(y < 0, y - 0.5, y + 0.5))
        y = tl.clamp(y, q_min, q_max)
        y = y.to(dtype)

        y_ptrs = Y_ptr + (row_off[:, None] * stride_y_row + col_off[None, :])
        tl.store(y_ptrs, y, mask=rc_mask)

        y_max = tl.max(y.abs(), axis=1)
        max_acc = tl.maximum(max_acc, y_max)

#rmsnorm_quant_infer(hidden_state, smooth_scale, self.weight, self.variance_epsilon, self.q_min, self.q_max, self.quant_dtype)

def rmsnorm_quant_infer_impl(
    x: torch.Tensor,
    smooth_scale: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    q_min: float,
    q_max: float,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.size(-1) == w.size(-1)
    shape = x.shape
    dim = shape[-1]
    X_2d = x.reshape(-1, dim)
    n_rows, n_cols = X_2d.shape

    y = torch.empty((n_rows, n_cols), dtype=dtype, device=X_2d.device)
    scale = torch.empty(n_rows, dtype=torch.float32, device=X_2d.device)

    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = COL_BLOCKING_THRESHOLD
    else:
        BLOCK_SIZE_N = align(x, n_cols, VEC_ALIGN_BYTES)

    grid = (_rmsnorm_fwd_grid_n_programs(n_rows, n_cols),)

    _rmsnorm_quant_infer_kernel[grid](
        x, y, w, scale,
        X_2d.stride(0), y.stride(0),
        n_rows, eps, q_min, q_max, 
        IS_INT8=(dtype == torch.int8),
        N_COLS=n_cols,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return y, scale.reshape((n_rows, 1))
    #return y.reshape((n_rows, n_cols)), scale.reshape((n_rows))
    

