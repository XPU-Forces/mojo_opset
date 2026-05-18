import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .utils import _block_size_n_pow2
from .utils import ilu_grid_dim_from_row_tasks
from .utils import libentry
from .utils import norm_fwd_heuristics


def _static_quant_grid_n_programs(n_rows: int, n_cols: int) -> int:
    block_m = norm_fwd_heuristics({"n_cols": n_cols})
    n_tasks = triton.cdiv(n_rows, block_m)
    return ilu_grid_dim_from_row_tasks(n_tasks)


@triton.heuristics({"BLOCK_SIZE_M": norm_fwd_heuristics})
@libentry()
@triton.jit
def _static_quant_kernel(
    output_ptr,
    input_ptr,
    scale_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    q_min: tl.constexpr,
    q_max: tl.constexpr,
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

    input_row_block = input_ptr + rows_off[:, None] * input_row_stride
    output_row_block = output_ptr + rows_off[:, None] * output_row_stride

    for col_offset in range(0, N_COLS, BLOCK_SIZE_N):
        cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
        cols_mask = cols_off < N_COLS
        block_mask = rows_mask[:, None] & cols_mask[None, :]

        x = tl.load(input_row_block + cols_off[None, :], mask=block_mask, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + cols_off, mask=cols_mask, other=1.0).to(tl.float32)

        val = libdevice.nearbyint(x / s[None, :]).to(tl.int32)
        val = tl.minimum(tl.maximum(val, q_min), q_max)
        result = val.to(tl.int8)

        tl.store(output_row_block + cols_off[None, :], result, mask=block_mask)


def static_quant_impl(
    input_tensor: torch.Tensor,
    scale: torch.Tensor,
    q_min: int = -128,
    q_max: int = 127,
) -> torch.Tensor:
    shape = input_tensor.shape
    n_cols = shape[-1]
    input_2d = input_tensor.reshape(-1, n_cols)
    n_rows = input_2d.shape[0]

    BLOCK_SIZE_N = _block_size_n_pow2(n_cols)

    output = torch.empty_like(input_2d, dtype=torch.int8)

    grid = (_static_quant_grid_n_programs(n_rows, n_cols),)

    _static_quant_kernel[grid](
        output,
        input_2d,
        scale,
        input_2d.stride(0),
        output.stride(0),
        n_rows,
        q_min=q_min,
        q_max=q_max,
        N_COLS=n_cols,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return output.reshape(shape)
