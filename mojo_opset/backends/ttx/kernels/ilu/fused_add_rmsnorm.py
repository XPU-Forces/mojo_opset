from typing import Tuple

import torch
import triton
import triton.language as tl

from .utils import VEC_ALIGN_BYTES
from .utils import ilu_grid_dim_from_row_tasks
from .utils import libentry
from mojo_opset.backends.ttx.kernels.utils import align
from mojo_opset.backends.ttx.kernels.utils import torch_to_triton_dtype

COL_BLOCKING_THRESHOLD = 2048

TOKEN_BLOCK_SIZE_TABLE = {
    2048: 4,
    1024: 8,
    # NOTE: tl.arange range must be power-of-2 on some backends.
    512: 16,
    256: 16,
    128: 32,
}

def _block_size_n_pow2(n_cols: int) -> int:
    # ILU backend requires tl.arange range to be power-of-2.
    if n_cols <= 128:
        return 128
    if n_cols <= 256:
        return 256
    if n_cols <= 512:
        return 512
    if n_cols <= 1024:
        return 1024
    return 2048


def rms_norm_fwd_heuristics(args):
    hidden_dim = args.get("n_cols", args.get("N_COLS"))
    if hidden_dim is None:
        raise KeyError("n_cols")
    if hidden_dim <= COL_BLOCKING_THRESHOLD:
        if hidden_dim in TOKEN_BLOCK_SIZE_TABLE:
            return TOKEN_BLOCK_SIZE_TABLE[hidden_dim]

        for dim_thresh, block_size in sorted(TOKEN_BLOCK_SIZE_TABLE.items()):
            if hidden_dim <= dim_thresh:
                return block_size
        return 1
    else:
        return 4


def _fused_add_rmsnorm_fwd_grid_n_programs(n_rows: int, n_cols: int) -> int:
    block_m = rms_norm_fwd_heuristics({"n_cols": n_cols})
    n_tasks = triton.cdiv(n_rows, block_m)
    return ilu_grid_dim_from_row_tasks(n_tasks)


@triton.heuristics({"BLOCK_SIZE_M": rms_norm_fwd_heuristics})
@libentry()
@triton.jit
def _fused_add_rmsnorm_fwd_kernel(
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
    eps,
    N_COLS: tl.constexpr,
    offset,
    casting_mode: tl.constexpr,
    S_dtype: tl.constexpr,
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
    R_ptr_row_block = R_ptr + rows_off[:, None] * R_row_stride
    S_ptr_row_block = S_ptr + rows_off[:, None] * S_row_stride
    Y_ptr_row_block = Y_ptr + rows_off[:, None] * Y_row_stride

    var_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for col_offset in range(0, N_COLS, BLOCK_SIZE_N):
        cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
        cols_mask = cols_off < N_COLS
        block_mask = rows_mask[:, None] & cols_mask[None, :]

        X_chunk = tl.load(X_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
        R_chunk = tl.load(R_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
        S_chunk = X_chunk + R_chunk
        tl.store(S_ptr_row_block + cols_off[None, :], S_chunk, mask=block_mask)

        S_chunk_f32 = S_chunk.to(tl.float32)
        var_acc += tl.sum(S_chunk_f32 * S_chunk_f32, axis=1)

    var = var_acc / N_COLS
    rstd_vec = tl.rsqrt(var + eps)
    tl.store(RSTD_ptr + rows_off * RSTD_row_stride, rstd_vec, mask=rows_mask)

    for col_offset in range(0, N_COLS, BLOCK_SIZE_N):
        cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
        cols_mask = cols_off < N_COLS
        block_mask = rows_mask[:, None] & cols_mask[None, :]

        S_chunk = tl.load(S_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
        W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)

        if casting_mode == 1:  # GEMMA
            S_chunk = S_chunk.to(tl.float32)
            W_chunk = W_chunk.to(tl.float32)
        elif casting_mode == 0:  # LLAMA
            S_chunk = S_chunk.to(tl.float32)

        if casting_mode == 0:  # LLAMA
            normed_S_chunk = (S_chunk * rstd_vec[:, None]).to(S_dtype)
        else:
            normed_S_chunk = S_chunk * rstd_vec[:, None]

        Y_chunk = normed_S_chunk * (W_chunk[None, :] + offset)

        if casting_mode == 1:  # GEMMA
            Y_chunk = Y_chunk.to(S_dtype)

        tl.store(Y_ptr_row_block + cols_off[None, :], Y_chunk, mask=block_mask)


def fused_add_rmsnorm_infer_impl(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    add_mode: str = "pre",
    eps: float = 1e-6,
    offset: float = 0.0,
    casting_mode: str = "llama",
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = hidden_states.shape
    dim = shape[-1]
    hidden_states_2d = hidden_states.reshape(-1, dim)
    residual_2d = residual.reshape(-1, dim)
    n_rows, n_cols = hidden_states_2d.shape

    BLOCK_SIZE_N = _block_size_n_pow2(n_cols)

    grid = (_fused_add_rmsnorm_fwd_grid_n_programs(n_rows, n_cols),)

    str_to_casting_mode = {"llama": 0, "gemma": 1, "none": -1}
    _casting_mode = str_to_casting_mode[casting_mode]

    rstd_dtype = torch.float32 if _casting_mode in (0, 1) else hidden_states.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=hidden_states.device)

    Y = torch.empty_like(hidden_states_2d)
    S = torch.empty_like(hidden_states_2d)

    _fused_add_rmsnorm_fwd_kernel[grid](
        Y,
        Y.stride(0),
        S,
        S.stride(0),
        hidden_states_2d,
        hidden_states_2d.stride(0),
        residual_2d,
        residual_2d.stride(0),
        weight,
        RSTD,
        RSTD.stride(0),
        n_rows,
        eps,
        N_COLS=n_cols,
        offset=offset,
        casting_mode=_casting_mode,
        S_dtype=torch_to_triton_dtype[hidden_states.dtype],
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    if add_mode == "pre":
        return Y.reshape(*shape), S.reshape(*shape)
    elif add_mode == "post":
        return Y.reshape(*shape), Y.reshape(*shape)
    else:
        raise ValueError(f"Invalid add_mode: {add_mode}. Must be 'pre' or 'post'.")
