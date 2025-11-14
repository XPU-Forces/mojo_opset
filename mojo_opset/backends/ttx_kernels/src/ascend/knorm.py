import torch
import triton
import triton.language as tl

import torch_npu
from triton.runtime.libentry import libentry

from .utils import VEC_ALIGN_BYTES
from .utils import align
from .utils import torch_to_triton_dtype

COL_BLOCKING_THRESHOLD = 4096


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1}),
        triton.Config({"BLOCK_SIZE_M": 2}),
        triton.Config({"BLOCK_SIZE_M": 4}),
        triton.Config({"BLOCK_SIZE_M": 8}),
        triton.Config({"BLOCK_SIZE_M": 16}),
        triton.Config({"BLOCK_SIZE_M": 24}),
        triton.Config({"BLOCK_SIZE_M": 32}),
    ],
    key=["N_ROWS", "N_COLS"],
    restore_value=["X_ptr"],
)
@libentry()
@triton.jit
def _k_rms_norm_inplace_kernel(
    X_ptr,
    W_ptr,
    q_head_num,
    kv_head_num,
    seq_len,
    stride_b,
    stride_s,
    stride_h,
    N_ROWS,
    N_COLS,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (N_ROWS + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M

        current_row_offsets = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        row_mask = current_row_offsets < N_ROWS

        _b_s_idx = current_row_offsets // kv_head_num
        b_idx = _b_s_idx // seq_len
        s_idx = _b_s_idx % seq_len
        h_k_idx = current_row_offsets % kv_head_num

        h_abs_idx = q_head_num + h_k_idx

        row_start_ptrs = X_ptr + (b_idx * stride_b + s_idx * stride_s + h_abs_idx * stride_h)

        ss_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for col_offset in range(0, N_COLS, BLOCK_SIZE_N):
            col_offsets = col_offset + tl.arange(0, BLOCK_SIZE_N)
            col_mask = col_offsets < N_COLS

            x_ptrs = row_start_ptrs[:, None] + col_offsets[None, :]

            x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
            ss_acc += tl.sum(x * x, axis=1)

        ss_acc = tl.where(row_mask, ss_acc, 0)
        mean_square = ss_acc / N_COLS
        rrms = tl.rsqrt(mean_square + eps)
        rrms = tl.where(row_mask, rrms, 0.0)

        for col_offset in range(0, N_COLS, BLOCK_SIZE_N):
            col_offsets = col_offset + tl.arange(0, BLOCK_SIZE_N)
            col_mask = col_offsets < N_COLS

            x_ptrs = row_start_ptrs[:, None] + col_offsets[None, :]
            w_ptrs = W_ptr + col_offsets

            x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=col_mask, other=0.0)

            x_f32 = x.to(tl.float32)
            w_f32 = w.to(tl.float32)

            x_normalized = x_f32 * rrms[:, None]
            y = x_normalized * w_f32[None, :]

            tl.store(
                x_ptrs,
                y.to(X_ptr.dtype.element_ty),
                mask=row_mask[:, None] & col_mask[None, :],
            )


def ttx_k_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float, q_head_num: int, kv_head_num: int):
    shape = x.shape
    assert x.ndim == 4, "Input must be a 4D tensor"
    bs, seq_len, total_head_num, head_dim = shape

    assert total_head_num == q_head_num + 2 * kv_head_num, "total_head_num does not match q_head_num + 2 * kv_head_num"

    n_rows = bs * seq_len * kv_head_num
    n_cols = head_dim

    if n_rows == 0:
        return x

    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = 2048
    else:
        BLOCK_SIZE_N = align(x, n_cols, VEC_ALIGN_BYTES)

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]

    grid = (num_programs,)

    stride_b, stride_s, stride_h, _ = x.stride()

    _k_rms_norm_inplace_kernel[grid](
        x,
        w,
        q_head_num,
        kv_head_num,
        seq_len,
        stride_b,
        stride_s,
        stride_h,
        N_ROWS=n_rows,
        N_COLS=n_cols,
        eps=eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return x