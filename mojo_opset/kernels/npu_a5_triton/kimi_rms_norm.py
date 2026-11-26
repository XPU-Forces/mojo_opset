import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import libentry

COL_BLOCK_SIZE = 8192


@libentry()
@triton.jit
def _kimi_rmsnorm_kernel(
    X_ptr, Y_ptr, W_ptr, stride_x_row, stride_y_row, n_rows, n_cols, eps,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = tl.cdiv(n_rows, BLOCK_SIZE_M)

    for row_task_id in range(pid, num_row_tasks, grid_size):
        rows = row_task_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        row_mask = rows < n_rows
        ss_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols = col_offset + tl.arange(0, BLOCK_SIZE_N)
            mask = row_mask[:, None] & (cols < n_cols)[None, :]
            x = tl.load(X_ptr + rows[:, None] * stride_x_row + cols[None, :],
                        mask=mask, other=0.0).to(tl.float32)
            ss_acc += tl.sum(x * x, axis=1)
        rrms = tl.rsqrt(ss_acc / n_cols + eps)
        rrms = tl.where(row_mask, rrms, 0.0)
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols = col_offset + tl.arange(0, BLOCK_SIZE_N)
            col_mask = cols < n_cols
            mask = row_mask[:, None] & col_mask[None, :]
            x = tl.load(X_ptr + rows[:, None] * stride_x_row + cols[None, :],
                        mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            normalized = (x * rrms[:, None]).to(X_ptr.dtype.element_ty).to(tl.float32)
            y = normalized * w[None, :]
            tl.store(Y_ptr + rows[:, None] * stride_y_row + cols[None, :],
                     y.to(Y_ptr.dtype.element_ty), mask=mask)


def kimi_rmsnorm_infer_impl(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    shape = x.shape
    x = x.contiguous()
    w = w.contiguous()
    x_2d = x.reshape(-1, shape[-1])
    n_rows, n_cols = x_2d.shape
    y = torch.empty_like(x_2d, dtype=torch.promote_types(x.dtype, w.dtype))
    if n_rows == 0:
        return y.reshape(shape)

    num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    _kimi_rmsnorm_kernel[(min(num_programs, n_rows),)](
        x_2d, y, w, x_2d.stride(0), y.stride(0), n_rows, n_cols, eps,
        BLOCK_SIZE_M=1, BLOCK_SIZE_N=min(COL_BLOCK_SIZE, triton.next_power_of_2(n_cols)),
    )
    return y.reshape(shape)


@torch.library.custom_op("mojo_npu_triton_a5::kimi_rms_norm", mutates_args=())
def kimi_rms_norm_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return kimi_rmsnorm_infer_impl(x, weight, eps)


@kimi_rms_norm_fwd.register_fake
def _kimi_rms_norm_fake(x, weight, eps):
    return torch.empty_like(x, dtype=torch.promote_types(x.dtype, weight.dtype),
                            memory_format=torch.contiguous_format)
