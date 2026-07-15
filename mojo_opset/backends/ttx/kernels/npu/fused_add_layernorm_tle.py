"""TLE/NPU Triton kernel for MojoResidualAddLayerNorm.

The operator first computes ``S = hidden_states + residual`` and then applies
LayerNorm on the last dimension:

    Y = (S - mean(S)) / sqrt(var(S) + eps) * weight + bias

Return semantics follow ``add_mode``:
- ``pre`` returns ``(Y, S)``
- ``post`` returns ``(Y, Y)``
"""

import os
import torch
import triton
import triton.language as tl

from .utils import libentry

from mojo_opset.backends.ttx.kernels.npu.utils import VEC_ALIGN_BYTES
from mojo_opset.backends.ttx.kernels.utils import align
from mojo_opset.backends.ttx.kernels.utils import ceil_div

# 当 hidden dim 很大时，不再尝试过细的列方向 blocking，而是按阈值裁剪。
COL_BLOCKING_THRESHOLD = 2048
SMALL_COL_FASTPATH_THRESHOLD = 512
BLOCK_SIZE_M_OVERRIDE_ENV = "MOJO_RALN_BLOCK_SIZE_M_OVERRIDE"
SMALL_COL_FASTPATH_THRESHOLD_ENV = "MOJO_RALN_SMALL_COL_FASTPATH_THRESHOLD"

# 经验表：hidden dim 越小，同一个 program 通常可以一次处理更多 row。
TOKEN_BLOCK_SIZE_TABLE = {
    # 256x2048 large-profile sweep prefers 6 rows/program over the old 4:
    # it reduces the number of row tasks from 64 to 43, improving per-program
    # work reuse without dropping parallelism below the effective Block Dim.
    2048: 6,
    1024: 8,
    512: 10,
    256: 18,
    # 128-wide rows are tiny: using too many rows per program leaves only a
    # handful of row tasks and under-utilizes vector cores.  A smaller row tile
    # creates more independent tasks while the extra W/B reload traffic remains
    # small for this shape.
    128: 4,
}


def layer_norm_fwd_heuristics(args):
    """Choose how many rows a single Triton program handles.

    直观理解：
    - 每一行都要沿最后一维做一次 LayerNorm
    - 如果 hidden dim 很小，那么处理一行的工作量不大，一个 program 就可以顺手多做几行
    - 如果 hidden dim 很大，那么一行已经很“重”了，就减少每次处理的行数
    """
    hidden_dim = args["n_cols"]
    block_size_m_override = _get_block_size_m_override(hidden_dim)
    if block_size_m_override is not None:
        return block_size_m_override

    if hidden_dim <= COL_BLOCKING_THRESHOLD:
        if hidden_dim in TOKEN_BLOCK_SIZE_TABLE:
            return TOKEN_BLOCK_SIZE_TABLE[hidden_dim]

        for dim_thresh, block_size in sorted(TOKEN_BLOCK_SIZE_TABLE.items()):
            if hidden_dim <= dim_thresh:
                return block_size
        return 1
    elif hidden_dim >= 7000:
        # Very wide rows have only a small number of row tasks.  Using fewer
        # rows per program increases the real Block Dim (64x8192 goes from 16
        # to 32 programs), which better matches torch-npu's AddLayerNorm launch
        # and reduces per-core serial column work.  Keep 4096 on the original
        # row tile because profiling showed its vector efficiency is better
        # with four rows per program.
        return 2
    else:
        return 4


def _get_block_size_m_override(hidden_dim):
    """Read optional perf-sweep override without changing default heuristic.

    Supported forms:
    - `2`: force all hidden dims to BLOCK_SIZE_M=2
    - `4096:2,7338:3,8192:2`: override selected hidden dims only
    """
    override = os.environ.get(BLOCK_SIZE_M_OVERRIDE_ENV)
    if not override:
        return None

    override = override.strip()
    if not override:
        return None

    if ":" not in override:
        return int(override)

    for item in override.split(","):
        if not item.strip():
            continue
        dim_text, block_text = item.split(":", 1)
        if int(dim_text.strip()) == hidden_dim:
            return int(block_text.strip())
    return None


def _get_small_col_fastpath_threshold():
    override = os.environ.get(SMALL_COL_FASTPATH_THRESHOLD_ENV)
    if not override:
        return SMALL_COL_FASTPATH_THRESHOLD
    return int(override.strip())


@triton.heuristics({"BLOCK_SIZE_M": layer_norm_fwd_heuristics})
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
    # 每个 program 负责若干个“行块任务”。
    # 这里的 row 指的是 reshape 成 2D 后的一个 token / 一个样本切片。
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

        # 第一趟：边算 S = X + R，边累计每一行的 sum 和 sum(x^2)。
        #
        # 例子：
        #   X_row = [1, 2, 3, 4]
        #   R_row = [10, 20, 30, 40]
        #   S_row = [11, 22, 33, 44]
        #
        # 后面 LayerNorm 需要这一行的 mean / var，因此这里顺手把统计量也累计起来。
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

        # LayerNorm 统计量：
        #   mean = E[S]
        #   var  = E[S^2] - E[S]^2
        #   rstd = 1 / sqrt(var + eps)
        mean_vec = mean_acc / n_cols
        var_vec = (var_acc / n_cols) - (mean_vec * mean_vec)
        rstd_vec = tl.rsqrt(var_vec + eps)

        # 注意：当前 infer_impl 的最终返回值不会直接暴露 Mean/RSTD，
        # 但这里仍然把它们写出，便于和 backward 公式保持一致，也方便后续排查。
        tl.store(Mean_ptr + rows_off * Mean_row_stride, mean_vec, mask=rows_mask)
        tl.store(RSTD_ptr + rows_off * RSTD_row_stride, rstd_vec, mask=rows_mask)

        # 第二趟：重新读取 S，并完成真正的 LayerNorm + affine:
        #
        #   S_hat = (S - mean) * rstd
        #   Y     = S_hat * W + B
        #
        # 这一步输出的 Y 就是最终的 normalized hidden state。
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


@triton.heuristics({"BLOCK_SIZE_M": layer_norm_fwd_heuristics})
@libentry()
@triton.jit
def _fused_add_layernorm_fwd_infer_kernel(
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
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # infer-only fast path:
    # 当前对外的推理接口只需要 Y 和 S，不需要把 mean/rstd 落到全局内存。
    # 因此这里保留相同数学流程，但去掉两次多余的 global store。
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

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            S_chunk = tl.load(S_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            S_chunk = S_chunk.to(tl.float32)
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)
            B_chunk = tl.load(B_ptr + cols_off, mask=cols_mask, other=0.0)

            normed_S_chunk = (S_chunk - mean_vec[:, None]) * rstd_vec[:, None]
            Y_chunk = normed_S_chunk * W_chunk[None, :] + B_chunk[None, :]
            tl.store(Y_ptr_row_block + cols_off[None, :], Y_chunk.to(Y_ptr.dtype.element_ty), mask=block_mask)


@triton.heuristics({"BLOCK_SIZE_M": layer_norm_fwd_heuristics})
@libentry()
@triton.jit
def _fused_add_layernorm_fwd_infer_small_col_kernel(
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
    n_rows,
    n_cols,
    eps,
    STORE_S: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # hidden dim <= 128 的 fast path。
    #
    # 通用 kernel 为了支持大 hidden dim，会分两趟：
    #   1. 写出 S = X + R，同时统计 mean/var
    #   2. 再从全局内存读回 S，计算 Y
    #
    # 对 128 这种很窄的行，所有列本来就在同一个 block 中。
    # 因此可以直接复用寄存器里的 S_chunk，省掉第二趟 S 的 global load。
    # 如果是 post 模式，S 不作为返回值，还能连 S 的 global store 一起省掉。
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    cols_off = tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols_off < n_cols
    W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)
    B_chunk = tl.load(B_ptr + cols_off, mask=cols_mask, other=0.0)
    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows
        block_mask = rows_mask[:, None] & cols_mask[None, :]

        X_ptr_row_block = X_ptr + rows_off[:, None] * X_row_stride
        R_ptr_row_block = R_ptr + rows_off[:, None] * R_row_stride
        S_ptr_row_block = S_ptr + rows_off[:, None] * S_row_stride
        Y_ptr_row_block = Y_ptr + rows_off[:, None] * Y_row_stride

        X_chunk = tl.load(X_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
        R_chunk = tl.load(R_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
        S_chunk = X_chunk + R_chunk

        if STORE_S:
            tl.store(S_ptr_row_block + cols_off[None, :], S_chunk, mask=block_mask)

        S_chunk_f32 = S_chunk.to(tl.float32)
        mean_vec = tl.sum(S_chunk_f32, axis=1) / n_cols
        var_vec = (tl.sum(S_chunk_f32 * S_chunk_f32, axis=1) / n_cols) - (mean_vec * mean_vec)
        rstd_vec = tl.rsqrt(var_vec + eps)

        normed_S_chunk = (S_chunk_f32 - mean_vec[:, None]) * rstd_vec[:, None]
        Y_chunk = normed_S_chunk * W_chunk[None, :] + B_chunk[None, :]
        tl.store(Y_ptr_row_block + cols_off[None, :], Y_chunk.to(Y_ptr.dtype.element_ty), mask=block_mask)


@triton.heuristics({"BLOCK_SIZE_M": layer_norm_fwd_heuristics})
@libentry()
@triton.jit
def _fused_add_layernorm_fwd_infer_small_col_post_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    W_ptr,
    B_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Small-col post-only path.
    # post 返回 `(Y, Y)`，不需要把 `S = X + R` 暴露给上层。由于 small-col
    # 一整行在单个列块中，S 可以留在寄存器里直接参与 mean/var 和 Y 计算，
    # 因此这里完全去掉 S 指针、STORE_S 分支和中间 S 的 global store/load。
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    cols_off = tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols_off < n_cols
    W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)
    B_chunk = tl.load(B_ptr + cols_off, mask=cols_mask, other=0.0)

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows
        block_mask = rows_mask[:, None] & cols_mask[None, :]

        X_ptr_row_block = X_ptr + rows_off[:, None] * X_row_stride
        R_ptr_row_block = R_ptr + rows_off[:, None] * R_row_stride
        Y_ptr_row_block = Y_ptr + rows_off[:, None] * Y_row_stride

        X_chunk = tl.load(X_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
        R_chunk = tl.load(R_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
        S_chunk_f32 = (X_chunk + R_chunk).to(tl.float32)

        mean_vec = tl.sum(S_chunk_f32, axis=1) / n_cols
        var_vec = (tl.sum(S_chunk_f32 * S_chunk_f32, axis=1) / n_cols) - (mean_vec * mean_vec)
        rstd_vec = tl.rsqrt(var_vec + eps)

        normed_S_chunk = (S_chunk_f32 - mean_vec[:, None]) * rstd_vec[:, None]
        Y_chunk = normed_S_chunk * W_chunk[None, :] + B_chunk[None, :]
        tl.store(Y_ptr_row_block + cols_off[None, :], Y_chunk.to(Y_ptr.dtype.element_ty), mask=block_mask)


@triton.heuristics({"BLOCK_SIZE_M": lambda args: ceil_div(4096, args["n_cols"])})
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
    # backward kernel 当前没有挂到 MojoResidualAddLayerNorm 的公开调用路径上，
    # 但它完整表达了这个 fused op 的反向数学。
    #
    # 可以把它理解成对下面两步一起求导：
    #   S = X + R
    #   Y = layer_norm(S) * W + B
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    # dW / dB 这里按 program 维度分桶累加，再做 atomic add。
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

        # 先累计 LayerNorm 反向里会用到的两个“行级标量”：
        #   sum(dY * W * x_hat)
        #   sum(dY * W)
        #
        # 这样第二趟就能按标准 LayerNorm 公式回推出 dS。
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

        # 第二趟真正计算对 S 的梯度：
        #
        #   dS = (dY*W - mean(dY*W) - x_hat*mean(dY*W*x_hat)) * rstd
        #
        # 如果 has_dS_out=True，说明除了 Y 这条分支以外，
        # 下游还直接消费了 S（例如 "pre" 模式返回的第二个张量），
        # 那么这条分支的梯度也要加回到 dS 上。
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


def fused_add_layernorm_infer_impl(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    add_mode: str = "pre",
    eps: float = 1e-6,
):
    """PyTorch-facing inference wrapper for the fused Triton kernel.

    这里做的主要是“把通用张量形状整理成 kernel 好处理的 2D 形状”。

    例子：
    - 原始输入如果是 `(B, T, D)`，例如 `(2, 8, 1024)`
    - 那么会 reshape 成 `(B*T, D)`，也就是 `(16, 1024)`
    - 因为 LayerNorm 永远只沿最后一维 `D` 做，所以前面的维度都可以摊平成 row

    返回语义：
    - pre:  `(layer_norm(X + R), X + R)`
    - post: `(layer_norm(X + R), layer_norm(X + R))`
    """
    shape = hidden_states.shape
    dim = shape[-1]

    # 常见 inference / perf case 已经是 `(tokens, hidden_dim)` 的 2D 连续张量。
    # 这种情况下再走 `reshape(-1, dim)` / `reshape(*shape)` 会生成额外的
    # view/reshape host op；kernel 本体很短时，这些 Python/eager 开销会很显眼。
    # 因此 2D contiguous 输入直接复用原张量，并在返回时也直接返回 2D 输出。
    use_2d_fast_wrapper = (
        hidden_states.dim() == 2
        and residual.dim() == 2
        and hidden_states.is_contiguous()
        and residual.is_contiguous()
    )
    if use_2d_fast_wrapper:
        X_2d = hidden_states
        R_2d = residual
    else:
        X_2d = hidden_states.reshape(-1, dim)
        R_2d = residual.reshape(-1, dim)
    n_rows, n_cols = X_2d.shape

    # 列方向 tile 的选择：
    # - hidden dim 特别大时直接截到阈值，避免单次列块过大
    # - 否则尽量做对齐，让向量读写更友好
    if n_cols > COL_BLOCKING_THRESHOLD:
        BLOCK_SIZE_N = COL_BLOCKING_THRESHOLD
    else:
        BLOCK_SIZE_N = align(X_2d, n_cols, VEC_ALIGN_BYTES)

    # Y: LayerNorm 输出
    # S: X + R，中间结果；在 pre 模式下它也会作为第二个返回值
    if add_mode not in ("pre", "post"):
        raise ValueError(f"Invalid add_mode: '{add_mode}'. Must be 'pre' or 'post'.")

    block_size_m = layer_norm_fwd_heuristics({"n_cols": n_cols})
    num_row_tasks = ceil_div(n_rows, block_size_m)
    try:
        num_vectorcores = triton.runtime.driver.active.utils.get_device_properties('npu')['num_vectorcore']
    except Exception:
        num_vectorcores = 48
    num_programs = min(num_vectorcores, num_row_tasks)
    grid = (num_programs,)

    # 输出总是按整理后的 2D row-major 视角写满，显式 empty 比 empty_like
    # 少做一些元数据推断；在 eager 小 kernel 场景里能削掉一点 host allocation 成本。
    Y = torch.empty((n_rows, n_cols), dtype=X_2d.dtype, device=X_2d.device)
    if n_cols <= _get_small_col_fastpath_threshold():
        if add_mode == "post":
            S = Y
            _fused_add_layernorm_fwd_infer_small_col_post_kernel[grid](
                Y,
                Y.stride(0),
                X_2d,
                X_2d.stride(0),
                R_2d,
                R_2d.stride(0),
                weight,
                bias,
                n_rows,
                n_cols,
                eps,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
        else:
            S = torch.empty((n_rows, n_cols), dtype=X_2d.dtype, device=X_2d.device)
            _fused_add_layernorm_fwd_infer_small_col_kernel[grid](
                Y,
                Y.stride(0),
                S,
                S.stride(0),
                X_2d,
                X_2d.stride(0),
                R_2d,
                R_2d.stride(0),
                weight,
                bias,
                n_rows,
                n_cols,
                eps,
                STORE_S=True,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
            )
    else:
        # In post mode the second return value is Y, not S = X + R.  Reuse Y
        # as the temporary S buffer so the wrapper does not allocate an extra
        # tensor only to discard it.  The generic kernel first writes S, then
        # reloads and overwrites the same block with final Y, so aliasing is
        # safe for row/column-disjoint program tiles.
        S = torch.empty((n_rows, n_cols), dtype=X_2d.dtype, device=X_2d.device) if add_mode == "pre" else Y
        _fused_add_layernorm_fwd_infer_kernel[grid](
            Y,
            Y.stride(0),
            S,
            S.stride(0),
            X_2d,
            X_2d.stride(0),
            R_2d,
            R_2d.stride(0),
            weight,
            bias,
            n_rows,
            n_cols,
            eps,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

    # 用同一套 kernel 同时兼容两种上层语义。
    if add_mode == "pre":
        if use_2d_fast_wrapper:
            return Y, S
        return Y.reshape(*shape), S.reshape(*shape)
    elif add_mode == "post":
        if use_2d_fast_wrapper:
            return Y, Y
        return Y.reshape(*shape), Y.reshape(*shape)
    raise ValueError(f"Invalid add_mode: '{add_mode}'. Must be 'pre' or 'post'.")
