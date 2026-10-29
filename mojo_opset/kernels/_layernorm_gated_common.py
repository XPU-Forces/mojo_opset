"""Shared Triton-Ascend implementation for fused gated normalization."""

import os

from functools import lru_cache

import torch
import triton
import triton.language as tl

MAX_ROWS_PER_BLOCK = 4
_TRUE_VALUES = {"1", "true", "yes", "on"}


@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,
    Y,
    W,
    B,
    Z,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    stride_z_token,
    stride_z_head,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    Z_IS_3D: tl.constexpr,
    Z_HEADS: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_GDC: tl.constexpr = False,
    Q=None,
    S=None,
    QUANT: tl.constexpr = False,
    EVEN_ROWS: tl.constexpr = False,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    group = tl.program_id(1)
    num_row_tasks = (M + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols[None, :] < N
    col_offsets = cols[None, :] + group * N
    if N == BLOCK_N:
        weight = tl.load(W + col_offsets).to(tl.float32)
    else:
        weight = tl.load(W + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        if N == BLOCK_N:
            bias = tl.load(B + col_offsets).to(tl.float32)
        else:
            bias = tl.load(B + col_offsets, mask=col_mask, other=0.0).to(
                tl.float32
            )

    for row_task_id in range(pid, num_row_tasks, num_programs):
        row_start = row_task_id * ROWS_PER_BLOCK
        rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
        if EVEN_ROWS:
            mask = col_mask
        else:
            mask = (rows[:, None] < M) & col_mask
        x_offsets = rows[:, None] * stride_x_row + col_offsets
        y_offsets = rows[:, None] * stride_y_row + col_offsets
        if EVEN_ROWS and N == BLOCK_N:
            x = tl.load(X + x_offsets).to(tl.float32)
        else:
            x = tl.load(X + x_offsets, mask=mask, other=0.0).to(tl.float32)

        if HAS_Z:
            if Z_IS_3D:
                z_row_offsets = (
                    (rows[:, None] // Z_HEADS) * stride_z_token
                    + (rows[:, None] % Z_HEADS) * stride_z_head
                )
            else:
                z_row_offsets = rows[:, None] * stride_z_row

        if HAS_Z and not NORM_BEFORE_GATE:
            if EVEN_ROWS and N == BLOCK_N:
                z = tl.load(Z + z_row_offsets + col_offsets).to(tl.float32)
            else:
                z = tl.load(
                    Z + z_row_offsets + col_offsets, mask=mask, other=0.0
                ).to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                x *= z * tl.sigmoid(z)
            elif ACTIVATION == "sigmoid":
                x *= tl.sigmoid(z)

        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=1) / N
            tl.store(Mean + group * M + rows, mean, mask=rows < M)
            if EVEN_ROWS and N == BLOCK_N:
                xbar = x - mean[:, None]
            else:
                xbar = tl.where(mask, x - mean[:, None], 0.0)
            var = tl.sum(xbar * xbar, axis=1) / N
        else:
            if EVEN_ROWS and N == BLOCK_N:
                xbar = x
            else:
                xbar = tl.where(mask, x, 0.0)
            var = tl.sum(xbar * xbar, axis=1) / N
            mean = 0.0

        rstd = tl.rsqrt(var + eps)
        tl.store(Rstd + group * M + rows, rstd, mask=rows < M)
        if IS_RMS_NORM:
            y = x * rstd[:, None] * weight
        else:
            y = (x - mean[:, None]) * rstd[:, None] * weight
        if HAS_BIAS:
            y += bias

        if HAS_Z and NORM_BEFORE_GATE:
            if EVEN_ROWS and N == BLOCK_N:
                z = tl.load(Z + z_row_offsets + col_offsets).to(tl.float32)
            else:
                z = tl.load(
                    Z + z_row_offsets + col_offsets, mask=mask, other=0.0
                ).to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                y *= z * tl.sigmoid(z)
            elif ACTIVATION == "sigmoid":
                y *= tl.sigmoid(z)

        if QUANT:
            if EVEN_ROWS and N == BLOCK_N:
                y = y.to(Y.dtype.element_ty).to(tl.float32)
            else:
                y = tl.where(mask, y.to(Y.dtype.element_ty).to(tl.float32), 0.0)
            scale = tl.maximum(tl.max(tl.abs(y)), 1e-10) / 448.0
            quantized = tl.clamp(y * (1.0 / scale), -448.0, 448.0).to(
                Q.dtype.element_ty
            )
            if EVEN_ROWS and N == BLOCK_N:
                tl.store(Q + rows[:, None] * N + col_offsets, quantized)
            else:
                tl.store(
                    Q + rows[:, None] * N + col_offsets,
                    quantized,
                    mask=mask,
                )
            tl.store(S + row_task_id, scale)
        else:
            if EVEN_ROWS and N == BLOCK_N:
                tl.store(Y + y_offsets, y)
            else:
                tl.store(Y + y_offsets, y, mask=mask)


@triton.jit
def _layer_norm_fwd_chunked_kernel(
    X,
    Y,
    W,
    B,
    Z,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    stride_z_token,
    stride_z_head,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    Z_IS_3D: tl.constexpr,
    Z_HEADS: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
    EVEN_ROWS: tl.constexpr = False,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    group = tl.program_id(1)
    num_row_tasks = (M + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    cols = tl.arange(0, BLOCK_N)

    for row_task_id in range(pid, num_row_tasks, num_programs):
        row_start = row_task_id * ROWS_PER_BLOCK
        rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
        row_mask = rows[:, None] < M
        if HAS_Z:
            if Z_IS_3D:
                z_row_offsets = (
                    (rows[:, None] // Z_HEADS) * stride_z_token
                    + (rows[:, None] % Z_HEADS) * stride_z_head
                )
            else:
                z_row_offsets = rows[:, None] * stride_z_row

        sum_acc = tl.zeros((ROWS_PER_BLOCK,), dtype=tl.float32)
        square_acc = tl.zeros((ROWS_PER_BLOCK,), dtype=tl.float32)
        for col_start in range(0, N, BLOCK_N):
            chunk_cols = col_start + cols
            col_offsets = chunk_cols[None, :] + group * N
            if EVEN_ROWS and N % BLOCK_N == 0:
                mask = None
                x = tl.load(
                    X + rows[:, None] * stride_x_row + col_offsets
                ).to(tl.float32)
            else:
                mask = row_mask & (chunk_cols[None, :] < N)
                x = tl.load(
                    X + rows[:, None] * stride_x_row + col_offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
            if HAS_Z and not NORM_BEFORE_GATE:
                if EVEN_ROWS and N % BLOCK_N == 0:
                    z = tl.load(Z + z_row_offsets + col_offsets).to(tl.float32)
                else:
                    z = tl.load(
                        Z + z_row_offsets + col_offsets,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    x *= z * tl.sigmoid(z)
                elif ACTIVATION == "sigmoid":
                    x *= tl.sigmoid(z)
            if not (EVEN_ROWS and N % BLOCK_N == 0):
                x = tl.where(mask, x, 0.0)
            sum_acc += tl.sum(x, axis=1)
            square_acc += tl.sum(x * x, axis=1)

        if IS_RMS_NORM:
            mean = 0.0
            var = square_acc / N
        else:
            mean = sum_acc / N
            var = tl.maximum(square_acc / N - mean * mean, 0.0)
        rstd = tl.rsqrt(var + eps)
        if IS_RMS_NORM:
            tl.store(Rstd + group * M + rows, rstd, mask=rows < M)
        else:
            tl.store(Mean + group * M + rows, mean, mask=rows < M)
            tl.store(Rstd + group * M + rows, rstd, mask=rows < M)

        for col_start in range(0, N, BLOCK_N):
            chunk_cols = col_start + cols
            col_offsets = chunk_cols[None, :] + group * N
            if EVEN_ROWS and N % BLOCK_N == 0:
                mask = None
                x = tl.load(
                    X + rows[:, None] * stride_x_row + col_offsets
                ).to(tl.float32)
                weight = tl.load(W + col_offsets).to(tl.float32)
                if HAS_BIAS:
                    bias = tl.load(B + col_offsets).to(tl.float32)
            else:
                mask = row_mask & (chunk_cols[None, :] < N)
                x = tl.load(
                    X + rows[:, None] * stride_x_row + col_offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                weight = tl.load(
                    W + col_offsets,
                    mask=chunk_cols[None, :] < N,
                    other=0.0,
                ).to(tl.float32)
                if HAS_BIAS:
                    bias = tl.load(
                        B + col_offsets,
                        mask=chunk_cols[None, :] < N,
                        other=0.0,
                    ).to(tl.float32)

            if HAS_Z and not NORM_BEFORE_GATE:
                if EVEN_ROWS and N % BLOCK_N == 0:
                    z = tl.load(Z + z_row_offsets + col_offsets).to(tl.float32)
                else:
                    z = tl.load(
                        Z + z_row_offsets + col_offsets,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    x *= z * tl.sigmoid(z)
                elif ACTIVATION == "sigmoid":
                    x *= tl.sigmoid(z)
            if IS_RMS_NORM:
                y = x * rstd[:, None] * weight
            else:
                y = (x - mean[:, None]) * rstd[:, None] * weight
            if HAS_BIAS:
                y += bias
            if HAS_Z and NORM_BEFORE_GATE:
                if EVEN_ROWS and N % BLOCK_N == 0:
                    z = tl.load(Z + z_row_offsets + col_offsets).to(tl.float32)
                else:
                    z = tl.load(
                        Z + z_row_offsets + col_offsets,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    y *= z * tl.sigmoid(z)
                elif ACTIVATION == "sigmoid":
                    y *= tl.sigmoid(z)
            y_offsets = rows[:, None] * stride_y_row + col_offsets
            if EVEN_ROWS and N % BLOCK_N == 0:
                tl.store(Y + y_offsets, y)
            else:
                tl.store(Y + y_offsets, y, mask=mask)


def is_batch_invariant() -> bool:
    """Read the task-defined batch-invariant switch on each dispatch."""

    return os.getenv("LN_GATED_BATCH_INVARIANT", "").strip().lower() in _TRUE_VALUES


@lru_cache(maxsize=None)
def _vector_core_count() -> int:
    try:
        from mojo_opset.kernels._npu_triton_utils import get_num_cores

        return int(get_num_cores("vector"))
    except (ImportError, KeyError, RuntimeError, TypeError):
        properties = triton.runtime.driver.active.utils.get_device_properties("npu")
        return int(properties["num_vectorcore"])


def calc_rows_per_block(M: int, N: int | None = None) -> int:
    """Choose a bounded row tile while honoring the exact batch-invariance rule."""

    if is_batch_invariant():
        return MAX_ROWS_PER_BLOCK
    override = os.getenv("LN_GATED_ROWS_PER_BLOCK")
    if override:
        return int(override)
    if N is None:
        rows_per_block = triton.next_power_of_2(
            triton.cdiv(M, 2 * _vector_core_count())
        )
        return min(rows_per_block, MAX_ROWS_PER_BLOCK)
    if N <= 128:
        # Large streaming tensors lose too much occupancy with a 64-row tile.
        target_rows = 32 if M * N >= 32 * 1024 * 1024 else 64
    elif N <= 256:
        target_rows = 32
    elif N <= 512:
        target_rows = 8
    elif N <= 1024:
        target_rows = 4
    else:
        target_rows = 2
    return min(target_rows, triton.next_power_of_2(M))


def _layer_norm_fwd_impl(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
    quant_heads: int = 0,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1

    z_is_3d = z is not None and z.ndim == 3
    if z is not None:
        assert z.stride(-1) == 1
        if z_is_3d:
            assert z.shape[0] * z.shape[1] == M
            assert z.shape[2] == N
        else:
            assert z.shape == (M, N)

    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.shape == (N,)
        assert bias.stride(-1) == 1

    if out is None:
        out = torch.empty_like(x)
    assert out.shape == x.shape
    assert out.stride(-1) == 1
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

    max_fused_size = 65536 // x.element_size()
    block_n = min(max_fused_size, triton.next_power_of_2(group_size))
    if group_size > block_n:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    if group_size != block_n and group_size <= 256:
        block_n = group_size
    use_chunked = False

    num_warps = min(max(block_n // 256, 1), 8)
    if os.getenv("LN_GATED_NUM_WARPS"):
        num_warps = int(os.environ["LN_GATED_NUM_WARPS"])
    rows_per_block = calc_rows_per_block(M, block_n)
    quant_args = {}
    if quant_heads:
        assert ngroups == 1 and M % quant_heads == 0
        num_warps = min(num_warps * quant_heads // rows_per_block, 16)
        rows_per_block = quant_heads
        q = torch.empty(
            (M // quant_heads, quant_heads * N),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
        s = torch.empty((M // quant_heads, 1), dtype=torch.float32, device=x.device)
        quant_args = {"Q": q, "S": s, "QUANT": True}

    row_tasks = triton.cdiv(M, rows_per_block)
    grid = (min(_vector_core_count(), row_tasks), ngroups)
    common_args = (
        x,
        out,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.stride(0),
        out.stride(0),
        z.stride(0) if z is not None and not z_is_3d else 0,
        z.stride(0) if z_is_3d else 0,
        z.stride(1) if z_is_3d else 0,
        M,
        group_size,
        eps,
    )
    meta = {
        "BLOCK_N": block_n,
        "ROWS_PER_BLOCK": rows_per_block,
        "HAS_BIAS": bias is not None,
        "HAS_Z": z is not None,
        "Z_IS_3D": z_is_3d,
        "Z_HEADS": z.shape[1] if z_is_3d else 1,
        "NORM_BEFORE_GATE": norm_before_gate,
        "IS_RMS_NORM": is_rms_norm,
        "EVEN_ROWS": M % rows_per_block == 0,
        "ACTIVATION": activation,
    }
    if use_chunked:
        _layer_norm_fwd_chunked_kernel[grid](
            *common_args,
            num_warps=num_warps,
            **meta,
        )
    else:
        _layer_norm_fwd_1pass_kernel[grid](
            *common_args,
            num_warps=num_warps,
            **meta,
            **quant_args,
        )
    return ((q, s) if quant_heads else out), mean, rstd
