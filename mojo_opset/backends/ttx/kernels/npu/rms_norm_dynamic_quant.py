# Copyright 2026, The FlagOS Contributors.

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from .utils import libentry

try:
    import triton.experimental.tle as tle
    HAS_TLE = True
except Exception:
    HAS_TLE = False


UB_LIMIT_BYTES = 192 * 1024
ESTIMATED_UB_BYTES_PER_COL = 54
BLOCK_N_ALIGN = 256
MIN_BLOCK_SIZE_N = 256
EXPERIMENTAL_SMALL_COLS_MAX_N = 4096
BLOCK_SIZE_M = 1
PREFILL_MAX_D = 4096

PREFILL_TOKEN_BLOCK_TABLE = {
    256: (4, 40),
    512: (4, 40),
    1024: (2, 80),
    1536: (2, 160),
    2048: (2, 80),
    4096: (1, 40),
}


def _ceil_div(a, b):
    return (a + b - 1) // b


def _ceil_to_multiple(value, multiple):
    return _ceil_div(value, multiple) * multiple


def _floor_to_multiple(value, multiple):
    return value // multiple * multiple


def _select_grid(total_rows):
    num_cores = triton.runtime.driver.active.utils.get_device_properties("npu")[
        "num_vectorcore"
    ]
    if total_rows <= num_cores:
        return total_rows

    if total_rows >= num_cores * 4:
        for g in range(num_cores, num_cores * 2):
            if total_rows % g == 0 and total_rows // g >= 16:
                return g

    for g in range(num_cores, num_cores // 3, -1):
        if total_rows % g == 0 and total_rows // g >= 2:
            return g

    return num_cores


def select_block_size_n(d):
    max_block_n = _floor_to_multiple(
        UB_LIMIT_BYTES // ESTIMATED_UB_BYTES_PER_COL, BLOCK_N_ALIGN
    )
    max_block_n = max(MIN_BLOCK_SIZE_N, max_block_n)
    target_block_n = _ceil_to_multiple(_ceil_div(d, 2), BLOCK_N_ALIGN)
    return max(MIN_BLOCK_SIZE_N, min(target_block_n, max_block_n))


def select_small_cols_block_size_n(d):
    max_block_n = _floor_to_multiple(
        UB_LIMIT_BYTES // ESTIMATED_UB_BYTES_PER_COL, BLOCK_N_ALIGN
    )
    max_block_n = max(MIN_BLOCK_SIZE_N, max_block_n)
    max_block_n = max(max_block_n, EXPERIMENTAL_SMALL_COLS_MAX_N)
    return _ceil_to_multiple(d, BLOCK_N_ALIGN) if d <= max_block_n else None


def should_use_small_cols(d, block_size_n=None):
    small_cols_block_n = select_small_cols_block_size_n(d)
    if small_cols_block_n is None:
        return False
    if block_size_n is None:
        return True
    return block_size_n >= small_cols_block_n


def should_use_no_mask_blocked(d, block_size_n=None):
    if should_use_small_cols(d, block_size_n):
        return False
    block_size_n = select_block_size_n(d) if block_size_n is None else block_size_n
    return d % block_size_n == 0


def should_use_cached2(d, block_size_n=None, smooth_scale=None, beta=None):
    if should_use_small_cols(d, block_size_n):
        return False
    block_size_n = select_block_size_n(d) if block_size_n is None else block_size_n
    return d == block_size_n * 2 and smooth_scale is not None and beta is None


def should_use_prefill_kernel(d, total_rows):
    if total_rows < 1024:
        return False
    if d > PREFILL_MAX_D:
        return False
    if select_small_cols_block_size_n(d) is None:
        return False
    return d in PREFILL_TOKEN_BLOCK_TABLE


def get_prefill_config(d, total_rows):
    if d in PREFILL_TOKEN_BLOCK_TABLE:
        return PREFILL_TOKEN_BLOCK_TABLE[d]
    if d <= 512:
        return 2, _select_grid(total_rows)
    elif d <= 2048:
        return 1, _select_grid(total_rows)
    else:
        return 1, _select_grid(total_rows)


@libentry()
@triton.jit
def _rms_norm_dynamic_quant_small_cols_kernel(
    x_ptr,
    gamma_ptr,
    smooth_ptr,
    beta_ptr,
    y_ptr,
    scale_ptr,
    total_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    HAS_SMOOTH: tl.constexpr,
    HAS_BETA: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < n_cols

    for row in tle.dsa.pipeline(pid, total_rows, grid_size):
        offsets = row * n_cols + cols
        x = tl.load(x_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

        square_sum = tl.sum(x * x, axis=0)
        rstd = tl.rsqrt(square_sum / n_cols + eps)

        q_input = x * rstd * gamma
        if HAS_BETA:
            beta = tl.load(beta_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            q_input = q_input + beta
        if HAS_SMOOTH:
            smooth = tl.load(smooth_ptr + cols, mask=col_mask, other=1.0).to(tl.float32)
            q_input = q_input * smooth

        max_abs = tl.max(tl.abs(q_input), axis=0)
        scale = max_abs / 127.0
        scale = tl.where(scale < 1.0e-6, 1.0, scale)
        tl.store(scale_ptr + row, scale)

        inv_scale = 127.0 / max_abs
        inv_scale = tl.where(max_abs < 1.27e-4, 1.0, inv_scale)
        q = q_input * inv_scale
        q = tl.math.floor(q + 0.5)
        tl.store(y_ptr + offsets, q.to(tl.int8), mask=col_mask)


@libentry()
@triton.jit
def _rms_norm_dynamic_quant_small_cols_kernel_non_tle(
    x_ptr,
    gamma_ptr,
    smooth_ptr,
    beta_ptr,
    y_ptr,
    scale_ptr,
    total_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    HAS_SMOOTH: tl.constexpr,
    HAS_BETA: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Non-TLE variant: uses range() loop instead of tle.dsa.pipeline."""
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < n_cols

    for row in range(pid, total_rows, grid_size):
        offsets = row * n_cols + cols
        x = tl.load(x_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

        square_sum = tl.sum(x * x, axis=0)
        rstd = tl.rsqrt(square_sum / n_cols + eps)

        q_input = x * rstd * gamma
        if HAS_BETA:
            beta = tl.load(beta_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            q_input = q_input + beta
        if HAS_SMOOTH:
            smooth = tl.load(smooth_ptr + cols, mask=col_mask, other=1.0).to(tl.float32)
            q_input = q_input * smooth

        max_abs = tl.max(tl.abs(q_input), axis=0)
        scale = max_abs / 127.0
        scale = tl.where(scale < 1.0e-6, 1.0, scale)
        tl.store(scale_ptr + row, scale)

        inv_scale = 127.0 / max_abs
        inv_scale = tl.where(max_abs < 1.27e-4, 1.0, inv_scale)
        q = q_input * inv_scale
        q = tl.math.floor(q + 0.5)
        tl.store(y_ptr + offsets, q.to(tl.int8), mask=col_mask)


@libentry()
@triton.jit
def _rms_norm_dynamic_quant_small_cols_dsa_smooth_kernel(
    x_ptr,
    gamma_ptr,
    smooth_ptr,
    y_ptr,
    scale_ptr,
    total_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    """small_cols variant with DSA weight pre-load + fusion (no beta)."""
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    cols = tl.arange(0, BLOCK_N)

    gamma_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=gamma_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    smooth_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=smooth_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    tle.dsa.copy(gamma_ptr + cols, gamma_ub, [BLOCK_N])
    tle.dsa.copy(smooth_ptr + cols, smooth_ub, [BLOCK_N])
    g = tle.dsa.to_tensor(gamma_ub).to(tl.float32)
    s = tle.dsa.to_tensor(smooth_ub).to(tl.float32)
    w = g * s  # fused weight, persistent in UB

    x_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=x_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )

    avg_factor: tl.constexpr = 1.0 / n_cols

    for row in tle.dsa.pipeline(pid, total_rows, grid_size):
        offsets = row * n_cols + cols
        tle.dsa.copy(x_ptr + offsets, x_ub, [BLOCK_N])

        x = tle.dsa.to_tensor(x_ub, writable=True).to(tl.float32)

        square_sum = tl.sum(x * x, axis=0)
        rstd = tl.rsqrt(square_sum * avg_factor + eps)

        q = x * rstd * w

        max_abs = tl.max(tl.abs(q), axis=0)
        scale = max_abs / 127.0
        scale = tl.where(scale < 1.0e-6, 1.0, scale)
        tl.store(scale_ptr + row, scale)

        inv_scale = 127.0 / max_abs
        inv_scale = tl.where(max_abs < 1.27e-4, 1.0, inv_scale)
        q_out = q * inv_scale
        q_out = tl.math.floor(q_out + 0.5)
        tl.store(y_ptr + offsets, q_out.to(tl.int8))


@libentry()
@triton.jit
def _rms_norm_dynamic_quant_small_cols_dsa_smooth_kernel_non_tle(
    x_ptr,
    gamma_ptr,
    smooth_ptr,
    y_ptr,
    scale_ptr,
    total_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    """Non-TLE variant: uses tl.load instead of tle.dsa.alloc/copy/to_tensor + pipeline."""
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    cols = tl.arange(0, BLOCK_N)

    g = tl.load(gamma_ptr + cols).to(tl.float32)
    s = tl.load(smooth_ptr + cols).to(tl.float32)
    w = g * s  # fused weight, fp32

    avg_factor: tl.constexpr = 1.0 / n_cols

    for row in range(pid, total_rows, grid_size):
        offsets = row * n_cols + cols
        x = tl.load(x_ptr + offsets).to(tl.float32)

        square_sum = tl.sum(x * x, axis=0)
        rstd = tl.rsqrt(square_sum * avg_factor + eps)

        q = x * rstd * w

        max_abs = tl.max(tl.abs(q), axis=0)
        scale = max_abs / 127.0
        scale = tl.where(scale < 1.0e-6, 1.0, scale)
        tl.store(scale_ptr + row, scale)

        inv_scale = 127.0 / max_abs
        inv_scale = tl.where(max_abs < 1.27e-4, 1.0, inv_scale)
        q_out = q * inv_scale
        q_out = tl.math.floor(q_out + 0.5)
        tl.store(y_ptr + offsets, q_out.to(tl.int8))


@libentry()
@triton.jit
def _rms_norm_dynamic_quant_small_cols_dsa_smooth_prefill_kernel(
    x_ptr,
    gamma_ptr,
    smooth_ptr,
    y_ptr,
    scale_ptr,
    total_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """BLOCK_M=2 row-batched kernel for prefill (small D, large total_rows)."""
    pid = tl.program_id(axis=0)
    gs = tl.num_programs(axis=0)
    cols = tl.arange(0, BLOCK_N)

    gamma_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=gamma_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    smooth_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=smooth_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    tle.dsa.copy(gamma_ptr + cols, gamma_ub, [BLOCK_N])
    tle.dsa.copy(smooth_ptr + cols, smooth_ub, [BLOCK_N])
    w = tle.dsa.to_tensor(gamma_ub).to(tl.float32) * tle.dsa.to_tensor(smooth_ub).to(
        tl.float32
    )

    x0_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=x_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    x1_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=x_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )

    af: tl.constexpr = 1.0 / n_cols

    for batch in range(pid * BLOCK_M, total_rows, gs * BLOCK_M):
        r0 = batch
        r1 = batch + 1

        tle.dsa.copy(x_ptr + r0 * n_cols + cols, x0_ub, [BLOCK_N])
        tle.dsa.copy(x_ptr + r1 * n_cols + cols, x1_ub, [BLOCK_N])

        v0 = tle.dsa.to_tensor(x0_ub, writable=True).to(tl.float32)
        v1 = tle.dsa.to_tensor(x1_ub, writable=True).to(tl.float32)

        rs0 = tl.rsqrt(tl.sum(v0 * v0, axis=0) * af + eps)
        rs1 = tl.rsqrt(tl.sum(v1 * v1, axis=0) * af + eps)

        q0 = v0 * rs0 * w
        q1 = v1 * rs1 * w

        m0 = tl.max(tl.abs(q0), axis=0)
        m1 = tl.max(tl.abs(q1), axis=0)

        tl.store(scale_ptr + r0, m0 / 127.0)
        tl.store(scale_ptr + r1, m1 / 127.0)

        qo0 = tl.math.floor(q0 * (127.0 / m0) + 0.5)
        qo1 = tl.math.floor(q1 * (127.0 / m1) + 0.5)

        tl.store(y_ptr + r0 * n_cols + cols, qo0.to(tl.int8))
        tl.store(y_ptr + r1 * n_cols + cols, qo1.to(tl.int8))


@libentry()
@triton.jit
def _rms_norm_dynamic_quant_small_cols_dsa_smooth_prefill_kernel_non_tle(
    x_ptr,
    gamma_ptr,
    smooth_ptr,
    y_ptr,
    scale_ptr,
    total_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Non-TLE variant of BLOCK_M=2 prefill kernel: tl.load instead of tle.dsa."""
    pid = tl.program_id(axis=0)
    gs = tl.num_programs(axis=0)
    cols = tl.arange(0, BLOCK_N)

    g = tl.load(gamma_ptr + cols).to(tl.float32)
    s = tl.load(smooth_ptr + cols).to(tl.float32)
    w = g * s

    af: tl.constexpr = 1.0 / n_cols

    for batch in range(pid * BLOCK_M, total_rows, gs * BLOCK_M):
        r0 = batch
        r1 = batch + 1

        v0 = tl.load(x_ptr + r0 * n_cols + cols).to(tl.float32)
        v1 = tl.load(x_ptr + r1 * n_cols + cols).to(tl.float32)

        rs0 = tl.rsqrt(tl.sum(v0 * v0, axis=0) * af + eps)
        rs1 = tl.rsqrt(tl.sum(v1 * v1, axis=0) * af + eps)

        q0 = v0 * rs0 * w
        q1 = v1 * rs1 * w

        m0 = tl.max(tl.abs(q0), axis=0)
        m1 = tl.max(tl.abs(q1), axis=0)

        tl.store(scale_ptr + r0, m0 / 127.0)
        tl.store(scale_ptr + r1, m1 / 127.0)

        qo0 = tl.math.floor(q0 * (127.0 / m0) + 0.5)
        qo1 = tl.math.floor(q1 * (127.0 / m1) + 0.5)

        tl.store(y_ptr + r0 * n_cols + cols, qo0.to(tl.int8))
        tl.store(y_ptr + r1 * n_cols + cols, qo1.to(tl.int8))


@libentry()
@triton.jit
def _rms_norm_dynamic_quant_small_cols_dsa_smooth_prefill_m4_kernel(
    x_ptr,
    gamma_ptr,
    smooth_ptr,
    y_ptr,
    scale_ptr,
    total_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """BLOCK_M=4 row-batched kernel for prefill (D=256, D=512)."""
    pid = tl.program_id(axis=0)
    gs = tl.num_programs(axis=0)
    cols = tl.arange(0, BLOCK_N)

    gamma_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=gamma_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    smooth_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=smooth_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    tle.dsa.copy(gamma_ptr + cols, gamma_ub, [BLOCK_N])
    tle.dsa.copy(smooth_ptr + cols, smooth_ub, [BLOCK_N])
    w = tle.dsa.to_tensor(gamma_ub).to(tl.float32) * tle.dsa.to_tensor(smooth_ub).to(
        tl.float32
    )

    x0_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=x_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    x1_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=x_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    x2_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=x_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    x3_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=x_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )

    af: tl.constexpr = 1.0 / n_cols

    for batch in range(pid * BLOCK_M, total_rows, gs * BLOCK_M):
        r0 = batch
        r1 = batch + 1
        r2 = batch + 2
        r3 = batch + 3

        tle.dsa.copy(x_ptr + r0 * n_cols + cols, x0_ub, [BLOCK_N])
        tle.dsa.copy(x_ptr + r1 * n_cols + cols, x1_ub, [BLOCK_N])
        tle.dsa.copy(x_ptr + r2 * n_cols + cols, x2_ub, [BLOCK_N])
        tle.dsa.copy(x_ptr + r3 * n_cols + cols, x3_ub, [BLOCK_N])

        v0 = tle.dsa.to_tensor(x0_ub, writable=True).to(tl.float32)
        v1 = tle.dsa.to_tensor(x1_ub, writable=True).to(tl.float32)
        v2 = tle.dsa.to_tensor(x2_ub, writable=True).to(tl.float32)
        v3 = tle.dsa.to_tensor(x3_ub, writable=True).to(tl.float32)

        rs0 = tl.rsqrt(tl.sum(v0 * v0, axis=0) * af + eps)
        rs1 = tl.rsqrt(tl.sum(v1 * v1, axis=0) * af + eps)
        rs2 = tl.rsqrt(tl.sum(v2 * v2, axis=0) * af + eps)
        rs3 = tl.rsqrt(tl.sum(v3 * v3, axis=0) * af + eps)

        q0 = v0 * rs0 * w
        q1 = v1 * rs1 * w
        q2 = v2 * rs2 * w
        q3 = v3 * rs3 * w

        m0 = tl.max(tl.abs(q0), axis=0)
        m1 = tl.max(tl.abs(q1), axis=0)
        m2 = tl.max(tl.abs(q2), axis=0)
        m3 = tl.max(tl.abs(q3), axis=0)

        tl.store(scale_ptr + r0, m0 / 127.0)
        tl.store(scale_ptr + r1, m1 / 127.0)
        tl.store(scale_ptr + r2, m2 / 127.0)
        tl.store(scale_ptr + r3, m3 / 127.0)

        qo0 = tl.math.floor(q0 * (127.0 / m0) + 0.5)
        qo1 = tl.math.floor(q1 * (127.0 / m1) + 0.5)
        qo2 = tl.math.floor(q2 * (127.0 / m2) + 0.5)
        qo3 = tl.math.floor(q3 * (127.0 / m3) + 0.5)

        tl.store(y_ptr + r0 * n_cols + cols, qo0.to(tl.int8))
        tl.store(y_ptr + r1 * n_cols + cols, qo1.to(tl.int8))
        tl.store(y_ptr + r2 * n_cols + cols, qo2.to(tl.int8))
        tl.store(y_ptr + r3 * n_cols + cols, qo3.to(tl.int8))


@libentry()
@triton.jit
def _rms_norm_dynamic_quant_small_cols_dsa_smooth_prefill_m4_kernel_non_tle(
    x_ptr,
    gamma_ptr,
    smooth_ptr,
    y_ptr,
    scale_ptr,
    total_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Non-TLE variant of BLOCK_M=4 prefill kernel: tl.load instead of tle.dsa."""
    pid = tl.program_id(axis=0)
    gs = tl.num_programs(axis=0)
    cols = tl.arange(0, BLOCK_N)

    g = tl.load(gamma_ptr + cols).to(tl.float32)
    s = tl.load(smooth_ptr + cols).to(tl.float32)
    w = g * s

    af: tl.constexpr = 1.0 / n_cols

    for batch in range(pid * BLOCK_M, total_rows, gs * BLOCK_M):
        r0 = batch
        r1 = batch + 1
        r2 = batch + 2
        r3 = batch + 3

        v0 = tl.load(x_ptr + r0 * n_cols + cols).to(tl.float32)
        v1 = tl.load(x_ptr + r1 * n_cols + cols).to(tl.float32)
        v2 = tl.load(x_ptr + r2 * n_cols + cols).to(tl.float32)
        v3 = tl.load(x_ptr + r3 * n_cols + cols).to(tl.float32)

        rs0 = tl.rsqrt(tl.sum(v0 * v0, axis=0) * af + eps)
        rs1 = tl.rsqrt(tl.sum(v1 * v1, axis=0) * af + eps)
        rs2 = tl.rsqrt(tl.sum(v2 * v2, axis=0) * af + eps)
        rs3 = tl.rsqrt(tl.sum(v3 * v3, axis=0) * af + eps)

        q0 = v0 * rs0 * w
        q1 = v1 * rs1 * w
        q2 = v2 * rs2 * w
        q3 = v3 * rs3 * w

        m0 = tl.max(tl.abs(q0), axis=0)
        m1 = tl.max(tl.abs(q1), axis=0)
        m2 = tl.max(tl.abs(q2), axis=0)
        m3 = tl.max(tl.abs(q3), axis=0)

        tl.store(scale_ptr + r0, m0 / 127.0)
        tl.store(scale_ptr + r1, m1 / 127.0)
        tl.store(scale_ptr + r2, m2 / 127.0)
        tl.store(scale_ptr + r3, m3 / 127.0)

        qo0 = tl.math.floor(q0 * (127.0 / m0) + 0.5)
        qo1 = tl.math.floor(q1 * (127.0 / m1) + 0.5)
        qo2 = tl.math.floor(q2 * (127.0 / m2) + 0.5)
        qo3 = tl.math.floor(q3 * (127.0 / m3) + 0.5)

        tl.store(y_ptr + r0 * n_cols + cols, qo0.to(tl.int8))
        tl.store(y_ptr + r1 * n_cols + cols, qo1.to(tl.int8))
        tl.store(y_ptr + r2 * n_cols + cols, qo2.to(tl.int8))
        tl.store(y_ptr + r3 * n_cols + cols, qo3.to(tl.int8))


@libentry()
@triton.jit
def _rms_norm_dynamic_quant_small_cols_dsa_smooth_prefill_m1_kernel(
    x_ptr,
    gamma_ptr,
    smooth_ptr,
    y_ptr,
    scale_ptr,
    total_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """BLOCK_M=1 prefill kernel for large D (e.g. 4096) using pipeline."""
    pid = tl.program_id(axis=0)
    gs = tl.num_programs(axis=0)
    cols = tl.arange(0, BLOCK_N)

    gamma_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=gamma_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    smooth_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=smooth_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    tle.dsa.copy(gamma_ptr + cols, gamma_ub, [BLOCK_N])
    tle.dsa.copy(smooth_ptr + cols, smooth_ub, [BLOCK_N])
    w = tle.dsa.to_tensor(gamma_ub).to(tl.float32) * tle.dsa.to_tensor(smooth_ub).to(
        tl.float32
    )

    x_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=x_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )

    af: tl.constexpr = 1.0 / n_cols

    for row in tle.dsa.pipeline(pid, total_rows, gs):
        tle.dsa.copy(x_ptr + row * n_cols + cols, x_ub, [BLOCK_N])
        v = tle.dsa.to_tensor(x_ub, writable=True).to(tl.float32)
        rs = tl.rsqrt(tl.sum(v * v, axis=0) * af + eps)
        q = v * rs * w
        m = tl.max(tl.abs(q), axis=0)
        tl.store(scale_ptr + row, m / 127.0)
        qo = tl.math.floor(q * (127.0 / m) + 0.5)
        tl.store(y_ptr + row * n_cols + cols, qo.to(tl.int8))


@libentry()
@triton.jit
def _rms_norm_dynamic_quant_small_cols_dsa_smooth_prefill_m1_kernel_non_tle(
    x_ptr,
    gamma_ptr,
    smooth_ptr,
    y_ptr,
    scale_ptr,
    total_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Non-TLE variant of BLOCK_M=1 prefill kernel: tl.load + range instead of tle.dsa."""
    pid = tl.program_id(axis=0)
    gs = tl.num_programs(axis=0)
    cols = tl.arange(0, BLOCK_N)

    g = tl.load(gamma_ptr + cols).to(tl.float32)
    s = tl.load(smooth_ptr + cols).to(tl.float32)
    w = g * s

    af: tl.constexpr = 1.0 / n_cols

    for row in range(pid, total_rows, gs):
        v = tl.load(x_ptr + row * n_cols + cols).to(tl.float32)
        rs = tl.rsqrt(tl.sum(v * v, axis=0) * af + eps)
        q = v * rs * w
        m = tl.max(tl.abs(q), axis=0)
        tl.store(scale_ptr + row, m / 127.0)
        qo = tl.math.floor(q * (127.0 / m) + 0.5)
        tl.store(y_ptr + row * n_cols + cols, qo.to(tl.int8))


@libentry()
@triton.jit
def _rms_norm_dynamic_quant_cached2_fused_kernel(
    x_ptr,
    gamma_ptr,
    smooth_ptr,
    y_ptr,
    scale_ptr,
    total_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    """cached2 with weight fusion: pre-combine w = gamma*smooth in fp32 at init."""
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    cols0 = tl.arange(0, BLOCK_N)
    cols1 = BLOCK_N + cols0

    g0_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=gamma_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    g1_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=gamma_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    s0_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=smooth_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    s1_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=smooth_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    tle.dsa.copy(gamma_ptr + cols0, g0_ub, [BLOCK_N])
    tle.dsa.copy(gamma_ptr + cols1, g1_ub, [BLOCK_N])
    tle.dsa.copy(smooth_ptr + cols0, s0_ub, [BLOCK_N])
    tle.dsa.copy(smooth_ptr + cols1, s1_ub, [BLOCK_N])

    g0 = tle.dsa.to_tensor(g0_ub).to(tl.float32)
    s0 = tle.dsa.to_tensor(s0_ub).to(tl.float32)
    w0 = g0 * s0

    g1 = tle.dsa.to_tensor(g1_ub).to(tl.float32)
    s1 = tle.dsa.to_tensor(s1_ub).to(tl.float32)
    w1 = g1 * s1

    x0_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=x_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )
    x1_ub = tle.dsa.alloc(
        [BLOCK_N], dtype=x_ptr.dtype.element_ty, mem_addr_space=tle.dsa.ascend.UB
    )

    avg_factor: tl.constexpr = 1.0 / n_cols

    for row in range(pid, total_rows, grid_size):
        offsets0 = row * n_cols + cols0
        offsets1 = row * n_cols + cols1

        tle.dsa.copy(x_ptr + offsets0, x0_ub, [BLOCK_N])
        tle.dsa.copy(x_ptr + offsets1, x1_ub, [BLOCK_N])
        x0 = tle.dsa.to_tensor(x0_ub, writable=True).to(tl.float32)
        x1 = tle.dsa.to_tensor(x1_ub, writable=True).to(tl.float32)

        square_sum = tl.sum(x0 * x0, axis=0) + tl.sum(x1 * x1, axis=0)
        rstd = tl.rsqrt(square_sum * avg_factor + eps)

        q0 = x0 * rstd * w0
        q1 = x1 * rstd * w1

        max0 = tl.max(tl.abs(q0), axis=0)
        max1 = tl.max(tl.abs(q1), axis=0)
        max_abs = tl.maximum(max0, max1)
        scale = max_abs / 127.0
        scale = tl.where(scale < 1.0e-6, 1.0, scale)
        tl.store(scale_ptr + row, scale)
        inv_scale = 127.0 / max_abs
        inv_scale = tl.where(max_abs < 1.27e-4, 1.0, inv_scale)

        q0_out = q0 * inv_scale
        q0_out = tl.math.floor(q0_out + 0.5)
        tl.store(y_ptr + offsets0, q0_out.to(tl.int8))

        q1_out = q1 * inv_scale
        q1_out = tl.math.floor(q1_out + 0.5)
        tl.store(y_ptr + offsets1, q1_out.to(tl.int8))


@libentry()
@triton.jit
def _rms_norm_dynamic_quant_cached2_fused_kernel_non_tle(
    x_ptr,
    gamma_ptr,
    smooth_ptr,
    y_ptr,
    scale_ptr,
    total_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    """Non-TLE variant of cached2_fused: tl.load instead of tle.dsa.alloc/copy/to_tensor."""
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    cols0 = tl.arange(0, BLOCK_N)
    cols1 = BLOCK_N + cols0

    g0 = tl.load(gamma_ptr + cols0).to(tl.float32)
    g1 = tl.load(gamma_ptr + cols1).to(tl.float32)
    s0 = tl.load(smooth_ptr + cols0).to(tl.float32)
    s1 = tl.load(smooth_ptr + cols1).to(tl.float32)
    w0 = g0 * s0
    w1 = g1 * s1

    avg_factor: tl.constexpr = 1.0 / n_cols

    for row in range(pid, total_rows, grid_size):
        offsets0 = row * n_cols + cols0
        offsets1 = row * n_cols + cols1

        x0 = tl.load(x_ptr + offsets0).to(tl.float32)
        x1 = tl.load(x_ptr + offsets1).to(tl.float32)

        square_sum = tl.sum(x0 * x0, axis=0) + tl.sum(x1 * x1, axis=0)
        rstd = tl.rsqrt(square_sum * avg_factor + eps)

        q0 = x0 * rstd * w0
        q1 = x1 * rstd * w1

        max0 = tl.max(tl.abs(q0), axis=0)
        max1 = tl.max(tl.abs(q1), axis=0)
        max_abs = tl.maximum(max0, max1)
        scale = max_abs / 127.0
        scale = tl.where(scale < 1.0e-6, 1.0, scale)
        tl.store(scale_ptr + row, scale)
        inv_scale = 127.0 / max_abs
        inv_scale = tl.where(max_abs < 1.27e-4, 1.0, inv_scale)

        q0_out = q0 * inv_scale
        q0_out = tl.math.floor(q0_out + 0.5)
        tl.store(y_ptr + offsets0, q0_out.to(tl.int8))

        q1_out = q1 * inv_scale
        q1_out = tl.math.floor(q1_out + 0.5)
        tl.store(y_ptr + offsets1, q1_out.to(tl.int8))


@libentry()
@triton.jit
def _rms_norm_dynamic_quant_no_mask_kernel(
    x_ptr,
    gamma_ptr,
    smooth_ptr,
    beta_ptr,
    y_ptr,
    scale_ptr,
    total_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    HAS_SMOOTH: tl.constexpr,
    HAS_BETA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    for row in range(pid, total_rows, grid_size):
        square_sum = tl.full((), 0.0, dtype=tl.float32)
        for col_start in tle.dsa.pipeline(0, n_cols, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            x = tl.load(x_ptr + row * n_cols + cols).to(tl.float32)
            square_sum += tl.sum(x * x, axis=0)

        rstd = tl.rsqrt(square_sum / n_cols + eps)

        max_abs = tl.full((), 0.0, dtype=tl.float32)
        for col_start in tle.dsa.pipeline(0, n_cols, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            x = tl.load(x_ptr + row * n_cols + cols).to(tl.float32)
            gamma = tl.load(gamma_ptr + cols).to(tl.float32)
            q_input = x * rstd * gamma
            if HAS_BETA:
                beta = tl.load(beta_ptr + cols).to(tl.float32)
                q_input = q_input + beta
            if HAS_SMOOTH:
                smooth = tl.load(smooth_ptr + cols).to(tl.float32)
                q_input = q_input * smooth
            max_abs = tl.maximum(max_abs, tl.max(tl.abs(q_input), axis=0))

        scale = max_abs / 127.0
        scale = tl.where(scale < 1.0e-6, 1.0, scale)
        tl.store(scale_ptr + row, scale)

        inv_scale = 1.0 / scale
        for col_start in tle.dsa.pipeline(0, n_cols, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            offsets = row * n_cols + cols
            x = tl.load(x_ptr + offsets).to(tl.float32)
            gamma = tl.load(gamma_ptr + cols).to(tl.float32)
            q = x * rstd * gamma
            if HAS_BETA:
                beta = tl.load(beta_ptr + cols).to(tl.float32)
                q = q + beta
            if HAS_SMOOTH:
                smooth = tl.load(smooth_ptr + cols).to(tl.float32)
                q = q * smooth
            q = q * inv_scale
            q = tl.math.floor(q + 0.5)
            q_i8 = q.to(tl.int8)
            tl.store(y_ptr + offsets, q_i8)


@libentry()
@triton.jit
def _rms_norm_dynamic_quant_no_mask_kernel_non_tle(
    x_ptr,
    gamma_ptr,
    smooth_ptr,
    beta_ptr,
    y_ptr,
    scale_ptr,
    total_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    HAS_SMOOTH: tl.constexpr,
    HAS_BETA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Non-TLE variant: range() loop instead of tle.dsa.pipeline for column iteration."""
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    for row in range(pid, total_rows, grid_size):
        square_sum = tl.full((), 0.0, dtype=tl.float32)
        for col_start in range(0, n_cols, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            x = tl.load(x_ptr + row * n_cols + cols).to(tl.float32)
            square_sum += tl.sum(x * x, axis=0)

        rstd = tl.rsqrt(square_sum / n_cols + eps)

        max_abs = tl.full((), 0.0, dtype=tl.float32)
        for col_start in range(0, n_cols, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            x = tl.load(x_ptr + row * n_cols + cols).to(tl.float32)
            gamma = tl.load(gamma_ptr + cols).to(tl.float32)
            q_input = x * rstd * gamma
            if HAS_BETA:
                beta = tl.load(beta_ptr + cols).to(tl.float32)
                q_input = q_input + beta
            if HAS_SMOOTH:
                smooth = tl.load(smooth_ptr + cols).to(tl.float32)
                q_input = q_input * smooth
            max_abs = tl.maximum(max_abs, tl.max(tl.abs(q_input), axis=0))

        scale = max_abs / 127.0
        scale = tl.where(scale < 1.0e-6, 1.0, scale)
        tl.store(scale_ptr + row, scale)

        inv_scale = 1.0 / scale
        for col_start in range(0, n_cols, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            offsets = row * n_cols + cols
            x = tl.load(x_ptr + offsets).to(tl.float32)
            gamma = tl.load(gamma_ptr + cols).to(tl.float32)
            q = x * rstd * gamma
            if HAS_BETA:
                beta = tl.load(beta_ptr + cols).to(tl.float32)
                q = q + beta
            if HAS_SMOOTH:
                smooth = tl.load(smooth_ptr + cols).to(tl.float32)
                q = q * smooth
            q = q * inv_scale
            q = tl.math.floor(q + 0.5)
            q_i8 = q.to(tl.int8)
            tl.store(y_ptr + offsets, q_i8)


@libentry()
@triton.jit
def _rms_norm_dynamic_quant_kernel(
    x_ptr,
    gamma_ptr,
    smooth_ptr,
    beta_ptr,
    y_ptr,
    scale_ptr,
    total_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    HAS_SMOOTH: tl.constexpr,
    HAS_BETA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    for row in range(pid, total_rows, grid_size):
        square_sum = tl.full((), 0.0, dtype=tl.float32)
        for col_start in tle.dsa.pipeline(0, n_cols, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            col_mask = cols < n_cols
            x = tl.load(x_ptr + row * n_cols + cols, mask=col_mask, other=0.0).to(
                tl.float32
            )
            square_sum += tl.sum(x * x, axis=0)

        rstd = tl.rsqrt(square_sum / n_cols + eps)

        max_abs = tl.full((), 0.0, dtype=tl.float32)
        for col_start in tle.dsa.pipeline(0, n_cols, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            col_mask = cols < n_cols
            x = tl.load(x_ptr + row * n_cols + cols, mask=col_mask, other=0.0).to(
                tl.float32
            )
            gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            q_input = x * rstd * gamma
            if HAS_BETA:
                beta = tl.load(beta_ptr + cols, mask=col_mask, other=0.0).to(
                    tl.float32
                )
                q_input = q_input + beta
            if HAS_SMOOTH:
                smooth = tl.load(smooth_ptr + cols, mask=col_mask, other=1.0).to(
                    tl.float32
                )
                q_input = q_input * smooth
            max_abs = tl.maximum(max_abs, tl.max(tl.abs(q_input), axis=0))

        scale = max_abs / 127.0
        scale = tl.where(scale < 1.0e-6, 1.0, scale)
        tl.store(scale_ptr + row, scale)

        inv_scale = 1.0 / scale
        for col_start in tle.dsa.pipeline(0, n_cols, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            col_mask = cols < n_cols
            offsets = row * n_cols + cols
            x = tl.load(x_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
            gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            q = x * rstd * gamma
            if HAS_BETA:
                beta = tl.load(beta_ptr + cols, mask=col_mask, other=0.0).to(
                    tl.float32
                )
                q = q + beta
            if HAS_SMOOTH:
                smooth = tl.load(smooth_ptr + cols, mask=col_mask, other=1.0).to(
                    tl.float32
                )
                q = q * smooth
            q = q * inv_scale
            q = tl.math.floor(q + 0.5)
            q_i8 = q.to(tl.int8)
            tl.store(y_ptr + offsets, q_i8, mask=col_mask)


@libentry()
@triton.jit
def _rms_norm_dynamic_quant_kernel_non_tle(
    x_ptr,
    gamma_ptr,
    smooth_ptr,
    beta_ptr,
    y_ptr,
    scale_ptr,
    total_rows: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    HAS_SMOOTH: tl.constexpr,
    HAS_BETA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Non-TLE variant: range() loop instead of tle.dsa.pipeline for column iteration."""
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    for row in range(pid, total_rows, grid_size):
        square_sum = tl.full((), 0.0, dtype=tl.float32)
        for col_start in range(0, n_cols, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            col_mask = cols < n_cols
            x = tl.load(x_ptr + row * n_cols + cols, mask=col_mask, other=0.0).to(
                tl.float32
            )
            square_sum += tl.sum(x * x, axis=0)

        rstd = tl.rsqrt(square_sum / n_cols + eps)

        max_abs = tl.full((), 0.0, dtype=tl.float32)
        for col_start in range(0, n_cols, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            col_mask = cols < n_cols
            x = tl.load(x_ptr + row * n_cols + cols, mask=col_mask, other=0.0).to(
                tl.float32
            )
            gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            q_input = x * rstd * gamma
            if HAS_BETA:
                beta = tl.load(beta_ptr + cols, mask=col_mask, other=0.0).to(
                    tl.float32
                )
                q_input = q_input + beta
            if HAS_SMOOTH:
                smooth = tl.load(smooth_ptr + cols, mask=col_mask, other=1.0).to(
                    tl.float32
                )
                q_input = q_input * smooth
            max_abs = tl.maximum(max_abs, tl.max(tl.abs(q_input), axis=0))

        scale = max_abs / 127.0
        scale = tl.where(scale < 1.0e-6, 1.0, scale)
        tl.store(scale_ptr + row, scale)

        inv_scale = 1.0 / scale
        for col_start in range(0, n_cols, BLOCK_N):
            cols = col_start + tl.arange(0, BLOCK_N)
            col_mask = cols < n_cols
            offsets = row * n_cols + cols
            x = tl.load(x_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
            gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            q = x * rstd * gamma
            if HAS_BETA:
                beta = tl.load(beta_ptr + cols, mask=col_mask, other=0.0).to(
                    tl.float32
                )
                q = q + beta
            if HAS_SMOOTH:
                smooth = tl.load(smooth_ptr + cols, mask=col_mask, other=1.0).to(
                    tl.float32
                )
                q = q * smooth
            q = q * inv_scale
            q = tl.math.floor(q + 0.5)
            q_i8 = q.to(tl.int8)
            tl.store(y_ptr + offsets, q_i8, mask=col_mask)


def rms_norm_dynamic_quant_impl(
    x: torch.Tensor,
    gamma: torch.Tensor,
    *,
    smooth_scale: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    block_size_n: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm + dynamic per-token quantization (int8).

    Args:
        x: ``(*, D)`` input tensor (float16 or bfloat16).
        gamma: RMSNorm weight, 1D of size D.
        smooth_scale: Optional per-channel smooth scale, 1D of size D.
        beta: Optional per-channel bias to add before smooth_scale mul, 1D of size D.
        epsilon: Epsilon for RMSNorm stability.
        block_size_n: Optional explicit BLOCK_N override.

    Returns:
        ``(y, scale)`` where ``y`` is int8 with same shape as ``x``,
        and ``scale`` is float32 with shape ``x.shape[:-1]``.
    """
    # ---- Input validation ----
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"Unsupported x dtype: {x.dtype}")
    if not x.is_contiguous():
        raise RuntimeError("x must be contiguous.")
    if not gamma.is_contiguous():
        raise RuntimeError("gamma must be contiguous.")

    shape = x.shape
    dim = shape[-1]

    if gamma.dim() != 1 or gamma.numel() != dim:
        raise ValueError(f"gamma must be 1D with {dim} elements, got {gamma.shape}.")
    if smooth_scale is not None:
        if not smooth_scale.is_contiguous():
            raise RuntimeError("smooth_scale must be contiguous.")
        if smooth_scale.dim() != 1 or smooth_scale.numel() != dim:
            raise ValueError(
                f"smooth_scale must be 1D with {dim} elements, got {smooth_scale.shape}."
            )
    if beta is not None:
        if not beta.is_contiguous():
            raise RuntimeError("beta must be contiguous.")
        if beta.dim() != 1 or beta.numel() != dim:
            raise ValueError(
                f"beta must be 1D with {dim} elements, got {beta.shape}."
            )

    # ---- Reshape to 2D ----
    x_2d = x.reshape(-1, dim)
    total_rows = x_2d.shape[0]
    d = dim

    # ---- Block size selection ----
    is_aligned = (d % BLOCK_N_ALIGN == 0)
    use_prefill = (
        should_use_prefill_kernel(d, total_rows)
        and smooth_scale is not None
        and beta is None
        and is_aligned
    )
    use_small_cols = should_use_small_cols(d, block_size_n)
    if block_size_n is None:
        block_size_n = select_block_size_n(d)
    use_cached2 = should_use_cached2(d, block_size_n, smooth_scale, beta)
    use_no_mask_blocked = should_use_no_mask_blocked(d, block_size_n)
    small_cols_block_n = select_small_cols_block_size_n(d)

    # ---- Allocate output ----
    y = torch.empty_like(x_2d, dtype=torch.int8)
    scale = torch.empty(total_rows, device=x.device, dtype=torch.float32)

    smooth_arg = smooth_scale if smooth_scale is not None else gamma
    beta_arg = beta if beta is not None else gamma

    has_smooth = smooth_scale is not None
    has_beta = beta is not None

    # ---- Dispatch to kernel ----
    if HAS_TLE:
        if use_prefill:
            block_m, prefill_grid = get_prefill_config(d, total_rows)
            prefill_grid = (prefill_grid,)

            if block_m == 4:
                _rms_norm_dynamic_quant_small_cols_dsa_smooth_prefill_m4_kernel[
                    prefill_grid
                ](
                    x_2d,
                    gamma,
                    smooth_scale,
                    y,
                    scale,
                    total_rows=total_rows,
                    n_cols=d,
                    eps=float(epsilon),
                    BLOCK_M=block_m,
                    BLOCK_N=small_cols_block_n,
                )
            elif block_m == 1:
                _rms_norm_dynamic_quant_small_cols_dsa_smooth_prefill_m1_kernel[
                    prefill_grid
                ](
                    x_2d,
                    gamma,
                    smooth_scale,
                    y,
                    scale,
                    total_rows=total_rows,
                    n_cols=d,
                    eps=float(epsilon),
                    BLOCK_M=block_m,
                    BLOCK_N=small_cols_block_n,
                )
            else:
                _rms_norm_dynamic_quant_small_cols_dsa_smooth_prefill_kernel[
                    prefill_grid
                ](
                    x_2d,
                    gamma,
                    smooth_scale,
                    y,
                    scale,
                    total_rows=total_rows,
                    n_cols=d,
                    eps=float(epsilon),
                    BLOCK_M=block_m,
                    BLOCK_N=small_cols_block_n,
                )
            return y.reshape(shape), scale.reshape(shape[:-1])

        grid = (_select_grid(total_rows),)

        if use_small_cols and has_smooth and not has_beta and is_aligned:
            _rms_norm_dynamic_quant_small_cols_dsa_smooth_kernel[grid](
                x_2d,
                gamma,
                smooth_scale,
                y,
                scale,
                total_rows=total_rows,
                n_cols=d,
                eps=float(epsilon),
                BLOCK_N=small_cols_block_n,
            )
        elif use_cached2:
            _rms_norm_dynamic_quant_cached2_fused_kernel[grid](
                x_2d,
                gamma,
                smooth_scale,
                y,
                scale,
                total_rows=total_rows,
                n_cols=d,
                eps=float(epsilon),
                BLOCK_N=block_size_n,
            )
        elif use_small_cols:
            _rms_norm_dynamic_quant_small_cols_kernel[grid](
                x_2d,
                gamma,
                smooth_arg,
                beta_arg,
                y,
                scale,
                total_rows=total_rows,
                n_cols=d,
                eps=float(epsilon),
                HAS_SMOOTH=has_smooth,
                HAS_BETA=has_beta,
                BLOCK_N=small_cols_block_n,
            )
        elif use_no_mask_blocked:
            _rms_norm_dynamic_quant_no_mask_kernel[grid](
                x_2d,
                gamma,
                smooth_arg,
                beta_arg,
                y,
                scale,
                total_rows=total_rows,
                n_cols=d,
                eps=float(epsilon),
                HAS_SMOOTH=has_smooth,
                HAS_BETA=has_beta,
                BLOCK_M=BLOCK_SIZE_M,
                BLOCK_N=block_size_n,
            )
        else:
            _rms_norm_dynamic_quant_kernel[grid](
                x_2d,
                gamma,
                smooth_arg,
                beta_arg,
                y,
                scale,
                total_rows=total_rows,
                n_cols=d,
                eps=float(epsilon),
                HAS_SMOOTH=has_smooth,
                HAS_BETA=has_beta,
                BLOCK_M=BLOCK_SIZE_M,
                BLOCK_N=block_size_n,
            )
    else:
        # Non-TLE path: uses range() loops and tl.load instead of tle.dsa.*
        if use_prefill:
            block_m, prefill_grid = get_prefill_config(d, total_rows)
            prefill_grid = (prefill_grid,)

            if block_m == 4:
                _rms_norm_dynamic_quant_small_cols_dsa_smooth_prefill_m4_kernel_non_tle[
                    prefill_grid
                ](
                    x_2d,
                    gamma,
                    smooth_scale,
                    y,
                    scale,
                    total_rows=total_rows,
                    n_cols=d,
                    eps=float(epsilon),
                    BLOCK_M=block_m,
                    BLOCK_N=small_cols_block_n,
                )
            elif block_m == 1:
                _rms_norm_dynamic_quant_small_cols_dsa_smooth_prefill_m1_kernel_non_tle[
                    prefill_grid
                ](
                    x_2d,
                    gamma,
                    smooth_scale,
                    y,
                    scale,
                    total_rows=total_rows,
                    n_cols=d,
                    eps=float(epsilon),
                    BLOCK_M=block_m,
                    BLOCK_N=small_cols_block_n,
                )
            else:
                _rms_norm_dynamic_quant_small_cols_dsa_smooth_prefill_kernel_non_tle[
                    prefill_grid
                ](
                    x_2d,
                    gamma,
                    smooth_scale,
                    y,
                    scale,
                    total_rows=total_rows,
                    n_cols=d,
                    eps=float(epsilon),
                    BLOCK_M=block_m,
                    BLOCK_N=small_cols_block_n,
                )
            return y.reshape(shape), scale.reshape(shape[:-1])

        grid = (_select_grid(total_rows),)

        if use_small_cols and has_smooth and not has_beta and is_aligned:
            _rms_norm_dynamic_quant_small_cols_dsa_smooth_kernel_non_tle[grid](
                x_2d,
                gamma,
                smooth_scale,
                y,
                scale,
                total_rows=total_rows,
                n_cols=d,
                eps=float(epsilon),
                BLOCK_N=small_cols_block_n,
            )
        elif use_cached2:
            _rms_norm_dynamic_quant_cached2_fused_kernel_non_tle[grid](
                x_2d,
                gamma,
                smooth_scale,
                y,
                scale,
                total_rows=total_rows,
                n_cols=d,
                eps=float(epsilon),
                BLOCK_N=block_size_n,
            )
        elif use_small_cols:
            _rms_norm_dynamic_quant_small_cols_kernel_non_tle[grid](
                x_2d,
                gamma,
                smooth_arg,
                beta_arg,
                y,
                scale,
                total_rows=total_rows,
                n_cols=d,
                eps=float(epsilon),
                HAS_SMOOTH=has_smooth,
                HAS_BETA=has_beta,
                BLOCK_N=small_cols_block_n,
            )
        elif use_no_mask_blocked:
            _rms_norm_dynamic_quant_no_mask_kernel_non_tle[grid](
                x_2d,
                gamma,
                smooth_arg,
                beta_arg,
                y,
                scale,
                total_rows=total_rows,
                n_cols=d,
                eps=float(epsilon),
                HAS_SMOOTH=has_smooth,
                HAS_BETA=has_beta,
                BLOCK_M=BLOCK_SIZE_M,
                BLOCK_N=block_size_n,
            )
        else:
            _rms_norm_dynamic_quant_kernel_non_tle[grid](
                x_2d,
                gamma,
                smooth_arg,
                beta_arg,
                y,
                scale,
                total_rows=total_rows,
                n_cols=d,
                eps=float(epsilon),
                HAS_SMOOTH=has_smooth,
                HAS_BETA=has_beta,
                BLOCK_M=BLOCK_SIZE_M,
                BLOCK_N=block_size_n,
            )

    return y.reshape(shape), scale.reshape(shape[:-1])
