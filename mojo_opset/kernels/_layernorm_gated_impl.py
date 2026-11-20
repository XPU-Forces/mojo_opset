"""Gated LayerNorm / RMSNorm forward (layernorm_gated) adapted for triton-ascend.

Port of ``sglang/kernels/ops/attention/fla/layernorm_gated.py`` (v0.5.22): one launch
computes normalization, optional gate, affine, and the optional per-token FP8
epilogue. Row tiles are proven independent of the per-row column reduction, so
Ascend can use much taller tiles than the CUDA ``MAX_ROWS_PER_BLOCK = 4``; the
batch-invariant mode keeps 4 to honour the upstream contract.

The launch is grid-stride over row tiles (``NS = min(vector cores, tiles)``) with
weight/bias hoisted out of the loop, which caps the block count and keeps the
affine operand resident in UB.
"""

import os

from functools import lru_cache

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores, libentry

# Upstream contract: batch-invariant mode requires this exact tile height.
MAX_ROWS_PER_BLOCK = 4
# Ascend-only upper bound for the dynamic tile height: 128 rows of BLOCK_N=128
# overflow UB at compile time on A2, so 64 is the ceiling.
MAX_ROWS_PER_BLOCK_ASCEND = 64
# Rows are reduced along columns only, so the tile height never enters the
# reduction tree; the budget below only bounds UB pressure.
_UB_SAFETY = 0.85
_UB_FALLBACK_BYTES = 192 * 1024
_UB_A5_BYTES = 256 * 1024
# Peak live fp32 tiles in flight (x, z, y, and reduction temporaries).
# 4.0 is the measured ceiling on A2: it yields 64 rows at BLOCK_N=128 (fastest
# in the tile sweep) while 128 rows fails to fit UB (BiShengHIR "ub" error).
_TILE_FP32_MULT = 4.0
def _launch_opts(chunked: bool = False, quant: bool = False) -> dict:
    """Ping-pong DMA pipelining: on, except on the two paths that lose it.

    Re-measured when the row tile dropped to 4-8 rows (it used to be 64, which
    left no UB for a second buffer and made this a loss): with the small tiles
    the gated path gains 2-23% and GroupNorm 2-14% from pipelining. Two paths
    stay off it: the chunked kernel (non-pow2 group_size) loses 31% (55.1 ->
    72.2 us on the tail shape), and the FP8 epilogue loses 41% on triton-ascend
    3.6.0 (81.3 -> 114.9 us on 32768x512/qh8), where the second buffer starves
    the per-token amax reduction.
    """
    forced = os.getenv("MOJO_LN_GATED_MULTIBUFFER", "").strip()
    if forced in ("0", "1"):
        return {"multibuffer": forced == "1"}
    return {"multibuffer": not (chunked or quant)}


def is_batch_invariant_mode_enabled() -> bool:
    value = os.getenv("LN_GATED_BATCH_INVARIANT", "").strip().lower()
    return value not in ("", "0", "false", "off", "no")


def _floor_pow2(value: int) -> int:
    return 1 << (int(value).bit_length() - 1) if value >= 1 else 1


@lru_cache(maxsize=1)
def _current_arch() -> str:
    try:
        return str(triton.runtime.driver.active.get_current_target().arch)
    except Exception:
        return ""


def _is_a5() -> bool:
    arch = _current_arch()
    return "950" in arch or "910_95" in arch


@lru_cache(maxsize=1)
def _ub_bytes() -> int:
    override = os.getenv("MOJO_LN_GATED_UB_KB")
    if override:
        return int(override) * 1024
    return _UB_A5_BYTES if _is_a5() else _UB_FALLBACK_BYTES


def _quant_supported() -> bool:
    # The task scopes the FP8 epilogue to Ascend 950; older parts may not lower
    # fp8 stores. The env knob exists for probing, not for shipping defaults.
    if os.getenv("MOJO_LN_GATED_ALLOW_FP8", "").strip() == "1":
        return True
    return _is_a5()


def calc_rows_per_block(M: int, block_n: int, quant_heads: int = 0, mem_mult: float = 0) -> int:
    """Row tile height: UB-bounded, core-aware, and constant under batch invariance."""
    # The quant epilogue reduces amax over one token, so the tile covers a whole token.
    if quant_heads:
        return quant_heads
    if is_batch_invariant_mode_enabled():
        return MAX_ROWS_PER_BLOCK
    multiple = mem_mult or _TILE_FP32_MULT
    rows_ub = max(1, int(_ub_bytes() * _UB_SAFETY // (block_n * 4 * multiple)))
    rows_ub = min(_floor_pow2(rows_ub), MAX_ROWS_PER_BLOCK_ASCEND)
    # Keep at least one tile per vector core for small row counts.
    rows_par = _floor_pow2(max(1, -(-M // max(1, get_num_cores()))))
    return max(1, min(rows_ub, rows_par))


def _maybe_libentry(fn):
    """Skip the cached launcher: triton-ascend 3.2.1 mis-packs defaulted None
    arguments through libentry (``TypeError: function takes exactly 25
    arguments (23 given)``). It only saves host-side launch cost, which the
    device-time contract does not measure, so it stays opt-in.
    """
    if os.getenv("MOJO_LN_GATED_LIBENTRY", "0").strip() == "1":
        return libentry()(fn)
    return fn


@_maybe_libentry
@triton.jit(do_not_specialize=["M"])
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    stride_z_token,
    stride_z_head,
    M,  # number of rows in X
    N: tl.constexpr,  # number of columns in X (one group)
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    Z_IS_3D: tl.constexpr,
    Z_HEADS: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
    FULL_ROWS: tl.constexpr,
    Q=None,
    S=None,
    QUANT: tl.constexpr = False,
    QH: tl.constexpr = 1,
    QUANT_GROUPS: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    group = tl.program_id(1)

    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N

    # Affine operands are group-local and constant across the row loop.
    if BLOCK_N == N:
        w = tl.load(W + group * N + cols).to(tl.float32)
    else:
        w = tl.load(W + group * N + cols, mask=col_mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        if BLOCK_N == N:
            b = tl.load(B + group * N + cols).to(tl.float32)
        else:
            b = tl.load(B + group * N + cols, mask=col_mask, other=0.0).to(tl.float32)

    num_tiles = (M + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    for tile in range(pid, num_tiles, num_programs):
        rows = tile * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
        rows64 = rows.to(tl.int64)
        row_mask = rows < M
        mask = row_mask[:, None] & col_mask[None, :]

        col_base = group * N + cols[None, :]
        # Tiles with no partial rows or columns skip masking entirely: the mask
        # maths is pure ALU overhead on a bandwidth-bound kernel.
        if FULL_ROWS and BLOCK_N == N:
            x = tl.load(X + rows64[:, None] * stride_x_row + col_base).to(tl.float32)
        else:
            x = tl.load(X + rows64[:, None] * stride_x_row + col_base, mask=mask, other=0.0).to(tl.float32)

        if Z_IS_3D:
            z_rows = (rows64 // Z_HEADS) * stride_z_token + (rows64 % Z_HEADS) * stride_z_head
        else:
            z_rows = rows64 * stride_z_row
        z_base = z_rows[:, None] + col_base

        if HAS_Z and not NORM_BEFORE_GATE:
            if FULL_ROWS and BLOCK_N == N:
                z = tl.load(Z + z_base).to(tl.float32)
            else:
                z = tl.load(Z + z_base, mask=mask, other=0.0).to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                x = x * z * tl.sigmoid(z)
            elif ACTIVATION == "sigmoid":
                x = x * tl.sigmoid(z)

        stats_offsets = group * M + rows
        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=1) / N
            tl.store(Mean + stats_offsets, mean, mask=row_mask)
            if FULL_ROWS and BLOCK_N == N:
                xbar = x - mean[:, None]
            else:
                xbar = tl.where(mask, x - mean[:, None], 0.0)
            var = tl.sum(xbar * xbar, axis=1) / N
        else:
            if FULL_ROWS and BLOCK_N == N:
                xbar = x
            else:
                xbar = tl.where(mask, x, 0.0)
            var = tl.sum(xbar * xbar, axis=1) / N
            mean = tl.zeros((ROWS_PER_BLOCK,), dtype=tl.float32)

        # 1/sqrt is used instead of tl.rsqrt: the ascend rsqrt accuracy table only
        # lists A2/A3, and 950 must compile from the same source.
        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Rstd + stats_offsets, rstd, mask=row_mask)

        if not IS_RMS_NORM:
            x_hat = (x - mean[:, None]) * rstd[:, None]
        else:
            x_hat = x * rstd[:, None]

        y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]

        if HAS_Z and NORM_BEFORE_GATE:
            if FULL_ROWS and BLOCK_N == N:
                z = tl.load(Z + z_base).to(tl.float32)
            else:
                z = tl.load(Z + z_base, mask=mask, other=0.0).to(tl.float32)
            if ACTIVATION == "swish" or ACTIVATION == "silu":
                y = y * z * tl.sigmoid(z)
            elif ACTIVATION == "sigmoid":
                y = y * tl.sigmoid(z)

        if QUANT:
            # Round to the activation dtype before amax so the fused path is
            # bit-exact with the two-pass per-token-group quantization. The bf16
            # cast round trip is folded away by the backend, so bf16 is narrowed
            # in the integer domain (round-to-nearest-even at 8 mantissa bits);
            # the fp16 trip survives codegen (measured: the stored scale matches
            # the oracle bitwise, which it could not if the cast were dropped).
            if Y.dtype.element_ty == tl.bfloat16:
                bits = y.to(tl.int32, bitcast=True)
                bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & -65536
                y = bits.to(tl.float32, bitcast=True)
            else:
                y = y.to(Y.dtype.element_ty).to(tl.float32)
            if not FULL_ROWS:
                y = tl.where(mask, y, 0.0)
            # torch on Ascend lowers "amax / 448.0" as a multiply by the rounded
            # reciprocal (measured: bitwise equal, while a true divide differs in
            # ~15% of rows), so the reciprocal form is what matches the oracle.
            if QUANT_GROUPS == 1:
                amax = tl.max(tl.max(tl.abs(y), axis=1), axis=0)
                y_s = tl.maximum(amax, 1e-10) * (1.0 / 448.0)
                y_q = tl.minimum(tl.maximum(y * (1.0 / y_s), -448.0), 448.0)
            else:
                # One amax per token, several tokens per tile: 4 tokens per tile
                # measured 78% vs 65% MBU on 950 (fewer scalar reductions).
                # One 2-D reduce per token: reshape to (tokens, QH*N) keeps a
                # single reduction and leaves the store layout unchanged.
                y2 = tl.reshape(y, (QUANT_GROUPS, QH * N))
                a2 = tl.max(tl.abs(y2), axis=1)
                y_s = tl.maximum(a2, 1e-10) * (1.0 / 448.0)
                # A *vector* "1.0 / y_s" lowers to an approximate reciprocal ~1
                # ulp off (a scalar one does not), which is enough to send a tie
                # to the wrong FP8 neighbour. One FMA Newton step restores the
                # correctly rounded value: measured bit-exact over 1M elements.
                inv = 1.0 / y_s
                inv = tl.math.fma(inv, tl.math.fma(-y_s, inv, 1.0), inv)
                q2 = tl.minimum(tl.maximum(y2 * inv[:, None], -448.0), 448.0)
                y_q = tl.reshape(q2, (ROWS_PER_BLOCK, BLOCK_N))
            if FULL_ROWS:
                tl.store(Q + rows64[:, None] * N + col_base, y_q.to(Q.dtype.element_ty))
            else:
                tl.store(Q + rows64[:, None] * N + col_base, y_q.to(Q.dtype.element_ty), mask=mask)
            if QUANT_GROUPS == 1:
                tl.store(S + tile, y_s)
            else:
                tl.store(S + tile * QUANT_GROUPS + tl.arange(0, QUANT_GROUPS), y_s)
        elif FULL_ROWS and BLOCK_N == N:
            tl.store(Y + rows64[:, None] * stride_y_row + col_base, y)
        else:
            tl.store(Y + rows64[:, None] * stride_y_row + col_base, y, mask=mask)


@triton.jit
def _chunk_affine(xc, wc, mean, rstd, IS_RMS_NORM: tl.constexpr):
    x_hat = xc * rstd[:, None] if IS_RMS_NORM else (xc - mean[:, None]) * rstd[:, None]
    return x_hat * wc[None, :]


@triton.jit
def _chunk_gate(value, zc, ACTIVATION: tl.constexpr):
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        return value * (zc * tl.sigmoid(zc))
    elif ACTIVATION == "sigmoid":
        return value * tl.sigmoid(zc)
    return value


@_maybe_libentry
@triton.jit(do_not_specialize=["M"])
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
    CHUNK_N: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    Z_IS_3D: tl.constexpr,
    Z_HEADS: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Column-chunked variant for group sizes that are not powers of two.

    A 256-wide masked tile over a 192-element row stride re-reads the next row's
    head on every row (measured 490 GB/s vs 878 GB/s for an exact-width tile),
    so the row is split into equal power-of-two chunks instead. The chunks are
    unrolled because Triton block shapes must stay power-of-two.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    group = tl.program_id(1)
    cols = tl.arange(0, CHUNK_N)
    base = group * N

    w0 = tl.load(W + base + cols).to(tl.float32)
    w1 = w0
    w2 = w0
    w3 = w0
    if N_CHUNKS >= 2:
        w1 = tl.load(W + base + CHUNK_N + cols).to(tl.float32)
    if N_CHUNKS >= 3:
        w2 = tl.load(W + base + 2 * CHUNK_N + cols).to(tl.float32)
    if N_CHUNKS >= 4:
        w3 = tl.load(W + base + 3 * CHUNK_N + cols).to(tl.float32)
    if HAS_BIAS:
        b0 = tl.load(B + base + cols).to(tl.float32)
        b1 = b0
        b2 = b0
        b3 = b0
        if N_CHUNKS >= 2:
            b1 = tl.load(B + base + CHUNK_N + cols).to(tl.float32)
        if N_CHUNKS >= 3:
            b2 = tl.load(B + base + 2 * CHUNK_N + cols).to(tl.float32)
        if N_CHUNKS >= 4:
            b3 = tl.load(B + base + 3 * CHUNK_N + cols).to(tl.float32)

    num_tiles = (M + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    for tile in range(pid, num_tiles, num_programs):
        rows = tile * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
        rows64 = rows.to(tl.int64)
        row_mask = rows < M
        mask = row_mask[:, None]
        x_row = rows64[:, None] * stride_x_row + base
        y_row = rows64[:, None] * stride_y_row + base
        if Z_IS_3D:
            z_rows = (rows64 // Z_HEADS) * stride_z_token + (rows64 % Z_HEADS) * stride_z_head
        else:
            z_rows = rows64 * stride_z_row
        z_row = z_rows[:, None] + base

        x0 = tl.load(X + x_row + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        x1 = x0
        x2 = x0
        x3 = x0
        if N_CHUNKS >= 2:
            x1 = tl.load(X + x_row + CHUNK_N + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        if N_CHUNKS >= 3:
            x2 = tl.load(X + x_row + 2 * CHUNK_N + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        if N_CHUNKS >= 4:
            x3 = tl.load(X + x_row + 3 * CHUNK_N + cols[None, :], mask=mask, other=0.0).to(tl.float32)

        z0 = x0
        z1 = x0
        z2 = x0
        z3 = x0
        if HAS_Z:
            z0 = tl.load(Z + z_row + cols[None, :], mask=mask, other=0.0).to(tl.float32)
            if N_CHUNKS >= 2:
                z1 = tl.load(Z + z_row + CHUNK_N + cols[None, :], mask=mask, other=0.0).to(tl.float32)
            if N_CHUNKS >= 3:
                z2 = tl.load(Z + z_row + 2 * CHUNK_N + cols[None, :], mask=mask, other=0.0).to(tl.float32)
            if N_CHUNKS >= 4:
                z3 = tl.load(Z + z_row + 3 * CHUNK_N + cols[None, :], mask=mask, other=0.0).to(tl.float32)
            if not NORM_BEFORE_GATE:
                x0 = _chunk_gate(x0, z0, ACTIVATION)
                if N_CHUNKS >= 2:
                    x1 = _chunk_gate(x1, z1, ACTIVATION)
                if N_CHUNKS >= 3:
                    x2 = _chunk_gate(x2, z2, ACTIVATION)
                if N_CHUNKS >= 4:
                    x3 = _chunk_gate(x3, z3, ACTIVATION)

        if not IS_RMS_NORM:
            total = tl.sum(x0, axis=1)
            if N_CHUNKS >= 2:
                total += tl.sum(x1, axis=1)
            if N_CHUNKS >= 3:
                total += tl.sum(x2, axis=1)
            if N_CHUNKS >= 4:
                total += tl.sum(x3, axis=1)
            mean = total / N
            tl.store(Mean + group * M + rows, mean, mask=row_mask)
            d0 = tl.where(mask, x0 - mean[:, None], 0.0)
            var = tl.sum(d0 * d0, axis=1)
            if N_CHUNKS >= 2:
                d1 = tl.where(mask, x1 - mean[:, None], 0.0)
                var += tl.sum(d1 * d1, axis=1)
            if N_CHUNKS >= 3:
                d2 = tl.where(mask, x2 - mean[:, None], 0.0)
                var += tl.sum(d2 * d2, axis=1)
            if N_CHUNKS >= 4:
                d3 = tl.where(mask, x3 - mean[:, None], 0.0)
                var += tl.sum(d3 * d3, axis=1)
            var = var / N
        else:
            var = tl.sum(tl.where(mask, x0, 0.0) * x0, axis=1)
            if N_CHUNKS >= 2:
                var += tl.sum(tl.where(mask, x1, 0.0) * x1, axis=1)
            if N_CHUNKS >= 3:
                var += tl.sum(tl.where(mask, x2, 0.0) * x2, axis=1)
            if N_CHUNKS >= 4:
                var += tl.sum(tl.where(mask, x3, 0.0) * x3, axis=1)
            var = var / N
            mean = tl.zeros((ROWS_PER_BLOCK,), dtype=tl.float32)

        rstd = 1.0 / tl.sqrt(var + eps)
        tl.store(Rstd + group * M + rows, rstd, mask=row_mask)

        y0 = _chunk_affine(x0, w0, mean, rstd, IS_RMS_NORM)
        if HAS_BIAS:
            y0 = y0 + b0[None, :]
        if HAS_Z and NORM_BEFORE_GATE:
            y0 = _chunk_gate(y0, z0, ACTIVATION)
        tl.store(Y + y_row + cols[None, :], y0, mask=mask)

        if N_CHUNKS >= 2:
            y1 = _chunk_affine(x1, w1, mean, rstd, IS_RMS_NORM)
            if HAS_BIAS:
                y1 = y1 + b1[None, :]
            if HAS_Z and NORM_BEFORE_GATE:
                y1 = _chunk_gate(y1, z1, ACTIVATION)
            tl.store(Y + y_row + CHUNK_N + cols[None, :], y1, mask=mask)
        if N_CHUNKS >= 3:
            y2 = _chunk_affine(x2, w2, mean, rstd, IS_RMS_NORM)
            if HAS_BIAS:
                y2 = y2 + b2[None, :]
            if HAS_Z and NORM_BEFORE_GATE:
                y2 = _chunk_gate(y2, z2, ACTIVATION)
            tl.store(Y + y_row + 2 * CHUNK_N + cols[None, :], y2, mask=mask)
        if N_CHUNKS >= 4:
            y3 = _chunk_affine(x3, w3, mean, rstd, IS_RMS_NORM)
            if HAS_BIAS:
                y3 = y3 + b3[None, :]
            if HAS_Z and NORM_BEFORE_GATE:
                y3 = _chunk_gate(y3, z3, ACTIVATION)
            tl.store(Y + y_row + 3 * CHUNK_N + cols[None, :], y3, mask=mask)


@_maybe_libentry
@triton.jit(do_not_specialize=["M"])
def _layer_norm_fwd_wide_kernel(
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
    M,
    N: tl.constexpr,
    eps,
    NG: tl.constexpr,
    GS: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Whole-row tiles for grouped normalization.

    The per-group grid reads 256 B slices out of 4 KB rows (measured 754 GB/s);
    loading the full row contiguously and reducing the [rows, NG, GS] view keeps
    the DMA sequential.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, NG * GS)
    groups = tl.arange(0, NG)
    col_mask = cols < N
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)

    num_tiles = (M + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    for tile in range(pid, num_tiles, num_programs):
        rows = tile * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
        rows64 = rows.to(tl.int64)
        row_mask = rows < M
        mask = row_mask[:, None] & col_mask[None, :]

        x = tl.load(X + rows64[:, None] * stride_x_row + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        z = x
        if HAS_Z:
            z = tl.load(Z + rows64[:, None] * stride_z_row + cols[None, :], mask=mask, other=0.0).to(tl.float32)
            if not NORM_BEFORE_GATE:
                x = _chunk_gate(x, z, ACTIVATION)

        x3 = tl.reshape(x, (ROWS_PER_BLOCK, NG, GS))
        if not IS_RMS_NORM:
            mean3 = tl.sum(x3, axis=2) / GS
            d3 = tl.where(row_mask[:, None, None], x3 - mean3[:, :, None], 0.0)
            var3 = tl.sum(d3 * d3, axis=2) / GS
        else:
            xbar3 = tl.where(row_mask[:, None, None], x3, 0.0)
            var3 = tl.sum(xbar3 * xbar3, axis=2) / GS
            mean3 = tl.zeros((ROWS_PER_BLOCK, NG), dtype=tl.float32)
        rstd3 = 1.0 / tl.sqrt(var3 + eps)

        stat_off = groups[None, :] * M + rows[:, None]
        if not IS_RMS_NORM:
            tl.store(Mean + stat_off, mean3, mask=row_mask[:, None])
        tl.store(Rstd + stat_off, rstd3, mask=row_mask[:, None])

        if not IS_RMS_NORM:
            x_hat3 = (x3 - mean3[:, :, None]) * rstd3[:, :, None]
        else:
            x_hat3 = x3 * rstd3[:, :, None]
        w3 = tl.reshape(w, (NG, GS))
        y = tl.reshape(x_hat3 * w3[None, :, :], (ROWS_PER_BLOCK, NG * GS))
        if HAS_BIAS:
            y = y + b[None, :]
        if HAS_Z and NORM_BEFORE_GATE:
            y = _chunk_gate(y, z, ACTIVATION)
        tl.store(Y + rows64[:, None] * stride_y_row + cols[None, :], y, mask=mask)


def _wide_plan(group_size: int, ngroups: int, element_size: int):
    """(ROWS, NG, GS) when a whole-row tile can carry every group."""
    if ngroups < 2 or (ngroups & (ngroups - 1)) or (group_size & (group_size - 1)):
        return None
    if group_size * ngroups > 65536 // element_size:
        return None
    return ngroups


def _chunk_plan(group_size: int):
    """(CHUNK_N, N_CHUNKS) when equal 64-wide chunks avoid mask/overlap waste."""
    if group_size == triton.next_power_of_2(group_size) or group_size % 64:
        return None
    chunks = group_size // 64
    if chunks not in (2, 3, 4):
        return None
    return 64, chunks


def _grid_x(num_tiles: int) -> int:
    """Blocks along the row axis.

    ``persistent`` (default) launches one block per vector core and walks the
    tiles grid-stride; ``all`` launches one block per tile, which trades block
    scheduling for a shorter per-block dependency chain.
    """
    if os.getenv("MOJO_LN_GATED_GRID", "persistent").strip() == "all":
        return min(num_tiles, 65535)
    return min(get_num_cores(), num_tiles)


def layer_norm_gated_fwd(
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
    """Forward of the gated normalization; mirrors sglang's ``_layer_norm_fwd``.

    Returns ``((q, s) if quant_heads else out, mean, rstd)``.
    """
    # Two call conventions coexist upstream: this package's functions layer passes
    # (x, w, b, eps, z=...), the reference layer proposed in mojo_opset#525 passes
    # (x, w, b, z, eps, ...) positionally. Normalise so either dispatcher works.
    if not isinstance(eps, (int, float)) or isinstance(z, (int, float)):
        eps, z = z, eps
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0, f"normalized width {N} is not a multiple of group_size {group_size}"
    ngroups = N // group_size
    assert x.stride(-1) == 1
    z_is_3d = z is not None and z.ndim == 3
    # A token-major view with stride_z_token == Z_HEADS * stride_z_head is still
    # affine in the flat row index; addressing it that way instead of r//H, r%H
    # keeps the access recognisable as a strided copy (measured 624 -> 805 GB/s
    # on 910B3 for the padded [T, H, D+32] view).
    z_row_stride = 0
    if z is not None:
        assert z.stride(-1) == 1
        if z_is_3d:
            assert z.shape[0] * z.shape[1] == M
            assert z.shape[2] == N
            if z.stride(0) == z.shape[1] * z.stride(1):
                z_row_stride = z.stride(1)
                z_is_3d = False
        else:
            assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1

    mean = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    rows_per_block = calc_rows_per_block(M, BLOCK_N, quant_heads)
    quant = {}
    if quant_heads:
        assert ngroups == 1 and M % quant_heads == 0
        if not _quant_supported():
            raise NotImplementedError("The fused FP8 epilogue is enabled on Ascend 950 only.")
        # Several tokens per tile when UB allows: fewer scalar reductions per
        # byte. Only for power-of-two, exact-width rows.
        groups = 1
        block_n = min(65536 // x.element_size(), triton.next_power_of_2(group_size))
        # The quant tile keeps y plus the quantised copy live, so it can afford
        # ~3 fp32 tiles per element. With a gate, z takes another slot and the
        # same budget thrashes UB: measured on 950 at 32768x512/qh8, tightening
        # the budget to ~5 tiles turns the gated row from 141.5 us (57.5% of the
        # real 117.5 MB) into 104.7 us (77.8%), while the ungated row keeps its
        # 3-tile budget (81.8 us vs 84.4 us at 5).
        budget = 5.0 if z is not None else 3.0
        per_token = int(_ub_bytes() * _UB_SAFETY // (block_n * 4 * budget))
        if block_n == group_size and M % (quant_heads * 2) == 0:
            groups = max(1, min(8, _floor_pow2(max(1, per_token // quant_heads))))
        rows_per_block = quant_heads * groups
        q = torch.empty(
            (M // quant_heads, quant_heads * N),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        )
        s = torch.empty((M // quant_heads, 1), dtype=torch.float32, device=x.device)
        quant = {"Q": q, "S": s, "QUANT": True}
        # The epilogue never writes Y; x only supplies the rounding dtype.
        y_out = x
    else:
        y_out = out

    num_tiles = triton.cdiv(M, rows_per_block)
    wide_ng = None if quant_heads else _wide_plan(group_size, ngroups, x.element_size())
    if wide_ng is not None:
        # The whole-row tile keeps the [rows, NG, GS] view live, so it needs a
        # larger budget than the plain kernel. On 950 the wide tile is 2.2x faster
        # at 2 rows than at 4 (1705 vs 787 GB/s), so the budget is tighter there.
        wide_mult = 8.0 if _is_a5() else 4.0
        wide_rows = calc_rows_per_block(M, wide_ng * group_size, mem_mult=wide_mult)
        wide_grid = (_grid_x(triton.cdiv(M, wide_rows)), 1)
        _layer_norm_fwd_wide_kernel[wide_grid](
            x,
            y_out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            y_out.stride(0),
            z_row_stride or (z.stride(0) if z is not None and not z_is_3d else 0),
            M,
            group_size * ngroups,
            eps,
            NG=wide_ng,
            GS=group_size,
            ROWS_PER_BLOCK=wide_rows,
            HAS_BIAS=bias is not None,
            HAS_Z=z is not None,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            ACTIVATION=activation,
            **_launch_opts(),
        )
        return out, mean, rstd

    chunk_plan = None if quant_heads else _chunk_plan(group_size)
    if chunk_plan is not None:
        chunk_n, n_chunks = chunk_plan
        chunk_rows = calc_rows_per_block(M, chunk_n)
        chunk_grid = (_grid_x(triton.cdiv(M, chunk_rows)), ngroups)
        _layer_norm_fwd_chunked_kernel[chunk_grid](
            x,
            y_out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            y_out.stride(0),
            z_row_stride or (z.stride(0) if z is not None and not z_is_3d else 0),
            z.stride(0) if z_is_3d else 0,
            z.stride(1) if z_is_3d else 0,
            M,
            group_size,
            eps,
            CHUNK_N=chunk_n,
            N_CHUNKS=n_chunks,
            ROWS_PER_BLOCK=chunk_rows,
            HAS_BIAS=bias is not None,
            HAS_Z=z is not None,
            Z_IS_3D=z_is_3d,
            Z_HEADS=z.shape[1] if z_is_3d else 1,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            ACTIVATION=activation,
            **_launch_opts(chunked=True),
        )
        return out, mean, rstd

    grid = (_grid_x(num_tiles), ngroups)
    _layer_norm_fwd_1pass_kernel[grid](
        x,
        y_out,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.stride(0),
        y_out.stride(0),
        z_row_stride or (z.stride(0) if z is not None and not z_is_3d else 0),
        z.stride(0) if z_is_3d else 0,
        z.stride(1) if z_is_3d else 0,
        M,
        group_size,
        eps,
        BLOCK_N=BLOCK_N,
        ROWS_PER_BLOCK=rows_per_block,
        HAS_BIAS=bias is not None,
        HAS_Z=z is not None,
        Z_IS_3D=z_is_3d,
        Z_HEADS=z.shape[1] if z_is_3d else 1,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        ACTIVATION=activation,
        FULL_ROWS=(M % rows_per_block == 0),
        QH=quant_heads if quant_heads else 1,
        QUANT_GROUPS=quant_heads and (rows_per_block // quant_heads) or 1,
        **_launch_opts(quant=bool(quant_heads)),
        **quant,
    )
    return ((q, s) if quant_heads else out), mean, rstd
