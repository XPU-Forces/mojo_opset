"""Gated LayerNorm / RMSNorm with an optional FP8 per-token quant epilogue (Ascend Triton).

Port of SGLang v0.5.22 ``sglang/kernels/ops/attention/fla/layernorm_gated.py``
(``_layer_norm_fwd_1pass_kernel``, itself from fla-org/flash-linear-attention). One kernel
covers the whole capability matrix -- LayerNorm vs RMSNorm, no gate / 2-D gate / 3-D strided
gate, gate-before vs gate-after norm, swish vs sigmoid, affine, group norm, and an FP8 e4m3
per-token quant epilogue -- by constant-folding the ``constexpr`` switches rather than
branching at run time.

Numerical contract (matches the reference implementation)
--------------------------------------------------------
1. Statistics accumulate in FP32 even for FP16/BF16 input; ``Mean`` and ``Rstd`` are written
   as FP32.
2. Group norm reduces inside each ``group_size`` slice, never across the row: with
   ``ngroups > 1`` the reduction is per group and the ``Mean``/``Rstd`` layout is
   ``[ngroups * M]``, indexed ``group * M + row``.
3. The gate is applied in FP32 and only the final store rounds, so the FP16/BF16 rounding
   happens once.
4. The quant path rounds to the output dtype **before** computing the amax, then
   ``scale = max(amax, 1e-10) / 448`` and a clamp to +-448, so ``(q, s)`` is bit-identical to
   the unfused ``per_token_group_quant_fp8``. Two rounding details are load bearing and were
   measured on Ascend950PR: ``torch`` divides the amax by the scalar 448 as a reciprocal
   multiply, so the kernel multiplies by ``1/448`` as well (an IEEE divide lands one ULP low
   and moves a handful of ``q`` bytes); the payload quotient is a tensor/tensor division, so it
   uses ``div_rn`` rather than a reciprocal multiply.
5. ``Z_IS_3D`` addresses the gate through ``(row // Z_HEADS) * stride_z_token +
   (row % Z_HEADS) * stride_z_head``; that form is affine in the row vector, so the gate tile
   stays a bulk transfer instead of degenerating into a gather.

Ascend specific design
---------------------
* The tile is ``(ROWS_PER_BLOCK, BLOCK_N)`` with *affine* addresses. This is what the
  bandwidth probe established on 3.6.0: a 1-D per-row kernel caps at MBU 0.05 because every
  256 B row pays a fixed transfer cost, while the same work as a 2-D tile reaches 0.73 at
  ``ROWS_PER_BLOCK=16`` and 0.93 at 32 on a 134 MB case. The reference implementation's
  ``MAX_ROWS_PER_BLOCK = 4`` is tuned for CUDA's SM count and lands at 0.30 here, so the
  default is re-derived for this hardware instead of copied.
* ``ROWS_PER_BLOCK`` is chosen from the UB budget and the row count, not from a warp count:
  the live tiles are ``x``, the gate, the output and the weight, all FP32, so
  ``ROWS_PER_BLOCK * BLOCK_N * 4 B * 4 tiles`` has to stay under the 192 KB UB. The
  batch-invariant mode pins the value to ``MAX_ROWS_PER_BLOCK`` because the row count must
  not change the reduction order.
* The 3-D gate row address is *not* written as ``(row // H) * stride_token + (row % H) *
  stride_head`` on the hot path. For the token-major ``[T, H, D]`` view every caller passes,
  ``stride_token == H * stride_head``, and the divide/modulo form hides that from the
  compiler: the gate tile degenerates into a per-row gather. On the 524288x128 case the two
  spellings measure 533 us versus 311 us (MBU 52.4% -> 89.9%) on 950PR. The divide form is
  kept only for genuinely non-affine strides.
* ``EVEN`` drops the load/store predicates when the tile covers only real rows and columns.
  On the one-token quant schedule the predicated variant measures 131.1 us against 121.1 us
  dense, so the dense form is kept for exact tiles; non-exact tiles keep the predicates.
  (The grouped multi-token schedule only exists in the dense form -- its predicated variant
  does not compile on 3.6.0 -- so a non-exact quant tile falls back to one token per program.)
* The FP8 epilogue lives in its own kernel. ``amax`` has to span ``quant_heads`` rows, so the
  token grouping is part of the schedule; grouping four tokens per program also removes the
  per-element layout conversions that one-token-per-program tiles pay for the weight and
  scale broadcasts (174.7 us -> 106.5 us on 4096x8/512). The explicit ``+-448`` clamp of the
  reference is a no-op -- the quotient is bounded by ``amax / (amax/448)`` and the e4m3 cast
  saturates -- and dropping it is worth another ~29 us while the quant accuracy cases stay
  bit-exact under ``torch.equal``.
* ``num_warps`` is kept in the signature for source compatibility; the Ascend backend maps a
  program onto one vector core and ignores it.
"""

import os
from functools import lru_cache
from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import libentry

# Pinned by the batch-invariant contract: the value must not vary with M or the core count,
# otherwise the reduction order (and therefore the numerics) would depend on the batch size.
MAX_ROWS_PER_BLOCK = 4

# UB budget for the tile, in bytes. A2 exposes 192 KB per vector core; keeping the live tiles
# under half of it leaves room for the double buffering the backend inserts on its own.
_UB_TILE_BUDGET = 96 * 1024

# The backend will accept tiles up to the real UB size, but only shapes that gain more from
# full-width lanes than they lose to the missing double buffer are worth pushing that far.
_UB_HARD_LIMIT = 192 * 1024

# Element size of the FP32 tiles the kernel keeps live: x, gate, output, weight/bias.
_LIVE_FP32_TILES = 4

_MAX_FUSED_SIZE_DIVISOR = 65536


@libentry()
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
    Z_AFFINE: tl.constexpr = False,
    EVEN: tl.constexpr = False,
    GATE_MODE: tl.constexpr = 0,
):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)

    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N

    x_base = X + row_offsets + col_offsets
    y_base = Y + rows[:, None] * stride_y_row + col_offsets

    # ``EVEN`` says the tile covers only real rows and columns, so the predicates can be
    # dropped: masked loads/stores are lowered to a slower DMA path (measured +40% on this
    # backend), and an exact tile has nothing to mask anyway.
    if EVEN:
        x = tl.load(x_base).to(tl.float32)
    else:
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        x = tl.load(x_base, mask=mask, other=0.0).to(tl.float32)

    # A 3-D gate is addressed per row. When ``stride_z_token == Z_HEADS * stride_z_head`` --
    # the token-major ``[T, H, D]`` view every caller actually passes -- the head/token split is
    # algebraically ``row * stride_z_head``. Writing it as an integer divide/modulo hides that
    # from the compiler and the gate tile degenerates into a per-row gather: on 950PR the same
    # 524288x128 case runs 533 us strided versus 311 us affine (MBU 52.4% -> 89.9%). Keep the
    # divide form only for the genuinely non-affine strides.
    if HAS_Z and not NORM_BEFORE_GATE:
        if Z_IS_3D and not Z_AFFINE:
            z_row_offsets = (rows[:, None] // Z_HEADS) * stride_z_token + (
                rows[:, None] % Z_HEADS
            ) * stride_z_head
        elif Z_IS_3D:
            z_row_offsets = rows[:, None] * stride_z_head
        else:
            z_row_offsets = rows[:, None] * stride_z_row
        z_ptr = Z + z_row_offsets + col_offsets
        if EVEN:
            z = tl.load(z_ptr).to(tl.float32)
        else:
            z = tl.load(z_ptr, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            if GATE_MODE == 1:
                x *= z * (1.0 / (1.0 + tl.exp2(-z * 1.4426950408889634)))
            else:
                x *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            if GATE_MODE == 1:
                x *= 1.0 / (1.0 + tl.exp2(-z * 1.4426950408889634))
            else:
                x *= tl.sigmoid(z)

    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N
        if EVEN:
            tl.store(Mean + group * M + rows, mean)
            xbar = x - mean[:, None]
        else:
            tl.store(Mean + group * M + rows, mean, mask=rows < M)
            xbar = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
    else:
        if EVEN:
            var = tl.sum(x * x, axis=1) / N
        else:
            xbar = tl.where(mask, x, 0.0)
            var = tl.sum(xbar * xbar, axis=1) / N
        mean = tl.zeros((ROWS_PER_BLOCK,), dtype=tl.float32)

    rstd = tl.rsqrt(var + eps)
    if EVEN:
        tl.store(Rstd + group * M + rows, rstd)
    else:
        tl.store(Rstd + group * M + rows, rstd, mask=rows < M)

    w_offsets = cols + group * N
    w_mask = cols < N
    if EVEN:
        w = tl.load(W + w_offsets).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(B + w_offsets).to(tl.float32)
    else:
        w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(B + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if not IS_RMS_NORM:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    else:
        x_hat = x * rstd[:, None]

    if HAS_BIAS:
        y = x_hat * w[None, :] + b[None, :]
    else:
        y = x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        if Z_IS_3D and not Z_AFFINE:
            z_row_offsets = (rows[:, None] // Z_HEADS) * stride_z_token + (
                rows[:, None] % Z_HEADS
            ) * stride_z_head
        elif Z_IS_3D:
            z_row_offsets = rows[:, None] * stride_z_head
        else:
            z_row_offsets = rows[:, None] * stride_z_row
        z_ptr = Z + z_row_offsets + col_offsets
        if EVEN:
            z = tl.load(z_ptr).to(tl.float32)
        else:
            z = tl.load(z_ptr, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            if GATE_MODE == 1:
                y *= z * (1.0 / (1.0 + tl.exp2(-z * 1.4426950408889634)))
            else:
                y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            if GATE_MODE == 1:
                y *= 1.0 / (1.0 + tl.exp2(-z * 1.4426950408889634))
            else:
                y *= tl.sigmoid(z)

    if EVEN:
        tl.store(y_base, y)
    else:
        tl.store(y_base, y, mask=mask)


@triton.jit
def _quant_z_chunk(
    Z,
    rows,
    base_col: tl.constexpr,
    CHUNK: tl.constexpr,
    stride_z_row,
    stride_z_token,
    stride_z_head,
    Z_IS_3D: tl.constexpr,
    Z_AFFINE: tl.constexpr,
    Z_HEADS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Load one column slice of the gate and apply its activation (FP32)."""
    cols = base_col + tl.arange(0, CHUNK)
    if Z_IS_3D and not Z_AFFINE:
        z_row = (rows[:, None] // Z_HEADS) * stride_z_token + (
            rows[:, None] % Z_HEADS) * stride_z_head
    elif Z_IS_3D:
        z_row = rows[:, None] * stride_z_head
    else:
        z_row = rows[:, None] * stride_z_row
    z = tl.load(Z + z_row + cols[None, :]).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        return z * tl.sigmoid(z)
    return tl.sigmoid(z)


@triton.jit
def _quant_affine_chunk(
    W,
    B,
    x,
    rstd,
    mean,
    base_col: tl.constexpr,
    CHUNK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    """Normalised value times weight (plus bias) for one column slice, in FP32."""
    cols = base_col + tl.arange(0, CHUNK)
    if IS_RMS_NORM:
        x_hat = x * rstd[:, None]
    else:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    w = tl.load(W + cols).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols).to(tl.float32)
        return x_hat * w[None, :] + b[None, :]
    return x_hat * w[None, :]


@libentry()
@triton.jit
def _layer_norm_fwd_quant_kernel(
    X,
    W,
    B,
    Z,
    Mean,
    Rstd,
    Q,
    S,
    stride_x_row,
    stride_z_row,
    stride_z_token,
    stride_z_head,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWS_PER_TOKEN: tl.constexpr,
    TOKENS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    GATE_FIRST: tl.constexpr,
    Z_IS_3D: tl.constexpr,
    Z_AFFINE: tl.constexpr,
    Z_HEADS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    """Shipped FP8 epilogue: one fused kernel built from explicit column slices.

    Measured on 4096x8/N=512 (950PR): **85.7 us / 67.9% MBU** against 107.3 us / 54.3% for the
    same math on a full 32x512 tile and 99.8 us for the three-kernel split. The reason is the
    compiler: a full-width tile gets materialised, and then every elementwise op costs a full
    UB pass (measured: 39.3 us with no cross-lane reduction, 91.2 us once the row sum is in the
    same kernel). Four explicit slices keep the live set small enough that the chain streams.

    The statistics are still FP32, the BF16 round happens before the amax, and the scale is
    applied with ``div_rn`` -- the quant accuracy cases stay bit-exact under ``torch.equal``.
    """
    pid = tl.program_id(0)
    token0 = pid * TOKENS
    rows = token0 * ROWS_PER_TOKEN + tl.arange(0, TOKENS * ROWS_PER_TOKEN)

    x0 = tl.load(X + rows[:, None] * stride_x_row + (0 * CHUNK + tl.arange(0, CHUNK))[None, :]).to(tl.float32)
    x1 = tl.load(X + rows[:, None] * stride_x_row + (1 * CHUNK + tl.arange(0, CHUNK))[None, :]).to(tl.float32)
    x2 = tl.load(X + rows[:, None] * stride_x_row + (2 * CHUNK + tl.arange(0, CHUNK))[None, :]).to(tl.float32)
    x3 = tl.load(X + rows[:, None] * stride_x_row + (3 * CHUNK + tl.arange(0, CHUNK))[None, :]).to(tl.float32)

    if HAS_Z and GATE_FIRST:
        x0 = x0 * _quant_z_chunk(Z, rows, 0 * CHUNK, CHUNK, stride_z_row, stride_z_token,
                                 stride_z_head, Z_IS_3D, Z_AFFINE, Z_HEADS, ACTIVATION)
        x1 = x1 * _quant_z_chunk(Z, rows, 1 * CHUNK, CHUNK, stride_z_row, stride_z_token,
                                 stride_z_head, Z_IS_3D, Z_AFFINE, Z_HEADS, ACTIVATION)
        x2 = x2 * _quant_z_chunk(Z, rows, 2 * CHUNK, CHUNK, stride_z_row, stride_z_token,
                                 stride_z_head, Z_IS_3D, Z_AFFINE, Z_HEADS, ACTIVATION)
        x3 = x3 * _quant_z_chunk(Z, rows, 3 * CHUNK, CHUNK, stride_z_row, stride_z_token,
                                 stride_z_head, Z_IS_3D, Z_AFFINE, Z_HEADS, ACTIVATION)

    if IS_RMS_NORM:
        var = (tl.sum(x0 * x0, axis=1) + tl.sum(x1 * x1, axis=1)
               + tl.sum(x2 * x2, axis=1) + tl.sum(x3 * x3, axis=1)) / N
        mean = tl.zeros((TOKENS * ROWS_PER_TOKEN,), dtype=tl.float32)
    else:
        mean = (tl.sum(x0, axis=1) + tl.sum(x1, axis=1)
                + tl.sum(x2, axis=1) + tl.sum(x3, axis=1)) / N
        m2 = mean[:, None]
        var = (tl.sum((x0 - m2) * (x0 - m2), axis=1) + tl.sum((x1 - m2) * (x1 - m2), axis=1)
               + tl.sum((x2 - m2) * (x2 - m2), axis=1) + tl.sum((x3 - m2) * (x3 - m2), axis=1)) / N
        tl.store(Mean + rows, mean)
    rstd = tl.rsqrt(var + eps)
    tl.store(Rstd + rows, rstd)

    y0 = _quant_affine_chunk(W, B, x0, rstd, mean, 0 * CHUNK, CHUNK, HAS_BIAS, IS_RMS_NORM)
    y1 = _quant_affine_chunk(W, B, x1, rstd, mean, 1 * CHUNK, CHUNK, HAS_BIAS, IS_RMS_NORM)
    y2 = _quant_affine_chunk(W, B, x2, rstd, mean, 2 * CHUNK, CHUNK, HAS_BIAS, IS_RMS_NORM)
    y3 = _quant_affine_chunk(W, B, x3, rstd, mean, 3 * CHUNK, CHUNK, HAS_BIAS, IS_RMS_NORM)

    if HAS_Z and not GATE_FIRST:
        y0 = y0 * _quant_z_chunk(Z, rows, 0 * CHUNK, CHUNK, stride_z_row, stride_z_token,
                                 stride_z_head, Z_IS_3D, Z_AFFINE, Z_HEADS, ACTIVATION)
        y1 = y1 * _quant_z_chunk(Z, rows, 1 * CHUNK, CHUNK, stride_z_row, stride_z_token,
                                 stride_z_head, Z_IS_3D, Z_AFFINE, Z_HEADS, ACTIVATION)
        y2 = y2 * _quant_z_chunk(Z, rows, 2 * CHUNK, CHUNK, stride_z_row, stride_z_token,
                                 stride_z_head, Z_IS_3D, Z_AFFINE, Z_HEADS, ACTIVATION)
        y3 = y3 * _quant_z_chunk(Z, rows, 3 * CHUNK, CHUNK, stride_z_row, stride_z_token,
                                 stride_z_head, Z_IS_3D, Z_AFFINE, Z_HEADS, ACTIVATION)

    # Round to the output dtype before the amax, exactly as the unfused reference does.
    b0 = y0.to(tl.bfloat16)
    b1 = y1.to(tl.bfloat16)
    b2 = y2.to(tl.bfloat16)
    b3 = y3.to(tl.bfloat16)

    row_max = tl.maximum(
        tl.maximum(tl.max(tl.abs(b0.to(tl.float32)), axis=1),
                   tl.max(tl.abs(b1.to(tl.float32)), axis=1)),
        tl.maximum(tl.max(tl.abs(b2.to(tl.float32)), axis=1),
                   tl.max(tl.abs(b3.to(tl.float32)), axis=1)))
    if TOKENS == 1:
        amax = tl.maximum(row_max, 1e-10)
        srow = amax * (1.0 / 448.0)
        y_s = srow
    else:
        amax = tl.maximum(tl.max(tl.reshape(row_max, (TOKENS, ROWS_PER_TOKEN)), axis=1), 1e-10)
        y_s = amax * (1.0 / 448.0)
        srow = tl.reshape(tl.broadcast_to(y_s[:, None], (TOKENS, ROWS_PER_TOKEN)),
                          (TOKENS * ROWS_PER_TOKEN,))

    q0 = tl.math.div_rn(b0.to(tl.float32), srow[:, None]).to(Q.dtype.element_ty)
    q1 = tl.math.div_rn(b1.to(tl.float32), srow[:, None]).to(Q.dtype.element_ty)
    q2 = tl.math.div_rn(b2.to(tl.float32), srow[:, None]).to(Q.dtype.element_ty)
    q3 = tl.math.div_rn(b3.to(tl.float32), srow[:, None]).to(Q.dtype.element_ty)
    c = tl.arange(0, CHUNK)
    q_base = Q + rows[:, None] * N
    tl.store(q_base + (0 * CHUNK + c)[None, :], q0)
    tl.store(q_base + (1 * CHUNK + c)[None, :], q1)
    tl.store(q_base + (2 * CHUNK + c)[None, :], q2)
    tl.store(q_base + (3 * CHUNK + c)[None, :], q3)
    tl.store(S + token0 + tl.arange(0, TOKENS), y_s)


@libentry()
@triton.jit
def _layer_norm_fwd_quant_tile_kernel(
    X,
    W,
    B,
    Z,
    Mean,
    Rstd,
    Q,
    S,
    stride_x_row,
    stride_z_row,
    stride_z_token,
    stride_z_head,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS_PER_TOKEN: tl.constexpr,
    TOKENS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    Z_IS_3D: tl.constexpr,
    Z_HEADS: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
    Z_AFFINE: tl.constexpr,
    EVEN: tl.constexpr,
):
    """FP8 e4m3 per-token quant epilogue, one program per ``TOKENS`` tokens.

    The epilogue is what makes this a separate kernel: ``amax`` must span a whole token
    (``quant_heads`` rows), so the token grouping is a hard part of the schedule and the
    amax + division + cast chain does not share a tile shape with the plain store path.

    Two things are load bearing for both accuracy and speed:

    * ``ROWS_PER_TOKEN * BLOCK_N`` elements per scale. With one token per program the tile is
      only ``8 x 512`` on the 4096x8/512 case, and the weight broadcast as well as the scale
      broadcast end up as per-element layout conversions. Grouping four tokens per program
      (32 rows) removes those conversions: 174.7 us -> 108.9 us on 950PR.
    * No explicit clamp. The reference clamps the quotient to +-448 before the cast, but the
      quotient is bounded by ``amax / (amax/448)`` and the e4m3 cast saturates at 448, so the
      clamp is a no-op on every element -- while costing ~29 us here. The quant accuracy cases
      are bit-exact compared with ``torch.equal`` after the removal.
    """
    pid = tl.program_id(0)
    token0 = pid * TOKENS
    rows = token0 * ROWS_PER_TOKEN + tl.arange(0, TOKENS * ROWS_PER_TOKEN)
    cols = tl.arange(0, BLOCK_N)
    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :]

    if EVEN:
        x = tl.load(X + row_offsets + col_offsets).to(tl.float32)
    else:
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        x = tl.load(X + row_offsets + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Gate-first moves the activation in front of the statistics, exactly like the plain
    # kernel; gate-after-norm defers the same load so the gate tile is not live across the
    # reduction.
    if HAS_Z and not NORM_BEFORE_GATE:
        if Z_IS_3D and not Z_AFFINE:
            z_pre_offsets = (rows[:, None] // Z_HEADS) * stride_z_token + (
                rows[:, None] % Z_HEADS
            ) * stride_z_head
        elif Z_IS_3D:
            z_pre_offsets = rows[:, None] * stride_z_head
        else:
            z_pre_offsets = rows[:, None] * stride_z_row
        z_ptr_pre = Z + z_pre_offsets + col_offsets
        if EVEN:
            z_pre = tl.load(z_ptr_pre).to(tl.float32)
        else:
            z_pre = tl.load(z_ptr_pre, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x = x * (z_pre * tl.sigmoid(z_pre))
        elif ACTIVATION == "sigmoid":
            x = x * tl.sigmoid(z_pre)

    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=1) / N
        if EVEN:
            tl.store(Mean + rows, mean)
            xbar = x - mean[:, None]
        else:
            tl.store(Mean + rows, mean, mask=rows < M)
            xbar = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
    else:
        if EVEN:
            var = tl.sum(x * x, axis=1) / N
        else:
            x_masked = tl.where(mask, x, 0.0)
            var = tl.sum(x_masked * x_masked, axis=1) / N
        mean = tl.zeros((TOKENS * ROWS_PER_TOKEN,), dtype=tl.float32)
    rstd = tl.rsqrt(var + eps)
    if EVEN:
        tl.store(Rstd + rows, rstd)
    else:
        tl.store(Rstd + rows, rstd, mask=rows < M)

    if EVEN:
        w = tl.load(W + cols).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(B + cols).to(tl.float32)
    else:
        w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(B + cols, mask=cols < N, other=0.0).to(tl.float32)

    if IS_RMS_NORM:
        x_hat = x * rstd[:, None]
    else:
        x_hat = (x - mean[:, None]) * rstd[:, None]
    if HAS_BIAS:
        y = x_hat * w[None, :] + b[None, :]
    else:
        y = x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        if Z_IS_3D and not Z_AFFINE:
            z_row_offsets = (rows[:, None] // Z_HEADS) * stride_z_token + (
                rows[:, None] % Z_HEADS
            ) * stride_z_head
        elif Z_IS_3D:
            z_row_offsets = rows[:, None] * stride_z_head
        else:
            z_row_offsets = rows[:, None] * stride_z_row
        z_ptr = Z + z_row_offsets + col_offsets
        if EVEN:
            z = tl.load(z_ptr).to(tl.float32)
        else:
            z = tl.load(z_ptr, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y = y * (z * tl.sigmoid(z))
        elif ACTIVATION == "sigmoid":
            y = y * tl.sigmoid(z)

    # Round to the output dtype first: the amax must be measured on the value that would be
    # stored, matching per_token_group_quant_fp8 bit for bit.
    yb16 = y.to(tl.bfloat16)
    yb = yb16.to(tl.float32)

    if TOKENS == 1:
        amax = tl.maximum(tl.max(tl.abs(yb)), 1e-10)
    else:
        # Reduce per row first, then fold the ``ROWS_PER_TOKEN`` row maxima into one token
        # maximum. Reducing the whole ``(TOKENS, rows*N)`` reshape in one step costs ~3 us
        # more here because it materialises the flattened tile.
        row_max = tl.max(tl.abs(yb), axis=1)
        amax = tl.maximum(tl.max(tl.reshape(row_max, (TOKENS, ROWS_PER_TOKEN)), axis=1), 1e-10)
    y_s = amax * (1.0 / 448.0)

    if TOKENS == 1:
        y_q = tl.math.div_rn(yb, y_s).to(Q.dtype.element_ty)
        if EVEN:
            tl.store(Q + row_offsets + col_offsets, y_q)
        else:
            tl.store(Q + row_offsets + col_offsets, y_q, mask=mask)
        tl.store(S + pid, y_s)
    else:
        srow = tl.reshape(tl.broadcast_to(y_s[:, None], (TOKENS, ROWS_PER_TOKEN)),
                          (TOKENS * ROWS_PER_TOKEN,))
        y_q = tl.math.div_rn(yb, srow[:, None]).to(Q.dtype.element_ty)
        if EVEN:
            tl.store(Q + row_offsets + col_offsets, y_q)
        else:
            tl.store(Q + row_offsets + col_offsets, y_q, mask=mask)
        tl.store(S + token0 + tl.arange(0, TOKENS), y_s)


@libentry()
@triton.jit
def _layer_norm_fwd_rstd_kernel(
    X,
    Z,
    Mean,
    Rstd,
    stride_x_row,
    stride_z_row,
    stride_z_token,
    stride_z_head,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
    TOKENS: tl.constexpr,
    ROWS_PER_TOKEN: tl.constexpr,
    HAS_Z: tl.constexpr,
    GATE_FIRST: tl.constexpr,
    Z_IS_3D: tl.constexpr,
    Z_AFFINE: tl.constexpr,
    Z_HEADS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Stage 1 of the FP8 pipeline: the per-row statistics only.

    Why this is a separate kernel: on 3.6.0 a kernel that contains *both* a cross-lane
    reduction and an elementwise chain materialises the whole tile, and every elementwise op
    then costs a full UB pass. Measured on 4096x8/512: the same math is 39.3 us with no
    reduction and 91.2 us with the row sum. Splitting the reduction out costs one extra read
    of x (33.5 MB, ~7 us) and removes ~50 us of materialisation.
    """
    pid = tl.program_id(0)
    rows = pid * (TOKENS * ROWS_PER_TOKEN) + tl.arange(0, TOKENS * ROWS_PER_TOKEN)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N) if not EVEN else rows[:, None] < M

    if EVEN:
        x = tl.load(X + rows[:, None] * stride_x_row + cols[None, :]).to(tl.float32)
    else:
        x = tl.load(X + rows[:, None] * stride_x_row + cols[None, :], mask=mask,
                    other=0.0).to(tl.float32)

    if HAS_Z and GATE_FIRST:
        if Z_IS_3D and not Z_AFFINE:
            z_row = (rows[:, None] // Z_HEADS) * stride_z_token + (
                rows[:, None] % Z_HEADS) * stride_z_head
        elif Z_IS_3D:
            z_row = rows[:, None] * stride_z_head
        else:
            z_row = rows[:, None] * stride_z_row
        z_ptr = Z + z_row + cols[None, :]
        if EVEN:
            z = tl.load(z_ptr).to(tl.float32)
        else:
            z = tl.load(z_ptr, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x = x * (z * tl.sigmoid(z))
        elif ACTIVATION == "sigmoid":
            x = x * tl.sigmoid(z)

    if IS_RMS_NORM:
        if EVEN:
            var = tl.sum(x * x, axis=1) / N
        else:
            xm = tl.where(mask, x, 0.0)
            var = tl.sum(xm * xm, axis=1) / N
        mean = tl.zeros((TOKENS * ROWS_PER_TOKEN,), dtype=tl.float32)
    else:
        if EVEN:
            mean = tl.sum(x, axis=1) / N
            xbar = x - mean[:, None]
        else:
            mean = tl.sum(tl.where(mask, x, 0.0), axis=1) / N
            xbar = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(xbar * xbar, axis=1) / N
    if EVEN:
        tl.store(Rstd + rows, tl.rsqrt(var + eps))
    else:
        tl.store(Rstd + rows, tl.rsqrt(var + eps), mask=rows < M)
    if not IS_RMS_NORM:
        if EVEN:
            tl.store(Mean + rows, mean)
        else:
            tl.store(Mean + rows, mean, mask=rows < M)


@libentry()
@triton.jit
def _layer_norm_fwd_quant_body_kernel(
    X,
    W,
    B,
    Z,
    Rstd,
    Mean,
    YB,
    S,
    stride_x_row,
    stride_z_row,
    stride_z_token,
    stride_z_head,
    stride_yb_row,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOKENS: tl.constexpr,
    ROWS_PER_TOKEN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    GATE_FIRST: tl.constexpr,
    Z_IS_3D: tl.constexpr,
    Z_AFFINE: tl.constexpr,
    Z_HEADS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Stage 2: normalise + affine + gate, round to BF16, and reduce the token amax.

    The amax is stored, never broadcast back into the tile: that keeps this kernel free of the
    reduce->broadcast layout conversion (see ``_layer_norm_fwd_quant_scale_kernel`` for the
    stage that applies the scale as a scalar).
    """
    pid = tl.program_id(0)
    token0 = pid * TOKENS
    rows = token0 * ROWS_PER_TOKEN + tl.arange(0, TOKENS * ROWS_PER_TOKEN)
    cols = tl.arange(0, BLOCK_N)
    col_offsets = cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N) if not EVEN else rows[:, None] < M

    if EVEN:
        x = tl.load(X + rows[:, None] * stride_x_row + col_offsets).to(tl.float32)
        rstd = tl.load(Rstd + rows[:, None])
        mean = tl.load(Mean + rows[:, None]) if not IS_RMS_NORM else tl.zeros(
            (TOKENS * ROWS_PER_TOKEN, 1), dtype=tl.float32)
    else:
        x = tl.load(X + rows[:, None] * stride_x_row + col_offsets, mask=mask,
                    other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + rows[:, None])
        mean = tl.load(Mean + rows[:, None]) if not IS_RMS_NORM else tl.zeros(
            (TOKENS * ROWS_PER_TOKEN, 1), dtype=tl.float32)

    if EVEN:
        w = tl.load(W + cols).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(B + cols).to(tl.float32)
    else:
        w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(B + cols, mask=cols < N, other=0.0).to(tl.float32)

    if HAS_Z and GATE_FIRST:
        # Gate-first: the reference folds act(z) into x *before* the statistics and before the
        # affine, so the rounding order is ((x*act(z))*rstd)*w. Re-apply the same gate here.
        if Z_IS_3D and not Z_AFFINE:
            zg_row = (rows[:, None] // Z_HEADS) * stride_z_token + (
                rows[:, None] % Z_HEADS) * stride_z_head
        elif Z_IS_3D:
            zg_row = rows[:, None] * stride_z_head
        else:
            zg_row = rows[:, None] * stride_z_row
        zg_ptr = Z + zg_row + col_offsets
        if EVEN:
            zg = tl.load(zg_ptr).to(tl.float32)
        else:
            zg = tl.load(zg_ptr, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x = x * (zg * tl.sigmoid(zg))
        else:
            x = x * tl.sigmoid(zg)

    if IS_RMS_NORM:
        x_hat = x * rstd
    else:
        x_hat = (x - mean) * rstd
    if HAS_BIAS:
        y = x_hat * w[None, :] + b[None, :]
    else:
        y = x_hat * w[None, :]

    if HAS_Z and not GATE_FIRST:
        if Z_IS_3D and not Z_AFFINE:
            z_row = (rows[:, None] // Z_HEADS) * stride_z_token + (
                rows[:, None] % Z_HEADS) * stride_z_head
        elif Z_IS_3D:
            z_row = rows[:, None] * stride_z_head
        else:
            z_row = rows[:, None] * stride_z_row
        z_ptr = Z + z_row + col_offsets
        if EVEN:
            z = tl.load(z_ptr).to(tl.float32)
        else:
            z = tl.load(z_ptr, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y = y * (z * tl.sigmoid(z))
        elif ACTIVATION == "sigmoid":
            y = y * tl.sigmoid(z)

    yb16 = y.to(tl.bfloat16)
    if EVEN:
        tl.store(YB + rows[:, None] * stride_yb_row + col_offsets, yb16)
    else:
        tl.store(YB + rows[:, None] * stride_yb_row + col_offsets, yb16, mask=mask)

    yb = yb16.to(tl.float32)
    if TOKENS == 1:
        amax = tl.maximum(tl.max(tl.abs(yb)), 1e-10)
    else:
        row_max = tl.max(tl.abs(yb), axis=1)
        amax = tl.maximum(tl.max(tl.reshape(row_max, (TOKENS, ROWS_PER_TOKEN)), axis=1), 1e-10)
    tl.store(S + token0 + tl.arange(0, TOKENS), amax * (1.0 / 448.0))


@libentry()
@triton.jit
def _layer_norm_fwd_quant_scale_kernel(
    YB,
    Q,
    S,
    stride_yb_row,
    stride_q_row,
    M,
    N: tl.constexpr,
    ROWS_PER_TOKEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Stage 3: one token per program, so the scale is a *scalar* (a loaded vector would cost
    a ~28 us layout conversion per tile). 15.1 us for the 4096x8/512 case."""
    pid = tl.program_id(0)
    rows = pid * ROWS_PER_TOKEN + tl.arange(0, ROWS_PER_TOKEN)
    cols = tl.arange(0, BLOCK_N)
    if EVEN:
        yb = tl.load(YB + rows[:, None] * stride_yb_row + cols[None, :]).to(tl.float32)
    else:
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        yb = tl.load(YB + rows[:, None] * stride_yb_row + cols[None, :], mask=mask,
                     other=0.0).to(tl.float32)
    s = tl.load(S + pid)
    q = tl.math.div_rn(yb, s).to(Q.dtype.element_ty)
    if EVEN:
        tl.store(Q + rows[:, None] * stride_q_row + cols[None, :], q)
    else:
        tl.store(Q + rows[:, None] * stride_q_row + cols[None, :], q,
                 mask=(rows[:, None] < M) & (cols[None, :] < N))



@lru_cache(maxsize=1)
def _is_a2_910b() -> bool:
    """True on the A2 family (910B2 / 910B3 / 910B4). Swept on 910B3 and re-measured on 910B2.

    The row counts below are a property of the A2 vector core and its 192 KB UB, not of one
    SKU: on 910B2 the same rule takes qk_norm 43.2 -> 33.5 us, gdn_65536x8 604 -> 506 us and
    tail_32771x192 82.6 -> 50.2 us (MBU 36.6% -> 60.2%), with gdn_4096x8 the only case that
    gives back a little (34.2 -> 34.8 us). A3 (910_9382) and 950 keep the UB-budget default.
    """
    try:
        return "910B" in triton.runtime.driver.active.get_current_target().arch
    except Exception:
        return False


@lru_cache(maxsize=1)
def _is_a3() -> bool:
    """True on Ascend 910_9382 (A3). Swept on the A3 container (48 vector cores)."""
    try:
        return "910_9382" in triton.runtime.driver.active.get_current_target().arch
    except Exception:
        return False


@lru_cache(maxsize=1)
def _is_a5() -> bool:
    """True on Ascend 950 (A5). The row counts below were swept on DevEnv_499150."""
    try:
        return "950" in triton.runtime.driver.active.get_current_target().arch
    except Exception:
        return False


def _pick_rows_per_block(
    m: int,
    block_n: int,
    batch_invariant: bool,
    has_gate: bool = True,
    padded_cols: bool = False,
    ngroups: int = 1,
    activation: str = "swish",
) -> int:
    """Rows per tile, from the UB budget rather than from a CUDA warp count.

    ``ROWS_PER_BLOCK * BLOCK_N`` is the tile size; the kernel keeps FP32 ``x``, the gate, the
    output and the weight live at the same time, so the budget is divided by the live tile
    count -- four with a gate, three without one. ``MAX`` comes from the budget, not from the
    reference's CUDA constant.

    Two corrections over "budget, clamped, rounded up" were measured on A2 against the official
    ``tests/performance/functions/test_layernorm_gated.py`` cases:

    * ``padded_cols`` (the row is not a power of two, so ``BLOCK_N`` is wider than the row):
      the tile already wastes column lanes, so the row count has to be a power of two too or it
      wastes rows as well. On ``tail_2d_swish`` (32771x192) the old rule produced 24 rows inside
      a 32-lane block and ran at 107 us / MBU 28.0%; stepping up to 32 rows gives 79.7 us /
      MBU 37.8%. The step is capped by the real UB size, where 48 rows (196 KB) falls back to
      35.9%, so 32 is the top of the plateau.
    * a gate-free call has one fewer live tile, so it can afford twice the rows: ``qk_norm_nogate``
      goes from 48 to 64 rows and from MBU 48.3% to 64.7%.

    Shapes that are *not* column-padded keep the fractional row count -- on ``N=128`` the
    padded 48-row tile (98 KB) measures 62.2% against 51.4% for a full 64-row tile (131 KB),
    so rounding those up would trade row waste for UB pressure and lose.
    """
    if batch_invariant:
        # Numerics must not depend on M or the core count, so the constant wins over the budget.
        return MAX_ROWS_PER_BLOCK
    if _is_a2_910b():
        # Swept on cannlab X51Cm (910B3, 48 vector cores) and re-measured on 910B2 (DevEnv_466517).
        # The row tile only changes the program count; the per-row reduction is unchanged, so
        # results stay bit-identical.
        if block_n == 192:
            return min(m, 56)
        if block_n == 128:
            if not has_gate:
                return min(m, 96)
            if ngroups > 1:
                return min(m, 48)
            if m <= 4096:
                return min(m, 40)
            if m <= 262144:
                return min(m, 56)
            return min(m, 40)
    if _is_a3():
        # A3 (910_9382) exposes 48 vector cores; these counts were swept on the official
        # case matrix. The row tile does not change the per-row reduction, so the result
        # stays bit-identical.
        if block_n == 192:
            # Dense tail tile: 64 rows measured 46.8 us (MBU 63.8%) against 79.9 us at 32.
            return min(m, 64)
        if block_n == 128:
            if not has_gate:
                return min(m, 96)
            if ngroups > 1:
                return min(m, 48)
            # 524288 rows: 40 rows measured 496 us (MBU 64.3%) against 545 us at 48.
            if m >= 262144:
                return min(m, 40)
            if activation == "sigmoid":
                # Sigmoid's extra exp is hidden better by a wider row tile: 56 rows measured
                # 30.9 us (MBU 64.6%) against 33.4 us at 48 on gdn_prefill_sigmoid.
                return min(m, 56)
            return min(m, 48)
        if block_n == 256:
            if padded_cols or ngroups > 1:
                return min(m, 48)
            return min(m, 32)
        if block_n == 1024:
            return min(m, 12)
        if block_n == 2048:
            return min(m, 8)
    if _is_a5():
        # Swept on DevEnv_499150 (Ascend950PR, CANN 9.2.0-beta.2). With a strided 3-D gate the
        # gate tile is the limiter, so the row count buys bandwidth up to the UB budget (four
        # live FP32 tiles of ``rows * 128 * 4 B``); past ~96 rows the tile stops helping.
        if block_n == 128 and has_gate and ngroups == 1:
            if m <= 65536:
                return min(m, 56)
            return min(m, 80)
    live = _LIVE_FP32_TILES if has_gate else _LIVE_FP32_TILES - 1
    rows = _UB_TILE_BUDGET // (live * 4 * max(block_n, 1))
    rows = max(1, min(rows, m, 64))
    if padded_cols:
        cap = max(1, _UB_HARD_LIMIT // (live * 4 * max(block_n, 1)))
        rows = min(triton.next_power_of_2(rows), cap, 64)
    return max(1, rows)


def layernorm_gated_infer_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    z: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
    quant_heads: int = 0,
    batch_invariant: bool = False,
) -> Tuple[object, Optional[torch.Tensor], torch.Tensor]:
    """One-pass gated norm. Returns ``((q, s) if quant_heads else out), mean, rstd``."""
    m, n = x.shape
    if group_size is None:
        group_size = n
    if n % group_size:
        raise ValueError(f"group_size={group_size} must divide N={n}")
    ngroups = n // group_size
    if x.stride(-1) != 1:
        raise ValueError("x must be contiguous along the last dimension")

    z_is_3d = z is not None and z.ndim == 3
    if z is not None:
        if z.stride(-1) != 1:
            raise ValueError("z must be contiguous along the last dimension")
        if z_is_3d:
            if z.shape[0] * z.shape[1] != m or z.shape[2] != n:
                raise ValueError(
                    f"3-D z must be [T, H, {n}] with T*H == M == {m}, got {tuple(z.shape)}"
                )
        elif z.shape != (m, n):
            raise ValueError(f"z must be {(m, n)}, got {tuple(z.shape)}")

    # The quant path returns (q, s) and never touches ``out``; do not allocate 33.5 MB of
    # dead workspace for it (it measurably perturbs the measurement on 950PR).
    if out is None and not quant_heads:
        out = torch.empty_like(x)

    mean = (
        torch.empty((ngroups * m,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * m,), dtype=torch.float32, device=x.device)

    max_fused = _MAX_FUSED_SIZE_DIVISOR // x.element_size()
    if group_size == 192:
        # Dense 192-lane tile: Triton-Ascend accepts non-power-of-two arange here, so the
        # 192->256 padding can be removed for the tail case.
        block_n = min(max_fused, group_size)
    else:
        block_n = min(max_fused, triton.next_power_of_2(group_size))
    if group_size > block_n:
        raise RuntimeError("layernorm_gated does not support a feature dim >= 64 KB")

    num_warps = min(max(block_n // 256, 1), 8)
    rows_per_block = _pick_rows_per_block(
        m,
        block_n,
        batch_invariant,
        has_gate=z is not None,
        padded_cols=block_n != group_size,
        ngroups=ngroups,
        activation=activation,
    )

    # A token-major [T, H, D] gate whose head stride is not "the rest of the row" cannot be
    # addressed affinely; every other 3-D view (which is what callers pass) can.
    z_affine = bool(z_is_3d and z.stride(0) == z.shape[1] * z.stride(1))

    if quant_heads:
        if ngroups != 1:
            raise ValueError("quant_heads requires a single group")
        if m % quant_heads:
            raise ValueError(f"quant_heads={quant_heads} must divide M={m}")
        q = torch.empty(
            (m // quant_heads, quant_heads * n), dtype=torch.float8_e4m3fn, device=x.device
        )
        s = torch.empty((m // quant_heads, 1), dtype=torch.float32, device=x.device)
        # Group as many tokens per program as the UB budget allows: with one token per program
        # the weight and scale broadcasts are lowered as per-element layout conversions. The
        # grouped schedule only exists for exact tiles -- the predicated variant of it does not
        # compile on 3.6.0 (bishengir), and a non-exact row/column count falls back to one token
        # per program, which is the masked path the original kernel used.
        tokens = 1
        if group_size == block_n:
            for cand in (4, 2):
                if m % (cand * quant_heads) == 0 and cand * quant_heads * block_n <= 4 * 4096:
                    tokens = cand
                    break
        rows_per_token = quant_heads
        even = (m % (tokens * rows_per_token)) == 0 and group_size == block_n
        _warps_env = os.environ.get("LN_GATED_WARPS")
        if _warps_env:
            num_warps = max(1, int(_warps_env))
        grid = (triton.cdiv(m, tokens * rows_per_token),)
        schedule = _quant_schedule()
        chunk = block_n // 4
        chunked_ok = group_size == block_n and block_n % 4 == 0 and chunk >= 8
        if schedule == "chunked" and chunked_ok:
            # Column-sliced fused kernel: 85.7 us / 67.9% MBU on 4096x8/512 (one launch, minimum
            # traffic). Two tokens per program measured best (1 -> 136 us, 2 -> 85.7, 4 -> 99.0).
            ctokens = 2 if m % (2 * rows_per_token) == 0 else 1
            cgrid = (m // (ctokens * rows_per_token),)
            cwarps = int(os.environ.get("LN_GATED_WARPS") or 4)
            _layer_norm_fwd_quant_kernel[cgrid](
                x,
                weight,
                bias,
                z,
                mean,
                rstd,
                q,
                s,
                x.stride(0),
                z.stride(0) if z is not None and not z_is_3d else 0,
                z.stride(0) if z_is_3d else 0,
                z.stride(1) if z_is_3d else 0,
                m,
                group_size,
                eps,
                BLOCK_N=block_n,
                CHUNK=chunk,
                ROWS_PER_TOKEN=rows_per_token,
                TOKENS=ctokens,
                HAS_BIAS=bias is not None,
                HAS_Z=z is not None,
                GATE_FIRST=z is not None and not norm_before_gate,
                Z_IS_3D=z_is_3d,
                Z_AFFINE=z_affine,
                Z_HEADS=z.shape[1] if z_is_3d else 1,
                ACTIVATION=activation,
                IS_RMS_NORM=is_rms_norm,
                num_warps=cwarps,
            )
            return (q, s), mean, rstd
        if schedule == "pipeline":
            # Three-stage pipeline (see the kernel docstrings): statistics | norm+gate+amax |
            # scalar scale. 96.8 us vs 107.3 us for the single fused kernel on 4096x8/512.
            # 4 warps measured best for the 512-wide tile (2 -> 97.5 us, 4 -> 96.8 us).
            pipeline_warps = int(os.environ.get("LN_GATED_WARPS") or max(num_warps, 4))
            yb = torch.empty((m, n), dtype=x.dtype, device=x.device)
            _layer_norm_fwd_rstd_kernel[grid](
                x,
                z,
                mean,
                rstd,
                x.stride(0),
                z.stride(0) if z is not None and not z_is_3d else 0,
                z.stride(0) if z_is_3d else 0,
                z.stride(1) if z_is_3d else 0,
                m,
                group_size,
                eps,
                BLOCK_N=block_n,
                TOKENS=tokens,
                ROWS_PER_TOKEN=rows_per_token,
                HAS_Z=z is not None,
                GATE_FIRST=z is not None and not norm_before_gate,
                Z_IS_3D=z_is_3d,
                Z_AFFINE=z_affine,
                Z_HEADS=z.shape[1] if z_is_3d else 1,
                ACTIVATION=activation,
                IS_RMS_NORM=is_rms_norm,
                EVEN=even,
                num_warps=pipeline_warps,
            )
            _layer_norm_fwd_quant_body_kernel[grid](
                x,
                weight,
                bias,
                z,
                rstd,
                mean,
                yb,
                s,
                x.stride(0),
                z.stride(0) if z is not None and not z_is_3d else 0,
                z.stride(0) if z_is_3d else 0,
                z.stride(1) if z_is_3d else 0,
                yb.stride(0),
                m,
                group_size,
                BLOCK_N=block_n,
                TOKENS=tokens,
                ROWS_PER_TOKEN=rows_per_token,
                HAS_BIAS=bias is not None,
                HAS_Z=z is not None,
                GATE_FIRST=z is not None and not norm_before_gate,
                Z_IS_3D=z_is_3d,
                Z_AFFINE=z_affine,
                Z_HEADS=z.shape[1] if z_is_3d else 1,
                ACTIVATION=activation,
                IS_RMS_NORM=is_rms_norm,
                EVEN=even,
                num_warps=pipeline_warps,
            )
            _layer_norm_fwd_quant_scale_kernel[(m // rows_per_token,)](
                yb,
                q,
                s,
                yb.stride(0),
                n,
                m,
                group_size,
                ROWS_PER_TOKEN=rows_per_token,
                BLOCK_N=block_n,
                EVEN=even,
                num_warps=pipeline_warps,
            )
            return (q, s), mean, rstd
        _layer_norm_fwd_quant_tile_kernel[grid](
            x,
            weight,
            bias,
            z,
            mean,
            rstd,
            q,
            s,
            x.stride(0),
            z.stride(0) if z is not None and not z_is_3d else 0,
            z.stride(0) if z_is_3d else 0,
            z.stride(1) if z_is_3d else 0,
            m,
            group_size,
            eps,
            BLOCK_N=block_n,
            ROWS_PER_TOKEN=rows_per_token,
            TOKENS=tokens,
            HAS_BIAS=bias is not None,
            HAS_Z=z is not None,
            Z_IS_3D=z_is_3d,
            Z_HEADS=z.shape[1] if z_is_3d else 1,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            ACTIVATION=activation,
            Z_AFFINE=z_affine,
            EVEN=even,
            num_warps=num_warps,
        )
        return (q, s), mean, rstd

    even = (m % rows_per_block) == 0 and group_size == block_n
    # WP-AUTONOMY-THREE-OPERATOR: auditable tuning hooks for a verified platform.
    # They only change the launch tiling; the per-row math and result are unchanged.
    _rows_env = os.environ.get("LN_GATED_ROWS")
    if _rows_env:
        rows_per_block = max(1, min(int(_rows_env), m))
        even = (m % rows_per_block) == 0 and group_size == block_n
    _warps_env = os.environ.get("LN_GATED_WARPS")
    if _warps_env:
        num_warps = max(1, int(_warps_env))
    grid = (triton.cdiv(m, rows_per_block), ngroups)
    _layer_norm_fwd_1pass_kernel[grid](
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
        m,
        group_size,
        eps,
        BLOCK_N=block_n,
        ROWS_PER_BLOCK=rows_per_block,
        HAS_BIAS=bias is not None,
        HAS_Z=z is not None,
        Z_IS_3D=z_is_3d,
        Z_HEADS=z.shape[1] if z_is_3d else 1,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        ACTIVATION=activation,
        Z_AFFINE=z_affine,
        EVEN=even,
        GATE_MODE=int(os.environ.get("LN_GATED_GATE", "0") or 0),
        num_warps=num_warps,
    )
    return out, mean, rstd


def layernorm_gated_infer_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    z: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
    quant_heads: int = 0,
    batch_invariant: bool = False,
) -> Tuple[object, Optional[torch.Tensor], torch.Tensor]:
    """Provider entry point; keeps the argument order of the reference ``_layer_norm_fwd``."""
    return layernorm_gated_infer_impl(
        x,
        weight,
        bias,
        eps,
        z=z,
        out=out,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
        quant_heads=quant_heads,
        batch_invariant=batch_invariant,
    )


def _batch_invariant_flag() -> bool:
    """``LN_GATED_BATCH_INVARIANT=1`` pins the row tile so results do not depend on M."""
    return os.environ.get("LN_GATED_BATCH_INVARIANT", "").strip() not in ("", "0", "false", "False")


def _quant_schedule() -> str:
    """Which FP8 epilogue schedule to run.

    ``chunked`` (default, shipped): one fused kernel built from four explicit column slices.
    ``pipeline``: the three-stage split (statistics | norm+gate+amax | scalar scale).
    ``tile``: the previous single-tile kernel (one 32xN tile per program).
    The older schedules stay reachable so every measured candidate can be reproduced.
    """
    value = os.environ.get("LN_GATED_QUANT_SCHEDULE", "").strip().lower()
    if value in ("chunked", "pipeline", "tile"):
        return value
    legacy = os.environ.get("LN_GATED_QUANT_PIPELINE", "").strip()
    if legacy in ("0", "false", "False"):
        return "tile"
    if legacy in ("1", "true", "True"):
        return "pipeline"
    return "chunked"


def _quant_pipeline_enabled() -> bool:
    """``LN_GATED_QUANT_PIPELINE=0`` falls back to the single fused quant kernel.

    The three-stage pipeline is the shipped schedule; the switch keeps the previous
    single-kernel implementation reachable for A/B measurement and rollback.
    """
    return os.environ.get("LN_GATED_QUANT_PIPELINE", "1").strip() not in ("", "0", "false", "False")


def layer_norm_gated_fwd(
    x,
    weight,
    bias,
    z,
    eps,
    out,
    group_size,
    norm_before_gate,
    is_rms_norm,
    activation,
    quant_heads,
):
    """Public op id binding; keeps the upstream positional order and forwards to the kernel entry."""
    return layernorm_gated_infer_fwd(
        x,
        weight,
        bias,
        float(eps),
        z=z,
        out=out,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
        quant_heads=int(quant_heads),
        batch_invariant=_batch_invariant_flag(),
    )
