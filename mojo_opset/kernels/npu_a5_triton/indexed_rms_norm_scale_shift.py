# SPDX-License-Identifier: Apache-2.0
"""A5 stride-correct composition: selected H16 native SIMT, H5376 staged loads,
H16384 full-row pass. Original regional fallbacks and FP32 math retained.
H16 BM256/32 warps selected by both 18- and 49-configuration autotunes.
"""

from functools import lru_cache

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from triton.language.extra.cann.extension import extract_slice


@triton.jit
def _rms_small_table_v2_overlap_v3_packed_xprefetch(
    out_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    rows,
    total_blocks,
    H: tl.constexpr,
    eps: tl.constexpr,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_CORES: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr,
    SHIFT_ROW_STRIDE: tl.constexpr,
    SCALE_ROW_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    cols = tl.arange(0, H).to(tl.int32)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    r_off = tl.arange(0, BLOCK_M).to(tl.int32)
    for b in range(pid, total_blocks, NUM_CORES):
        r = b * BLOCK_M + r_off
        mask_m = r < rows
        mask = mask_m[:, None] & (cols[None, :] < H)
        index = tl.load(indices_ptr + r * INDEX_STRIDE, mask=mask_m, other=0).to(tl.int32)
        x = tl.load(x_ptr + r[:, None] * X_ROW_STRIDE + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=1) / H + eps)
        u = x * rstd[:, None]
        # row-invariant weight folded into the per-condition table row, so the
        # per-element chain is (x*rstd)*A + B instead of
        # (x*rstd)*weight*(1+scale)+shift
        coef = tl.zeros([BLOCK_M, H], dtype=tl.float32)
        shift = tl.zeros([BLOCK_M, H], dtype=tl.float32)
        for c in tl.static_range(C):
            t_sh = tl.load(shift_ptr + c * SHIFT_ROW_STRIDE + cols).to(tl.float32)
            t_sc = tl.load(scale_ptr + c * SCALE_ROW_STRIDE + cols).to(tl.float32)
            sel = index[:, None] == c
            coef = tl.where(sel, weight * (1.0 + t_sc), coef)
            shift = tl.where(sel, t_sh, shift)
        out = u * coef + shift
        tl.store(out_ptr + r[:, None] * H + cols[None, :], out, mask=mask)


@triton.jit
def _rms_medium_table_v2_overlap_v3_packed_xprefetch(
    out_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    rows,
    H: tl.constexpr,
    eps: tl.constexpr,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_CORES: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr,
    SHIFT_ROW_STRIDE: tl.constexpr,
    SCALE_ROW_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
):
    """Full tiles only (rows % BLOCK_M == 0); table hoisted out of the loop."""
    pid = tl.program_id(0).to(tl.int32)
    cols = tl.arange(0, H).to(tl.int32)
    r_off = tl.arange(0, BLOCK_M).to(tl.int32)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    nb = rows // BLOCK_M
    if C == 1:
        b0 = tl.load(shift_ptr + cols).to(tl.float32)
        a0 = weight * (1.0 + tl.load(scale_ptr + cols).to(tl.float32))
        for b in range(pid, nb, NUM_CORES):
            r = b * BLOCK_M + r_off
            x = tl.load(x_ptr + r[:, None] * X_ROW_STRIDE + cols[None, :]).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(x * x, axis=1) / H + eps)
            u = x * rstd[:, None]
            out = u * a0[None, :] + b0[None, :]
            tl.store(out_ptr + r[:, None] * H + cols[None, :], out)
    elif C == 2:
        b0 = tl.load(shift_ptr + cols).to(tl.float32)
        b1 = tl.load(shift_ptr + SHIFT_ROW_STRIDE + cols).to(tl.float32)
        a0 = weight * (1.0 + tl.load(scale_ptr + cols).to(tl.float32))
        a1 = weight * (1.0 + tl.load(scale_ptr + SCALE_ROW_STRIDE + cols).to(tl.float32))
        for b in range(pid, nb, NUM_CORES):
            r = b * BLOCK_M + r_off
            index = tl.load(indices_ptr + r * INDEX_STRIDE).to(tl.int32)
            x = tl.load(x_ptr + r[:, None] * X_ROW_STRIDE + cols[None, :]).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(x * x, axis=1) / H + eps)
            u = x * rstd[:, None]
            sel = index[:, None] == 0
            coef = tl.where(sel, a0[None, :], a1[None, :])
            sh = tl.where(sel, b0[None, :], b1[None, :])
            out = u * coef + sh
            tl.store(out_ptr + r[:, None] * H + cols[None, :], out)
    else:
        b0 = tl.load(shift_ptr + cols).to(tl.float32)
        b1 = tl.load(shift_ptr + SHIFT_ROW_STRIDE + cols).to(tl.float32)
        b2 = tl.load(shift_ptr + 2 * SHIFT_ROW_STRIDE + cols).to(tl.float32)
        b3 = tl.load(shift_ptr + 3 * SHIFT_ROW_STRIDE + cols).to(tl.float32)
        a0 = weight * (1.0 + tl.load(scale_ptr + cols).to(tl.float32))
        a1 = weight * (1.0 + tl.load(scale_ptr + SCALE_ROW_STRIDE + cols).to(tl.float32))
        a2 = weight * (1.0 + tl.load(scale_ptr + 2 * SCALE_ROW_STRIDE + cols).to(tl.float32))
        a3 = weight * (1.0 + tl.load(scale_ptr + 3 * SCALE_ROW_STRIDE + cols).to(tl.float32))
        for b in range(pid, nb, NUM_CORES):
            r = b * BLOCK_M + r_off
            row0 = b * BLOCK_M
            row1 = row0 + 1
            i0 = tl.load(indices_ptr + row0 * INDEX_STRIDE).to(tl.int32)
            i1 = tl.load(indices_ptr + row1 * INDEX_STRIDE).to(tl.int32)
            x = tl.load(x_ptr + r[:, None] * X_ROW_STRIDE + cols[None, :]).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(x * x, axis=1) / H + eps)
            x0 = extract_slice(x, (0, 0), (1, H), (1, 1))
            x1 = extract_slice(x, (1, 0), (1, H), (1, 1))
            r0 = extract_slice(rstd, (0,), (1,), (1,))
            r1 = extract_slice(rstd, (1,), (1,), (1,))
            if i0 == 0:
                result0 = (x0 * r0[:, None]) * a0[None, :] + b0[None, :]
                tl.store(out_ptr + row0 * H + cols[None, :], result0)
            elif i0 == 1:
                result0 = (x0 * r0[:, None]) * a1[None, :] + b1[None, :]
                tl.store(out_ptr + row0 * H + cols[None, :], result0)
            elif i0 == 2:
                result0 = (x0 * r0[:, None]) * a2[None, :] + b2[None, :]
                tl.store(out_ptr + row0 * H + cols[None, :], result0)
            else:
                result0 = (x0 * r0[:, None]) * a3[None, :] + b3[None, :]
                tl.store(out_ptr + row0 * H + cols[None, :], result0)
            if i1 == 0:
                result1 = (x1 * r1[:, None]) * a0[None, :] + b0[None, :]
                tl.store(out_ptr + row1 * H + cols[None, :], result1)
            elif i1 == 1:
                result1 = (x1 * r1[:, None]) * a1[None, :] + b1[None, :]
                tl.store(out_ptr + row1 * H + cols[None, :], result1)
            elif i1 == 2:
                result1 = (x1 * r1[:, None]) * a2[None, :] + b2[None, :]
                tl.store(out_ptr + row1 * H + cols[None, :], result1)
            else:
                result1 = (x1 * r1[:, None]) * a3[None, :] + b3[None, :]
                tl.store(out_ptr + row1 * H + cols[None, :], result1)


@triton.jit
def _rms_wide2_v2_overlap_v3_packed_xprefetch(
    out_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    rows,
    nb,
    chunk,
    H: tl.constexpr,
    eps: tl.constexpr,
    NUM_CORES: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr,
    SHIFT_ROW_STRIDE: tl.constexpr,
    SCALE_ROW_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    cols = tl.arange(0, H).to(tl.int32)
    r_off = tl.arange(0, 2).to(tl.int32)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    lo = pid * chunk
    hi = tl.minimum(lo + chunk, nb)
    for b in range(lo, hi):
        b0 = 2 * b
        b1 = b0 + 1
        i0 = tl.load(indices_ptr + b0 * INDEX_STRIDE).to(tl.int32)
        i1 = tl.load(indices_ptr + b1 * INDEX_STRIDE).to(tl.int32)
        xb = tl.load(x_ptr + (2 * b + r_off[:, None]) * X_ROW_STRIDE + cols[None, :])
        sb0 = tl.load(shift_ptr + i0 * SHIFT_ROW_STRIDE + cols)
        cb0 = tl.load(scale_ptr + i0 * SCALE_ROW_STRIDE + cols)
        sb1 = tl.load(shift_ptr + i1 * SHIFT_ROW_STRIDE + cols)
        cb1 = tl.load(scale_ptr + i1 * SCALE_ROW_STRIDE + cols)
        x = xb.to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, axis=1) / H + eps)
        x0 = extract_slice(x, (0, 0), (1, H), (1, 1))
        r0 = extract_slice(inv, (0,), (1,), (1,))
        s0 = sb0.to(tl.float32)
        c0 = cb0.to(tl.float32)
        tl.store(
            out_ptr + b0 * H + cols[None, :], (x0 * r0[:, None]) * weight[None, :] * (1.0 + c0[None, :]) + s0[None, :]
        )
        x1 = extract_slice(x, (1, 0), (1, H), (1, 1))
        r1 = extract_slice(inv, (1,), (1,), (1,))
        s1 = sb1.to(tl.float32)
        c1 = cb1.to(tl.float32)
        tl.store(
            out_ptr + b1 * H + cols[None, :], (x1 * r1[:, None]) * weight[None, :] * (1.0 + c1[None, :]) + s1[None, :]
        )


@triton.jit
def _rms_generic_v2_overlap_v3_packed_xprefetch(
    out_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    rows,
    total_blocks,
    chunk,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_TILES: tl.constexpr,
    NUM_CORES: tl.constexpr,
    CHUNKED: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr,
    SHIFT_ROW_STRIDE: tl.constexpr,
    SCALE_ROW_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    cols = tl.arange(0, BLOCK_N).to(tl.int32)
    if CHUNKED:
        lo = pid * chunk
        hi = tl.minimum(lo + chunk, total_blocks)
    else:
        lo = pid
        hi = total_blocks
    step = 1 if CHUNKED else NUM_CORES

    if NUM_TILES == 1:
        wmask = cols < H
        weight = tl.load(weight_ptr + cols, mask=wmask, other=0.0).to(tl.float32)
        for b in range(lo, hi, step):
            xb = tl.load(x_ptr + b * X_ROW_STRIDE + cols, mask=wmask, other=0.0)
            index = tl.load(indices_ptr + b * INDEX_STRIDE).to(tl.int32)
            sb = tl.load(shift_ptr + index * SHIFT_ROW_STRIDE + cols, mask=wmask, other=0.0)
            cb = tl.load(scale_ptr + index * SCALE_ROW_STRIDE + cols, mask=wmask, other=0.0)
            x = xb.to(tl.float32)
            shift = sb.to(tl.float32)
            scale = cb.to(tl.float32)
            rstd = tl.rsqrt(tl.sum(x * x, axis=0) / H + eps)
            out = (x * rstd) * weight * (1.0 + scale) + shift
            tl.store(out_ptr + b * H + cols, out, mask=wmask)
    else:
        for b in range(lo, hi, step):
            index = tl.load(indices_ptr + b * INDEX_STRIDE).to(tl.int32)
            sumsq = tl.zeros([], dtype=tl.float32)
            for j in range(NUM_TILES):
                c = j * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
                x = tl.load(x_ptr + b * X_ROW_STRIDE + c, mask=c < H, other=0.0).to(tl.float32)
                sumsq += tl.sum(x * x, axis=0)
            rstd = tl.rsqrt(sumsq / H + eps)
            for j in range(NUM_TILES):
                c = j * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
                xmask = c < H
                x = tl.load(x_ptr + b * X_ROW_STRIDE + c, mask=xmask, other=0.0).to(tl.float32)
                weight = tl.load(weight_ptr + c, mask=xmask, other=0.0).to(tl.float32)
                shift = tl.load(shift_ptr + index * SHIFT_ROW_STRIDE + c, mask=xmask, other=0.0).to(tl.float32)
                scale = tl.load(scale_ptr + index * SCALE_ROW_STRIDE + c, mask=xmask, other=0.0).to(tl.float32)
                out = (x * rstd) * weight * (1.0 + scale) + shift
                tl.store(out_ptr + b * H + c, out, mask=xmask)


@triton.jit
def _rms_h16_simt_v2_overlap_v3_packed_xprefetch(
    out_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    C: tl.constexpr,
    BM: tl.constexpr,
    CORES: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr,
    SHIFT_ROW_STRIDE: tl.constexpr,
    SCALE_ROW_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, H)
    local_rows = tl.arange(0, BM)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    for block in range(pid, tl.cdiv(ROWS, BM), CORES):
        rows = block * BM + local_rows
        index = tl.load(indices_ptr + rows * INDEX_STRIDE).to(tl.int32)
        x = tl.load(x_ptr + rows[:, None] * X_ROW_STRIDE + cols[None, :]).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, axis=1) / H + EPS)
        normalized = (x * inv[:, None]) * weight[None, :]
        shift = tl.load(shift_ptr + index[:, None] * SHIFT_ROW_STRIDE + cols[None, :]).to(tl.float32)
        scale = tl.load(scale_ptr + index[:, None] * SCALE_ROW_STRIDE + cols[None, :]).to(tl.float32)
        output = normalized * (1.0 + scale) + shift
        tl.store(out_ptr + rows[:, None] * H + cols[None, :], output)


@triton.jit
def _rms_wide2_staged_v2_overlap_v3_packed_xprefetch(
    out_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    rows,
    nb,
    chunk,
    H: tl.constexpr,
    eps: tl.constexpr,
    NUM_CORES: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr,
    SHIFT_ROW_STRIDE: tl.constexpr,
    SCALE_ROW_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    cols = tl.arange(0, H).to(tl.int32)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    lo = pid * chunk
    hi = tl.minimum(lo + chunk, nb)
    pair_cols = tl.arange(0, 2 * H).to(tl.int32)
    if lo < hi:
        xb = tl.load(x_ptr + 2 * lo * X_ROW_STRIDE + pair_cols)
        for b in range(lo, hi):
            b0 = 2 * b
            b1 = b0 + 1
            i0 = tl.load(indices_ptr + b0 * INDEX_STRIDE).to(tl.int32)
            i1 = tl.load(indices_ptr + b1 * INDEX_STRIDE).to(tl.int32)
            sb0 = tl.load(shift_ptr + i0 * SHIFT_ROW_STRIDE + cols)
            cb0 = tl.load(scale_ptr + i0 * SCALE_ROW_STRIDE + cols)
            sb1 = tl.load(shift_ptr + i1 * SHIFT_ROW_STRIDE + cols)
            cb1 = tl.load(scale_ptr + i1 * SCALE_ROW_STRIDE + cols)
            nextx = xb
            if b + 1 < hi:
                nextx = tl.load(x_ptr + (b0 + 2) * X_ROW_STRIDE + pair_cols)
            x = tl.reshape(xb, (2, H)).to(tl.float32)
            inv = tl.rsqrt(tl.sum(x * x, axis=1) / H + eps)
            x0 = extract_slice(x, (0, 0), (1, H), (1, 1))
            r0 = extract_slice(inv, (0,), (1,), (1,))
            s0 = sb0.to(tl.float32)
            c0 = cb0.to(tl.float32)
            tl.store(
                out_ptr + b0 * H + cols[None, :],
                (x0 * r0[:, None]) * weight[None, :] * (1.0 + c0[None, :]) + s0[None, :],
            )
            x1 = extract_slice(x, (1, 0), (1, H), (1, 1))
            r1 = extract_slice(inv, (1,), (1,), (1,))
            s1 = sb1.to(tl.float32)
            c1 = cb1.to(tl.float32)
            tl.store(
                out_ptr + b1 * H + cols[None, :],
                (x1 * r1[:, None]) * weight[None, :] * (1.0 + c1[None, :]) + s1[None, :],
            )
            xb = nextx


@triton.jit
def _rms_large_board_best_20261114(
    out_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    CORES: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr,
    SHIFT_ROW_STRIDE: tl.constexpr,
    SCALE_ROW_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    cols = tl.arange(0, H).to(tl.int32)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    for row in range(pid, ROWS, CORES):
        xb = tl.load(x_ptr + row * X_ROW_STRIDE + cols)
        index = tl.load(indices_ptr + row * INDEX_STRIDE).to(tl.int32)
        cb = tl.load(scale_ptr + index * SCALE_ROW_STRIDE + cols)
        sb = tl.load(shift_ptr + index * SHIFT_ROW_STRIDE + cols)
        x = xb.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / H + EPS)
        value = x * rstd
        value = value * weight
        scale = cb.to(tl.float32)
        value = value * (1.0 + scale)
        shift = sb.to(tl.float32)
        tl.store(out_ptr + row * H + cols, value + shift)


@triton.jit
def _rms_wide2_prefetch_overlap_v3_packed_xprefetch(
    out_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    rows,
    nb,
    chunk,
    H: tl.constexpr,
    eps: tl.constexpr,
    NUM_CORES: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr,
    SHIFT_ROW_STRIDE: tl.constexpr,
    SCALE_ROW_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    cols = tl.arange(0, H).to(tl.int32)
    r_off = tl.arange(0, 2).to(tl.int32)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    lo = pid * chunk
    hi = tl.minimum(lo + chunk, nb)
    if lo < hi:
        xb = tl.load(x_ptr + (2 * lo + r_off[:, None]) * X_ROW_STRIDE + cols[None, :])
        for b in range(lo, hi):
            b0 = 2 * b
            b1 = b0 + 1
            i0 = tl.load(indices_ptr + b0 * INDEX_STRIDE).to(tl.int32)
            i1 = tl.load(indices_ptr + b1 * INDEX_STRIDE).to(tl.int32)
            sb0 = tl.load(shift_ptr + i0 * SHIFT_ROW_STRIDE + cols)
            cb0 = tl.load(scale_ptr + i0 * SCALE_ROW_STRIDE + cols)
            sb1 = tl.load(shift_ptr + i1 * SHIFT_ROW_STRIDE + cols)
            cb1 = tl.load(scale_ptr + i1 * SCALE_ROW_STRIDE + cols)
            nextx = xb
            if b + 1 < hi:
                nextx = tl.load(x_ptr + (b0 + 2 + r_off[:, None]) * X_ROW_STRIDE + cols[None, :])
            x = xb.to(tl.float32)
            inv = tl.rsqrt(tl.sum(x * x, axis=1) / H + eps)
            x0 = extract_slice(x, (0, 0), (1, H), (1, 1))
            r0 = extract_slice(inv, (0,), (1,), (1,))
            s0 = sb0.to(tl.float32)
            c0 = cb0.to(tl.float32)
            tl.store(
                out_ptr + b0 * H + cols[None, :],
                (x0 * r0[:, None]) * weight[None, :] * (1.0 + c0[None, :]) + s0[None, :],
            )
            x1 = extract_slice(x, (1, 0), (1, H), (1, 1))
            r1 = extract_slice(inv, (1,), (1,), (1,))
            s1 = sb1.to(tl.float32)
            c1 = cb1.to(tl.float32)
            tl.store(
                out_ptr + b1 * H + cols[None, :],
                (x1 * r1[:, None]) * weight[None, :] * (1.0 + c1[None, :]) + s1[None, :],
            )
            xb = nextx


def _launch(x, weight, shift, scale, indices, eps, core_count):
    rows, hidden = x.shape
    strides = dict(
        X_ROW_STRIDE=x.stride(0),
        SHIFT_ROW_STRIDE=shift.stride(0),
        SCALE_ROW_STRIDE=scale.stride(0),
        INDEX_STRIDE=indices.stride(0),
    )
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    # Two-row packed input loads require adjacent rows. Read a padded or
    # broadcast x view directly with the generic kernel instead of copying.
    if x.stride(0) != hidden:
        block_n = hidden if hidden <= 8192 else 8192
        num_tiles = (hidden + block_n - 1) // block_n
        cores = min(core_count, rows)
        _rms_generic_v2_overlap_v3_packed_xprefetch[(cores,)](
            out,
            x,
            weight,
            shift,
            scale,
            indices,
            rows,
            rows,
            1,
            H=hidden,
            eps=eps,
            BLOCK_N=block_n,
            NUM_TILES=num_tiles,
            NUM_CORES=cores,
            CHUNKED=False,
            **strides,
        )
        return out
    if hidden == 16 and rows >= core_count * 1024 and rows % 256 == 0:
        cores = core_count
        _rms_h16_simt_v2_overlap_v3_packed_xprefetch[(cores,)](
            out,
            x,
            weight,
            shift,
            scale,
            indices,
            ROWS=rows,
            H=hidden,
            EPS=eps,
            C=scale.shape[0],
            BM=256,
            CORES=cores,
            num_warps=32,
            compile_mode="simt_only",
            **strides,
        )
        return out

    if hidden == 16384:
        cores = min(core_count, rows)
        _rms_large_board_best_20261114[(cores,)](
            out,
            x,
            weight,
            shift,
            scale,
            indices,
            ROWS=rows,
            H=hidden,
            EPS=eps,
            CORES=cores,
            multibuffer=False,
            **strides,
        )
        return out

    if hidden == 5376 and rows >= 4096 and rows % 2 == 0:
        nb = rows // 2
        cores = min(core_count, nb)
        chunk = (nb + cores - 1) // cores
        _rms_wide2_staged_v2_overlap_v3_packed_xprefetch[(cores,)](
            out,
            x,
            weight,
            shift,
            scale,
            indices,
            rows,
            nb,
            chunk,
            H=hidden,
            eps=eps,
            NUM_CORES=cores,
            **strides,
        )
        return out

    cond = scale.shape[0]
    is_pow2 = hidden & (hidden - 1) == 0

    # narrow hidden: whole C x H table selected per row (2-D row batching)
    if is_pow2 and hidden <= 64 and cond <= 16:
        bm = 256 if rows % 256 == 0 else 128
        total_blocks = (rows + bm - 1) // bm
        num_cores = min(core_count, total_blocks)
        _rms_small_table_v2_overlap_v3_packed_xprefetch[(num_cores,)](
            out,
            x,
            weight,
            shift,
            scale,
            indices,
            rows,
            total_blocks,
            H=hidden,
            eps=eps,
            C=cond,
            BLOCK_M=bm,
            NUM_CORES=num_cores,
            **strides,
        )
        return out

    # C4 resident-table path: branch-local consumption avoids table vsel.
    if hidden == 4096 and cond == 4 and rows % 2 == 0:
        num_cores = min(core_count, rows // 2)
        _rms_medium_table_v2_overlap_v3_packed_xprefetch[(num_cores,)](
            out,
            x,
            weight,
            shift,
            scale,
            indices,
            rows,
            H=hidden,
            eps=eps,
            C=cond,
            BLOCK_M=2,
            NUM_CORES=num_cores,
            multibuffer=False,
            **strides,
        )
        return out

    # medium pow2 hidden, few conditions: hoisted table + broadcast select
    if is_pow2 and 64 < hidden <= 2048 and cond in (1, 2, 4) and rows % 2 == 0:
        num_cores = min(core_count, rows // 2)
        _rms_medium_table_v2_overlap_v3_packed_xprefetch[(num_cores,)](
            out,
            x,
            weight,
            shift,
            scale,
            indices,
            rows,
            H=hidden,
            eps=eps,
            C=cond,
            BLOCK_M=2,
            NUM_CORES=num_cores,
            **strides,
        )
        return out

    if hidden <= 8192:
        block_n, num_tiles = hidden, 1
    else:
        block_n, num_tiles = 8192, (hidden + 8191) // 8192

    # wide rows with enough work per core: two consecutive rows per
    # iteration (contiguous DMA, loads of both rows issued up front).
    if num_tiles == 1 and hidden >= 3072 and rows >= 4096 and rows % 2 == 0:
        nb = rows // 2
        num_cores = min(core_count, nb)
        chunk = (nb + num_cores - 1) // num_cores
        if hidden == 3072 and cond == 4:
            _rms_wide2_prefetch_overlap_v3_packed_xprefetch[(num_cores,)](
                out,
                x,
                weight,
                shift,
                scale,
                indices,
                rows,
                nb,
                chunk,
                H=hidden,
                eps=eps,
                NUM_CORES=num_cores,
                **strides,
            )
        else:
            _rms_wide2_v2_overlap_v3_packed_xprefetch[(num_cores,)](
                out,
                x,
                weight,
                shift,
                scale,
                indices,
                rows,
                nb,
                chunk,
                H=hidden,
                eps=eps,
                NUM_CORES=num_cores,
                **strides,
            )
        return out

    total_blocks = rows
    num_cores = min(core_count, total_blocks)
    chunked = rows >= 8192 or (rows >= 4096 and hidden <= 3072)
    chunk = (total_blocks + num_cores - 1) // num_cores
    _rms_generic_v2_overlap_v3_packed_xprefetch[(num_cores,)](
        out,
        x,
        weight,
        shift,
        scale,
        indices,
        rows,
        total_blocks,
        chunk,
        H=hidden,
        eps=eps,
        BLOCK_N=block_n,
        NUM_TILES=num_tiles,
        NUM_CORES=num_cores,
        CHUNKED=chunked,
        **strides,
    )
    return out


@lru_cache(maxsize=None)
def _num_vector_cores(device_index):
    return int(triton.runtime.driver.active.utils.get_device_properties(device_index)["num_vectorcore"])


def _validate_layout(x, weight, shift, scale):
    if x.stride(1) != 1 or weight.stride(0) != 1 or shift.stride(1) != 1 or scale.stride(1) != 1:
        raise ValueError("A5 requires unit stride in the feature dimension of x, weight, shift, and scale")


@torch.library.custom_op("mojo_npu_triton_a5::indexed_rms_norm_scale_shift_infer", mutates_args=())
def indexed_rms_norm_scale_shift_infer_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    _validate_layout(x, weight, shift, scale)
    if x.shape[0] == 0:
        return torch.empty_like(x, memory_format=torch.contiguous_format)
    with torch.npu.device(x.device):
        return _launch(x, weight, shift, scale, indices, eps, _num_vector_cores(x.device.index))


@indexed_rms_norm_scale_shift_infer_fwd.register_fake
def _indexed_rms_norm_scale_shift_fake(x, weight, shift, scale, indices, eps):
    _validate_layout(x, weight, shift, scale)
    return torch.empty_like(x, memory_format=torch.contiguous_format)
