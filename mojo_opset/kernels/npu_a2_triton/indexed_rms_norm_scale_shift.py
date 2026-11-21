# SPDX-License-Identifier: Apache-2.0
"""A2 RMSNorm indexed affine: direct input/weight/table/index element strides; no layout copies."""

from functools import lru_cache

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _rms_norm_indexed_scale_shift_kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    rows,
    total_blocks,
    num_cores,
    H: tl.constexpr,
    C: tl.constexpr,
    TABLE_N: tl.constexpr,
    FULL_ROWS: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_TILES: tl.constexpr,
    SX: tl.constexpr,
    XC: tl.constexpr,
    SW: tl.constexpr,
    SS: tl.constexpr,
    CS: tl.constexpr,
    SC: tl.constexpr,
    CC: tl.constexpr,
    SI: tl.constexpr,
    USE_PARAM_CACHE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    cols = tl.arange(0, BLOCK_N).to(tl.int32)
    if BLOCK_M == 1:
        if NUM_TILES == 1:
            mask = cols < H
            weight = tl.load(weight_ptr + cols * SW, mask=mask, other=0.0).to(tl.float32)
            if USE_PARAM_CACHE:
                cached_index = tl.full((), -1, tl.int64)
                cached_shift = tl.zeros((BLOCK_N,), tl.float32)
                cached_gain = tl.zeros((BLOCK_N,), tl.float32)
            for b in range(pid, total_blocks, num_cores):
                x = tl.load(x_ptr + b * SX + cols * XC, mask=mask, other=0.0).to(tl.float32)
                index = tl.load(indices_ptr + b * SI)
                if USE_PARAM_CACHE:
                    if index != cached_index:
                        cached_shift = tl.load(shift_ptr + index * SS + cols * CS, mask=mask, other=0.0).to(tl.float32)
                        cached_gain = 1.0 + tl.load(scale_ptr + index * SC + cols * CC, mask=mask, other=0.0).to(
                            tl.float32
                        )
                        cached_index = index
                    shift = cached_shift
                else:
                    shift = tl.load(shift_ptr + index * SS + cols * CS, mask=mask, other=0.0).to(tl.float32)
                    scale = tl.load(scale_ptr + index * SC + cols * CC, mask=mask, other=0.0).to(tl.float32)
                if H >= 8192 and H & H - 1 == 0:
                    rstd = tl.rsqrt(tl.sum((x * x)[None, :], axis=1) / H + eps)
                else:
                    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / H + eps)
                if USE_PARAM_CACHE:
                    out = x * (rstd * weight) * cached_gain + shift
                else:
                    out = x * (rstd * weight) * (1.0 + scale) + shift
                tl.store(out_ptr + b * H + cols, out, mask=mask)
        else:
            for b in range(pid, total_blocks, num_cores):
                index = tl.load(indices_ptr + b * SI)
                sumsq = tl.zeros([], dtype=tl.float32)
                for j in tl.static_range(NUM_TILES):
                    c = j * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
                    x = tl.load(x_ptr + b * SX + c * XC, mask=c < H, other=0.0).to(tl.float32)
                    sumsq += tl.sum(x * x, axis=0)
                rstd = tl.rsqrt(sumsq / H + eps)
                for j in tl.static_range(NUM_TILES):
                    c = j * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
                    mask = c < H
                    x = tl.load(x_ptr + b * SX + c * XC, mask=mask, other=0.0).to(tl.float32)
                    weight = tl.load(weight_ptr + c * SW, mask=mask, other=0.0).to(tl.float32)
                    shift = tl.load(shift_ptr + index * SS + c * CS, mask=mask, other=0.0).to(tl.float32)
                    scale = tl.load(scale_ptr + index * SC + c * CC, mask=mask, other=0.0).to(tl.float32)
                    out = x * (rstd * weight) * (1.0 + scale) + shift
                    tl.store(out_ptr + b * H + c, out, mask=mask)
    else:
        if (H <= 128 or (H <= 512 and C <= 4)) and BLOCK_N == H:
            table_cols = tl.arange(0, TABLE_N)
            if SS == H and CS == 1:
                shift_table = tl.load(shift_ptr + table_cols, table_cols < C * H, 0)
            else:
                shift_table = tl.load(shift_ptr + (table_cols // H * SS + table_cols % H * CS), table_cols < C * H, 0)
            if SC == H and CC == 1:
                scale_table = tl.load(scale_ptr + table_cols, table_cols < C * H, 0)
            else:
                scale_table = tl.load(scale_ptr + (table_cols // H * SC + table_cols % H * CC), table_cols < C * H, 0)
        for b in range(pid, total_blocks, num_cores):
            r = (b * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int32)
            mask_m = tl.full((BLOCK_M,), True, tl.int1) if FULL_ROWS else r < rows
            mask = mask_m[:, None] & (cols[None, :] < H)
            index = tl.load(indices_ptr + r * SI, mask=mask_m, other=0).to(tl.int32)
            x = tl.load(x_ptr + r[:, None] * SX + cols[None, :] * XC, mask=mask, other=0.0).to(tl.float32)
            weight = tl.load(weight_ptr + cols * SW, mask=cols < H, other=0.0).to(tl.float32)
            if (H <= 128 or (H <= 512 and C <= 4)) and BLOCK_N == H:
                table_index = tl.reshape(index[:, None] * H + cols[None, :], (BLOCK_M * BLOCK_N,))
                shift = tl.reshape(tl.gather(shift_table, table_index, 0), (BLOCK_M, BLOCK_N)).to(tl.float32)
                scale = tl.reshape(tl.gather(scale_table, table_index, 0), (BLOCK_M, BLOCK_N)).to(tl.float32)
            else:
                shift = tl.load(shift_ptr + index[:, None] * SS + cols[None, :] * CS, mask=mask, other=0.0).to(
                    tl.float32
                )
                scale = tl.load(scale_ptr + index[:, None] * SC + cols[None, :] * CC, mask=mask, other=0.0).to(
                    tl.float32
                )
            rstd = tl.rsqrt(tl.sum(x * x, axis=1) / H + eps)
            out = x * (rstd[:, None] * weight[None, :]) * (1.0 + scale) + shift
            tl.store(out_ptr + r[:, None] * H + cols[None, :], out, mask=mask)


def _pick_tile(hidden):
    """Return (BLOCK_M, BLOCK_N, NUM_TILES).

    Wide hidden uses a 128-aligned row width instead of next_pow2 to avoid
    wasting vector lanes on masked columns (e.g. 5376 -> 5376, not 8192).
    """
    if hidden <= 128:
        block_n = triton.next_power_of_2(hidden)
        block_m = max(1, 4096 // block_n)
        return (block_m, block_n, 1)
    if hidden <= 8192:
        block_n = (hidden + 127) // 128 * 128
        block_m = max(1, 4096 // block_n)
        return (block_m, block_n, 1)
    block_n = 8192
    return (1, block_n, (hidden + 8192 - 1) // 8192)


@triton.jit
def _a2_table_select(
    out_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    rows,
    blocks,
    cores,
    H: tl.constexpr,
    C: tl.constexpr,
    eps: tl.constexpr,
    M: tl.constexpr,
    FULL_ROWS: tl.constexpr,
    SX: tl.constexpr,
    XC: tl.constexpr,
    SW: tl.constexpr,
    SS: tl.constexpr,
    CS: tl.constexpr,
    SC: tl.constexpr,
    CC: tl.constexpr,
    SI: tl.constexpr,
):
    col = tl.arange(0, H)
    rowoff = tl.arange(0, M)
    w = tl.load(weight_ptr + col * SW).to(tl.float32)
    s0 = tl.load(scale_ptr + col * CC).to(tl.float32)
    b0 = tl.load(shift_ptr + col * CS).to(tl.float32)
    if C >= 2:
        s1 = tl.load(scale_ptr + SC + col * CC).to(tl.float32)
        b1 = tl.load(shift_ptr + SS + col * CS).to(tl.float32)
    if C == 4:
        s2 = tl.load(scale_ptr + 2 * SC + col * CC).to(tl.float32)
        b2 = tl.load(shift_ptr + 2 * SS + col * CS).to(tl.float32)
        s3 = tl.load(scale_ptr + 3 * SC + col * CC).to(tl.float32)
        b3 = tl.load(shift_ptr + 3 * SS + col * CS).to(tl.float32)
    for block in range(tl.program_id(0), blocks, cores):
        row = block * M + rowoff
        x = tl.load(x_ptr + row[:, None] * SX + col[None, :] * XC, FULL_ROWS | (row[:, None] < rows), 0).to(tl.float32)
        idx = tl.load(indices_ptr + row * SI, FULL_ROWS | (row < rows), 0).to(tl.int32)
        rstd = tl.rsqrt(tl.sum(x * x, 1) / H + eps)
        if C == 1:
            scale = s0[None, :]
            shift = b0[None, :]
        elif C == 2:
            scale = tl.where(idx[:, None] == 0, s0[None, :], s1[None, :])
            shift = tl.where(idx[:, None] == 0, b0[None, :], b1[None, :])
        else:
            scale = tl.where(
                idx[:, None] == 0,
                s0[None, :],
                tl.where(idx[:, None] == 1, s1[None, :], tl.where(idx[:, None] == 2, s2[None, :], s3[None, :])),
            )
            shift = tl.where(
                idx[:, None] == 0,
                b0[None, :],
                tl.where(idx[:, None] == 1, b1[None, :], tl.where(idx[:, None] == 2, b2[None, :], b3[None, :])),
            )
        out = x * (rstd[:, None] * w[None, :]) * (1.0 + scale) + shift
        tl.store(out_ptr + row[:, None] * H + col[None, :], out, FULL_ROWS | (row[:, None] < rows))


@triton.jit
def _a2_pair_rows(
    out_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    pairs,
    cores,
    H: tl.constexpr,
    eps: tl.constexpr,
    SX: tl.constexpr,
    XC: tl.constexpr,
    SW: tl.constexpr,
    SS: tl.constexpr,
    CS: tl.constexpr,
    SC: tl.constexpr,
    CC: tl.constexpr,
    SI: tl.constexpr,
):
    col = tl.arange(0, H)
    w = tl.load(weight_ptr + col * SW).to(tl.float32)
    for pair in range(tl.program_id(0), pairs, cores):
        row0 = 2 * pair
        row1 = row0 + 1
        x0 = tl.load(x_ptr + row0 * SX + col * XC).to(tl.float32)
        x1 = tl.load(x_ptr + row1 * SX + col * XC).to(tl.float32)
        i0 = tl.load(indices_ptr + row0 * SI)
        i1 = tl.load(indices_ptr + row1 * SI)
        s0 = tl.load(shift_ptr + i0 * SS + col * CS).to(tl.float32)
        c0 = tl.load(scale_ptr + i0 * SC + col * CC).to(tl.float32)
        s1 = tl.load(shift_ptr + i1 * SS + col * CS).to(tl.float32)
        c1 = tl.load(scale_ptr + i1 * SC + col * CC).to(tl.float32)
        r0 = tl.rsqrt(tl.sum(x0 * x0, 0) / H + eps)
        r1 = tl.rsqrt(tl.sum(x1 * x1, 0) / H + eps)
        tl.store(out_ptr + row0 * H + col, x0 * (r0 * w) * (1.0 + c0) + s0)
        tl.store(out_ptr + row1 * H + col, x1 * (r1 * w) * (1.0 + c1) + s1)


@triton.jit
def _a2_full_row(
    out_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    rows,
    cores,
    H: tl.constexpr,
    eps: tl.constexpr,
    SX: tl.constexpr,
    XC: tl.constexpr,
    SW: tl.constexpr,
    SS: tl.constexpr,
    CS: tl.constexpr,
    SC: tl.constexpr,
    CC: tl.constexpr,
    SI: tl.constexpr,
):
    col = tl.arange(0, H)
    for row in range(tl.program_id(0), rows, cores):
        x = tl.load(x_ptr + row * SX + col * XC).to(tl.float32)
        idx = tl.load(indices_ptr + row * SI)
        rstd = tl.rsqrt(tl.sum((x * x)[None, :], 1) / H + eps)
        w = tl.load(weight_ptr + col * SW).to(tl.float32)
        u = x * (rstd * w)
        scale = tl.load(scale_ptr + idx * SC + col * CC).to(tl.float32)
        u = u * (1.0 + scale)
        shift = tl.load(shift_ptr + idx * SS + col * CS).to(tl.float32)
        tl.store(out_ptr + row * H + col, u + shift)


@triton.jit
def _a2_table_select_native(
    out_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    rows,
    blocks,
    cores,
    H: tl.constexpr,
    C: tl.constexpr,
    eps: tl.constexpr,
    M: tl.constexpr,
    SX: tl.constexpr,
    XC: tl.constexpr,
    SW: tl.constexpr,
    SS: tl.constexpr,
    CS: tl.constexpr,
    SC: tl.constexpr,
    CC: tl.constexpr,
    SI: tl.constexpr,
):
    col = tl.arange(0, H)
    rowoff = tl.arange(0, M)
    w = tl.load(weight_ptr + col * SW).to(tl.float32)
    s0 = tl.load(scale_ptr + col * CC)
    b0 = tl.load(shift_ptr + col * CS)
    if C >= 2:
        s1 = tl.load(scale_ptr + SC + col * CC)
        b1 = tl.load(shift_ptr + SS + col * CS)
    if C == 4:
        s2 = tl.load(scale_ptr + 2 * SC + col * CC)
        b2 = tl.load(shift_ptr + 2 * SS + col * CS)
        s3 = tl.load(scale_ptr + 3 * SC + col * CC)
        b3 = tl.load(shift_ptr + 3 * SS + col * CS)
    for block in range(tl.program_id(0), blocks, cores):
        row = block * M + rowoff
        x = tl.load(x_ptr + row[:, None] * SX + col[None, :] * XC, row[:, None] < rows, 0).to(tl.float32)
        idx = tl.load(indices_ptr + row * SI, row < rows, 0).to(tl.int32)
        rstd = tl.rsqrt(tl.sum(x * x, 1) / H + eps)
        if C == 1:
            scale = s0[None, :]
            shift = b0[None, :]
        elif C == 2:
            scale = tl.where(idx[:, None] == 0, s0[None, :], s1[None, :])
            shift = tl.where(idx[:, None] == 0, b0[None, :], b1[None, :])
        else:
            scale = tl.where(
                idx[:, None] == 0,
                s0[None, :],
                tl.where(idx[:, None] == 1, s1[None, :], tl.where(idx[:, None] == 2, s2[None, :], s3[None, :])),
            )
            shift = tl.where(
                idx[:, None] == 0,
                b0[None, :],
                tl.where(idx[:, None] == 1, b1[None, :], tl.where(idx[:, None] == 2, b2[None, :], b3[None, :])),
            )
        out = x * (rstd[:, None] * w[None, :]) * (1.0 + scale.to(tl.float32)) + shift.to(tl.float32)
        tl.store(out_ptr + row[:, None] * H + col[None, :], out, row[:, None] < rows)


def _launch(x, weight, shift, scale, indices, eps, num_cores):
    (rows, hidden) = x.shape
    out = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)
    (block_m, block_n, num_tiles) = _pick_tile(hidden)
    total_blocks = (rows + block_m - 1) // block_m
    grid = (min(total_blocks, num_cores),)
    if hidden == 512 and shift.shape[0] in (1, 2, 4):
        select_m = block_m
        select_blocks = (rows + select_m - 1) // select_m
        select_grid = (min(select_blocks, num_cores),)
        _a2_table_select_native[select_grid](
            out,
            x,
            weight,
            shift,
            scale,
            indices,
            rows,
            select_blocks,
            select_grid[0],
            hidden,
            shift.shape[0],
            eps,
            select_m,
            SX=x.stride(0),
            XC=x.stride(1),
            SW=weight.stride(0),
            SS=shift.stride(0),
            CS=shift.stride(1),
            SC=scale.stride(0),
            CC=scale.stride(1),
            SI=indices.stride(0),
        )
        return out
    if 8192 < hidden <= 16384 and hidden & hidden - 1 == 0:
        full_grid = (min(rows, num_cores),)
        _a2_full_row[full_grid](
            out,
            x,
            weight,
            shift,
            scale,
            indices,
            rows,
            full_grid[0],
            hidden,
            eps,
            SX=x.stride(0),
            XC=x.stride(1),
            SW=weight.stride(0),
            SS=shift.stride(0),
            CS=shift.stride(1),
            SC=scale.stride(0),
            CC=scale.stride(1),
            SI=indices.stride(0),
        )
        return out
    if 2048 < hidden <= 3072 and rows >= 128 and (rows % 2 == 0):
        pairs = rows // 2
        pair_grid = (min(pairs, num_cores),)
        _a2_pair_rows[pair_grid](
            out,
            x,
            weight,
            shift,
            scale,
            indices,
            pairs,
            pair_grid[0],
            hidden,
            eps,
            SX=x.stride(0),
            XC=x.stride(1),
            SW=weight.stride(0),
            SS=shift.stride(0),
            CS=shift.stride(1),
            SC=scale.stride(0),
            CC=scale.stride(1),
            SI=indices.stride(0),
        )
        return out
    if 1024 < hidden <= 2048 and hidden & hidden - 1 == 0 and (shift.shape[0] in (1, 2, 4)):
        select_m = 2
        select_blocks = (rows + select_m - 1) // select_m
        select_grid = (min(select_blocks, num_cores),)
        _a2_table_select[select_grid](
            out,
            x,
            weight,
            shift,
            scale,
            indices,
            rows,
            select_blocks,
            select_grid[0],
            hidden,
            shift.shape[0],
            eps,
            select_m,
            rows % select_m == 0,
            SX=x.stride(0),
            XC=x.stride(1),
            SW=weight.stride(0),
            SS=shift.stride(0),
            CS=shift.stride(1),
            SC=scale.stride(0),
            CC=scale.stride(1),
            SI=indices.stride(0),
        )
        return out
    _rms_norm_indexed_scale_shift_kernel[grid](
        out,
        x,
        weight,
        shift,
        scale,
        indices,
        rows,
        total_blocks,
        grid[0],
        H=hidden,
        C=shift.shape[0],
        TABLE_N=triton.next_power_of_2(shift.shape[0] * hidden),
        FULL_ROWS=rows % block_m == 0,
        eps=eps,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NUM_TILES=num_tiles,
        SX=x.stride(0),
        XC=x.stride(1),
        SW=weight.stride(0),
        SS=shift.stride(0),
        CS=shift.stride(1),
        SC=scale.stride(0),
        CC=scale.stride(1),
        SI=indices.stride(0),
        USE_PARAM_CACHE=hidden == 5376 and (shift.stride(0) != hidden or scale.stride(0) != hidden),
    )
    return out


@lru_cache(maxsize=None)
def _num_vector_cores(device_index):
    return int(triton.runtime.driver.active.utils.get_device_properties(device_index)["num_vectorcore"])


@torch.library.custom_op("mojo_npu_triton_a2::indexed_rms_norm_scale_shift_infer", mutates_args=())
def indexed_rms_norm_scale_shift_infer_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if x.shape[0] == 0:
        return torch.empty_like(x, memory_format=torch.contiguous_format)
    with torch.npu.device(x.device):
        return _launch(x, weight, shift, scale, indices, eps, _num_vector_cores(x.device.index))


@indexed_rms_norm_scale_shift_infer_fwd.register_fake
def _indexed_rms_norm_scale_shift_fake(x, weight, shift, scale, indices, eps):
    return torch.empty_like(x, memory_format=torch.contiguous_format)
