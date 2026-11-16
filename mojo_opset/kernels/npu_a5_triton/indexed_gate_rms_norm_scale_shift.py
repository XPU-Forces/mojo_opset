# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend A5 fused indexed gating, BF16 residual, RMSNorm and affine.

Derived from vllm-project/vllm-omni at d613fdb62c432a342681e88b307d36b54f6ed5f0,
vllm_omni/diffusion/layers/indexed_modulation.py.
"""

import torch
import triton
import triton.language as tl

from triton.runtime import driver

_MAX_1D_GRID = 65535
_MAX_BLOCK_N = 8192


@triton.jit
def _indexed_gate_rms_shift_kernel(
    residual_out_ptr,
    modulated_out_ptr,
    residual_ptr,
    gate_ptr,
    branch_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    rows,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CORES: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
    EXACT: tl.constexpr,
    KEEP_I64: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    offs = tl.arange(0, BLOCK_N).to(tl.int32)

    if EXACT:
        # ---- mask-free main loop (BLOCK_N == HIDDEN) ----
        rvec = tl.arange(0, ROWS_PER_ITER).to(tl.int32)
        weight = tl.load(weight_ptr + offs).to(tl.float32)
        n_full = (rows // (NUM_CORES * ROWS_PER_ITER)) * (NUM_CORES * ROWS_PER_ITER)
        for row0 in range(pid * ROWS_PER_ITER, n_full, NUM_CORES * ROWS_PER_ITER):
            rows_v = row0 + rvec
            index = tl.load(indices_ptr + rows_v)
            if not KEEP_I64:
                index = index.to(tl.int32)
            base = rows_v[:, None] * HIDDEN + offs[None, :]
            residual = tl.load(residual_ptr + base).to(tl.float32)
            gate = tl.load(gate_ptr + index[:, None] * HIDDEN + offs[None, :]).to(tl.float32)
            branch = tl.load(branch_ptr + base).to(tl.float32)
            updated_bf = (residual + gate * branch).to(tl.bfloat16)
            tl.store(residual_out_ptr + base, updated_bf)
            updated = updated_bf.to(tl.float32)
            variance = tl.sum(updated * updated, axis=1) / HIDDEN
            normalized = updated * tl.rsqrt(variance[:, None] + EPS) * weight[None, :]
            shift = tl.load(shift_ptr + index[:, None] * HIDDEN + offs[None, :]).to(tl.float32)
            scale = tl.load(scale_ptr + index[:, None] * HIDDEN + offs[None, :]).to(tl.float32)
            output = normalized * (1.0 + scale) + shift
            tl.store(modulated_out_ptr + base, output.to(tl.bfloat16))
        # ---- row remainder ----
        for row in range(n_full + pid, rows, NUM_CORES):
            row_i = row.to(tl.int32)
            index = tl.load(indices_ptr + row_i)
            if not KEEP_I64:
                index = index.to(tl.int32)
            base = row_i * HIDDEN
            residual = tl.load(residual_ptr + base + offs).to(tl.float32)
            gate = tl.load(gate_ptr + index * HIDDEN + offs).to(tl.float32)
            branch = tl.load(branch_ptr + base + offs).to(tl.float32)
            updated = (residual + gate * branch).to(tl.bfloat16).to(tl.float32)
            tl.store(residual_out_ptr + base + offs, updated.to(tl.bfloat16))
            variance = tl.sum(updated * updated, axis=0) / HIDDEN
            normalized = updated * tl.rsqrt(variance + EPS) * weight
            shift = tl.load(shift_ptr + index * HIDDEN + offs).to(tl.float32)
            scale = tl.load(scale_ptr + index * HIDDEN + offs).to(tl.float32)
            output = normalized * (1.0 + scale) + shift
            tl.store(modulated_out_ptr + base + offs, output.to(tl.bfloat16))
    else:
        # ---- masked path (HIDDEN 不是 BLOCK_N，尾块必须显式排除) ----
        mask = offs < HIDDEN
        for row in range(pid, rows, NUM_CORES):
            row_i = row.to(tl.int32)
            index = tl.load(indices_ptr + row_i)
            if not KEEP_I64:
                index = index.to(tl.int32)
            base = row_i * HIDDEN
            residual = tl.load(residual_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            gate = tl.load(gate_ptr + index * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
            branch = tl.load(branch_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            updated_bf = (residual + gate * branch).to(tl.bfloat16)
            tl.store(residual_out_ptr + base + offs, updated_bf, mask=mask)
            updated = updated_bf.to(tl.float32)
            weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            # other=0.0 保证越界 lane 的 updated 必为 0，无需额外 tl.where
            variance = tl.sum(updated * updated, axis=0) / HIDDEN
            normalized = updated * tl.rsqrt(variance + EPS) * weight
            shift = tl.load(shift_ptr + index * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
            scale = tl.load(scale_ptr + index * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
            output = normalized * (1.0 + scale) + shift
            tl.store(modulated_out_ptr + base + offs, output.to(tl.bfloat16), mask=mask)


@triton.jit
def _indexed_gate_rms_shift_cached_kernel(
    residual_out_ptr,
    modulated_out_ptr,
    residual_ptr,
    gate_ptr,
    branch_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    rows,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CORES: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
    EXACT: tl.constexpr,
    NCOND: tl.constexpr,
    CP2: tl.constexpr,
    CACHE_GATE: tl.constexpr,
    CACHE_AFFINE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    offs = tl.arange(0, BLOCK_N).to(tl.int32)

    if EXACT:
        # ---- mask-free main loop (BLOCK_N == HIDDEN) ----
        rvec = tl.arange(0, ROWS_PER_ITER).to(tl.int32)
        weight = tl.load(weight_ptr + offs).to(tl.float32)
        table_offsets = tl.arange(0, CP2 * HIDDEN).to(tl.int32)
        table_mask = table_offsets < NCOND * HIDDEN
        if CACHE_GATE:
            gate_table = tl.load(gate_ptr + table_offsets, mask=table_mask, other=0.0)
        if CACHE_AFFINE:
            shift_table = tl.load(shift_ptr + table_offsets, mask=table_mask, other=0.0)
            scale_table = tl.load(scale_ptr + table_offsets, mask=table_mask, other=0.0)
        n_full = (rows // (NUM_CORES * ROWS_PER_ITER)) * (NUM_CORES * ROWS_PER_ITER)
        for row0 in range(pid * ROWS_PER_ITER, n_full, NUM_CORES * ROWS_PER_ITER):
            rows_v = row0 + rvec
            index = tl.load(indices_ptr + rows_v).to(tl.int32)
            base = rows_v[:, None] * HIDDEN + offs[None, :]
            residual = tl.load(residual_ptr + base).to(tl.float32)
            table_indices = tl.reshape(index[:, None] * HIDDEN + offs[None, :], [ROWS_PER_ITER * HIDDEN])
            if CACHE_GATE:
                gate = tl.reshape(tl.gather(gate_table, table_indices, 0), [ROWS_PER_ITER, HIDDEN]).to(tl.float32)
            else:
                gate = tl.load(gate_ptr + index[:, None] * HIDDEN + offs[None, :]).to(tl.float32)
            branch = tl.load(branch_ptr + base).to(tl.float32)
            updated_bf = (residual + gate * branch).to(tl.bfloat16)
            tl.store(residual_out_ptr + base, updated_bf)
            updated = updated_bf.to(tl.float32)
            variance = tl.sum(updated * updated, axis=1) / HIDDEN
            normalized = updated * tl.rsqrt(variance[:, None] + EPS) * weight[None, :]
            if CACHE_AFFINE:
                shift = tl.reshape(tl.gather(shift_table, table_indices, 0), [ROWS_PER_ITER, HIDDEN]).to(tl.float32)
                scale = tl.reshape(tl.gather(scale_table, table_indices, 0), [ROWS_PER_ITER, HIDDEN]).to(tl.float32)
            else:
                shift = tl.load(shift_ptr + index[:, None] * HIDDEN + offs[None, :]).to(tl.float32)
                scale = tl.load(scale_ptr + index[:, None] * HIDDEN + offs[None, :]).to(tl.float32)
            output = normalized * (1.0 + scale) + shift
            tl.store(modulated_out_ptr + base, output.to(tl.bfloat16))
        # ---- row remainder ----
        for row in range(n_full + pid, rows, NUM_CORES):
            row_i = row.to(tl.int32)
            index = tl.load(indices_ptr + row_i).to(tl.int32)
            base = row_i * HIDDEN
            residual = tl.load(residual_ptr + base + offs).to(tl.float32)
            gate = tl.load(gate_ptr + index * HIDDEN + offs).to(tl.float32)
            branch = tl.load(branch_ptr + base + offs).to(tl.float32)
            updated = (residual + gate * branch).to(tl.bfloat16).to(tl.float32)
            tl.store(residual_out_ptr + base + offs, updated.to(tl.bfloat16))
            variance = tl.sum(updated * updated, axis=0) / HIDDEN
            normalized = updated * tl.rsqrt(variance + EPS) * weight
            shift = tl.load(shift_ptr + index * HIDDEN + offs).to(tl.float32)
            scale = tl.load(scale_ptr + index * HIDDEN + offs).to(tl.float32)
            output = normalized * (1.0 + scale) + shift
            tl.store(modulated_out_ptr + base + offs, output.to(tl.bfloat16))
    else:
        # ---- masked path (HIDDEN 不是 BLOCK_N，尾块必须显式排除) ----
        mask = offs < HIDDEN
        for row in range(pid, rows, NUM_CORES):
            row_i = row.to(tl.int32)
            index = tl.load(indices_ptr + row_i).to(tl.int32)
            base = row_i * HIDDEN
            residual = tl.load(residual_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            gate = tl.load(gate_ptr + index * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
            branch = tl.load(branch_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            updated_bf = (residual + gate * branch).to(tl.bfloat16)
            tl.store(residual_out_ptr + base + offs, updated_bf, mask=mask)
            updated = updated_bf.to(tl.float32)
            weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            # other=0.0 保证越界 lane 的 updated 必为 0，无需额外 tl.where
            variance = tl.sum(updated * updated, axis=0) / HIDDEN
            normalized = updated * tl.rsqrt(variance + EPS) * weight
            shift = tl.load(shift_ptr + index * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
            scale = tl.load(scale_ptr + index * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
            output = normalized * (1.0 + scale) + shift
            tl.store(modulated_out_ptr + base + offs, output.to(tl.bfloat16), mask=mask)


_TWO_TILE_SPLIT = 4096
_SMALL_ROWS = 64


@triton.jit
def _indexed_gate_rms_shift_row_kernel(
    residual_out_ptr,
    modulated_out_ptr,
    residual_ptr,
    gate_ptr,
    branch_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SINGLE_CONDITION: tl.constexpr,
):
    """rows 很小时一行一个 program，用满所有 vector core。

    2-D tile 要求 ROWS_PER_ITER >= 2，rows=2/11 只能占 1/6 个核；
    逐行 grid=(rows,) 在 2/11/1 行上实测 3.3-3.5us（原 4.5us）。

    单行只有一遍流水，关键路径 = index 载入 -> 5 个数据载入 -> 归约 -> 写回，
    所以把 shift/scale/weight 全部提到最前面，与 residual/branch 并行发射，
    实测 3.55/3.39/3.26 -> 3.04/3.00/2.77us，且消除了原实现的运行间抖动。
    """
    if SINGLE_CONDITION:
        row_i = tl.program_id(0).to(tl.int32)
        offs = tl.arange(0, BLOCK_N).to(tl.int32)
        base = row_i * HIDDEN
        # For a one-row condition table every valid index is necessarily zero.
        # Specialize metadata, never inspect/cache tensor values on the host.
        index = tl.full((), 0, tl.int32)
        gbase = index * HIDDEN + offs
        residual = tl.load(residual_ptr + base + offs).to(tl.float32)
        gate = tl.load(gate_ptr + gbase).to(tl.float32)
        branch = tl.load(branch_ptr + base + offs).to(tl.float32)
        shift = tl.load(shift_ptr + gbase).to(tl.float32)
        scale = tl.load(scale_ptr + gbase).to(tl.float32)
        weight = tl.load(weight_ptr + offs).to(tl.float32)
        updated_bf = (residual + gate * branch).to(tl.bfloat16)
        updated = updated_bf.to(tl.float32)
        variance = tl.sum(updated * updated, axis=0) / HIDDEN
        normalized = updated * tl.rsqrt(variance + EPS) * weight
        output = normalized * (1.0 + scale) + shift
        tl.store(residual_out_ptr + base + offs, updated_bf)
        tl.store(modulated_out_ptr + base + offs, output.to(tl.bfloat16))
    else:
        row_i = tl.program_id(0).to(tl.int32)
        offs = tl.arange(0, BLOCK_N).to(tl.int32)
        base = row_i * HIDDEN
        residual = tl.load(residual_ptr + base + offs).to(tl.float32)
        branch = tl.load(branch_ptr + base + offs).to(tl.float32)
        weight = tl.load(weight_ptr + offs).to(tl.float32)
        # For a one-row condition table every valid index is necessarily zero.
        # Specialize metadata, never inspect/cache tensor values on the host.
        index = tl.load(indices_ptr + row_i).to(tl.int32)
        gbase = index * HIDDEN + offs
        gate = tl.load(gate_ptr + gbase).to(tl.float32)
        shift = tl.load(shift_ptr + gbase).to(tl.float32)
        scale = tl.load(scale_ptr + gbase).to(tl.float32)
        updated_bf = (residual + gate * branch).to(tl.bfloat16)
        updated = updated_bf.to(tl.float32)
        variance = tl.sum(updated * updated, axis=0) / HIDDEN
        normalized = updated * tl.rsqrt(variance + EPS) * weight
        output = normalized * (1.0 + scale) + shift
        tl.store(residual_out_ptr + base + offs, updated_bf)
        tl.store(modulated_out_ptr + base + offs, output.to(tl.bfloat16))


@triton.jit
def _indexed_gate_rms_shift_two_tile_kernel(
    residual_out_ptr,
    modulated_out_ptr,
    residual_ptr,
    gate_ptr,
    branch_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    rows,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    """H 在 (4096, 8192] 且非对齐时拆成 4096 + (H-4096) 两 tile。

    单 tile 会按 next_pow2(H) 展开（5375→8192，34% 无效 lane）；两 tile 把
    H-4096 直接作为第二块宽度（5375→4096+1279、7168→4096+3072），无效 lane
    降到 0，在 VEC bound 的 H=5375/7168 上实测比 pow2 取整再快 2.5%/3.7%。
    """
    pid = tl.program_id(0).to(tl.int32)
    offs_a = tl.arange(0, BLOCK_A).to(tl.int32)
    offs_b = tl.arange(0, BLOCK_B).to(tl.int32) + BLOCK_A
    mask_b = offs_b < HIDDEN
    wa = tl.load(weight_ptr + offs_a).to(tl.float32)
    wb = tl.load(weight_ptr + offs_b, mask=mask_b, other=0.0).to(tl.float32)
    for row in range(pid, rows, NUM_CORES):
        row_i = row.to(tl.int32)
        index = tl.load(indices_ptr + row_i).to(tl.int32)
        base = row_i * HIDDEN
        ia = index * HIDDEN + offs_a
        ib = index * HIDDEN + offs_b
        ra = tl.load(residual_ptr + base + offs_a).to(tl.float32)
        ga = tl.load(gate_ptr + ia).to(tl.float32)
        ba = tl.load(branch_ptr + base + offs_a).to(tl.float32)
        rb = tl.load(residual_ptr + base + offs_b, mask=mask_b, other=0.0).to(tl.float32)
        gb = tl.load(gate_ptr + ib, mask=mask_b, other=0.0).to(tl.float32)
        bb = tl.load(branch_ptr + base + offs_b, mask=mask_b, other=0.0).to(tl.float32)
        ua = (ra + ga * ba).to(tl.bfloat16)
        ub = (rb + gb * bb).to(tl.bfloat16)
        tl.store(residual_out_ptr + base + offs_a, ua)
        tl.store(residual_out_ptr + base + offs_b, ub, mask=mask_b)
        fa = ua.to(tl.float32)
        fb = ub.to(tl.float32)
        variance = (tl.sum(fa * fa, axis=0) + tl.sum(fb * fb, axis=0)) / HIDDEN
        rsqrt = tl.rsqrt(variance + EPS)
        sa = tl.load(shift_ptr + ia).to(tl.float32)
        ca = tl.load(scale_ptr + ia).to(tl.float32)
        sb = tl.load(shift_ptr + ib, mask=mask_b, other=0.0).to(tl.float32)
        cb = tl.load(scale_ptr + ib, mask=mask_b, other=0.0).to(tl.float32)
        oa = fa * rsqrt * wa
        ob = fb * rsqrt * wb
        tl.store(modulated_out_ptr + base + offs_a, (oa * (1.0 + ca) + sa).to(tl.bfloat16))
        tl.store(modulated_out_ptr + base + offs_b, (ob * (1.0 + cb) + sb).to(tl.bfloat16), mask=mask_b)


_TINY_HIDDEN = 32
_TABLE_MAX_ELEMS = 4096
_TABLE_ROWS = 128


@triton.jit
def _indexed_gate_rms_shift_table_kernel(
    residual_out_ptr,
    modulated_out_ptr,
    residual_ptr,
    gate_ptr,
    branch_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    rows,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    NUM_CORES: tl.constexpr,
    NCOND: tl.constexpr,
    CP2: tl.constexpr,
):
    """极窄行（HIDDEN <= 32，即单行 <= 64B）改用「条件表 + UB gather」。

    极窄行时 gate/shift/scale 的逐行 load 每行只有 HIDDEN*2 字节，小 DMA 次数
    正比于行数，DMA 发射率成为瓶颈：69888x16 实测只有 215GB/s（去掉这三个
    gather 后同 kernel 可达 888GB/s）。这里把整张条件表
    （NCOND*HIDDEN <= 4096 元素，L2 常驻）一次性搬进 UB，再用 tl.gather 选行，
    小 DMA 次数从 O(rows) 降到 O(1)，该 case 实测 215 -> 594GB/s（72.8 -> 26.4us）。

    交叉点：逐行 DMA 约 18 cycle/行（18/HIDDEN cycle/元素），UB gather 约
    0.2 cycle/元素，HIDDEN < 64 时 gather 更省；只在该机制实测过的 HIDDEN <= 32
    上启用，其余形状仍走逐行 load。
    """
    pid = tl.program_id(0).to(tl.int32)
    n: tl.constexpr = BLOCK_R * HIDDEN
    tn: tl.constexpr = CP2 * HIDDEN
    foffs = tl.arange(0, n).to(tl.int32)
    rvec = tl.arange(0, BLOCK_R).to(tl.int32)
    hvec = tl.arange(0, HIDDEN).to(tl.int32)
    tall = tl.arange(0, tn).to(tl.int32)
    tmask = tall < NCOND * HIDDEN
    # 表保持 bf16：tl.gather 的代价按元素计，bf16 比 fp32 少一半搬运，
    # 69888x16 实测 29.1us -> 26.4us。
    gate_tab = tl.load(gate_ptr + tall, mask=tmask, other=0.0)
    shift_tab = tl.load(shift_ptr + tall, mask=tmask, other=0.0)
    scale_tab = tl.load(scale_ptr + tall, mask=tmask, other=0.0)
    weight = tl.load(weight_ptr + hvec).to(tl.float32)
    limit = rows * HIDDEN
    for r0 in range(pid * BLOCK_R, rows, NUM_CORES * BLOCK_R):
        rows_v = r0 + rvec
        rmask = rows_v < rows
        index = tl.load(indices_ptr + rows_v, mask=rmask, other=0).to(tl.int32)
        gidx = tl.reshape(index[:, None] * HIDDEN + hvec[None, :], [n])
        fbase = r0 * HIDDEN + foffs
        fmask = fbase < limit
        residual = tl.load(residual_ptr + fbase, mask=fmask, other=0.0).to(tl.float32)
        branch = tl.load(branch_ptr + fbase, mask=fmask, other=0.0).to(tl.float32)
        gate = tl.gather(gate_tab, gidx, 0).to(tl.float32)
        updated_bf = (residual + gate * branch).to(tl.bfloat16)
        tl.store(residual_out_ptr + fbase, updated_bf, mask=fmask)
        updated = tl.reshape(updated_bf.to(tl.float32), [BLOCK_R, HIDDEN])
        variance = tl.sum(updated * updated, axis=1) / HIDDEN
        normalized = updated * tl.rsqrt(variance[:, None] + EPS) * weight[None, :]
        shift = tl.reshape(tl.gather(shift_tab, gidx, 0).to(tl.float32), [BLOCK_R, HIDDEN])
        scale = tl.reshape(tl.gather(scale_tab, gidx, 0).to(tl.float32), [BLOCK_R, HIDDEN])
        output = normalized * (1.0 + scale) + shift
        tl.store(modulated_out_ptr + fbase, tl.reshape(output, [n]).to(tl.bfloat16), mask=fmask)


def _plan(hidden):
    """Return (mode, block_n, rows_per_iter).

    mode 0: EXACT 2-D tile（BLOCK_N == HIDDEN）
    mode 1: 单 tile masked（BLOCK_N = next_pow2(HIDDEN)）
    mode 2: 两 tile（HIDDEN 在 (4096, 8192]，BLOCK_A=4096 + BLOCK_B=HIDDEN-4096）

    A5 实测（56 vector core，248KB UB）：BLOCK_N=3072/2048 时 RPI=2 明显优于 3/4，
    更大 RPI 只增加 UB 压力；H<=64 的窄行用 32 行/迭代补足访存并行度。
    """
    exact = hidden <= 5376 and hidden % 8 == 0
    if exact:
        if hidden <= 64:
            return 0, hidden, 32
        return 0, hidden, 2
    if _TWO_TILE_SPLIT < hidden <= _MAX_BLOCK_N:
        return 2, _TWO_TILE_SPLIT, 1
    return 1, min(triton.next_power_of_2(hidden), _MAX_BLOCK_N), 1


@triton.jit
def _indexed_gate_rms_shift_native_wide_kernel(
    residual_out_ptr,
    modulated_out_ptr,
    residual_ptr,
    gate_ptr,
    branch_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    rows,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    offs = tl.arange(0, HIDDEN).to(tl.int32)
    weight = tl.load(weight_ptr + offs).to(tl.float32)
    for row in range(pid, rows, NUM_CORES):
        row_i = row.to(tl.int32)
        index = tl.load(indices_ptr + row_i).to(tl.int32)
        base = row_i * HIDDEN + offs
        gbase = index * HIDDEN + offs
        residual = tl.load(residual_ptr + base).to(tl.float32)
        gate = tl.load(gate_ptr + gbase).to(tl.float32)
        branch = tl.load(branch_ptr + base).to(tl.float32)
        updated_bf = (residual + gate * branch).to(tl.bfloat16)
        tl.store(residual_out_ptr + base, updated_bf)
        updated = updated_bf.to(tl.float32)
        variance = tl.sum(updated * updated, axis=0) / HIDDEN
        normalized = updated * tl.rsqrt(variance + EPS) * weight
        shift = tl.load(shift_ptr + gbase).to(tl.float32)
        scale = tl.load(scale_ptr + gbase).to(tl.float32)
        output = normalized * (1.0 + scale) + shift
        tl.store(modulated_out_ptr + base, output.to(tl.bfloat16))


def _forward(residual, gate, branch, weight, shift, scale, indices, eps):
    residual = residual.contiguous()
    gate = gate.contiguous()
    branch = branch.contiguous()
    weight = weight.contiguous()
    shift = shift.contiguous()
    scale = scale.contiguous()
    indices = indices.contiguous()

    rows, hidden = residual.shape
    residual_out = torch.empty_like(residual)
    modulated_out = torch.empty_like(residual)
    if rows == 0:
        return residual_out, modulated_out

    mode, block_n, rows_per_iter = _plan(hidden)
    num_cores = int(driver.active.utils.get_device_properties(residual.device.index)["num_vectorcore"])
    conditions = gate.shape[0]
    if mode == 0 and rows > _SMALL_ROWS and hidden <= _TINY_HIDDEN and conditions * hidden <= _TABLE_MAX_ELEMS:
        grid = (min(triton.cdiv(rows, _TABLE_ROWS), num_cores, _MAX_1D_GRID),)
        _indexed_gate_rms_shift_table_kernel[grid](
            residual_out,
            modulated_out,
            residual,
            gate,
            branch,
            weight,
            shift,
            scale,
            indices,
            rows,
            prevec_max_fused_ops_num=2,
            enable_auto_vectorize_v2=True,
            hfusion_enable_multiple_consumer_fusion=True,
            HIDDEN=hidden,
            EPS=eps,
            BLOCK_R=_TABLE_ROWS,
            NUM_CORES=grid[0],
            NCOND=conditions,
            CP2=triton.next_power_of_2(conditions),
        )
        return residual_out, modulated_out
    if mode == 0 and rows <= _SMALL_ROWS:
        grid = (rows,)
        _indexed_gate_rms_shift_row_kernel[grid](
            residual_out,
            modulated_out,
            residual,
            gate,
            branch,
            weight,
            shift,
            scale,
            indices,
            prevec_max_fused_ops_num=2,
            HIDDEN=hidden,
            EPS=eps,
            BLOCK_N=block_n,
            SINGLE_CONDITION=(conditions == 1),
        )
        return residual_out, modulated_out
    if mode == 2 and hidden % 8 != 0:
        grid = (min(rows, num_cores, _MAX_1D_GRID),)
        _indexed_gate_rms_shift_native_wide_kernel[grid](
            residual_out,
            modulated_out,
            residual,
            gate,
            branch,
            weight,
            shift,
            scale,
            indices,
            rows,
            HIDDEN=hidden,
            EPS=eps,
            NUM_CORES=grid[0],
            prevec_max_fused_ops_num=2,
        )
        return residual_out, modulated_out
    if mode == 2:
        grid = (min(rows, num_cores, _MAX_1D_GRID),)
        _indexed_gate_rms_shift_two_tile_kernel[grid](
            residual_out,
            modulated_out,
            residual,
            gate,
            branch,
            weight,
            shift,
            scale,
            indices,
            rows,
            prevec_max_fused_ops_num=2,
            HIDDEN=hidden,
            EPS=eps,
            BLOCK_A=block_n,
            BLOCK_B=max(32, hidden - block_n),
            NUM_CORES=grid[0],
        )
        return residual_out, modulated_out

    if mode == 0 and 32 < hidden <= 3072 and rows >= 1024 and conditions * hidden <= 16384:
        grid = (min(triton.cdiv(rows, rows_per_iter), num_cores, _MAX_1D_GRID),)
        _indexed_gate_rms_shift_cached_kernel[grid](
            residual_out,
            modulated_out,
            residual,
            gate,
            branch,
            weight,
            shift,
            scale,
            indices,
            rows,
            HIDDEN=hidden,
            EPS=eps,
            BLOCK_N=block_n,
            NUM_CORES=grid[0],
            ROWS_PER_ITER=rows_per_iter,
            EXACT=True,
            NCOND=conditions,
            CP2=triton.next_power_of_2(conditions),
            CACHE_GATE=True,
            CACHE_AFFINE=False,
            prevec_max_fused_ops_num=2,
        )
        return residual_out, modulated_out

    grid = (min(triton.cdiv(rows, rows_per_iter), num_cores, _MAX_1D_GRID),)
    _indexed_gate_rms_shift_kernel[grid](
        residual_out,
        modulated_out,
        residual,
        gate,
        branch,
        weight,
        shift,
        scale,
        indices,
        rows,
        prevec_max_fused_ops_num=2,
        HIDDEN=hidden,
        EPS=eps,
        BLOCK_N=block_n,
        NUM_CORES=grid[0],
        ROWS_PER_ITER=rows_per_iter,
        EXACT=(mode == 0),
        KEEP_I64=(mode == 0 and hidden == 3072 and rows >= 32768),
        **({"multibuffer": False} if 64 < rows <= 1024 else {}),
    )
    return residual_out, modulated_out


@torch.library.custom_op("mojo_npu_triton_a5::indexed_gate_rms_norm_scale_shift_infer", mutates_args=())
def indexed_gate_rms_norm_scale_shift_infer_fwd(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Device queries and launches must use the input device, even if a different
    # NPU is current. Keeping them inside the leaf also permits fullgraph capture.
    with torch.npu.device(residual.device):
        return _forward(residual, gate, branch, weight, shift, scale, indices, eps)


@indexed_gate_rms_norm_scale_shift_infer_fwd.register_fake
def _fake(residual, gate, branch, weight, shift, scale, indices, eps):
    return (
        torch.empty_like(residual, memory_format=torch.contiguous_format),
        torch.empty_like(residual, memory_format=torch.contiguous_format),
    )
