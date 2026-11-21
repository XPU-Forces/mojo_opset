# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend A2 fused indexed gating, BF16 residual, RMSNorm and affine.

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
    CONDITIONS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    offs = tl.arange(0, BLOCK_N).to(tl.int32)

    if EXACT:
        # ---- mask-free main loop (BLOCK_N == HIDDEN) ----
        rvec = tl.arange(0, ROWS_PER_ITER).to(tl.int32)
        weight = tl.load(weight_ptr + offs).to(tl.float32)
        if HIDDEN <= 64:
            table_off = tl.arange(0, triton.next_power_of_2(CONDITIONS * HIDDEN))
            gate_table = tl.load(gate_ptr + table_off, table_off < CONDITIONS * HIDDEN, 0).to(tl.float32)
            shift_table = tl.load(shift_ptr + table_off, table_off < CONDITIONS * HIDDEN, 0).to(tl.float32)
            scale_table = tl.load(scale_ptr + table_off, table_off < CONDITIONS * HIDDEN, 0).to(tl.float32)
        n_full = (rows // (NUM_CORES * ROWS_PER_ITER)) * (NUM_CORES * ROWS_PER_ITER)
        for row0 in range(pid * ROWS_PER_ITER, n_full, NUM_CORES * ROWS_PER_ITER):
            rows_v = row0 + rvec
            index = tl.load(indices_ptr + rows_v).to(tl.int32)
            base = rows_v[:, None] * HIDDEN + offs[None, :]
            residual = tl.load(residual_ptr + base).to(tl.float32)
            if HIDDEN <= 64:
                lookup = tl.reshape(index[:, None] * HIDDEN + offs[None, :], (ROWS_PER_ITER * BLOCK_N,))
                gate = tl.reshape(tl.gather(gate_table, lookup, 0), (ROWS_PER_ITER, BLOCK_N))
            else:
                gate = tl.load(gate_ptr + index[:, None] * HIDDEN + offs[None, :]).to(tl.float32)
            branch = tl.load(branch_ptr + base).to(tl.float32)
            updated = (residual + gate * branch).to(tl.bfloat16).to(tl.float32)
            tl.store(residual_out_ptr + base, updated.to(tl.bfloat16))
            variance = tl.sum(updated * updated, axis=1) / HIDDEN
            normalized = updated * tl.rsqrt(variance[:, None] + EPS) * weight[None, :]
            if HIDDEN <= 64:
                lookup = tl.reshape(index[:, None] * HIDDEN + offs[None, :], (ROWS_PER_ITER * BLOCK_N,))
                shift = tl.reshape(tl.gather(shift_table, lookup, 0), (ROWS_PER_ITER, BLOCK_N))
            else:
                shift = tl.load(shift_ptr + index[:, None] * HIDDEN + offs[None, :]).to(tl.float32)
            if HIDDEN <= 64:
                lookup = tl.reshape(index[:, None] * HIDDEN + offs[None, :], (ROWS_PER_ITER * BLOCK_N,))
                scale = tl.reshape(tl.gather(scale_table, lookup, 0), (ROWS_PER_ITER, BLOCK_N))
            else:
                scale = tl.load(scale_ptr + index[:, None] * HIDDEN + offs[None, :]).to(tl.float32)
            output = normalized * (1.0 + scale) + shift
            tl.store(modulated_out_ptr + base, output.to(tl.bfloat16))
        # ---- row remainder ----
        for row in range(n_full + pid, rows, NUM_CORES):
            row_i = row.to(tl.int32)
            index = tl.load(indices_ptr + row_i).to(tl.int32)
            base = row_i * HIDDEN
            residual = tl.load(residual_ptr + base + offs).to(tl.float32)
            if HIDDEN <= 64:
                gate = tl.gather(gate_table, index * HIDDEN + offs, 0)
            else:
                gate = tl.load(gate_ptr + index * HIDDEN + offs).to(tl.float32)
            branch = tl.load(branch_ptr + base + offs).to(tl.float32)
            updated = (residual + gate * branch).to(tl.bfloat16).to(tl.float32)
            tl.store(residual_out_ptr + base + offs, updated.to(tl.bfloat16))
            variance = tl.sum(updated * updated, axis=0) / HIDDEN
            normalized = updated * tl.rsqrt(variance + EPS) * weight
            if HIDDEN <= 64:
                shift = tl.gather(shift_table, index * HIDDEN + offs, 0)
            else:
                shift = tl.load(shift_ptr + index * HIDDEN + offs).to(tl.float32)
            if HIDDEN <= 64:
                scale = tl.gather(scale_table, index * HIDDEN + offs, 0)
            else:
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
            updated = (residual + gate * branch).to(tl.bfloat16).to(tl.float32)
            tl.store(residual_out_ptr + base + offs, updated.to(tl.bfloat16), mask=mask)
            weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            square = tl.where(mask, updated * updated, 0.0)
            variance = tl.sum(square, axis=0) / HIDDEN
            normalized = updated * tl.rsqrt(variance + EPS) * weight
            shift = tl.load(shift_ptr + index * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
            scale = tl.load(scale_ptr + index * HIDDEN + offs, mask=mask, other=0.0).to(tl.float32)
            output = normalized * (1.0 + scale) + shift
            tl.store(modulated_out_ptr + base + offs, output.to(tl.bfloat16), mask=mask)


def _plan(hidden):
    """Return (block_n, rows_per_iter, exact) for this hidden size.

    ROWS_PER_ITER is capped so that the 2-D tile stays inside the UB budget:
    four live tiles of BLOCK_N * ROWS_PER_ITER FP32 words ~= 160 KB.
    The mask-free path needs BLOCK_N == HIDDEN, at least two rows per iteration
    (R=1 makes the compiler reject the 2-D tile) and BLOCK_N <= 5376, which is the
    largest tile that still fits with two rows.
    """
    exact = hidden <= 5376 and hidden % 8 == 0
    block_n = hidden if exact else min(triton.next_power_of_2(hidden), _MAX_BLOCK_N)
    if not exact:
        return block_n, 1, False
    if block_n <= 64:
        rows = 64
    elif block_n <= 2048:
        rows = 4
    elif block_n <= 3072:
        rows = 3
    else:
        rows = 2
    return block_n, rows, True


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

    block_n, rows_per_iter, exact = _plan(hidden)
    num_cores = int(driver.active.utils.get_device_properties(residual.device.index)["num_vectorcore"])
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
        HIDDEN=hidden,
        EPS=eps,
        BLOCK_N=block_n,
        NUM_CORES=grid[0],
        ROWS_PER_ITER=rows_per_iter,
        EXACT=exact,
        CONDITIONS=gate.shape[0],
    )
    return residual_out, modulated_out


@torch.library.custom_op("mojo_npu_triton_a2::indexed_gate_rms_norm_scale_shift_infer", mutates_args=())
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
