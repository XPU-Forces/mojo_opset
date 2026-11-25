# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend A2 single-path gated residual, RMSNorm and indexed affine.

Derived from vllm-project/vllm-omni at d613fdb62c432a342681e88b307d36b54f6ed5f0,
vllm_omni/diffusion/layers/indexed_modulation.py.
"""

import torch
import triton
import triton.language as tl

from triton.runtime import driver

_ROW_TILE_BUDGET_BYTES = 176128
_BYTES_PER_ELEMENT_PER_ROW = 16
_MAX_ROWS_PER_ITER = 64
_MAX_1D_GRID = 65535
_SUPPORTED_HIDDEN = 5376


def _rows_per_iter(block_n):
    return max(1, min(_MAX_ROWS_PER_ITER, _ROW_TILE_BUDGET_BYTES // (_BYTES_PER_ELEMENT_PER_ROW * block_n)))


@triton.jit
def _indexed_gate_rms_norm_scale_shift_kernel(
    residual_out_ptr,
    modulated_out_ptr,
    residual_ptr,
    gate_ptr,
    branch_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    gate_stride,
    shift_stride,
    scale_stride,
    rows,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CORES: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    col = tl.arange(0, BLOCK_N).to(tl.int32)
    rvec = tl.arange(0, ROWS_PER_ITER).to(tl.int32)
    weight = tl.load(weight_ptr + col)
    step = NUM_CORES * ROWS_PER_ITER
    n_full = (rows // step) * step

    # Full groups: mask-free 2-D hot loop, BF16 inputs staged in UB.
    for row0 in range(pid * ROWS_PER_ITER, n_full, step):
        rows_v = row0 + rvec
        index = tl.load(indices_ptr + rows_v).to(tl.int32)
        base = rows_v[:, None] * HIDDEN + col[None, :]
        residual = tl.load(residual_ptr + base)
        gate = tl.load(gate_ptr + index[:, None] * gate_stride + col[None, :])
        branch = tl.load(branch_ptr + base)
        updated = (residual.to(tl.float32) + gate.to(tl.float32) * branch.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        tl.store(residual_out_ptr + base, updated.to(tl.bfloat16))
        variance = tl.sum(updated * updated, axis=1) / HIDDEN
        normalized = updated * tl.rsqrt(variance[:, None] + EPS) * weight[None, :].to(tl.float32)
        shift = tl.load(shift_ptr + index[:, None] * shift_stride + col[None, :])
        scale = tl.load(scale_ptr + index[:, None] * scale_stride + col[None, :])
        output = normalized * (1.0 + scale.to(tl.float32)) + shift.to(tl.float32)
        tl.store(modulated_out_ptr + base, output.to(tl.bfloat16))

    # Tail: the remaining rows, one narrow row per program turn.
    for row in range(n_full + pid, rows, NUM_CORES):
        row_i = row.to(tl.int32)
        index = tl.load(indices_ptr + row_i).to(tl.int32)
        base = row_i * HIDDEN + col
        residual = tl.load(residual_ptr + base)
        gate = tl.load(gate_ptr + index * gate_stride + col)
        branch = tl.load(branch_ptr + base)
        updated = (residual.to(tl.float32) + gate.to(tl.float32) * branch.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        tl.store(residual_out_ptr + base, updated.to(tl.bfloat16))
        variance = tl.sum(updated * updated, axis=0) / HIDDEN
        normalized = updated * tl.rsqrt(variance + EPS) * weight.to(tl.float32)
        shift = tl.load(shift_ptr + index * shift_stride + col)
        scale = tl.load(scale_ptr + index * scale_stride + col)
        output = normalized * (1.0 + scale.to(tl.float32)) + shift.to(tl.float32)
        tl.store(modulated_out_ptr + base, output.to(tl.bfloat16))


def _forward(residual, gate, branch, weight, shift, scale, indices, eps):
    rows, hidden = residual.shape
    if hidden != _SUPPORTED_HIDDEN:
        raise ValueError("Ascend A2 indexed gated RMSNorm requires hidden == 5376")
    residual = residual.contiguous()
    branch = branch.contiguous()
    weight = weight.contiguous()
    indices = indices.contiguous()
    if gate.stride(1) != 1 or shift.stride(1) != 1 or scale.stride(1) != 1:
        gate = gate.contiguous()
        shift = shift.contiguous()
        scale = scale.contiguous()
    residual_out = torch.empty_like(residual)
    modulated_out = torch.empty_like(residual)
    if rows == 0:
        return residual_out, modulated_out

    num_cores = int(driver.active.utils.get_device_properties(residual.device.index)["num_vectorcore"])
    rows_per_iter = _rows_per_iter(hidden)
    grid = (min(triton.cdiv(rows, rows_per_iter), num_cores, _MAX_1D_GRID),)
    _indexed_gate_rms_norm_scale_shift_kernel[grid](
        residual_out,
        modulated_out,
        residual,
        gate,
        branch,
        weight,
        shift,
        scale,
        indices,
        gate.stride(0),
        shift.stride(0),
        scale.stride(0),
        rows,
        HIDDEN=hidden,
        EPS=eps,
        BLOCK_N=hidden,
        NUM_CORES=grid[0],
        ROWS_PER_ITER=rows_per_iter,
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
    if residual.shape[1] != _SUPPORTED_HIDDEN:
        raise ValueError("Ascend A2 indexed gated RMSNorm requires hidden == 5376")
    return (
        torch.empty_like(residual, memory_format=torch.contiguous_format),
        torch.empty_like(residual, memory_format=torch.contiguous_format),
    )
