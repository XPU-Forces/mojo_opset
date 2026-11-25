# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend A5 single-path gated residual, RMSNorm and indexed affine.

Derived from vllm-project/vllm-omni at d613fdb62c432a342681e88b307d36b54f6ed5f0,
vllm_omni/diffusion/layers/indexed_modulation.py.
"""

import torch
import triton
import triton.language as tl

from triton.runtime import driver


@triton.jit
def _indexed_gate_rms_norm_scale_shift_kernel(
    residual_out,
    output,
    residual,
    gate,
    branch,
    weight,
    shift,
    scale,
    indices,
    rows,
    G0: tl.constexpr,
    G1: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    CORES: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    columns = tl.arange(0, BLOCK_H)
    rvec = tl.arange(0, 2)
    col_ok = columns < H
    col_mask = tl.broadcast_to(col_ok[None, :], [2, BLOCK_H])
    w = tl.load(weight + columns, mask=col_ok, other=0.0).to(tl.float32)

    # Full two-row groups: no row predicate.
    n_full = (rows // (CORES * 2)) * (CORES * 2)
    for row0 in range(pid * 2, n_full, CORES * 2):
        index0 = tl.load(indices + row0).to(tl.int32)
        index1 = tl.load(indices + row0 + 1).to(tl.int32)
        idx = tl.where(rvec == 0, index0, index1)
        r = row0 + rvec
        offsets = r[:, None] * H + columns[None, :]
        x = tl.load(residual + offsets, mask=col_mask, other=0.0).to(tl.float32)
        g = tl.load(gate + idx[:, None] * G0 + columns[None, :] * G1, mask=col_mask, other=0.0).to(tl.float32)
        b = tl.load(branch + offsets, mask=col_mask, other=0.0).to(tl.float32)
        updated_bf16 = (x + g * b).to(tl.bfloat16)
        tl.store(residual_out + offsets, updated_bf16, mask=col_mask)
        updated = updated_bf16.to(tl.float32)
        variance = tl.sum(updated * updated, axis=1) / H
        normalized = updated * tl.rsqrt(variance[:, None] + EPS) * w[None, :]
        t = tl.load(shift + idx[:, None] * T0 + columns[None, :] * T1, mask=col_mask, other=0.0).to(tl.float32)
        s = tl.load(scale + idx[:, None] * S0 + columns[None, :] * S1, mask=col_mask, other=0.0).to(tl.float32)
        result = normalized * (1.0 + s) + t
        tl.store(output + offsets, result.to(tl.bfloat16), mask=col_mask)

    # Row remainder: one row per program, same math.
    for row in range(n_full + pid, rows, CORES):
        row_i = row.to(tl.int32)
        idx = tl.load(indices + row_i).to(tl.int32)
        offsets = row_i * H + columns
        x = tl.load(residual + offsets, mask=col_ok, other=0.0).to(tl.float32)
        g = tl.load(gate + idx * G0 + columns * G1, mask=col_ok, other=0.0).to(tl.float32)
        b = tl.load(branch + offsets, mask=col_ok, other=0.0).to(tl.float32)
        updated_bf16 = (x + g * b).to(tl.bfloat16)
        tl.store(residual_out + offsets, updated_bf16, mask=col_ok)
        updated = updated_bf16.to(tl.float32)
        variance = tl.sum(updated * updated, axis=0) / H
        normalized = updated * tl.rsqrt(variance + EPS) * w
        t = tl.load(shift + idx * T0 + columns * T1, mask=col_ok, other=0.0).to(tl.float32)
        s = tl.load(scale + idx * S0 + columns * S1, mask=col_ok, other=0.0).to(tl.float32)
        result = normalized * (1.0 + s) + t
        tl.store(output + offsets, result.to(tl.bfloat16), mask=col_ok)


def _forward(residual, gate, branch, weight, shift, scale, indices, eps):
    rows, hidden = residual.shape
    # The backend's strided-column gather can leave an unwritten temporary.
    # Original model tables have column stride one and need no copies here.
    if gate.stride(1) != 1:
        gate = gate.contiguous()
    if shift.stride(1) != 1:
        shift = shift.contiguous()
    if scale.stride(1) != 1:
        scale = scale.contiguous()
    residual = residual.contiguous()
    branch = branch.contiguous()
    weight = weight.contiguous()
    indices = indices.contiguous()
    updated = torch.empty_like(residual)
    result = torch.empty_like(residual)
    if rows == 0:
        return updated, result

    num_cores = int(driver.active.utils.get_device_properties(residual.device.index)["num_vectorcore"])
    _indexed_gate_rms_norm_scale_shift_kernel[(num_cores,)](
        updated,
        result,
        residual,
        gate,
        branch,
        weight,
        shift,
        scale,
        indices,
        rows,
        G0=gate.stride(0),
        G1=gate.stride(1),
        T0=shift.stride(0),
        T1=shift.stride(1),
        S0=scale.stride(0),
        S1=scale.stride(1),
        H=hidden,
        BLOCK_H=(hidden + 7) // 8 * 8,
        CORES=num_cores,
        EPS=eps,
        prevec_max_fused_ops_num=2,
        multibuffer=False,
    )
    return updated, result


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
