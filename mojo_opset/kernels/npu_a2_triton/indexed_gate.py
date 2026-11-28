# SPDX-License-Identifier: Apache-2.0
"""Single-path indexed gate with a zero row in each bounded UB table."""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from triton.language.extra.cann import extension as al
from triton.runtime import driver


@lru_cache(maxsize=None)
def _vector_cores(device):
    return driver.active.utils.get_device_properties(device)["num_vectorcore"]


@triton.jit
def _load_gate(gate, cols, start, H: tl.constexpr, C: tl.constexpr, SG: tl.constexpr, GC: tl.constexpr):
    lane = tl.arange(0, GC)
    row = (start + lane).to(tl.int64)
    active = (lane[:, None] < GC - 1) & (row[:, None] < C) & (cols[None, :] < H)
    return tl.load(gate + row[:, None] * SG + cols[None, :], active, other=0)


@triton.jit
def _row_address(ids, lane, start, C: tl.constexpr, GC: tl.constexpr):
    index = al.get_element(ids, (lane,)).to(tl.int64)
    local = (index - start).to(tl.uint64)
    active = local < tl.minimum(GC - 1, C - start)
    return tl.where(active, local.to(tl.int32), GC - 1), active


@triton.jit
def _dense(x, other, rows, cols, R: tl.constexpr, H: tl.constexpr, SX: tl.constexpr, SO: tl.constexpr):
    mask = (rows[:, None] < R) & (cols[None, :] < H)
    a = tl.load(x + rows[:, None] * SX + cols[None, :], mask, other=0)
    b = tl.load(other + rows[:, None] * SO + cols[None, :], mask, other=0)
    return a, b


@triton.jit
def _indexed_gate_a2_resident_suffix(
    out,
    x,
    gate,
    other,
    indices,
    R: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    SX: tl.constexpr,
    SG: tl.constexpr,
    SO: tl.constexpr,
    SY: tl.constexpr,
    SI: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    GC: tl.constexpr,
    NC: tl.constexpr,
    NR: tl.constexpr,
):
    pid = tl.program_id(0)
    row_core, col_core = pid // NC, pid % NC
    row_lanes = tl.arange(0, BR)
    col_lanes = tl.arange(0, BC)
    for col_tile in range(col_core, tl.cdiv(H, BC), NC):
        cols = col_tile * BC + col_lanes
        first = _load_gate(gate, cols, 0, H, C, SG, GC)
        initial_rows = (row_core * BR + row_lanes).to(tl.int64)
        a, b = _dense(x, other, initial_rows, cols, R, H, SX, SO)
        for row_tile in range(row_core, tl.cdiv(R, BR), NR):
            rows = (row_tile * BR + row_lanes).to(tl.int64)
            valid_rows = rows < R
            ids = tl.load(indices + rows * SI, valid_rows, other=0)
            selected = tl.full((BR, BC), 0, gate.dtype.element_ty)
            for lane in range(BR):
                address, _ = _row_address(ids, lane, 0, C, GC)
                value = al.extract_slice(first, (address, 0), (1, BC), (1, 1))
                selected = al.insert_slice(selected, value, (lane, 0), (1, BC), (1, 1))
            for start in range(GC - 1, C, GC - 1):
                table = _load_gate(gate, cols, start, H, C, SG, GC)
                for lane in range(BR):
                    address, active = _row_address(ids, lane, start, C, GC)
                    value = al.extract_slice(table, (address, 0), (1, BC), (1, 1))
                    prior = al.extract_slice(selected, (lane, 0), (1, BC), (1, 1))
                    selected = al.insert_slice(selected, tl.where(active, value, prior), (lane, 0), (1, BC), (1, 1))
            mask = valid_rows[:, None] & (cols[None, :] < H)
            next_a, next_b = _dense(x, other, rows + NR * BR, cols, R, H, SX, SO)
            result = a.to(tl.float32) + selected.to(tl.float32) * b.to(tl.float32)
            tl.store(out + rows[:, None] * SY + cols[None, :], result, mask)
            a, b = next_a, next_b


def _launch(output, x, gate, other, indices):
    rows, hidden = x.shape
    if rows == 0 or hidden == 0:
        return
    cores = _vector_cores(x.device.index)
    columns_per_core = triton.cdiv(hidden, triton.cdiv(cores, rows))
    gate_rows = min(64, triton.next_power_of_2(max(2, gate.shape[0] + 1)))
    column_budget = (32 * 1024) // (gate_rows * gate.element_size())
    tile_cols = min(8192, column_budget, triton.next_power_of_2(min(hidden, max(64, columns_per_core))))
    tile_rows = min(32, max(1, 4096 // tile_cols), triton.next_power_of_2(rows))
    column_cores = min(triton.cdiv(hidden, tile_cols), cores)
    row_cores = min(triton.cdiv(rows, tile_rows), cores // column_cores)
    _indexed_gate_a2_resident_suffix[(row_cores * column_cores,)](
        output,
        x,
        gate,
        other,
        indices,
        rows,
        hidden,
        gate.shape[0],
        x.stride(0),
        gate.stride(0),
        other.stride(0),
        output.stride(0),
        indices.stride(0),
        tile_rows,
        tile_cols,
        gate_rows,
        column_cores,
        row_cores,
        enable_fp_fusion=False,
        multibuffer=False,
    )


@torch.library.custom_op("mojo_npu_triton_a2::indexed_gate_infer", mutates_args=())
def indexed_gate_infer_fwd(
    x: torch.Tensor, gate: torch.Tensor, other: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    with torch.npu.device(x.device):
        output = torch.empty_like(x, memory_format=torch.contiguous_format)
        if x.numel() == 0:
            return output
        # Keep unit-column-stride views of packed projection tables in place.
        if x.stride(1) != 1:
            x = x.contiguous()
        if gate.stride(1) != 1:
            gate = gate.contiguous()
        if other.stride(1) != 1:
            other = other.contiguous()
        _launch(output, x, gate, other, indices)
        return output


@indexed_gate_infer_fwd.register_fake
def _fake(x, gate, other, indices):
    return torch.empty_like(x, memory_format=torch.contiguous_format)
