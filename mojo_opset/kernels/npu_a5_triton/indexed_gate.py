# SPDX-License-Identifier: Apache-2.0
"""Triton-Ascend indexed gate for Ascend 950PR."""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from triton.runtime import driver


@lru_cache(maxsize=None)
def _vector_cores(device_index):
    return driver.active.utils.get_device_properties(device_index)["num_vectorcore"]


@triton.jit
def _indexed_gate_kernel(
    out,
    x,
    gate,
    other,
    indices,
    H: tl.constexpr,
    TOTAL: tl.constexpr,
    SX: tl.constexpr,
    SG: tl.constexpr,
    SO: tl.constexpr,
    SY: tl.constexpr,
    SI: tl.constexpr,
    BLOCK: tl.constexpr,
    TILES_PER_ROW: tl.constexpr,
    WIDE_OFFSETS: tl.constexpr,
):
    column_lane = tl.arange(0, BLOCK)
    program = tl.program_id(0)
    programs = tl.num_programs(0)
    per_program = TOTAL // programs
    remainder = TOTAL % programs
    first = program * per_program + tl.minimum(program, remainder)
    stop = first + per_program + (program < remainder).to(tl.int32)
    for tile in tl.range(first, stop, loop_unroll_factor=2):
        row = tile // TILES_PER_ROW
        columns = (tile - row * TILES_PER_ROW) * BLOCK + column_lane
        in_columns = columns < H
        if WIDE_OFFSETS:
            row = row.to(tl.int64)
        index = tl.load(indices + row * SI)
        gate_row = index.to(tl.int64) if WIDE_OFFSETS else index.to(tl.int32)
        g = tl.load(gate + gate_row * SG + columns, in_columns, other=0).to(tl.float32)
        a = tl.load(x + row * SX + columns, in_columns, other=0).to(tl.float32)
        b = tl.load(other + row * SO + columns, in_columns, other=0).to(tl.float32)
        tl.store(out + row * SY + columns, a + g * b, in_columns)


def _launch(output, x, gate, other, indices):
    rows, hidden = x.shape
    if rows == 0 or hidden == 0:
        return
    block = 2048
    tiles_per_row = triton.cdiv(hidden, block)
    total = rows * tiles_per_row
    wide_offsets = (
        max(
            rows * max(x.stride(0), other.stride(0), output.stride(0), indices.stride(0)),
            gate.shape[0] * gate.stride(0),
        )
        >= 2**31
    )
    _indexed_gate_kernel[(min(total, _vector_cores(x.device.index)),)](
        output,
        x,
        gate,
        other,
        indices,
        hidden,
        total,
        x.stride(0),
        gate.stride(0),
        other.stride(0),
        output.stride(0),
        indices.stride(0),
        block,
        tiles_per_row,
        wide_offsets,
        enable_fp_fusion=False,
    )


@torch.library.custom_op("mojo_npu_triton_a5::indexed_gate_infer", mutates_args=())
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
