# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted for Ascend NPU by the mojo_opset contributors.
"""DeepSeek mHC residual collapse: weighted FP32 sum over BF16 hc streams."""

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores
from mojo_opset.kernels._npu_triton_utils import libentry
from mojo_opset.kernels._triton_utils import input_guard

# Whole-row tile cap, found by trial against the target's Unified Buffer:
# A5 (256KB UB) compiles and runs up to 12288 elements per tile; A2 (192KB
# UB) scales proportionally (12288 * 192/256 = 9216). One token per task
# maximizes the contiguous burst per stream; intermediate widths pay
# tail-lane waste unless they divide hidden_size exactly.
HC_COLLAPSE_BLOCK_H = 9216


@libentry()
@triton.jit
def _hc_collapse_kernel(
    pre_ptr,
    x_ptr,
    out_ptr,
    pre_stride_t,
    pre_stride_m,
    x_stride_t,
    x_stride_m,
    x_stride_h,
    num_tokens,
    hidden_size,
    num_h_blocks,
    HC_MULT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """out[t, h] = sum_m pre[t, m] * x[t, m, h] accumulated in FP32.

    One task collapses one (token, hidden-block) tile; with the whole-row
    tile a task is exactly one token. Each program owns one contiguous task
    range: the per-core load differs by at most one tile, and a token's
    tiles stay local so its pre-mix row stays hot in L2.
    """
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    total_tasks = num_tokens * num_h_blocks
    tasks_per_core = (total_tasks + grid_size - 1) // grid_size
    task_start = pid * tasks_per_core
    task_end = tl.minimum(task_start + tasks_per_core, total_tasks)

    cols = tl.arange(0, BLOCK_H)

    for task_id in range(task_start, task_end):
        token_idx = (task_id // num_h_blocks).to(tl.int64)
        block_idx = task_id % num_h_blocks
        offsets = block_idx * BLOCK_H + cols
        col_mask = offsets < hidden_size

        pre_row = pre_ptr + token_idx * pre_stride_t
        x_row = x_ptr + token_idx * x_stride_t
        out_row = out_ptr + token_idx * hidden_size

        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for mix_idx in tl.static_range(HC_MULT):
            pre = tl.load(pre_row + mix_idx * pre_stride_m).to(tl.float32)
            x = tl.load(
                x_row + mix_idx * x_stride_m + offsets * x_stride_h,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)
            acc += pre * x

        tl.store(out_row + offsets, acc.to(out_ptr.dtype.element_ty), mask=col_mask)

@input_guard(make_contiguous=True)
def hc_collapse_impl(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    num_tokens, hc_mult, hidden_size = x.shape
    out = torch.empty(num_tokens, hidden_size, dtype=x.dtype, device=x.device)
    if num_tokens == 0 or hidden_size == 0:
        return out

    block_h = min(hidden_size, HC_COLLAPSE_BLOCK_H)
    num_h_blocks = (hidden_size + block_h - 1) // block_h
    grid = (min(get_num_cores("vector"), num_tokens * num_h_blocks),)

    _hc_collapse_kernel[grid](
        pre_mix,
        x,
        out,
        pre_mix.stride(0),
        pre_mix.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        num_tokens,
        hidden_size,
        num_h_blocks,
        HC_MULT=hc_mult,
        BLOCK_H=block_h,
    )
    return out


@torch.library.custom_op("mojo_npu_triton_a2::hc_collapse_fwd", mutates_args=())
def hc_collapse_fwd(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return hc_collapse_impl(x, pre_mix)


@hc_collapse_fwd.register_fake
def _hc_collapse_fwd_fake(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape[0], x.shape[2], dtype=x.dtype, device=x.device)
