# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted for Ascend NPU by the mojo_opset contributors.
"""mHC post block: combine a sublayer output with the HC residual streams."""

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores
from mojo_opset.kernels._npu_triton_utils import libentry

# Tile caps by hc_mult, found by trial against the target's Unified Buffer:
# the FP32 accumulator is a [hc_mult, BLOCK_H] tile, so the cap shrinks as
# hc_mult grows. A5 (256KB UB) compiles and runs [4, 6144] (also [8, 3072],
# [16, 1536], [2, 12288]); [4, 7168] and above fail in bisheng/tvming. A2
# (192KB UB) scales every cap proportionally (x 192/256). One token per task
# maximizes the contiguous burst per stream; intermediate widths pay
# tail-lane waste unless they divide hidden_size exactly.
_MHC_POST_TILE_CAPS = ((2, 12288), (4, 6144), (8, 3072))
_MHC_POST_FALLBACK_ELEMENTS = 16384


def _mhc_post_tile_cap(hc_mult: int) -> int:
    for max_mult, cap in _MHC_POST_TILE_CAPS:
        if hc_mult <= max_mult:
            return cap
    return max(256, _MHC_POST_FALLBACK_ELEMENTS // hc_mult)


def _largest_divisor_at_most(value: int, cap: int) -> int:
    """Largest divisor of ``value`` not above ``cap``; ``value`` itself when prime."""
    divisor = value
    while divisor > cap:
        for factor in range(2, int(divisor**0.5) + 1):
            if divisor % factor == 0:
                divisor //= factor
                break
        else:
            return value  # Prime: only the trivial divisor fits the budget.
    return divisor


@libentry()
@triton.jit
def _mhc_post_kernel(
    x_ptr,
    res_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    x_stride_t,
    x_stride_h,
    res_stride_t,
    res_stride_m,
    res_stride_h,
    post_stride_t,
    post_stride_m,
    comb_stride_t,
    comb_stride_i,
    comb_stride_j,
    num_tokens,
    hidden_size,
    num_h_blocks,
    HC_MULT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """out[t, j, h] = sum_i comb[t, i, j] * res[t, i, h] + post[t, j] * x[t, h].

    All math is FP32; only the store rounds to the residual dtype. The comb
    mix is summed over input streams first and the post term is added last,
    matching the einsum-then-add order of the Torch reference. One task
    updates one (token, hidden-block) tile across every output stream: each
    residual row is loaded exactly once and broadcast into the [hc_mult,
    BLOCK_H] accumulator. Each program owns one contiguous task range, so
    the per-core load differs by at most one tile and a token's tiles stay
    local (its gates stay hot in L2).
    """
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    total_tasks = num_tokens * num_h_blocks
    tasks_per_core = (total_tasks + grid_size - 1) // grid_size
    task_start = pid * tasks_per_core
    task_end = tl.minimum(task_start + tasks_per_core, total_tasks)

    cols = tl.arange(0, BLOCK_H)
    rows = tl.arange(0, HC_MULT)

    for task_id in range(task_start, task_end):
        token_idx = (task_id // num_h_blocks).to(tl.int64)
        block_idx = task_id % num_h_blocks
        offsets = block_idx * BLOCK_H + cols
        col_mask = offsets < hidden_size

        x_row = x_ptr + token_idx * x_stride_t
        res_row = res_ptr + token_idx * res_stride_t
        post_row = post_ptr + token_idx * post_stride_t
        comb_row = comb_ptr + token_idx * comb_stride_t
        # The output is freshly allocated and contiguous.
        out_row = out_ptr + token_idx * HC_MULT * hidden_size

        acc = tl.zeros((HC_MULT, BLOCK_H), dtype=tl.float32)
        for in_idx in tl.static_range(HC_MULT):
            res = tl.load(
                res_row + in_idx * res_stride_m + offsets * res_stride_h,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)
            comb = tl.load(comb_row + in_idx * comb_stride_i + rows * comb_stride_j).to(tl.float32)
            acc += comb[:, None] * res[None, :]
        x_tile = tl.load(x_row + offsets * x_stride_h, mask=col_mask, other=0.0).to(tl.float32)
        post = tl.load(post_row + rows * post_stride_m).to(tl.float32)
        acc += post[:, None] * x_tile[None, :]

        tl.store(
            out_row + rows[:, None] * hidden_size + offsets[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=col_mask[None, :],
        )


def mhc_post_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hc_mult, hidden_size = residual.shape
    out = torch.empty(num_tokens, hc_mult, hidden_size, dtype=residual.dtype, device=residual.device)
    if num_tokens == 0 or hidden_size == 0 or hc_mult == 0:
        return out

    block_h = min(hidden_size, _mhc_post_tile_cap(hc_mult))
    if block_h < hidden_size and hidden_size % block_h:
        # An exact split beats one maximal tile plus a mostly-masked tail.
        divisor = _largest_divisor_at_most(hidden_size, block_h)
        if divisor > 1:
            block_h = divisor
    num_h_blocks = (hidden_size + block_h - 1) // block_h
    grid = (min(get_num_cores("vector"), num_tokens * num_h_blocks),)

    _mhc_post_kernel[grid](
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        out,
        x.stride(0),
        x.stride(1),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        post_layer_mix.stride(0),
        post_layer_mix.stride(1),
        comb_res_mix.stride(0),
        comb_res_mix.stride(1),
        comb_res_mix.stride(2),
        num_tokens,
        hidden_size,
        num_h_blocks,
        HC_MULT=hc_mult,
        BLOCK_H=block_h,
    )
    return out


@torch.library.custom_op("mojo_npu_triton_a5::mhc_post_fwd", mutates_args=())
def mhc_post_fwd(
    x: torch.Tensor, residual: torch.Tensor, post_layer_mix: torch.Tensor, comb_res_mix: torch.Tensor
) -> torch.Tensor:
    return mhc_post_impl(x, residual, post_layer_mix, comb_res_mix)


@mhc_post_fwd.register_fake
def _mhc_post_fwd_fake(
    x: torch.Tensor, residual: torch.Tensor, post_layer_mix: torch.Tensor, comb_res_mix: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(residual, memory_format=torch.contiguous_format)
