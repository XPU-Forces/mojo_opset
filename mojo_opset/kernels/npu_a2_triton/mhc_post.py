# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted for Ascend NPU by the mojo_opset contributors.
"""mHC post block: out[t, j, h] = post[t, j] * x[t, h] + sum_i comb[t, i, j] * res[t, i, h].

FP32 accumulation over a resident [hc_mult, BLOCK_H] stream block, rows
sliced per fused broadcast multiply-add; the store rounds once to the
residual dtype. When num_tokens >= num cores the grid splits only along
tokens (balanced contiguous ranges) and each core walks the consecutive
h_blocks of its tokens; below that threshold the flattened
(token, h_block) task list fills the machine.
"""

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores
from mojo_opset.kernels._npu_triton_utils import libentry
from mojo_opset.kernels._triton_utils import input_guard

_UB_BYTES_A2 = 196608
_MHC_POST_SPLIT_FLOOR = 256
_MHC_POST_ALIGN_D = 16


def _ceil_align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _mhc_post_block_h(hc_mult, hidden_size, num_tokens, num_cores):
    """Resident-budget tile cap plus fill-core halving for small batches."""
    max_block = _UB_BYTES_A2 // (8 * (hc_mult + 2))
    d_outer = 1
    block = hidden_size
    while (num_tokens * d_outer <= num_cores // 2 or block > max_block) and block > _MHC_POST_SPLIT_FLOOR:
        d_outer *= 2
        block = hidden_size // d_outer
    return min(hidden_size, max(_MHC_POST_SPLIT_FLOOR, _ceil_align(block, _MHC_POST_ALIGN_D)))


@triton.jit
def _mhc_post_tile(
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
    token_idx,
    block_idx,
    hidden_size,
    HC_MULT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_H)
    rows = tl.arange(0, HC_MULT)
    offsets = block_idx * BLOCK_H + cols
    col_mask = offsets < hidden_size

    x_row = x_ptr + token_idx * x_stride_t
    res_row = res_ptr + token_idx * res_stride_t
    post_row = post_ptr + token_idx * post_stride_t
    comb_row = comb_ptr + token_idx * comb_stride_t
    out_row = out_ptr + token_idx * HC_MULT * hidden_size

    hout = tl.load(x_row + offsets * x_stride_h, mask=col_mask, other=0.0).to(tl.float32)
    streams = tl.load(
        res_row + rows[:, None] * res_stride_m + offsets[None, :] * res_stride_h,
        mask=col_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    for out_idx in tl.range(HC_MULT):
        post = tl.load(post_row + out_idx * post_stride_m).to(tl.float32)
        acc = hout[None, :] * post
        for in_idx in tl.static_range(HC_MULT):
            comb = tl.load(comb_row + in_idx * comb_stride_i + out_idx * comb_stride_j).to(tl.float32)
            stream = tl.extra.cann.extension.extract_slice(streams, [in_idx, 0], [1, BLOCK_H], [1, 1])
            acc += stream * comb
        tl.store(
            out_row + out_idx * hidden_size + offsets[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=col_mask[None, :],
        )


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
    TOKEN_MAJOR: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    if TOKEN_MAJOR:
        # Token-only split: core p takes base + (p < extra) tokens.
        base = num_tokens // grid_size
        extra = num_tokens % grid_size
        token_start = pid * base + tl.minimum(pid, extra)
        token_count = base + tl.where(pid < extra, 1, 0)
        for token in range(token_start, token_start + token_count):
            token_idx = token.to(tl.int64)
            for block_idx in range(0, num_h_blocks):
                _mhc_post_tile(
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
                    token_idx,
                    block_idx,
                    hidden_size,
                    HC_MULT=HC_MULT,
                    BLOCK_H=BLOCK_H,
                )
    else:
        # Flattened (token, h_block) tasks fill all cores.
        total_tasks = num_tokens * num_h_blocks
        tasks_per_core = (total_tasks + grid_size - 1) // grid_size
        task_start = pid * tasks_per_core
        task_end = tl.minimum(task_start + tasks_per_core, total_tasks)
        for task_id in range(task_start, task_end):
            token_idx = (task_id // num_h_blocks).to(tl.int64)
            block_idx = task_id % num_h_blocks
            _mhc_post_tile(
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
                token_idx,
                block_idx,
                hidden_size,
                HC_MULT=HC_MULT,
                BLOCK_H=BLOCK_H,
            )


@input_guard()
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

    num_cores = get_num_cores("vector")
    block_h = _mhc_post_block_h(hc_mult, hidden_size, num_tokens, num_cores)
    num_h_blocks = (hidden_size + block_h - 1) // block_h
    token_major = num_tokens >= num_cores
    grid = (min(num_cores, num_tokens if token_major else num_tokens * num_h_blocks),)

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
        TOKEN_MAJOR=token_major,
    )
    return out


@torch.library.custom_op("mojo_npu_triton_a2::mhc_post_fwd", mutates_args=())
def mhc_post_fwd(
    x: torch.Tensor, residual: torch.Tensor, post_layer_mix: torch.Tensor, comb_res_mix: torch.Tensor
) -> torch.Tensor:
    return mhc_post_impl(x, residual, post_layer_mix, comb_res_mix)


@mhc_post_fwd.register_fake
def _mhc_post_fwd_fake(
    x: torch.Tensor, residual: torch.Tensor, post_layer_mix: torch.Tensor, comb_res_mix: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(residual, memory_format=torch.contiguous_format)
