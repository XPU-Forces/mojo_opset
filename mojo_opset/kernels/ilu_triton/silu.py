# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Row-wise Triton SiLU (same launch pattern as ilu/swiglu.py v1).

import torch
import triton
import triton.language as tl

from .utils import libentry


def calculate_settings(n):
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


@libentry()
@triton.jit
def _silu_fwd_kernel_v1(x_ptr, y_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)
    x_ptr += program_id * stride
    y_ptr += program_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x_row = tl.load(x_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    y_row = x_row * tl.sigmoid(x_row)
    tl.store(y_ptr + col_offsets, y_row, mask=mask)


@libentry()
@triton.jit
def _silu_bwd_kernel_v1(dy_ptr, x_ptr, dx_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)
    dy_ptr += program_id * stride
    x_ptr += program_id * stride
    dx_ptr += program_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dy_row = tl.load(dy_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    x_row = tl.load(x_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    sig_x = tl.sigmoid(x_row)
    dsilu_dx = sig_x * (1.0 + x_row * (1.0 - sig_x))
    dx_row = dy_row * dsilu_dx
    tl.store(dx_ptr + col_offsets, dx_row, mask=mask)


def silu_fwd_impl(x: torch.Tensor) -> torch.Tensor:
    ori_shape = x.shape
    n_cols = ori_shape[-1]
    x_2d = x.reshape(-1, n_cols).contiguous()
    n_rows = x_2d.shape[0]
    y = torch.empty_like(x_2d)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    _silu_fwd_kernel_v1[(n_rows,)](
        x_2d,
        y,
        x_2d.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y.reshape(*ori_shape)


def silu_bwd_impl(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ori_shape = dy.shape
    n_cols = ori_shape[-1]
    dy_2d = dy.reshape(-1, n_cols).contiguous()
    x_2d = x.reshape(-1, n_cols).contiguous()
    n_rows = dy_2d.shape[0]
    dx = torch.empty_like(x_2d)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    _silu_bwd_kernel_v1[(n_rows,)](
        dy_2d,
        x_2d,
        dx,
        dx.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return dx.reshape(*ori_shape)
