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
# GELU tanh approximation (same as npu/gelu.py), row-wise launch like ilu/silu.py.

import torch
import triton
import triton.language as tl

from .silu import calculate_settings
from .utils import libentry


@triton.jit
def _tanh(x):
    # tl.tanh is missing on some Triton builds; tanh(x) == 2 * sigmoid(2x) - 1
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def gelu_tanh_approx(x):
    sqrt_2_over_pi = 0.7978845608028654
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    return 0.5 * x * (1 + _tanh(tanh_arg))


@libentry()
@triton.jit
def _gelu_fwd_kernel_v1(x_ptr, y_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)
    x_ptr += program_id * stride
    y_ptr += program_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x_row = tl.load(x_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    y_row = gelu_tanh_approx(x_row)
    tl.store(y_ptr + col_offsets, y_row, mask=mask)


@libentry()
@triton.jit
def _gelu_bwd_kernel_v1(dy_ptr, x_ptr, dx_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)
    dy_ptr += program_id * stride
    x_ptr += program_id * stride
    dx_ptr += program_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dy_row = tl.load(dy_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    x_row = tl.load(x_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    sqrt_2_over_pi = 0.7978845608028654
    x_cubed = x_row * x_row * x_row
    tanh_arg = sqrt_2_over_pi * (x_row + 0.044715 * x_cubed)
    tanh_result = _tanh(tanh_arg)
    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = 0.5 * x_row * (1 - tanh_sq) * (sqrt_2_over_pi * (1 + 3 * 0.044715 * x_row * x_row))
    dgelu_dx = term1 + term2
    dx_row = dy_row * dgelu_dx
    tl.store(dx_ptr + col_offsets, dx_row, mask=mask)


def gelu_fwd_impl(x: torch.Tensor) -> torch.Tensor:
    ori_shape = x.shape
    n_cols = ori_shape[-1]
    x_2d = x.reshape(-1, n_cols).contiguous()
    n_rows = x_2d.shape[0]
    y = torch.empty_like(x_2d)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    _gelu_fwd_kernel_v1[(n_rows,)](
        x_2d,
        y,
        x_2d.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y.reshape(*ori_shape)


def gelu_bwd_impl(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ori_shape = dy.shape
    n_cols = ori_shape[-1]
    dy_2d = dy.reshape(-1, n_cols).contiguous()
    x_2d = x.reshape(-1, n_cols).contiguous()
    n_rows = dy_2d.shape[0]
    dx = torch.empty_like(x_2d)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    _gelu_bwd_kernel_v1[(n_rows,)](
        dy_2d,
        x_2d,
        dx,
        dx.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return dx.reshape(*ori_shape)
