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
# Kernel adapted from trike/srcs/swiglu/swiglu.py (v1 row kernel).
# Ref: _swiglu_forward_kernel_v1:
# https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py#L15

import torch
import triton
import triton.language as tl

from .silu import calculate_settings
from .utils import libentry


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


@libentry()
@triton.jit
def _swiglu_forward_kernel_v1(
    a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    c_row = _silu(a_row) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


def swiglu_fwd_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Forward SwiGLU: silu(a) * b. Matches MojoSwiGLU(gate_out, up_out).
    """
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a_2d = a.reshape(-1, n_cols).contiguous()
    b_2d = b.reshape(-1, n_cols).contiguous()
    n_rows = a_2d.shape[0]
    if b_2d.shape != a_2d.shape:
        raise ValueError(f"SwiGLU inputs must match shape; got {a_2d.shape} vs {b_2d.shape}")

    o = torch.empty_like(a_2d)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel_v1[(n_rows,)](
        a_2d,
        b_2d,
        o,
        o.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return o.reshape(*ori_shape)
