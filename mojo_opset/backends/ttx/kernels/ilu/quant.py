from typing import Optional

import torch
import triton
import triton.language as tl

from .utils import ilu_grid_dim_from_row_tasks, libentry


def dynamic_quant_impl(
    input_tensor: torch.Tensor,
    scale_tensor: Optional[torch.Tensor],
):
    dims = input_tensor.shape[-1]
    input_2d = input_tensor.reshape(-1, dims).contiguous()
    output_2d = torch.empty_like(input_2d, dtype=torch.int8)
    quant_scale = torch.empty(input_2d.shape[0], device=input_tensor.device, dtype=torch.float32)
    has_scale = scale_tensor is not None
    if scale_tensor is None:
        scale_tensor = torch.empty(0, device=input_tensor.device, dtype=torch.float32)

    block_size = triton.next_power_of_2(dims)
    n_rows = input_2d.shape[0]
    grid = lambda META: (ilu_grid_dim_from_row_tasks(n_rows),)
    _dynamic_quant_kernel[grid](
        input_2d,
        scale_tensor,
        output_2d,
        quant_scale,
        input_2d.stride(0),
        output_2d.stride(0),
        n_rows,
        dims,
        HAS_SCALE=has_scale,
        BLOCK_SIZE=block_size,
    )

    output = output_2d.reshape(*input_tensor.shape)
    return output, quant_scale.reshape(*input_tensor.shape[:-1], 1)


@libentry()
@triton.jit
def _dynamic_quant_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    quant_scale_ptr,
    input_stride,
    output_stride,
    n_rows,
    n_cols: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)
    if row_id >= n_rows:
        return
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    input_row = tl.load(input_ptr + row_id * input_stride + offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_SCALE:
        smooth_scale = tl.load(scale_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        input_row = input_row * smooth_scale

    max_abs = tl.max(tl.abs(input_row), axis=0)
    scale = max_abs / 127.0
    scale = tl.where(scale < 1e-6, 1.0, scale)
    tl.store(quant_scale_ptr + row_id, scale)

    quant = input_row / scale
    quant = tl.where(quant < 0, quant - 0.5, quant + 0.5)
    quant = tl.cast(quant, dtype=tl.int8, overflow_mode="saturate")
    tl.store(output_ptr + row_id * output_stride + offsets, quant, mask=mask)
