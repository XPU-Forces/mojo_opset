from typing import Optional, Tuple

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


def dequant_impl(
    input_tensor: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize a quantized tensor using a per-channel scale.

    Computes ``output[..., c] = input[..., c].to(float32) * scale[c]``
    and stores the result in ``output_dtype`` (same convention as ``MojoDequant``).

    Args:
        input_tensor: Quantized input tensor of shape ``(..., K)``.
        scale: Per-channel scale tensor whose flattened length equals ``K``
            (the last dimension / number of columns of ``input_tensor``).
        output_dtype: Target floating-point dtype for the output.

    Returns:
        Dequantized tensor of the same shape as ``input_tensor`` in ``output_dtype``.
    """

    dims = input_tensor.shape[-1]
    scale_flat = scale.reshape(-1)
    if scale_flat.numel() != dims:
        raise ValueError(
            f"dequant scale must have one entry per channel: got scale.numel()={scale_flat.numel()}, "
            f"expected {dims} (input last dim)."
        )

    total_tokens = input_tensor.numel() // dims
    grid = (ilu_grid_dim_from_row_tasks(total_tokens),)

    output_tensor = torch.empty_like(input_tensor, dtype=output_dtype)
    align_dims = triton.next_power_of_2(dims)

    input_2d = input_tensor.view(-1, dims)
    output_2d = output_tensor.view(-1, dims)
    scale_channels = scale_flat.contiguous()

    dequant_kernel[grid](
        input_2d,
        scale_channels,
        output_2d,
        total_tokens=total_tokens,
        dims=dims,
        align_dims=align_dims,
        BLOCK_SIZE_N=256,
    )

    return output_tensor


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 8}),
        triton.Config({"BLOCK_SIZE_M": 16}),
        triton.Config({"BLOCK_SIZE_M": 32}),
        triton.Config({"BLOCK_SIZE_M": 64}),
    ],
    key=["dims"],
)
@triton.jit
def dequant_kernel(
    input,
    scale,
    output,
    total_tokens: tl.constexpr,
    dims: tl.constexpr,
    align_dims: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Per-channel dequantization kernel.

    For each token ``t`` and column ``c``:
        ``output[t, c] = input[t, c] * scale[c]``

    Args:
        input: Pointer to the quantized input, flattened to ``[total_tokens, dims]``.
        scale: Pointer to the per-channel scale, contiguous layout ``[dims]``.
        output: Pointer to the dequantized output, flattened to ``[total_tokens, dims]``.
        total_tokens: Total number of tokens (product of all leading dimensions).
        dims: Number of columns (last dimension size).
        align_dims: ``dims`` rounded up to the next power of 2.
        BLOCK_SIZE_M: Number of tokens processed per program iteration.
        BLOCK_SIZE_N: Number of columns processed per inner loop iteration.
    """
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_tasks = (total_tokens + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for task_id in range(pid, num_tasks, grid_size):
        block_start = task_id * BLOCK_SIZE_M
        element_off = block_start + tl.arange(0, BLOCK_SIZE_M)
        element_mask = element_off < total_tokens

        for col_block_offset in tl.static_range(0, align_dims, BLOCK_SIZE_N):
            dims_off = col_block_offset + tl.arange(0, BLOCK_SIZE_N)
            dims_mask = dims_off < dims
            block_mask = (element_mask[:, None] & dims_mask[None, :])

            scale_vals = tl.load(scale + dims_off, mask=dims_mask, other=1.0)

            input_offset = element_off[:, None] * dims + dims_off[None, :]
            input_vals = tl.load(input + input_offset, mask=block_mask, other=0)

            output_vals = input_vals * scale_vals[None, :]

            tl.store(output + input_offset, output_vals, mask=block_mask)

from .utils import libentry


# Single-pass row fits in this many fp32 lanes. Above that we fall back to the
# 2-pass column-blocked kernel below. 32K covers all common hidden sizes
# (e.g. 4096 / 8192 / 16384) for an int8 quant kernel.
_MAX_SINGLE_PASS_BLOCK = 32768
# Column tile width used by the 2-pass fallback kernel.
_FALLBACK_COL_TILE = 2048


def _calculate_settings(n_cols: int) -> Tuple[int, int]:
    """Pick BLOCK_SIZE and num_warps for the single-pass kernel."""
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE >= 32768:
        num_warps = 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    else:
        num_warps = 4
    return BLOCK_SIZE, num_warps


@libentry()
@triton.jit
def _dynamic_quant_row_kernel(
    x_ptr,
    scale_ptr,
    y_ptr,
    qscale_ptr,
    stride_x_row,
    stride_y_row,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per token. Loads a whole row, computes per-row quant scale, stores int8."""
    pid = tl.program_id(axis=0).to(tl.int64)
    x_ptr = x_ptr + pid * stride_x_row
    y_ptr = y_ptr + pid * stride_y_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x = tl.load(x_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(scale_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    x_scaled = x * s
    abs_max = tl.max(tl.abs(x_scaled), axis=0)

    qscale = abs_max / 127.0
    inv_qscale = tl.where(qscale > 0.0, 1.0 / qscale, 0.0)

    q = x_scaled * inv_qscale
    # Round-half-away-from-zero, matching the NPU reference kernel.
    q = tl.where(q < 0.0, q - 0.5, q + 0.5)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    q_int8 = q.to(tl.int8)

    tl.store(y_ptr + col_offsets, q_int8, mask=mask)
    tl.store(qscale_ptr + pid, qscale)


@libentry()
@triton.jit
def _dynamic_quant_row_kernel_blocked(
    x_ptr,
    scale_ptr,
    y_ptr,
    qscale_ptr,
    stride_x_row,
    stride_y_row,
    n_cols: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fallback for very wide rows: 2-pass column-blocked.
        Pass 1: scan the row to find max(|x*s|).
        Pass 2: re-scan and write int8 with the now-known per-row qscale.
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    x_row_ptr = x_ptr + pid * stride_x_row
    y_row_ptr = y_ptr + pid * stride_y_row

    abs_max = tl.zeros((), dtype=tl.float32)
    for col_off in range(0, n_cols, BLOCK_SIZE_N):
        cols = col_off + tl.arange(0, BLOCK_SIZE_N)
        m = cols < n_cols
        x = tl.load(x_row_ptr + cols, mask=m, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + cols, mask=m, other=0.0).to(tl.float32)
        cur = tl.max(tl.abs(x * s), axis=0)
        abs_max = tl.maximum(abs_max, cur)

    qscale = abs_max / 127.0
    inv_qscale = tl.where(qscale > 0.0, 1.0 / qscale, 0.0)
    tl.store(qscale_ptr + pid, qscale)

    for col_off in range(0, n_cols, BLOCK_SIZE_N):
        cols = col_off + tl.arange(0, BLOCK_SIZE_N)
        m = cols < n_cols
        x = tl.load(x_row_ptr + cols, mask=m, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + cols, mask=m, other=0.0).to(tl.float32)
        q = (x * s) * inv_qscale
        q = tl.where(q < 0.0, q - 0.5, q + 0.5)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(y_row_ptr + cols, q.to(tl.int8), mask=m)


def dynamic_quant_impl(
    input_tensor: torch.Tensor,
    scale_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamic per-token int8 quantization with an optional per-channel pre-scale.

    Args:
        input_tensor: (..., D) float tensor.
        scale_tensor: (D,) float tensor; per-channel multiplier applied before
            computing the per-token max. Pass `torch.ones(D)` for plain dynamic quant.

    Returns:
        output_tensor: (..., D) int8.
        quant_scale  : (...,)  float32, such that output * quant_scale ≈ input.
    """
    assert input_tensor.shape[-1] == scale_tensor.shape[-1], (
        f"scale last-dim {scale_tensor.shape[-1]} must match input last-dim "
        f"{input_tensor.shape[-1]}"
    )

    ori_shape = input_tensor.shape
    n_cols = ori_shape[-1]

    x_2d = input_tensor.reshape(-1, n_cols).contiguous()
    n_rows = x_2d.shape[0]

    output = torch.empty_like(x_2d, dtype=torch.int8)
    quant_scale = torch.empty(n_rows, device=input_tensor.device, dtype=torch.float32)

    scale_1d = scale_tensor.reshape(-1).contiguous()

    grid = (n_rows,)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE <= _MAX_SINGLE_PASS_BLOCK:
        BLOCK_SIZE, num_warps = _calculate_settings(n_cols)
        _dynamic_quant_row_kernel[grid](
            x_2d,
            scale_1d,
            output,
            quant_scale,
            x_2d.stride(0),
            output.stride(0),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        _dynamic_quant_row_kernel_blocked[grid](
            x_2d,
            scale_1d,
            output,
            quant_scale,
            x_2d.stride(0),
            output.stride(0),
            n_cols=n_cols,
            BLOCK_SIZE_N=_FALLBACK_COL_TILE,
            num_warps=8,
        )

    return (
        output.reshape(*ori_shape),
        quant_scale.reshape(*ori_shape[:-1], 1),
    )
