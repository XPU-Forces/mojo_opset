import torch
import triton
import triton.language as tl

from .utils import ilu_grid_dim_from_row_tasks


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

