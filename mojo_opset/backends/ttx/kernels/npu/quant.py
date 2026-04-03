import torch
import triton
import triton.language as tl

from mojo_opset.backends.ttx.kernels.npu.utils import get_num_cores


def dynamic_quant_impl(
    input_tensor: torch.Tensor,
    scale_tensor: torch.Tensor,
):
    num_programs = get_num_cores()

    grid = (num_programs,)
    dims = input_tensor.shape[-1]

    device = input_tensor.device
    output_tensor = torch.empty_like(input_tensor, dtype=torch.int8)

    quant_scale_tensor = torch.empty(*input_tensor.shape[:-1], device=device, dtype=torch.float32)
    align_dims = triton.next_power_of_2(dims)

    # Flatten all leading dimensions into one: [B, S, N, D] or [B, S, D] -> [T, D]
    total_tokens = input_tensor.numel() // dims
    input_2d = input_tensor.view(-1, dims)
    output_2d = output_tensor.view(-1, dims)
    quant_scale_1d = quant_scale_tensor.view(-1)

    scale_dynamic_quant_kernel[grid](
        input_2d,
        scale_tensor,
        output_2d,
        quant_scale_1d,
        total_tokens=total_tokens,
        dims=dims,
        align_dims=align_dims,
        BLOCK_SIZE_N=256,
    )

    return output_tensor, quant_scale_tensor


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 8, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 16, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 32, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 64, "multibuffer": True}),
        triton.Config({"BLOCK_SIZE_M": 8, "multibuffer": False}),
        triton.Config({"BLOCK_SIZE_M": 16, "multibuffer": False}),
        triton.Config({"BLOCK_SIZE_M": 32, "multibuffer": False}),
        triton.Config({"BLOCK_SIZE_M": 64, "multibuffer": False}),
    ],
    key=["dims"],
)
@triton.jit
def scale_dynamic_quant_kernel(
    input,
    scale,
    output,
    quant_scale,
    total_tokens: tl.constexpr,
    dims: tl.constexpr,
    align_dims: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Dynamic quantization kernel for N-dim tensors, quantizing only the last dimension.

    Args:
        input: Pointer to input tensor, flattened to [total_tokens, dims]
        scale: Pointer to scale tensor with shape [dims]
        output: Pointer to output tensor (int8), flattened to [total_tokens, dims]
        quant_scale: Pointer to quantization scale tensor, flattened to [total_tokens]
        total_tokens: Total number of tokens (product of all leading dimensions)
        dims: Number of columns (last dimension size)
        align_dims: Aligned column size (power of 2)
        BLOCK_SIZE_M: Block size for M dimension (token dimension)
        BLOCK_SIZE_N: Block size for N dimension (dims)

    Memory layout: [..., dims] where ... represents any number of leading dimensions
    Scale shape: [dims] (broadcast to all tokens)
    Output shape: [..., dims] (int8)
    Quantization scale shape: [...] (one scale per token)
    """

    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_tasks = (total_tokens + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for task_id in range(pid, num_tasks, grid_size):
        block_start = task_id * BLOCK_SIZE_M
        element_off = block_start + tl.arange(0, BLOCK_SIZE_M)

        element_mask = element_off < total_tokens
        max_abs_accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for col_block_offset in range(0, align_dims, BLOCK_SIZE_N):
            dims_off = col_block_offset + tl.arange(0, BLOCK_SIZE_N)
            dims_mask = dims_off < dims

            # Memory layout: [total_tokens, dims]
            input_offset = element_off[:, None] * dims + dims_off[None, :]

            input_ptr = input + input_offset
            scale_ptr = scale + dims_off

            block_mask = element_mask[:, None] & dims_mask[None, :]
            input_vals = tl.load(input_ptr, mask=block_mask, other=0.0).to(tl.float32)
            scale_vals = tl.load(scale_ptr, mask=dims_mask, other=0.0).to(tl.float32)
            scaled_vals = input_vals * scale_vals

            current_max = tl.max(tl.abs(scaled_vals), axis=1)

            max_abs_accumulator = tl.maximum(max_abs_accumulator, current_max)

        final_max_abs = max_abs_accumulator
        current_quant_scale = final_max_abs / 127.0
        quant_scale_ptr = quant_scale + element_off
        tl.store(quant_scale_ptr, current_quant_scale, mask=element_mask)

        for col_block_offset in range(0, align_dims, BLOCK_SIZE_N):
            dims_off = col_block_offset + tl.arange(0, BLOCK_SIZE_N)
            dims_mask = dims_off < dims

            # Calculate input and output pointer offsets
            input_offset = element_off[:, None] * dims + dims_off[None, :]

            input_ptr = input + input_offset
            output_ptr = output + input_offset
            scale_ptr = scale + dims_off

            block_mask = element_mask[:, None] & dims_mask[None, :]

            input_vals = tl.load(input_ptr, mask=block_mask, other=0.0).to(tl.float32)
            scale_vals = tl.load(scale_ptr, mask=dims_mask, other=0.0).to(tl.float32)

            # Apply scaling and quantization
            scaled_vals = input_vals * scale_vals
            quant_vals = scaled_vals / current_quant_scale[:, None]
            quant_vals = tl.where(quant_vals < 0, quant_vals - 0.5, quant_vals + 0.5)
            quant_vals_int8 = tl.cast(quant_vals, dtype=tl.int8, overflow_mode="saturate")

            tl.store(output_ptr, quant_vals_int8, mask=block_mask)
