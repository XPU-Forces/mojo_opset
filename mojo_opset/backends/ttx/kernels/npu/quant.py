import torch
import triton
import triton.language as tl


def quant_int8_infer_impl(
    input_tensor: torch.Tensor,
    scale_tensor: torch.Tensor,
):
    try:
        import triton.runtime.driver as driver

        num_programs = driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"]
    except AttributeError:
        num_programs = 48

    grid = (num_programs,)
    dims = input_tensor.shape[-1]

    device = input_tensor.device
    output_tensor = torch.empty_like(input_tensor, dtype=torch.int8)

    quant_scale_tensor = torch.empty(*input_tensor.shape[:-1], device=device, dtype=torch.float32)
    # quant_scale_tensor = torch.empty(batch, seqlen, device=device, dtype=torch.float32)
    align_dims = triton.next_power_of_2(dims)

    # TODO merge 3d and 4d quant
    if input_tensor.ndim == 3:
        batch, seqlen, _ = input_tensor.shape
        scale_dynamic_quant_kernel_3d[grid](
            input_tensor,
            scale_tensor,
            output_tensor,
            quant_scale_tensor.view(-1),  # Flatten for easier indexing
            batch=batch,
            seqlen=seqlen,
            dims=dims,
            align_dims=align_dims,
            BLOCK_SIZE_N=256,
        )
    elif input_tensor.ndim == 4:
        batch, seqlen, num_head, _ = input_tensor.shape
        scale_dynamic_quant_kernel_4d[grid](
            input_tensor,
            scale_tensor,
            output_tensor,
            quant_scale_tensor.view(-1),  # Flatten for easier indexing
            batch=batch,
            seqlen=seqlen,
            num_head=num_head,
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
def scale_dynamic_quant_kernel_3d(
    input,
    scale,
    output,
    quant_scale,
    batch: tl.constexpr,
    seqlen: tl.constexpr,
    dims: tl.constexpr,
    align_dims: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    3D dynamic quantization kernel

    Args:
        input: Pointer to input tensor with shape [batch, seqlen, dims]
        scale: Pointer to scale tensor with shape [dims]
        output: Pointer to output tensor with shape [batch, seqlen, dims] (int8)
        quant_scale: Pointer to quantization scale tensor with shape [batch, seqlen]
        batch: Number of batches
        seqlen: Number of seqlen per batch
        dims: Number of columns
        align_dims: Aligned column size (power of 2)
        BLOCK_SIZE_M: Block size for M dimension (seqlen)
        BLOCK_SIZE_N: Block size for N dimension (dims)

    Memory layout: [batch, seqlen, dims]
    Scale shape: [dims] (broadcast to all batches and seqlen)
    Output shape: [batch, seqlen, dims] (int8)
    Quantization scale shape: [batch, seqlen]
    """

    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    total_elements = batch * seqlen
    num_tasks = (total_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for task_id in range(pid, num_tasks, grid_size):
        block_start = task_id * BLOCK_SIZE_M
        element_off = block_start + tl.arange(0, BLOCK_SIZE_M)

        batch_idx = element_off // seqlen
        row_idx = element_off % seqlen

        element_mask = element_off < total_elements
        max_abs_accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for col_block_offset in range(0, align_dims, BLOCK_SIZE_N):
            dims_off = col_block_offset + tl.arange(0, BLOCK_SIZE_N)
            dims_mask = dims_off < dims

            # Memory layout: [batch, seqlen, dims]
            input_offset = batch_idx[:, None] * seqlen * dims + row_idx[:, None] * dims + dims_off[None, :]

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
            input_offset = batch_idx[:, None] * seqlen * dims + row_idx[:, None] * dims + dims_off[None, :]

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


# quant
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
    key=["cols"],
)
@triton.jit
def scale_dynamic_quant_kernel_4d(
    input,
    scale,
    output,
    quant_scale,
    batch: tl.constexpr,
    seqlen: tl.constexpr,
    num_head: tl.constexpr,
    dims: tl.constexpr,
    align_dims: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    4D dynamic quantization kernel

    Args:
        input: Pointer to input tensor with shape [batch, seqlen, num_head, dims]
        scale: Pointer to scale tensor with shape [dims]
        output: Pointer to output tensor with shape [batch, seqlen, num_head, dims] (int8)
        quant_scale: Pointer to quantization scale tensor with shape [batch, seqlen, num_head]
        batch: Number of batches
        seqlen: Number of seqlen per batch
        dims: Number of columns
        align_dims: Aligned column size (power of 2)
        BLOCK_SIZE_M: Block size for M dimension (seqlen)
        BLOCK_SIZE_N: Block size for N dimension (dims)

    Memory layout: [batch, seqlen, num_head, dims]
    Scale shape: [dims] (broadcast to all batches and seqlen)
    Output shape: [batch, seqlen, num_head, dims] (int8)
    Quantization scale shape: [batch, seqlen, num_head]
    """

    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    total_elements = batch * seqlen * num_head
    num_tasks = (total_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for task_id in range(pid, num_tasks, grid_size):
        block_start = task_id * BLOCK_SIZE_M
        element_off = block_start + tl.arange(0, BLOCK_SIZE_M)

        batch_idx = element_off // (seqlen * num_head)
        seq_idx = (element_off // num_head) % seqlen
        row_idx = element_off % num_head

        element_mask = element_off < total_elements

        max_abs_accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for col_block_offset in range(0, align_dims, BLOCK_SIZE_N):
            cols_off = col_block_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < dims

            # get input position
            # mem layout: [batch, seqlen, rows, cols]
            input_offset = batch_idx[:, None] * seqlen * num_head * dims + seq_idx[:, None] * num_head * dims + row_idx[:, None] * dims + cols_off[None, :]

            input_ptr = input + input_offset
            scale_ptr = scale + cols_off

            block_mask = element_mask[:, None] & cols_mask[None, :]

            input_vals = tl.load(input_ptr, mask=block_mask, other=0.0).to(tl.float32)
            scale_vals = tl.load(scale_ptr, mask=cols_mask, other=0.0).to(tl.float32)

            scaled_vals = input_vals * scale_vals
            current_max = tl.max(tl.abs(scaled_vals), axis=1)
            max_abs_accumulator = tl.maximum(max_abs_accumulator, current_max)

        final_max_abs = max_abs_accumulator
        current_quant_scale = final_max_abs / 127.0

        quant_scale_ptr = quant_scale + element_off
        tl.store(quant_scale_ptr, current_quant_scale, mask=element_mask)

        for col_block_offset in range(0, align_dims, BLOCK_SIZE_N):
            cols_off = col_block_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < dims

            input_offset = batch_idx[:, None] * seqlen * num_head * dims + seq_idx[:, None] * num_head * dims + row_idx[:, None] * dims + cols_off[None, :]

            input_ptr = input + input_offset
            output_ptr = output + input_offset
            scale_ptr = scale + cols_off

            block_mask = element_mask[:, None] & cols_mask[None, :]

            input_vals = tl.load(input_ptr, mask=block_mask, other=0.0).to(tl.float32)
            scale_vals = tl.load(scale_ptr, mask=cols_mask, other=0.0).to(tl.float32)

            scaled_vals = input_vals * scale_vals
            quant_vals = scaled_vals / current_quant_scale[:, None]
            quant_vals = tl.where(quant_vals < 0, quant_vals - 0.5, quant_vals + 0.5)
            # rounded_vals = quant_vals + 0.5
            # quant_vals_int8 = quant_vals.to(tl.int8)
            quant_vals_int8 = tl.cast(quant_vals, dtype=tl.int8, overflow_mode="saturate")

            tl.store(output_ptr, quant_vals_int8, mask=block_mask)
