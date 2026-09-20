"""torch_npu GEMM leaves; quantized reduction retains the original BF16 contract."""

import torch
import torch_npu


@torch.library.custom_op("mojo_npu_torch_npu::group_gemm", mutates_args=())
def group_gemm_fwd(
    input: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor, trans_weight: bool
) -> torch.Tensor:
    if input.dtype == torch.float32:
        raise NotImplementedError("NPU grouped matmul does not support float32")
    # The M-split vendor kernel rejects a transposed input layout.
    input = input.contiguous()
    if trans_weight:
        weight = weight.transpose(1, 2).contiguous()
    weights = [w.contiguous() for w in weight]
    ends = [int(x) for x in group_list.cumsum(0).tolist()]
    return torch.cat(
        torch_npu.npu_grouped_matmul(
            [input],
            weights,
            group_type=0,
            group_list=ends,
        ),
        dim=0,
    )


@group_gemm_fwd.register_fake
def _group_fake(input, weight, group_list, trans_weight):
    return input.new_empty((input.shape[0], weight.shape[1 if trans_weight else 2]))


@torch.library.custom_op("mojo_npu_torch_npu::quant_gemm", mutates_args=())
def quant_gemm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
    trans_weight: bool,
) -> torch.Tensor:
    if trans_weight:
        weight = weight.t().contiguous()
    kernel_dtype = output_dtype
    if weight_scale.dtype == torch.bfloat16 and output_dtype not in (torch.bfloat16, torch.int32):
        kernel_dtype = torch.bfloat16
    output = torch_npu.npu_quant_matmul(
        input,
        weight,
        weight_scale.flatten(),
        pertoken_scale=input_scale.flatten(),
        output_dtype=kernel_dtype,
    )
    return output.to(output_dtype)


@quant_gemm_fwd.register_fake
def _quant_fake(input, weight, input_scale, weight_scale, output_dtype, trans_weight):
    return input.new_empty((input.shape[0], weight.shape[0 if trans_weight else 1]), dtype=output_dtype)


@torch.library.custom_op("mojo_npu_torch_npu::quant_batch_gemm_reduce_sum", mutates_args=())
def _quant_batch_gemm_reduce_sum(
    input: torch.Tensor, weight: torch.Tensor, x1_scale: torch.Tensor, x2_scale: torch.Tensor, trans_weight: bool
) -> torch.Tensor:
    if trans_weight:
        weight = weight.transpose(1, 2).contiguous()
    if torch_npu.get_npu_format(weight) != 29:
        weight = torch_npu.npu_format_cast(weight, 29)
    return torch_npu.npu_quant_matmul_reduce_sum(
        input,
        weight,
        x1_scale=x1_scale,
        x2_scale=x2_scale.to(torch.bfloat16),
    )


@_quant_batch_gemm_reduce_sum.register_fake
def _quant_batch_fake(input, weight, x1_scale, x2_scale, trans_weight):
    return input.new_empty((input.shape[1], weight.shape[1 if trans_weight else 2]), dtype=torch.bfloat16)


def quant_batch_gemm_reduce_sum_fwd(
    input: torch.Tensor, weight: torch.Tensor, x1_scale: torch.Tensor, x2_scale: torch.Tensor, trans_weight: bool
) -> torch.Tensor:
    # Mojo master test_gemm.py/test_linear.py skipped this entire NPU matrix
    # for the CANN 8.2 issue, without a version or shape condition. Keep the
    # original launcher above for future porting, but do not expose it as usable.
    raise NotImplementedError(
        "torch_npu quant_batch_gemm_reduce_sum is disabled: the original master "
        "skipped all NPU cases due to the CANN 8.2 issue."
    )
