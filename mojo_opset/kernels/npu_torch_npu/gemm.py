"""torch_npu GEMM leaves; quantized reduction retains the original BF16 contract."""

import os

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
    num_experts = weight.shape[0]
    if group_list.ndim != 1 or group_list.numel() != num_experts:
        raise ValueError("group_list must contain one count per expert")
    # Routing-produced device counts partition input; validate CPU metadata and
    # opt-in debug runs, without imposing a host sync on every NPU GEMM.
    device_counts = group_list.device.type in ("npu", "privateuseone")
    if num_experts <= 128 and device_counts and os.getenv("MOJO_NPU_VALIDATE_GROUP_COUNTS", "0") != "1":
        if input.shape[0] == 0:
            return input.new_empty((0, weight.shape[-1]))
        if num_experts == 0:
            raise ValueError("nonempty input requires at least one expert")
        weights = [weight.contiguous()] if num_experts == 1 else [w.contiguous() for w in weight.unbind(0)]
        outputs = torch_npu.npu_grouped_matmul(
            [input], weights,
            group_list=group_list.to(device=input.device, dtype=torch.int64).cumsum(0),
            split_item=2, group_type=0,
        )
        if len(outputs) != 1:
            raise RuntimeError("expected one M-split grouped matmul output")
        return outputs[0]
    counts = group_list.to(device="cpu", dtype=torch.int64).tolist()
    if any(size < 0 for size in counts) or sum(counts) != input.shape[0]:
        raise ValueError("group_list must contain nonnegative row counts summing to M")
    group_list = group_list.to(device=input.device, dtype=torch.int64)
    outputs = []
    token_start = 0
    # CANN accepts at most 128 weight tensors per grouped matmul.
    for start in range(0, len(counts), 128):
        end = min(start + 128, len(counts))
        token_end = token_start + sum(counts[start:end])
        if token_end != token_start:
            # With split_item=2, CANN treats a single weight as a stacked
            # [groups, K, N] tensor, including a one-group final chunk.
            chunk = weight[start:end]
            weights = [chunk.contiguous()] if end - start == 1 else [w.contiguous() for w in chunk.unbind(0)]
            outputs.extend(
                torch_npu.npu_grouped_matmul(
                    [input[token_start:token_end]],
                    weights,
                    group_list=group_list[start:end].cumsum(0),
                    split_item=2,
                    group_type=0,
                )
            )
        token_start = token_end
    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=0) if outputs else input.new_empty((0, weight.shape[-1]))


@group_gemm_fwd.register_fake
def _group_fake(input, weight, group_list, trans_weight):
    return input.new_empty((input.shape[0], weight.shape[1 if trans_weight else 2]))


@torch.library.custom_op("mojo_npu_torch_npu::group_gemm_wgrad", mutates_args=())
def _group_wgrad(dy: torch.Tensor, input: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    gradient = torch.zeros(
        (counts.numel(), input.shape[-1], dy.shape[-1]),
        device=input.device,
        dtype=torch.float32,
    )
    if input.shape[0] == 0:
        return gradient.to(input.dtype)
    torch_npu.npu_grouped_matmul_add_(
        gradient,
        input.contiguous(),
        dy.contiguous(),
        counts.to(device=input.device, dtype=torch.int64).cumsum(0),
        transpose_x=True,
        transpose_weight=False,
        group_type=2,
    )
    return gradient.to(input.dtype)


@_group_wgrad.register_fake
def _group_wgrad_fake(dy, input, counts):
    return input.new_empty((counts.numel(), input.shape[-1], dy.shape[-1]))


def group_gemm_bwd(dy, input, weight, group_list, trans_weight, need_input, need_weight):
    dx = group_gemm_fwd(dy.contiguous(), weight, group_list, not trans_weight) if need_input else None
    dw = _group_wgrad(dy, input, group_list) if need_weight else None
    if dw is not None and trans_weight:
        dw = dw.transpose(-1, -2).contiguous()
    return dx, dw


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
