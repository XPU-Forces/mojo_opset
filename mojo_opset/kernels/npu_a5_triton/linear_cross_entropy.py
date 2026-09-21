import os

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import get_num_cores


@triton.jit
def _accumulate_bf16_into_fp32_kernel(
    output_ptr,
    update_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    core_id = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    num_blocks = tl.cdiv(numel, BLOCK_SIZE)
    for block_id in tl.range(core_id, num_blocks, NUM_CORES):
        indices = block_id * BLOCK_SIZE + offsets
        mask = indices < numel
        output = tl.load(output_ptr + indices, mask=mask).to(tl.float32)
        update = tl.load(update_ptr + indices, mask=mask).to(tl.float32)
        tl.store(output_ptr + indices, output + update, mask=mask)


@triton.jit
def _fused_ce_zloss_logits_stats_fwd_kernel(
    logits_base_ptr,
    labels_base_ptr,
    losses_base_ptr,
    lse_base_ptr,
    lse_abs_base_ptr,
    z_losses_base_ptr,
    token_accuracy_base_ptr,
    z_loss_weight,
    ignore_index,
    M,
    V: tl.constexpr,
    stride_logits_m,
    BLOCK_V: tl.constexpr,
    RETURN_ACCURACY: tl.constexpr,
):
    for row in range(tl.program_id(0).to(tl.int64), M, 48):
        logits_ptr = logits_base_ptr + row * stride_logits_m
        label = tl.load(labels_base_ptr + row).to(tl.int64)
        valid_label = (label != ignore_index) & (label >= 0) & (label < V)
        safe_label = tl.where(valid_label, label, 0)

        max_ce = float("-inf")
        exp_sum_ce = 0.0
        max_abs = float("-inf")
        exp_sum_abs = 0.0
        argmax_idx = V

        for start_v in range(0, V, BLOCK_V):
            v_offs = start_v + tl.arange(0, BLOCK_V)
            mask = v_offs < V
            x = tl.load(logits_ptr + v_offs, mask=mask, other=0.0).to(tl.float32)

            block_max_ce = tl.max(tl.where(mask, x, float("-inf")), axis=0)
            max_ce = tl.maximum(max_ce, block_max_ce)

            abs_x = tl.abs(x)
            block_max_abs = tl.max(tl.where(mask, abs_x, float("-inf")), axis=0)
            max_abs = tl.maximum(max_abs, block_max_abs)

        tl.debug_barrier()

        for start_v in range(0, V, BLOCK_V):
            v_offs = start_v + tl.arange(0, BLOCK_V)
            mask = v_offs < V
            x = tl.load(logits_ptr + v_offs, mask=mask, other=0.0).to(tl.float32)
            exp_sum_ce += tl.sum(tl.where(mask, tl.exp(x - max_ce), 0.0), axis=0)

            abs_x = tl.abs(x)
            exp_sum_abs += tl.sum(tl.where(mask, tl.exp(abs_x - max_abs), 0.0), axis=0)

            if RETURN_ACCURACY:
                is_max_mask = mask & (x == max_ce)
                masked_offsets = tl.where(is_max_mask, v_offs, V)
                argmax_idx = tl.minimum(argmax_idx, tl.min(masked_offsets))

        target_logit = tl.load(logits_ptr + safe_label).to(tl.float32)
        lse_ce = max_ce + tl.log(exp_sum_ce)
        lse_abs = max_abs + tl.log(exp_sum_abs)
        ce_loss = lse_ce - target_logit
        z_loss = z_loss_weight * lse_abs * lse_abs

        tl.store(losses_base_ptr + row, tl.where(valid_label, ce_loss + z_loss, 0.0))
        tl.store(lse_base_ptr + row, lse_ce)
        tl.store(lse_abs_base_ptr + row, lse_abs)
        tl.store(z_losses_base_ptr + row, tl.where(valid_label, z_loss, 0.0))

        if RETURN_ACCURACY:
            is_correct = tl.where(valid_label & (argmax_idx == label), 1.0, 0.0)
            tl.store(token_accuracy_base_ptr + row, is_correct)


@triton.jit
def _cross_entropy_backward_kernel(
    logits_base_ptr,
    dloss_base_ptr,
    logsumexp_base_ptr,
    lse_abs_base_ptr,
    labels_base_ptr,
    dlogits_base_ptr,
    num_tokens,
    dloss_row_stride,
    LOGITS_ROW_STRIDE: tl.constexpr,
    DLOGITS_ROW_STRIDE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    Z_LOSS_WEIGHT: tl.constexpr,
):
    n_blocks = tl.cdiv(VOCAB_SIZE, BLOCK_SIZE)
    for tid in range(tl.program_id(0), num_tokens * n_blocks, 48):
        row_idx = tid // n_blocks
        block_idx = tid % n_blocks

        logits_ptr = logits_base_ptr + row_idx * LOGITS_ROW_STRIDE
        dlogits_ptr = dlogits_base_ptr + row_idx * DLOGITS_ROW_STRIDE
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < VOCAB_SIZE

        label_idx = tl.load(labels_base_ptr + row_idx).to(tl.int32)
        valid_label = (label_idx >= 0) & (label_idx < VOCAB_SIZE)
        dloss = tl.where(valid_label, tl.load(dloss_base_ptr + row_idx * dloss_row_stride), 0.0)
        x = tl.load(logits_ptr + col_offsets, mask=mask, other=0.0)

        logsumexp = tl.load(logsumexp_base_ptr + row_idx)
        probs = tl.exp(x.to(tl.float32) - logsumexp)
        dlogits = tl.where(col_offsets == label_idx, probs - 1.0, probs) * dloss

        if Z_LOSS_WEIGHT != 0.0:
            lse_abs = tl.load(lse_abs_base_ptr + row_idx)
            softmax_abs = tl.exp(tl.abs(x) - lse_abs)
            sign_x = tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))
            dlogits += dloss * (2.0 * Z_LOSS_WEIGHT * lse_abs * softmax_abs * sign_x)

        tl.store(dlogits_ptr + col_offsets, dlogits, mask=mask)


# Function-layer wrapper implementation.
def _pick_power2_block(vocab_size: int, default: int, env_name: str) -> int:
    requested = int(os.getenv(env_name, str(default)))
    requested = max(requested, 1)
    return min(triton.next_power_of_2(requested), triton.next_power_of_2(vocab_size))


def _chunked_loss_chunk_size() -> int:
    return int(
        os.getenv(
            "MARIANA_CHUNKED_LOSS_CHUNK_SIZE",
            os.getenv("SEED_KERNELS_CHUNKED_LOSS_CHUNK_SIZE", "2048"),
        )
    )


def _use_true_fp32_matmul() -> bool:
    value = os.getenv("MOJO_CHUNKED_LOSS_FP32_GEMM_OUTPUT", "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def _matmul_fp32(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    try:
        return torch.mm(a, b, out_dtype=torch.float32, out=out)
    except TypeError:
        # torch-npu 2.7 has no out_dtype argument. Compute on the exact FP32
        # representation of the BF16 operands instead of widening a BF16 result.
        if _use_true_fp32_matmul():
            return torch.mm(a.float(), b.float(), out=out)
        result = torch.mm(a, b).float()
        if out is not None:
            out.copy_(result)
            return out
    return result


def _matmul_logits(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _use_true_fp32_matmul():
        return _matmul_fp32(a, b)
    return torch.mm(a, b)


def _accumulate_bf16_into_fp32(output: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    assert output.device.type == "npu" and update.device == output.device
    assert output.dtype == torch.float32 and update.dtype == torch.bfloat16
    assert output.shape == update.shape and output.is_contiguous() and update.is_contiguous()
    if output.numel() == 0:
        return output
    num_cores = get_num_cores("vector")
    _accumulate_bf16_into_fp32_kernel[(num_cores,)](
        output,
        update,
        output.numel(),
        BLOCK_SIZE=4096,
        NUM_CORES=num_cores,
        num_warps=8,
    )
    return output


def _addmm_fp32(input_: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    try:
        return torch.addmm(input=input_, mat1=mat1, mat2=mat2, out_dtype=torch.float32, out=out)
    except TypeError:
        assert input_.data_ptr() == out.data_ptr()
        if _use_true_fp32_matmul():
            return torch.addmm(input=input_, mat1=mat1.float(), mat2=mat2.float(), out=out)
        update = torch.mm(mat1, mat2)
        if update.dtype == torch.float32:
            out.add_(update)
            return out
        return _accumulate_bf16_into_fp32(out, update)


def _pick_block_v(vocab_size: int) -> int:
    if vocab_size <= 1024:
        default = 1024
    elif vocab_size <= 2048:
        default = 2048
    else:
        default = 4096
    return _pick_power2_block(vocab_size, default, "SEED_KERNELS_CE_FWD_BLOCK_V")


def _fused_ce_zloss_logits_stats_fwd(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss: torch.Tensor,
    zloss: torch.Tensor,
    token_accuracy: torch.Tensor,
    ignore_index: int,
    z_loss_weight: float = 0.0,
    calc_acc: bool = False,
    lse: torch.Tensor = None,
    lse_abs: torch.Tensor = None,
):
    M, V = logits.shape
    if lse is None:
        lse = torch.empty(M, dtype=torch.float32, device=logits.device)
    if lse_abs is None:
        lse_abs = torch.empty(M, dtype=torch.float32, device=logits.device)
    num_cores = get_num_cores("vector")
    _fused_ce_zloss_logits_stats_fwd_kernel[(num_cores,)](
        logits,
        labels,
        loss,
        lse,
        lse_abs,
        zloss,
        token_accuracy,
        z_loss_weight,
        ignore_index,
        M,
        V,
        logits.stride(0),
        BLOCK_V=_pick_block_v(V),
        RETURN_ACCURACY=calc_acc,
        num_warps=8,
    )
    return lse, lse_abs


def _cross_entropy_backward(
    dloss: torch.Tensor,
    logits: torch.Tensor,
    lse: torch.Tensor,
    labels: torch.Tensor,
    z_loss_weight: float,
    lse_abs: torch.Tensor,
    dlogits_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    num_tokens, vocab_size = logits.shape
    dlogits = torch.empty(num_tokens, vocab_size, device=logits.device, dtype=dlogits_dtype)
    block_size = _pick_power2_block(vocab_size, 4096, "SEED_KERNELS_CE_BWD_BLOCK_SIZE")
    num_cores = get_num_cores("vector")
    _cross_entropy_backward_kernel[(num_cores,)](
        logits,
        dloss,
        lse,
        lse_abs,
        labels,
        dlogits,
        num_tokens,
        dloss.stride(0),
        logits.stride(0),
        dlogits.stride(0),
        VOCAB_SIZE=vocab_size,
        BLOCK_SIZE=block_size,
        Z_LOSS_WEIGHT=z_loss_weight,
        num_warps=8,
    )
    return dlogits


def chunked_linear_ce_fwd_impl(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    z_loss_weight: float,
    calc_acc: bool,
    align_precision: bool,
):
    del align_precision
    use_fp32_gemm = inputs.dtype == weight.dtype == torch.float32
    x_compute = inputs if use_fp32_gemm else inputs.bfloat16()
    weight_compute = weight if use_fp32_gemm else weight.bfloat16()
    tokens = x_compute.shape[0]
    loss = torch.empty(tokens, dtype=torch.float32, device=inputs.device)
    zloss = torch.empty_like(loss)
    accuracy = torch.empty(tokens if calc_acc else 0, dtype=torch.float32, device=inputs.device)
    lse = torch.empty_like(loss)
    lse_abs = torch.empty_like(loss)
    chunk_size = _chunked_loss_chunk_size()
    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        logits = _matmul_logits(x_compute[start:end], weight_compute.T)
        _fused_ce_zloss_logits_stats_fwd(
            logits,
            labels[start:end],
            loss[start:end],
            zloss[start:end],
            accuracy[start:end] if calc_acc else loss[start:end],
            ignore_index,
            z_loss_weight,
            calc_acc,
            lse[start:end],
            lse_abs[start:end],
        )
    return loss, zloss, accuracy, lse, lse_abs


def chunked_linear_ce_bwd_impl(
    grad_loss: torch.Tensor,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    lse_abs: torch.Tensor,
    ignore_index: int,
    z_loss_weight: float,
):
    use_fp32_gemm = inputs.dtype == weight.dtype == torch.float32
    x_compute = inputs if use_fp32_gemm else inputs.bfloat16()
    weight_compute = weight if use_fp32_gemm else weight.bfloat16()
    dx = torch.zeros_like(inputs)
    dweight = torch.zeros_like(weight, dtype=torch.float32)
    grad_loss = grad_loss.masked_fill(labels == ignore_index, 0.0)
    chunk_size = _chunked_loss_chunk_size()
    for start in range(0, inputs.shape[0], chunk_size):
        end = min(start + chunk_size, inputs.shape[0])
        logits = _matmul_logits(x_compute[start:end], weight_compute.T)
        dlogits = _cross_entropy_backward(
            grad_loss[start:end], logits, lse[start:end], labels[start:end], z_loss_weight, lse_abs[start:end], x_compute.dtype
        )
        if inputs.dtype == torch.bfloat16:
            torch.mm(dlogits, weight_compute, out=dx[start:end])
        else:
            _matmul_fp32(dlogits, weight_compute, out=dx[start:end])
        _addmm_fp32(dweight, dlogits.T, x_compute[start:end], out=dweight)
    return dx, dweight.to(weight.dtype)


@torch.library.custom_op("mojo_npu_triton_a5::linear_cross_entropy_fwd", mutates_args=())
def linear_cross_entropy_fwd(
    inputs: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, ignore_index: int,
    z_loss_weight: float, calc_acc: bool, align_precision: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return chunked_linear_ce_fwd_impl(
        inputs, weight, labels, ignore_index, z_loss_weight, calc_acc, align_precision
    )


@linear_cross_entropy_fwd.register_fake
def _linear_cross_entropy_fwd_fake(
    inputs: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, ignore_index: int,
    z_loss_weight: float, calc_acc: bool, align_precision: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del weight, labels, ignore_index, z_loss_weight, align_precision
    vector = torch.empty(inputs.shape[0], dtype=torch.float32, device=inputs.device)
    accuracy = torch.empty(inputs.shape[0] if calc_acc else 0, dtype=torch.float32, device=inputs.device)
    return vector, torch.empty_like(vector), accuracy, torch.empty_like(vector), torch.empty_like(vector)


@torch.library.custom_op("mojo_npu_triton_a5::linear_cross_entropy_bwd", mutates_args=())
def linear_cross_entropy_bwd(
    grad_loss: torch.Tensor, inputs: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
    lse: torch.Tensor, lse_abs: torch.Tensor, ignore_index: int, z_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return chunked_linear_ce_bwd_impl(
        grad_loss, inputs, weight, labels, lse, lse_abs, ignore_index, z_loss_weight
    )


@linear_cross_entropy_bwd.register_fake
def _linear_cross_entropy_bwd_fake(
    grad_loss: torch.Tensor, inputs: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
    lse: torch.Tensor, lse_abs: torch.Tensor, ignore_index: int, z_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del grad_loss, labels, lse, lse_abs, ignore_index, z_loss_weight
    return torch.empty_like(inputs), torch.empty_like(weight)


@torch.library.custom_op(
    "mojo_npu_triton_a5::linear_cross_entropy_and_zloss_fwd",
    mutates_args=(),
)
def linear_cross_entropy_and_zloss_fwd(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    z_loss_weight: float,
    calc_acc: bool,
    align_precision: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return chunked_linear_ce_fwd_impl(
        inputs, weight, labels, ignore_index, z_loss_weight, calc_acc, align_precision
    )


@linear_cross_entropy_and_zloss_fwd.register_fake
def _linear_cross_entropy_and_zloss_fwd_fake(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    z_loss_weight: float,
    calc_acc: bool,
    align_precision: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del weight, labels, ignore_index, z_loss_weight, align_precision
    vector = torch.empty(inputs.shape[0], dtype=torch.float32, device=inputs.device)
    accuracy = torch.empty(inputs.shape[0] if calc_acc else 0, dtype=torch.float32, device=inputs.device)
    return vector, torch.empty_like(vector), accuracy, torch.empty_like(vector), torch.empty_like(vector)


@torch.library.custom_op(
    "mojo_npu_triton_a5::linear_cross_entropy_and_zloss_bwd",
    mutates_args=(),
)
def linear_cross_entropy_and_zloss_bwd(
    grad_loss: torch.Tensor,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    lse_abs: torch.Tensor,
    ignore_index: int,
    z_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return chunked_linear_ce_bwd_impl(
        grad_loss, inputs, weight, labels, lse, lse_abs, ignore_index, z_loss_weight
    )


@linear_cross_entropy_and_zloss_bwd.register_fake
def _linear_cross_entropy_and_zloss_bwd_fake(
    grad_loss: torch.Tensor,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    lse_abs: torch.Tensor,
    ignore_index: int,
    z_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del grad_loss, labels, lse, lse_abs, ignore_index, z_loss_weight
    return torch.empty_like(inputs), torch.empty_like(weight)
