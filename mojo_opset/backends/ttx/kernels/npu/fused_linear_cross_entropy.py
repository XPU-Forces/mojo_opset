from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from triton.runtime.libentry import libentry


def _noop_decorator(*args, **kwargs):
    def decorator(fn):
        return fn

    return decorator


try:
    import torch_npu

    amp_custom_fwd = torch_npu.npu.amp.custom_fwd
    amp_custom_bwd = torch_npu.npu.amp.custom_bwd
except Exception:
    amp_custom_fwd = _noop_decorator
    amp_custom_bwd = _noop_decorator


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}),
        triton.Config({"BLOCK_SIZE": 8192}),
        triton.Config({"BLOCK_SIZE": 16384}),
    ],
    key=["n_cols"],
    restore_value=["X_ptr"],
)
@libentry()
@triton.jit
def _cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    weight_ptr,
    loss_ptr,
    z_loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    softcap,
    OVERWRITE_GRAD_LOGITS: tl.constexpr,
    IS_BACKWARD: tl.constexpr,
    RETURN_Z_LOSS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now. Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.
    """

    # https://github.com/triton-lang/triton/issues/1058
    # If B*T*V is too large, program_id * stride will overflow out of int32, so we convert to int64
    program_id = tl.program_id(0).to(tl.int64)

    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    X_ptr += program_id * X_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    if HAS_WEIGHT:
        weight_y = tl.load(weight_ptr + y).cast(tl.float32)

    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
    # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

    # [Online softmax] first pass: find max + sum
    m = float("-inf")  # m is the max value. use the notation from the paper
    d = 0.0  # d is the sum. use the notation from the paper
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)  # we need to store the original value of X_y for the loss calculation

    if HAS_SOFTCAPPING:
        ori_X_y = softcap * tl.math.tanh(ori_X_y / softcap)

    # Label smoothing is a general case of normal cross entropy
    # See the full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issue-2503665310
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_mask = X_offsets < n_cols
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_mask,
            other=float("-inf"),
            # Ensure float32 precision for softmax calculation
        ).cast(tl.float32)

        if HAS_SOFTCAPPING:
            X_block = softcap * tl.math.tanh(X_block / softcap)
        block_max = tl.max(X_block)
        if label_smoothing > 0:
            X_block2 = tl.load(
                X_ptr + X_offsets,
                mask=X_mask,
                other=0.0,
            ).cast(tl.float32)
            # scale X beforehand to avoid overflow
            if HAS_WEIGHT:
                weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                scaled_x_sum += tl.sum(-eps * X_block2 * weight_block).to(tl.float32)
            else:
                scaled_x_sum += tl.sum(-eps * X_block2).to(tl.float32)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    # log (sum(e^(X_i))) = log (sum(e ^ (max(X) * e ^ (X_i - max(X)))))
    #                    = log (e^(max(X)) * sum(e ^ (X_i - max(X))))
    #                    = max(X) + log (sum(e ^ (X_i - max(X)))) = m + log d
    lse = m + tl.log(d)

    if OVERWRITE_GRAD_LOGITS:

        # [Online Softmax] Second pass: compute gradients
        # For 'mean' reduction, gradients are normalized by number of non-ignored elements (N)
        # dx_y = (softmax(x_y) - 1) / N
        # dx_i = softmax(x_i) / N, i != y
        # For label smoothing:
        # dx_i = (softmax(x_i) - label_smoothing / V) / N, V = n_cols, i != y
        # dx_y = (softmax(x_y) - label_smoothing / V - (1 - label_smoothing)) / N
        #      = dx_i - (1 - label_smoothing) / N
        # With Z loss:
        # dx_i = ((1 + 2 * lse_square_scale * lse) * softmax(x_i) - label_smoothing / V) / N, i != y
        # dx_y = dx_i - (1 - label_smoothing) / N
        # For 'sum' reduction, no normalization is applied:
        # dx_y = softmax(x_y) - 1
        # dx_i = softmax(x_i), for i ≠ y

        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)

            y_is_in_this_block = (y >= i) and (y < i + BLOCK_SIZE)

            X_block = tl.load(
                X_ptr + X_offsets,
                mask=X_offsets < n_cols,
                other=float("-inf"),
                # Ensure float32 precision for softmax calculation
            ).cast(tl.float32)
            if HAS_SOFTCAPPING:
                intermediate = tl.math.tanh(X_block / softcap)
                X_block = softcap * intermediate

            if not HAS_WEIGHT:
                # softmax(x_i)
                X_block = tl.exp(X_block - m) / d
                # derivative of z-loss: 2 * lse_square_scale * lse * softmax(x_i)
                X_block += 2 * lse_square_scale * lse * X_block
                # smoothing term
                X_block += -eps
                # special handle dx_y
                if y_is_in_this_block:
                    is_target_class_mask = X_offsets == y
                    X_block -= is_target_class_mask * (1 - label_smoothing)
                # X_block = tl.where(X_offsets != y, X_block, X_block - (1 - label_smoothing))
                # reduction scale
                if reduction == "mean":
                    X_block = X_block / n_non_ignore
            else:
                weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                softmax_X = tl.exp(X_block - m) / d
                # derivative of original_loss
                dloss_ori = (1 - label_smoothing) * softmax_X
                # specially handle dx_y
                # dloss_ori = tl.where(X_offsets != y, dloss_ori, dloss_ori - (1 - label_smoothing))
                if y_is_in_this_block:
                    is_target_class_mask = X_offsets == y
                    dloss_ori -= is_target_class_mask * (1 - label_smoothing)
                dloss_ori = dloss_ori * weight_y
                # derivative of smooth_loss
                dloss_smooth = eps * (-weight_block + softmax_X * weight_sum)
                # derivative of z-loss
                dz_loss = 2 * lse_square_scale * lse * softmax_X
                # reduction scale
                if reduction == "mean":
                    dloss_ori = dloss_ori / sum_non_ignore_weight
                    dloss_smooth = dloss_smooth / sum_non_ignore_weight
                    # TODO: Implement weighted z_loss. Currently, z_loss is not scaled by weight.
                    dz_loss = dz_loss / n_non_ignore
                # derivative of total_loss
                X_block = dloss_ori + dloss_smooth + dz_loss

            # chain rule softcapping
            # d(softcap * tanh(x / softcap)) = (1 - tanh^2(x / softcap))
            if HAS_SOFTCAPPING:
                X_block = X_block * (1 - intermediate * intermediate)

            tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    if not IS_BACKWARD:

        loss_ptr += program_id * loss_stride
        if RETURN_Z_LOSS:
            z_loss_ptr += program_id * loss_stride

        # We need tl.debug_barrier() to ensure the new result of X_ptr is written as mentioned in
        # https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34
        # tl.debug_barrier()

        # Calculate the loss

        # loss = log (softmax(X_y)) = log ((e ^ (X_y - max(X)) / sum(e ^ (X - max(X))))
        #      = (X_y - max(X)) - log(sum(e ^ (X - max(X))))
        #      = X_y - m - log d = X_y - lse
        # sum(e ^ (X - max(X))) must >= 1 because the max term is e ^ 0 = 1
        # So we can safely calculate log (softmax(X_y)) without overflow
        loss = lse - ori_X_y
        if HAS_WEIGHT:
            loss = weight_y * loss

        # Original loss = H(q, p),  with label smoothing regularization = H(q', p) and (label_smoothing / V) = eps
        # H(q', p) = (1 - label_smoothing) * H(q, p) + label_smoothing * H(u, p)
        #          = (1 - label_smoothing) * H(q, p) + eps * sum(logsoftmax(x_i))
        # By using m (global max of xi) and d (sum of e^(xi-m)), we can simplify as:
        #          = (1 - label_smoothing) * H(q, p) + (sum(-eps * x_i) + label_smoothing * (m + logd))
        # Refer to H(q', p) in section 7 of the paper: https://arxiv.org/pdf/1512.00567
        # pytorch: https://github.com/pytorch/pytorch/blob/2981534f54d49fa3a9755c9b0855e7929c2527f0/aten/src/ATen/native/LossNLL.cpp#L516
        # See full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issuecomment-2333753087
        if label_smoothing > 0:
            if HAS_WEIGHT:
                smooth_loss = scaled_x_sum + eps * lse * weight_sum
            else:
                smooth_loss = scaled_x_sum + label_smoothing * lse
            loss = loss * (1 - label_smoothing) + smooth_loss

        # An auxiliary loss, z_loss
        # Refer to Page14 Loss function section in the paper PaLM: https://www.jmlr.org/papers/v24/22-1144.html
        z_loss = lse_square_scale * lse * lse
        # Normalize the loss by the number of non-ignored elements if reduction is "mean"
        if reduction == "mean":
            if HAS_WEIGHT:
                loss = loss / sum_non_ignore_weight
            else:
                loss = loss / n_non_ignore
            # TODO: Implement weighted z_loss. Currently, z_loss is not scaled by weight.
            z_loss = z_loss / n_non_ignore
        loss += z_loss

        tl.store(loss_ptr, loss)
        if RETURN_Z_LOSS:
            tl.store(z_loss_ptr, z_loss)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}),
        triton.Config({"BLOCK_SIZE": 8192}),
        triton.Config({"BLOCK_SIZE": 16384}),
    ],
    key=["n_cols"],
    restore_value=["X_ptr"],
)
@libentry()
@triton.jit
def _cross_entropy_prime_kernel(
    X_ptr,
    Y_ptr,
    loss_ptr,
    n_rows,  # Now we need n_rows to know the loop boundary
    n_cols,
    X_stride_row,  # Stride to jump between rows
    Y_stride_row,  # Stride to jump between rows for Y
    loss_stride_row,  # Stride to jump between rows for loss
    n_non_ignore,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    OVERWRITE_GRAD_LOGITS: tl.constexpr,
    IS_BACKWARD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This is the modified version of the cross-entropy kernel that uses a fixed grid size.
    Each program_id corresponds to a "worker" that processes multiple rows in a strided loop.
    """
    # program_id is the worker ID, from 0 to grid_size-1
    pid = tl.program_id(0)
    # num_programs is the total number of workers (e.g., 48)
    num_programs = tl.num_programs(0)

    # The outer loop iterates through all rows assigned to this worker
    for row_idx in range(pid, n_rows, num_programs):
        # --- Pointer calculations for the current row ---
        # Note: We use row_idx for offsets now, not program_id
        current_X_ptr = X_ptr + row_idx * X_stride_row
        current_Y_ptr = Y_ptr + row_idx * Y_stride_row

        y = tl.load(current_Y_ptr)

        m = float("-inf")
        d = 0.0
        ori_X_y = 0.0

        if y != ignore_index:
            ori_X_y = tl.load(current_X_ptr + y).cast(tl.float32)

        scaled_x_sum = 0.0
        eps = label_smoothing / n_cols

        # First pass: calculate lse (log-sum-exp)
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            X_mask = X_offsets < n_cols
            X_block = tl.load(
                current_X_ptr + X_offsets,
                mask=X_mask,
                other=float("-inf"),
            ).cast(tl.float32)

            block_max = tl.max(X_block, axis=0)  # Use axis=0 for clarity, it's a 1D reduction
            if label_smoothing > 0:
                X_block2 = tl.load(
                    current_X_ptr + X_offsets,
                    mask=X_mask,
                    other=0.0,
                ).cast(tl.float32)
                scaled_x_sum += tl.sum(-eps * X_block2, axis=0).to(tl.float32)
            m_new = tl.maximum(m, block_max)
            d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new), axis=0)
            m = m_new

        lse = m + tl.log(d)

        if OVERWRITE_GRAD_LOGITS:
            # Second pass: calculate gradient and overwrite X in-place
            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                X_mask = X_offsets < n_cols

                X_block = tl.load(
                    current_X_ptr + X_offsets,
                    mask=X_mask,
                    other=float("-inf"),
                ).cast(tl.float32)

                X_block = tl.exp(X_block - m) / d
                X_block += 2 * lse_square_scale * lse * X_block
                X_block += -eps
                if reduction == "mean":
                    X_block = X_block / n_non_ignore

                tl.store(current_X_ptr + X_offsets, X_block, mask=X_mask)

        if not IS_BACKWARD:
            current_loss_ptr = loss_ptr + row_idx * loss_stride_row
            # Calculate and store the loss for the current row
            loss = lse - ori_X_y

            if label_smoothing > 0:
                smooth_loss = scaled_x_sum + label_smoothing * lse
                loss = loss * (1 - label_smoothing) + smooth_loss

            z_loss = lse_square_scale * lse * lse
            if reduction == "mean":
                loss = loss / n_non_ignore
                z_loss = z_loss / n_non_ignore
            loss += z_loss

            tl.store(current_loss_ptr, loss)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 1024}),
        triton.Config({"BLOCK_SIZE_M": 12, "BLOCK_SIZE_N": 1024}),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 1024}),
        triton.Config({"BLOCK_SIZE_M": 24, "BLOCK_SIZE_N": 1024}),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 1024}),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 2048}),
        triton.Config({"BLOCK_SIZE_M": 12, "BLOCK_SIZE_N": 2048}),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 2048}),
        triton.Config({"BLOCK_SIZE_M": 24, "BLOCK_SIZE_N": 2048}),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 2048}),
        triton.Config({"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 4096}),
        triton.Config({"BLOCK_SIZE_M": 12, "BLOCK_SIZE_N": 4096}),
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 4096}),
        triton.Config({"BLOCK_SIZE_M": 24, "BLOCK_SIZE_N": 4096}),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 4096}),
    ],
    key=["n_cols"],
    restore_value=["X_ptr"],
)
@libentry()
@triton.jit
def _element_mul_kernel(
    X_ptr,
    grad_output_ptr,
    n_rows,
    n_cols,
    X_stride_row,  # Stride for moving between rows
    # X_stride_col is assumed to be 1 for contiguous rows
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Multiplies each element of a 2D tensor X with a scalar grad_output.
    This kernel is adapted for NPU-style parallelism with a fixed grid size.

    Parameters:
    X_ptr: Pointer to the input 2D tensor.
    grad_output_ptr: Pointer to the scalar gradient output value.
    n_rows (int): The number of rows in the input tensor.
    n_cols (int): The number of columns in the input tensor.
    X_stride_row (int): The stride to move from one row to the next.
    BLOCK_SIZE_M (int): The number of rows to process in one tile.
    BLOCK_SIZE_N (int): The number of columns to process in one tile.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    grad_output = tl.load(grad_output_ptr)

    for row_block_start in range(pid * BLOCK_SIZE_M, n_rows, num_programs * BLOCK_SIZE_M):
        row_offsets = row_block_start + tl.arange(0, BLOCK_SIZE_M)
        row_mask = row_offsets < n_rows

        for col_block_start in range(0, n_cols, BLOCK_SIZE_N):
            col_offsets = col_block_start + tl.arange(0, BLOCK_SIZE_N)
            col_mask = col_offsets < n_cols

            X_ptr_tile = X_ptr + row_offsets[:, None] * X_stride_row + col_offsets[None, :]

            mask_2d = row_mask[:, None] & col_mask[None, :]

            X_tile = tl.load(X_ptr_tile, mask=mask_2d, other=0.0)

            result_tile = X_tile * grad_output

            tl.store(X_ptr_tile, result_tile, mask=mask_2d)


def fused_linear_cross_entropy_fwd_impl(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ce_weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    accum_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"
    device = _input.device

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the increase in memory = BT x V
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048
    BT, H = _input.shape
    V = weight.shape[0]

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # (BT + inc_factor - 1) // inc_factor
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    grad_input = torch.zeros_like(_input, device=device)
    if accum_dtype is None:
        grad_weight = torch.zeros_like(weight, device=device) if weight.requires_grad else None
        grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
    else:
        grad_weight = torch.zeros_like(weight, dtype=accum_dtype, device=device) if weight.requires_grad else None
        grad_bias = torch.zeros_like(bias, dtype=accum_dtype, device=device) if bias is not None else None

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device) if return_z_loss else None

    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()

    # assert total_n_non_ignore == len(target), (
    #     "TritonX limits the maximum number of programs to 65535. To support more rows, TTX kernels cap the grid launch count to the number of vector cores and introduce a loop. As a result, ignore_index is temporarily unsupported because Triton does not allow control-flow statements like 'continue'."
    # )
    total_sum_non_ignore_ce_weight = total_n_non_ignore
    ce_weight_sum = 0.0
    if ce_weight is not None:
        assert ce_weight.shape[0] == V, f"If given, weight has to be a Tensor of size V. Got: {ce_weight.shape}"
        assert torch.is_floating_point(
            ce_weight
        ), f"If given, weight has to be a Tensor of floating point dtype. Got: {ce_weight.dtype}"
        total_sum_non_ignore_ce_weight = (
            torch.gather(ce_weight, dim=0, index=target.masked_select(target_mask)).sum().item()
        )
        ce_weight_sum = ce_weight.sum().item()
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]  # chunk_size x H

        logits_chunk = _input_chunk @ weight.t()  # chunk_size x V
        if bias is not None:
            logits_chunk = logits_chunk + bias

        target_chunk = target[start_idx:end_idx]  # chunk_size,

        n_rows = logits_chunk.shape[0]

        target_chunk_mask = target_chunk != ignore_index

        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size,
        z_loss_1d_slice = z_loss_1d[start_idx:end_idx] if return_z_loss else None

        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        if (ce_weight is None) and (not return_z_loss) and (softcap is None):
            num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]

            _cross_entropy_prime_kernel[(num_programs,)](
                X_ptr=logits_chunk,
                Y_ptr=target_chunk,
                loss_ptr=loss_1d_slice,
                n_rows=n_rows,
                n_cols=V,
                X_stride_row=logits_chunk.stride(0),
                Y_stride_row=target_chunk.stride(0),
                loss_stride_row=loss_1d_slice.stride(0),
                n_non_ignore=total_n_non_ignore,
                ignore_index=ignore_index,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                reduction=reduction,
                OVERWRITE_GRAD_LOGITS=True,
                IS_BACKWARD=False,
            )
            loss_1d_slice.masked_fill_(~target_chunk_mask, 0.0)

            grad_logits_incorrect_chunk = logits_chunk

            row_indices = torch.arange(0, n_rows, device=device)
            valid_rows = row_indices[target_chunk_mask]
            valid_targets = target_chunk[target_chunk_mask]

            if valid_rows.numel() > 0:
                g_y_incorrect = grad_logits_incorrect_chunk[valid_rows, valid_targets]
                correction_term = 1 - label_smoothing
                if reduction == "mean":
                    correction_term /= total_n_non_ignore
                g_y_correct = g_y_incorrect - correction_term
                grad_logits_incorrect_chunk[valid_rows, valid_targets] = g_y_correct

            grad_logits_chunk = grad_logits_incorrect_chunk.masked_fill_(~target_chunk_mask.unsqueeze(1), 0.0)

        else:
            # Here we calculate the gradient of logits_chunk in place so we can save memory.
            _cross_entropy_kernel[(n_rows,)](
                X_ptr=logits_chunk,
                X_stride=logits_chunk.stride(-2),
                Y_ptr=target_chunk,
                Y_stride=target_chunk.stride(-1),  # always 1
                weight_ptr=ce_weight,
                loss_ptr=loss_1d_slice,
                z_loss_ptr=z_loss_1d_slice,
                loss_stride=loss_1d_slice.stride(-1),  # always 1
                n_cols=V,
                n_non_ignore=total_n_non_ignore,
                sum_non_ignore_weight=total_sum_non_ignore_ce_weight,
                weight_sum=ce_weight_sum,
                ignore_index=ignore_index,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                reduction=reduction,
                OVERWRITE_GRAD_LOGITS=True,
                IS_BACKWARD=False,
                softcap=softcap,
                RETURN_Z_LOSS=return_z_loss,
                HAS_WEIGHT=True if ce_weight is not None else False,
                HAS_SOFTCAPPING=True if softcap is not None else False,
            )

        grad_logits_chunk = logits_chunk  # chunk_size x V
        grad_input[start_idx:end_idx] = torch.matmul(grad_logits_chunk, weight)

        if grad_weight is not None:
            grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk).float()

        if bias is not None:
            torch.add(
                input=grad_bias,
                other=logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )

    # Need extra calculations for backward if reduction=='none'. Not supporting reduction='none' now.
    loss = torch.sum(loss_1d)
    z_loss = torch.sum(z_loss_1d) if return_z_loss else None

    # Cast back to original dtype
    grad_weight = grad_weight.to(weight.dtype) if grad_weight is not None else None
    grad_bias = grad_bias.to(bias.dtype) if grad_bias is not None else None

    return loss, z_loss, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_bwd_impl(
    grad_output: torch.Tensor,
    grad_input: torch.Tensor,
    grad_weight: Optional[torch.Tensor] = None,
    grad_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
        # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
        BT, H = grad_input.shape
        n_rows = BT

        num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
        _element_mul_kernel[(num_programs,)](
            grad_input,
            grad_output,
            n_rows,
            H,
            grad_input.stride(-2),
        )

        # handle grad_weight
        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V

            _element_mul_kernel[(num_programs,)](
                grad_weight,
                grad_output,
                n_rows,
                H,
                grad_weight.stride(-2),
            )

        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V

            _element_mul_kernel[(num_programs,)](
                grad_bias,
                grad_output,
                n_rows,
                1,
                grad_bias.stride(-1),
            )
    return grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_1d_fwd_impl(
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ce_weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"
    device = _input.device

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the increase in memory = BT x V
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048
    BT, H = _input.shape
    V = weight.shape[0]

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # (BT + inc_factor - 1) // inc_factor
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device) if return_z_loss else None

    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()

    # assert total_n_non_ignore == len(target), (
    #     "TritonX limits the maximum number of programs to 65535. To support more rows, TTX kernels cap the grid launch count to the number of vector cores and introduce a loop. As a result, ignore_index is temporarily unsupported because Triton does not allow control-flow statements like 'continue'."
    # )
    total_sum_non_ignore_ce_weight = total_n_non_ignore
    ce_weight_sum = 0.0
    if ce_weight is not None:
        assert ce_weight.shape[0] == V, f"If given, weight has to be a Tensor of size V. Got: {ce_weight.shape}"
        assert torch.is_floating_point(
            ce_weight
        ), f"If given, weight has to be a Tensor of floating point dtype. Got: {ce_weight.dtype}"
        total_sum_non_ignore_ce_weight = (
            torch.gather(ce_weight, dim=0, index=target.masked_select(target_mask)).sum().item()
        )
        ce_weight_sum = ce_weight.sum().item()
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]  # chunk_size x H

        logits_chunk = _input_chunk @ weight.t()  # chunk_size x V
        if bias is not None:
            logits_chunk = logits_chunk + bias

        target_chunk = target[start_idx:end_idx]  # chunk_size,

        n_rows = logits_chunk.shape[0]

        target_chunk_mask = target_chunk != ignore_index

        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size,
        z_loss_1d_slice = z_loss_1d[start_idx:end_idx] if return_z_loss else None

        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        if (ce_weight is None) and (not return_z_loss) and (softcap is None):
            num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]

            _cross_entropy_prime_kernel[(num_programs,)](
                X_ptr=logits_chunk,
                Y_ptr=target_chunk,
                loss_ptr=loss_1d_slice,
                n_rows=n_rows,
                n_cols=V,
                X_stride_row=logits_chunk.stride(0),
                Y_stride_row=target_chunk.stride(0),
                loss_stride_row=loss_1d_slice.stride(0),
                n_non_ignore=total_n_non_ignore,
                ignore_index=ignore_index,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                reduction="none",
                OVERWRITE_GRAD_LOGITS=False,
                IS_BACKWARD=False,
            )
            loss_1d_slice.masked_fill_(~target_chunk_mask, 0.0)

        else:
            # Here we calculate the gradient of logits_chunk in place so we can save memory.
            _cross_entropy_kernel[(n_rows,)](
                X_ptr=logits_chunk,
                X_stride=logits_chunk.stride(-2),
                Y_ptr=target_chunk,
                Y_stride=target_chunk.stride(-1),  # always 1
                weight_ptr=ce_weight,
                loss_ptr=loss_1d_slice,
                z_loss_ptr=z_loss_1d_slice,
                loss_stride=loss_1d_slice.stride(-1),  # always 1
                n_cols=V,
                n_non_ignore=total_n_non_ignore,
                sum_non_ignore_weight=total_sum_non_ignore_ce_weight,
                weight_sum=ce_weight_sum,
                ignore_index=ignore_index,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                reduction="none",
                OVERWRITE_GRAD_LOGITS=False,
                IS_BACKWARD=False,
                softcap=softcap,
                RETURN_Z_LOSS=return_z_loss,
                HAS_WEIGHT=True if ce_weight is not None else False,
                HAS_SOFTCAPPING=True if softcap is not None else False,
            )

    return loss_1d, z_loss_1d


def fused_linear_cross_entropy_1d_bwd_impl(
    grad_output: torch.Tensor,
    _input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ce_weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    softcap: Optional[float] = None,
    accum_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = _input.device

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the increase in memory = BT x V
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048
    BT, H = _input.shape
    V = weight.shape[0]

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # (BT + inc_factor - 1) // inc_factor
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

    grad_input = torch.zeros_like(_input, device=device)
    if accum_dtype is None:
        grad_weight = torch.zeros_like(weight, device=device) if weight is not None else None
        grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
    else:
        grad_weight = torch.zeros_like(weight, dtype=accum_dtype, device=device) if weight is not None else None
        grad_bias = torch.zeros_like(bias, dtype=accum_dtype, device=device) if bias is not None else None

    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()

    # assert total_n_non_ignore == len(target), (
    #     "TritonX limits the maximum number of programs to 65535. To support more rows, TTX kernels cap the grid launch count to the number of vector cores and introduce a loop. As a result, ignore_index is temporarily unsupported because Triton does not allow control-flow statements like 'continue'."
    # )
    total_sum_non_ignore_ce_weight = total_n_non_ignore
    ce_weight_sum = 0.0
    if ce_weight is not None:
        assert ce_weight.shape[0] == V, f"If given, weight has to be a Tensor of size V. Got: {ce_weight.shape}"
        assert torch.is_floating_point(
            ce_weight
        ), f"If given, weight has to be a Tensor of floating point dtype. Got: {ce_weight.dtype}"
        total_sum_non_ignore_ce_weight = (
            torch.gather(ce_weight, dim=0, index=target.masked_select(target_mask)).sum().item()
        )
        ce_weight_sum = ce_weight.sum().item()
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]  # chunk_size x H

        logits_chunk = _input_chunk @ weight.t()  # chunk_size x V
        if bias is not None:
            logits_chunk = logits_chunk + bias

        target_chunk = target[start_idx:end_idx]  # chunk_size,

        grad_output_chunk = grad_output[start_idx:end_idx]  # chunk_size,

        n_rows = logits_chunk.shape[0]

        target_chunk_mask = target_chunk != ignore_index

        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        if (ce_weight is None) and (softcap is None):
            num_programs = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]

            _cross_entropy_prime_kernel[(num_programs,)](
                X_ptr=logits_chunk,
                Y_ptr=target_chunk,
                loss_ptr=None,
                n_rows=n_rows,
                n_cols=V,
                X_stride_row=logits_chunk.stride(0),
                Y_stride_row=target_chunk.stride(0),
                loss_stride_row=0,
                n_non_ignore=total_n_non_ignore,
                ignore_index=ignore_index,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                reduction="none",
                OVERWRITE_GRAD_LOGITS=True,
                IS_BACKWARD=True,
            )

            grad_logits_incorrect_chunk = logits_chunk

            row_indices = torch.arange(0, n_rows, device=device)
            valid_rows = row_indices[target_chunk_mask]
            valid_targets = target_chunk[target_chunk_mask]

            if valid_rows.numel() > 0:
                g_y_incorrect = grad_logits_incorrect_chunk[valid_rows, valid_targets]
                correction_term = 1 - label_smoothing
                g_y_correct = g_y_incorrect - correction_term
                grad_logits_incorrect_chunk[valid_rows, valid_targets] = g_y_correct

            grad_logits_chunk = grad_logits_incorrect_chunk.masked_fill_(~target_chunk_mask.unsqueeze(1), 0.0)

        else:
            # Here we calculate the gradient of logits_chunk in place so we can save memory.
            _cross_entropy_kernel[(n_rows,)](
                X_ptr=logits_chunk,
                X_stride=logits_chunk.stride(-2),
                Y_ptr=target_chunk,
                Y_stride=target_chunk.stride(-1),  # always 1
                weight_ptr=ce_weight,
                loss_ptr=None,
                z_loss_ptr=None,
                loss_stride=0,  # always 1
                n_cols=V,
                n_non_ignore=total_n_non_ignore,
                sum_non_ignore_weight=total_sum_non_ignore_ce_weight,
                weight_sum=ce_weight_sum,
                ignore_index=ignore_index,
                lse_square_scale=lse_square_scale,
                label_smoothing=label_smoothing,
                reduction="none",
                OVERWRITE_GRAD_LOGITS=True,
                IS_BACKWARD=True,
                softcap=softcap,
                RETURN_Z_LOSS=False,
                HAS_WEIGHT=True if ce_weight is not None else False,
                HAS_SOFTCAPPING=True if softcap is not None else False,
            )

        grad_logits_chunk = logits_chunk.mul_(grad_output_chunk.unsqueeze(1))  # chunk_size x V
        grad_input[start_idx:end_idx] = torch.matmul(grad_logits_chunk, weight)

        if grad_weight is not None:
            grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk).float()

        if bias is not None:
            torch.add(
                input=grad_bias,
                other=logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )

    # Cast back to original dtype
    grad_weight = grad_weight.to(weight.dtype) if grad_weight is not None else None
    grad_bias = grad_bias.to(bias.dtype) if grad_bias is not None else None

    return grad_input, grad_weight, grad_bias
