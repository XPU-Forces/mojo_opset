"""Original mojo CE family; independent of the ext-derived chunked CE APIs."""

from typing import Optional

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl


class LinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, labels, bias, ce_weight, options, implementation):
        forward, ctx.backward_kernel = load_impl("linear_cross_entropy", implementation, require_backward=True)
        ctx.options = options
        ctx.device_type = inputs.device.type
        ctx.autocast_enabled = torch.is_autocast_enabled(inputs.device.type)
        ctx.autocast_dtype = torch.get_autocast_dtype(inputs.device.type)
        loss, zloss, dx, dw, db = forward(
            inputs,
            weight,
            labels,
            bias,
            ce_weight,
            *options,
            ctx.needs_input_grad[1],
        )
        if dx is None:
            ctx.save_for_backward(inputs, weight, labels, bias, ce_weight, None, None, None)
        else:
            # The original reduced TTX path precomputes gradients in forward;
            # retain only those buffers, not the otherwise-unused input activation.
            ctx.save_for_backward(None, None, None, None, None, dx, dw, db)
        return loss, zloss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss, grad_zloss):
        inputs, weight, labels, bias, ce_weight, dx, dw, db = ctx.saved_tensors
        with torch.autocast(ctx.device_type, enabled=ctx.autocast_enabled, dtype=ctx.autocast_dtype):
            gradients = ctx.backward_kernel(
                grad_loss,
                grad_zloss,
                inputs,
                weight,
                labels,
                bias,
                ce_weight,
                *ctx.options,
                dx,
                dw,
                db,
            )
        return *gradients[:2], None, gradients[2], None, None, None


def linear_cross_entropy(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ce_weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    accum_dtype: Optional[torch.dtype] = None,
    *,
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Original mojo CE (MojoFusedLinearCrossEntropyFunction), with options preserved.

    Returns (loss, z_loss_or_None). Z-loss uses
    lse_square_scale * logsumexp(logits)**2, unlike the ext-derived
    linear_cross_entropy_v2_and_zloss, which uses abs(logits).

    Historical provider differences remain: Triton applies z-loss regardless
    of return_z_loss and ignores its diagnostic gradient. The reference adds
    mean z-loss only when return_z_loss=True, folds its gradient into the loss
    gradient, and ignores softcap/accum_dtype.
    """
    if input_tensor.ndim != 2 or weight.ndim != 2 or target.shape != input_tensor.shape[:1]:
        raise ValueError("expected input[N,H], weight[V,H], target[N]")
    if input_tensor.shape[1] != weight.shape[1] or not input_tensor.shape[0]:
        raise ValueError("input and weight must have matching H and nonempty N")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")
    options = (ignore_index, lse_square_scale, label_smoothing, reduction, softcap, return_z_loss, accum_dtype)
    return LinearCrossEntropyFunction.apply(
        input_tensor,
        weight,
        target,
        bias,
        ce_weight,
        options,
        implementation,
    )
