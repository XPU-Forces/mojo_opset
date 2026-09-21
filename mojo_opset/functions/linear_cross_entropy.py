from typing import Optional

import torch

from torch.autograd.function import once_differentiable

from ._dispatch import load_impl


class LinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        weight,
        labels,
        ignore_index,
        z_loss_weight,
        calc_acc,
        align_precision,
        op_id,
        implementation,
    ):
        forward, backward = load_impl(op_id, implementation, require_backward=True)
        loss, zloss, accuracy, lse, lse_abs = forward(
            inputs, weight, labels, ignore_index, z_loss_weight, calc_acc, align_precision
        )
        ctx.backward_kernel = backward
        ctx.ignore_index = ignore_index
        ctx.z_loss_weight = z_loss_weight
        ctx.save_for_backward(inputs, weight, labels, lse, lse_abs)
        ctx.mark_non_differentiable(zloss, accuracy, lse, lse_abs)
        return loss, zloss, accuracy, lse, lse_abs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss, _grad_zloss, _grad_accuracy, _grad_lse, _grad_lse_abs):
        inputs, weight, labels, lse, lse_abs = ctx.saved_tensors
        grad_inputs, grad_weight = ctx.backward_kernel(
            grad_loss.contiguous(),
            inputs,
            weight,
            labels,
            lse,
            lse_abs,
            ctx.ignore_index,
            ctx.z_loss_weight,
        )
        return grad_inputs, grad_weight, None, None, None, None, None, None, None


def _reduce(values: torch.Tensor, labels: torch.Tensor, ignore_index: int, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.float().sum()
    if reduction == "mean":
        return values.float().sum() / (labels != ignore_index).sum().clamp(min=1)
    raise ValueError(f"reduction must be 'none', 'sum', or 'mean', got {reduction!r}.")


def _run(
    op_id: str,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int,
    z_loss_weight: float,
    reduction: str,
    calc_acc: bool,
    align_precision: bool,
    implementation: Optional[str],
):
    loss, zloss, accuracy, _lse, _lse_abs = LinearCrossEntropyFunction.apply(
        inputs,
        weight,
        labels,
        ignore_index,
        z_loss_weight,
        calc_acc,
        align_precision,
        op_id,
        implementation,
    )
    return _reduce(loss, labels, ignore_index, reduction), _reduce(zloss, labels, ignore_index, reduction), accuracy


def linear_cross_entropy(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
    calc_acc: bool = False,
    align_precision: bool = True,
    implementation: Optional[str] = None,
):
    """Chunked linear cross entropy without z-loss.

    Returns loss, or (loss, accuracy) when calc_acc=True.
    """
    loss, _zloss, accuracy = _run(
        "linear_cross_entropy",
        inputs,
        weight,
        labels,
        ignore_index=ignore_index,
        z_loss_weight=0.0,
        reduction=reduction,
        calc_acc=calc_acc,
        align_precision=align_precision,
        implementation=implementation,
    )
    return (loss, accuracy) if calc_acc else loss


def linear_cross_entropy_and_zloss(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    z_loss_weight: float,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
    calc_acc: bool = False,
    align_precision: bool = True,
    implementation: Optional[str] = None,
):
    """Chunked CE with z_loss_weight * logsumexp(abs(logits))**2.

    Returns (loss, zloss), or (loss, accuracy, zloss) when calc_acc=True.
    Loss includes z-loss and its gradient; zloss and accuracy are diagnostics
    without their own gradients.
    """
    loss, zloss, accuracy = _run(
        "linear_cross_entropy_and_zloss",
        inputs,
        weight,
        labels,
        ignore_index=ignore_index,
        z_loss_weight=z_loss_weight,
        reduction=reduction,
        calc_acc=calc_acc,
        align_precision=align_precision,
        implementation=implementation,
    )
    return (loss, accuracy, zloss) if calc_acc else (loss, zloss)
