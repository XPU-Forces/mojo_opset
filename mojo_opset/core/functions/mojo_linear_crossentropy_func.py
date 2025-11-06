import os
import torch
from ..mojo_function import MojoFuncBase, mojo_func_dispatcher
import torch.nn.functional as F


@mojo_func_dispatcher
class MojoFusedLinearCrossEntropyFunction(MojoFuncBase):
    @staticmethod
    def forward_dump(
        ctx,
        input_tensor,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        accum_dtype,
    ):
        pass

    @staticmethod
    def forward_ref(
        ctx,
        input_tensor,
        weight,
        target,
        bias,
        ce_weight,
        ignore_index,
        lse_square_scale,
        label_smoothing,
        reduction,
        softcap,
        return_z_loss,
        accum_dtype,
    ):
        logits_ref = F.linear(input_tensor, weight, bias).float()

        loss_ref = F.cross_entropy(
            logits_ref,
            target,
            weight=ce_weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

        z_loss_ref = None
        if return_z_loss:
            with torch.no_grad():
                valid_logits = logits_ref[target != ignore_index]
                if valid_logits.numel() > 0:
                    lse_ref = torch.logsumexp(valid_logits, dim=-1)

                    z_loss_calc = lse_square_scale * torch.sum(lse_ref * lse_ref) / (target != ignore_index).sum()
                    loss_ref += z_loss_calc
                    z_loss_ref = z_loss_calc

        ctx.save_for_backward(input_tensor, weight, target, bias, ce_weight)
        ctx.ignore_index = ignore_index
        ctx.label_smoothing = label_smoothing
        ctx.reduction = reduction
        ctx.return_z_loss = return_z_loss
        ctx.lse_square_scale = lse_square_scale

        if return_z_loss:
            if z_loss_ref is None:
                z_loss_ref = torch.tensor(0.0, device=loss_ref.device, dtype=loss_ref.dtype)
            return loss_ref, z_loss_ref
        else:
            return loss_ref

    @staticmethod
    def backward_dump(ctx, grad_loss, grad_z_loss):
        pass

    @staticmethod
    def backward_ref(ctx, grad_loss, grad_z_loss=None):
        input_tensor, weight, target, bias, ce_weight = ctx.saved_tensors

        ignore_index = ctx.ignore_index
        label_smoothing = ctx.label_smoothing
        reduction = ctx.reduction
        return_z_loss = ctx.return_z_loss
        lse_square_scale = ctx.lse_square_scale

        input_with_grad = input_tensor.detach().clone().requires_grad_(True)
        weight_with_grad = weight.detach().clone().requires_grad_(True)
        bias_with_grad = bias.detach().clone().requires_grad_(True) if bias is not None else None

        with torch.enable_grad():
            logits_ref = F.linear(input_with_grad, weight_with_grad, bias_with_grad).float()

            loss_ref = F.cross_entropy(
                logits_ref,
                target,
                weight=ce_weight,
                ignore_index=ignore_index,
                reduction=reduction,
                label_smoothing=label_smoothing,
            )

            if return_z_loss:
                valid_logits = logits_ref[target != ignore_index]
                if valid_logits.numel() > 0:
                    lse_ref = torch.logsumexp(valid_logits, dim=-1)
                    z_loss_calc = lse_square_scale * torch.sum(lse_ref * lse_ref) / (target != ignore_index).sum()
                    loss_ref += z_loss_calc

        total_grad = grad_loss
        if return_z_loss and grad_z_loss is not None:
            total_grad += grad_z_loss

        loss_ref.backward(gradient=total_grad)

        grad_input = input_with_grad.grad
        grad_weight = weight_with_grad.grad
        grad_bias = bias_with_grad.grad if bias_with_grad is not None else None

        return grad_input, grad_weight, None, grad_bias, None, None, None, None, None, None, None, None
