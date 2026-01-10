from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..function import MojoFunction


class MojoFusedLinearCrossEntropyFunction(MojoFunction):
    """
    MojoFusedLinearCrossEntropyFunction implements the fused linear cross entropy loss.
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: torch.Tensor,
        ce_weight: torch.Tensor,
        ignore_index: int,
        lse_square_scale: float,
        label_smoothing: float,
        reduction: str,
        softcap: float,
        return_z_loss: bool,
        accum_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss by avoiding
        the materialization of the large logits tensor. Since Cross Entropy Loss is the last layer, we can
        compute the gradient at the forward pass. By doing so, we don't have to store the _input and target
        for the backward pass.

        Args:
            ctx: Context object for the backward.
            input_tensor (torch.Tensor): (B*T, H) where B is batch size, T is sequence length, H is hidden dimension.
            weight (torch.Tensor): (V, H) where V is the number of classes, H is hidden dimension.
            target (torch.Tensor): (B*T) where each value is in [0, V-1].
            bias (torch.Tensor): (V) where V is the number of classes.
            ce_weight (torch.Tensor): (V) where V is the number of classes.
            ignore_index (int): Index to ignore in the target.
            lse_square_scale (float): Scale factor for the z-loss.
            label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
            reduction (str): Reduction method for the loss.
            softcap (float): Softcap value for the loss.
            return_z_loss (bool): Whether to return the z-loss.
            accum_dtype (torch.dtype): the dtype of intermediate result buffers for weight and bias gradient accumulations.
            Recommended to set `accum_dtype` to higher precision, e.g. `torch.float32`, if the training is unstable with original dtype. Default: `None`, performing accumulations in original dtype.

        Returns:
            torch.Tensor: Loss tensor.
        """

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
            return loss_ref, None

    @staticmethod
    def backward(
        ctx,
        grad_loss: torch.Tensor,
        grad_z_loss: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor, None, None, None, None, None, None]:
        """
        Backward pass for fused linear cross entropy loss.

        Args:
            ctx: Context object for the backward.
            grad_loss (torch.Tensor): Gradient tensor for the loss. shape [BSZ, SEQ]
            grad_z_loss (Optional[torch.Tensor]): Gradient tensor for the z-loss. shape [BSZ, SEQ]

        Returns:
            tuple[torch.Tensor, torch.Tensor, None, torch.Tensor, None, None, None, None, None, None]:
                grad_input: torch.Tensor, shape [BSZ, SEQ, VOCAB_SIZE]
                grad_weight: torch.Tensor, shape [VOCAB_SIZE, HEAD_DIM]
                grad_bias: torch.Tensor, shape [VOCAB_SIZE]
        """

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


class MojoFusedLinearCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        self.kwargs = kwargs

    def forward(
        self,
        lin_weight: torch.Tensor,
        _input: torch.Tensor,
        target: torch.Tensor,
        bias=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            lin_weight (torch.Tensor): Linear layer weight tensor. shape [VOCAB_SIZE, HEAD_DIM]
            _input (torch.Tensor): Input tensor. shape [BSZ, SEQ, HEAD_DIM]
            target (torch.Tensor): Target tensor. shape [BSZ, SEQ]
            bias (Optional[torch.Tensor]): Linear layer bias tensor. shape [VOCAB_SIZE]

        Returns:
            torch.Tensor: Loss tensor. shape [BSZ, SEQ]
        """

        return MojoFusedLinearCrossEntropyFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            self.kwargs.get("ce_weight", None),
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
            **self.kwargs,
        )
