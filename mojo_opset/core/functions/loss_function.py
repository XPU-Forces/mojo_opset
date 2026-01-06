import torch.nn as nn

from ..function import MojoFunction


class MojoFusedLinearCrossEntropyFunction(MojoFunction):
    @staticmethod
    def forward(
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
    def backward(ctx, grad_loss, grad_z_loss=None):
        pass


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

    def forward(self, lin_weight, _input, target, bias=None):
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
