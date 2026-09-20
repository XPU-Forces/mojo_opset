from typing import Optional

import torch

from mojo_opset import functions


class LinearCrossEntropyLoss(torch.nn.Module):
    """Original mojo loss module, retaining (weight, input, target, bias) order."""

    def __init__(self, ignore_index=-100, lse_square_scale=0.0, label_smoothing=0.0,
                 reduction="mean", *, ce_weight=None, softcap=None, return_z_loss=False,
                 accum_dtype=None, implementation: Optional[str] = None):
        super().__init__()
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.register_buffer("ce_weight", ce_weight)
        self.softcap = softcap
        self.return_z_loss = return_z_loss
        self.accum_dtype = accum_dtype
        self.implementation = implementation

    def forward(self, lin_weight, _input, target, bias=None):
        return functions.linear_cross_entropy(
            _input, lin_weight, target, bias, self.ce_weight, self.ignore_index,
            self.lse_square_scale, self.label_smoothing, self.reduction,
            self.softcap, self.return_z_loss, self.accum_dtype, implementation=self.implementation,
        )
