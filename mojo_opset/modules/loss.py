from typing import Optional

import torch

from mojo_opset import functions


class LinearCrossEntropyLoss(torch.nn.Module):
    """Chunked CE, called with (weight, input, labels).

    Returns loss, or (loss, per-token accuracy) when calc_acc=True.
    """

    def __init__(self, ignore_index=-100, reduction="mean", *, calc_acc=False,
                 align_precision=True, implementation: Optional[str] = None):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.calc_acc = calc_acc
        self.align_precision = align_precision
        self.implementation = implementation

    def forward(self, lin_weight, _input, labels):
        return functions.linear_cross_entropy(
            _input, lin_weight, labels, ignore_index=self.ignore_index,
            reduction=self.reduction, calc_acc=self.calc_acc,
            align_precision=self.align_precision, implementation=self.implementation,
        )
