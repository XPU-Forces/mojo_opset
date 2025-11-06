import os

import torch
from ..mojo_function import MojoFuncBase, mojo_func_dispatcher
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


@mojo_func_dispatcher
class MojoSiluFunction(MojoFuncBase):
    @staticmethod
    def forward_dump(ctx, input):
        pass

    @staticmethod
    def forward_ref(ctx, input):
        sigmoid_x = torch.sigmoid(input)
        ctx.save_for_backward(input)
        return input * sigmoid_x

    @staticmethod
    def backward_dump(ctx, grad_output):
        pass

    @staticmethod
    def backward_ref(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output * torch.sigmoid(input) * (1 + input * (1 - torch.sigmoid(input)))
        return grad_input
