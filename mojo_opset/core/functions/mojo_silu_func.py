import torch

from mojo_opset.utils.logging import get_logger

from ..mojo_function import MojoFuncBase
from ..mojo_function import mojo_func_dispatcher

logger = get_logger(__name__)


@mojo_func_dispatcher
class MojoSiluFunction(MojoFuncBase):
    @staticmethod
    def forward_ref(ctx, input):
        sigmoid_x = torch.sigmoid(input)
        ctx.save_for_backward(input)
        return input * sigmoid_x

    @staticmethod
    def backward_ref(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output * torch.sigmoid(input) * (1 + input * (1 - torch.sigmoid(input)))
        return grad_input
