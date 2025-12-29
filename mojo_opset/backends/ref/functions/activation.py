import torch

from mojo_opset.core.functions.activation import MojoSiluFunction
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class RefSiluFunction(MojoSiluFunction):
    @staticmethod
    def forward(ctx, input):
        sigmoid_x = torch.sigmoid(input)
        ctx.save_for_backward(input)
        return input * sigmoid_x

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output * torch.sigmoid(input) * (1 + input * (1 - torch.sigmoid(input)))
        return grad_input
