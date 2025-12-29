import torch
import torch.nn.functional as F

from ..function import MojoFunction
from ..function import mojo_func_dispatcher


@mojo_func_dispatcher
class MojoRMSNormFunction(MojoFunction):
    @staticmethod
    def forward_ref(ctx, input, weight, eps):
        normalized_shape = (input.shape[-1],)
        y = F.rms_norm(input, normalized_shape, weight=weight, eps=eps)

        ctx.save_for_backward(input, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps

        return y

    @staticmethod
    def backward_ref(ctx, grad_output):
        input, weight, _ = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps

        input_with_grad = input.detach().clone().requires_grad_(True)
        weight_with_grad = weight.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            y_ref = F.rms_norm(input_with_grad, normalized_shape, weight=weight_with_grad, eps=eps)

        y_ref.backward(gradient=grad_output)

        grad_input = input_with_grad.grad
        grad_weight = weight_with_grad.grad

        return grad_input, grad_weight, None
