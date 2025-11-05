import os
import torch
from ..mojo_function import MojoFuncBase, mojo_func_dispatcher
import torch.nn.functional as F


@mojo_func_dispatcher
class MojoRMSNormFunction(MojoFuncBase):
    @staticmethod
    def forward_dump(ctx, input, weight, eps):
        pass

    @staticmethod
    def forward_ref(ctx, input, weight, eps):
        normalized_shape = (input.shape[-1],)
        y = F.rms_norm(input, normalized_shape, weight=weight, eps=eps)

        ctx.save_for_backward(input, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps

        return y

    @staticmethod
    def backward_dump(ctx, grad_output):
        pass

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
