import torch
import torch.nn.functional as F

from ..function import MojoFunction


class MojoRMSNormFunction(MojoFunction):
    """
    RMSNorm function.
    The RMSNorm function is defined as: RMSNorm(x) = x * (weight / sqrt(mean(x^2) + eps)).
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """
        Forward pass of RMSNorm.
        The normalized dimension is the last dimension of input.

        Args:
            ctx: Context object for the backward.
            input (torch.Tensor): Input tensor.
            weight (torch.Tensor): Weight tensor.
            eps (float): Epsilon value for numerical stability.

        Returns:
            torch.Tensor: Result of the RMSNorm activation.
        """
        normalized_shape = (input.shape[-1],)
        y = F.rms_norm(input, normalized_shape, weight=weight, eps=eps)

        ctx.save_for_backward(input, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps

        return y

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Backward pass of RMSNorm.

        Args:
            ctx: Context object for the backward.
            grad_output (torch.Tensor): Gradient of the output tensor.

        Returns:
            torch.Tensor: Gradient of the input tensor.
        """
        input, weight = ctx.saved_tensors
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
