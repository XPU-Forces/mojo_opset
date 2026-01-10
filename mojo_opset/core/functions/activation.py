import torch

from ..function import MojoFunction


class MojoSiluFunction(MojoFunction):
    """
    Implements the SiLU (Sigmoid Linear Unit) activation function as default.
    The SiLU activation is defined as: SiLU(x) = x * sigmoid(x).
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of SiLU.

        Args:
            ctx: Context object for the backward.
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Result of the SiLU activation.
        """
        sigmoid_x = torch.sigmoid(input)
        ctx.save_for_backward(input)
        return input * sigmoid_x

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Backward pass of SiLU.

        Args:
            ctx: Context object for the backward.
            grad_output (torch.Tensor): Gradient of the output tensor.

        Returns:
            torch.Tensor: Gradient of the input tensor.
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output * torch.sigmoid(input) * (1 + input * (1 - torch.sigmoid(input)))
        return grad_input


def mojo_silu(
    input: torch.Tensor,
) -> torch.Tensor:
    """
    SiLU activation function.

    Args:
        input (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Result of the SiLU activation.
    """
    return MojoSiluFunction.apply(input)
