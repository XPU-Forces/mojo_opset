import torch

from mojo_opset.training.function import MojoFunction
from mojo_opset.training.kernel import MojoKernel
from mojo_opset.training.module import MojoModule


class MojoSiluKernel(MojoKernel):
    """
    Implements the SiLU (Sigmoid Linear Unit) activation function as default.
    The SiLU activation is defined as: SiLU(x) = x * sigmoid(x).
    """

    @staticmethod
    def forward(
        input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of SiLU.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Result of the SiLU activation.
        """
        sigmoid_x = torch.sigmoid(input)
        return input * sigmoid_x

    @staticmethod
    def backward(
        grad_output: torch.Tensor,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Backward pass of SiLU.

        Args:
            grad_output (torch.Tensor): Gradient of the output tensor.

        Returns:
            torch.Tensor: Gradient of the input tensor.
        """
        grad_input = grad_output * torch.sigmoid(input) * (1 + input * (1 - torch.sigmoid(input)))
        return grad_input


class MojoSiluFunction(MojoFunction):
    """
    SiLU activation function.
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
        output = MojoSiluKernel.forward(input)
        ctx.save_for_backward(input)
        return output

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
        grad_input = MojoSiluKernel.backward(grad_output, input)
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


class MojoSiluModule(MojoModule):
    """
    SiLU activation module.
    """

    def __init__(
        self,
        name="",
        layer_idx=0,
    ):
        super().__init__(name, layer_idx)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of SiLU.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Result of the SiLU activation.
        """
        return MojoSiluFunction.apply(input)
