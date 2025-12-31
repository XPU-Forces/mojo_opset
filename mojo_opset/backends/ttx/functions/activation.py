from mojo_opset.backends.ttx.kernels import silu_bwd
from mojo_opset.backends.ttx.kernels import silu_fwd
from mojo_opset.core import MojoSiluFunction


class TTXSiluFunction(MojoSiluFunction):
    supported_platforms_list = ["npu"]

    @staticmethod
    def forward(ctx, input):
        """
        Forward pass of SiLU function.

        Args:
            input: Input tensor

        Returns:
            y: Output tensor y = silu(input) = input * sigmoid(input)
        """
        y = silu_fwd(input)
        ctx.save_for_backward(input)
        return y

    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass of SiLU function.

        Args:
            dy: Gradient w.r.t. output

        Returns:
            dx: Gradient w.r.t. input
        """
        (input,) = ctx.saved_tensors
        dx = silu_bwd(dy, input)
        return dx
