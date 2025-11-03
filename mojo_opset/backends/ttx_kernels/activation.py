import torch
from mojo_opset.backends.ttx_kernels.src.ascend.gelu import ttx_gelu
from mojo_opset.backends.ttx_kernels.src.ascend.silu import ttx_silu, silu_fwd, silu_bwd
from mojo_opset.backends.ttx_kernels.src.ascend.swiglu import ttx_silu_mul


from mojo_opset.core import MojoGelu, MojoSilu, MojoSiluFunction, MojoSiluMul


class TTXGelu(MojoGelu, default_priority=0):
    def forward_std(self, hidden_state: torch.Tensor):
        return ttx_gelu(hidden_state)


class TTXSilu(MojoSilu, default_priority=0):
    def forward_std(self, hidden_state: torch.Tensor):
        return ttx_silu(hidden_state)


class TTXSiluMul(MojoSiluMul, default_priority=0):
    def forward_std(self, gate_out: torch.Tensor, up_out: torch.Tensor):
        return ttx_silu_mul(gate_out, up_out)


class TTXSiluFunction(MojoSiluFunction):
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
