import torch

from ..operator import MojoOperator


class MojoGelu(MojoOperator):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with GELU activation.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Same shape as input with element-wise GELU applied.
        """
        return torch.nn.functional.gelu(x)


class MojoSilu(MojoOperator):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SiLU (Swish) activation.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Same shape as input with element-wise SiLU applied.

        Notes:
            Uses torch.nn.functional.silu; preserves dtype and device.
            SiLU is defined as x * sigmoid(x).
        """
        return torch.nn.functional.silu(x)


class MojoSwiGLU(MojoOperator):
    def forward(self, gate_out: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.

        Args:
            gate_out (torch.Tensor): Gate output tensor of any shape.
            up_out (torch.Tensor): Up output tensor of any shape.

        Returns:
            torch.Tensor: Same shape as gate_out with element-wise SwiGLU applied.

        Notes:
            SwiGLU is defined as SiLU(gate_out) * up_out.
        """
        return torch.nn.functional.silu(gate_out) * up_out
