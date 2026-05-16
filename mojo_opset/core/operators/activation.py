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
    def __init__(self, swiglu_limit: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.swiglu_limit = swiglu_limit

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
            If ``swiglu_limit > 0``, ``up_out`` is clamped to
            ``[-swiglu_limit, swiglu_limit]`` and ``gate_out`` is clamped to
            a maximum of ``swiglu_limit`` before activation.
        """
        if self.swiglu_limit > 0:
            up_out = torch.clamp(up_out, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate_out = torch.clamp(gate_out, max=self.swiglu_limit)
        return torch.nn.functional.silu(gate_out) * up_out

    def extra_repr(self) -> str:
        return f"{self.swiglu_limit=}".replace("self.", "")
