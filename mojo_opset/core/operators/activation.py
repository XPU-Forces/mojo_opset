import math

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


class MojoRotateActivation(MojoOperator):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hadamard rotation activation.

        Applies a normalized Walsh-Hadamard transform to the last dimension of the input.
        The input is zero-padded to the next power of 2 if needed, multiplied by the
        Hadamard matrix, scaled by ``dim ** -0.5``, then truncated back.

        Args:
            x (torch.Tensor): Input tensor with shape ``(*, D)``, where ``D`` is the
                feature dimension. Supports arbitrary leading dimensions.

        Returns:
            torch.Tensor: Same shape as input after Hadamard rotation.
        """
        from .misc import hadamard

        x_shape = x.shape
        dim = x.shape[-1]
        x = x.reshape(-1, dim)
        dim_padded = 2 ** math.ceil(math.log2(dim))

        if dim != dim_padded:
            x = torch.nn.functional.pad(x, (0, dim_padded - dim))
        hadamard_tensor = hadamard(dim_padded, dtype=x.dtype, device=x.device)
        out = torch.nn.functional.linear(x, hadamard_tensor) * dim**-0.5
        return out[..., :dim].reshape(*x_shape)
