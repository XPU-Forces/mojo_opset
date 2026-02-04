import torch
import math
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


class MojoIndexerRotateActivation(MojoOperator):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with GELU activation.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Same shape as input with element-wise GELU applied.
        """

        def hadamard(n: int, dtype, device):
            """Torch version hadamard matrix generation"""
            if n < 1:
                lg2 = 0
            else:
                lg2 = int(math.log(n, 2))

            if 2**lg2 != n:
                raise ValueError(f"n must be a power of 2, but got {n}")

            H = torch.tensor([1], dtype=dtype, device=device)
            for i in range(0, lg2):
                H = torch.vstack((torch.hstack((H, H)), torch.hstack((H, -H))))
            return H
        
        hidden_size = x.size(-1)
        x_shape = x.shape
        dim = x.shape[-1]
        x = x.reshape(-1, dim)
        log_dim = math.ceil(math.log2(dim))
        dim_padded = 2**log_dim
        if dim != dim_padded:
            x = torch.nn.functional.pad(x, (0, dim_padded - dim))
        out = torch.nn.functional.linear(x, torch.tensor(hadamard(dim_padded, dtype=float, device=x.device), dtype=x.dtype, device=x.device))
        out = out * hidden_size**-0.5
        return out[..., :dim].reshape(*x_shape)