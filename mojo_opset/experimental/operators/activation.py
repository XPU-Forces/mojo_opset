import math

import torch

from mojo_opset.core.operator import MojoOperator
from mojo_opset.core.operators.misc import hadamard


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
        x_shape = x.shape
        dim = x.shape[-1]
        x = x.reshape(-1, dim)
        dim_padded = 2 ** math.ceil(math.log2(dim))

        if dim != dim_padded:
            x = torch.nn.functional.pad(x, (0, dim_padded - dim))
        hadamard_tensor = hadamard(dim_padded, dtype=x.dtype, device=x.device)
        out = torch.nn.functional.linear(x, hadamard_tensor) * dim**-0.5
        return out[..., :dim].reshape(*x_shape)

__all__ = [
    "MojoRotateActivation",
]
