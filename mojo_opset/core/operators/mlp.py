
import torch

from ..operator import MojoOperator


class MojoSwiGLUMLP(MojoOperator):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
    ):
        """
        SwiGLU (Swish Gated Linear Unit) is a specific type of Gated Linear Unit
        that uses the Swish (SiLU) activation function. It's often used in
        Transformer-based models (like LLaMA, PaLM) as a replacement for standard FFNs.


        Args:
            input_size (int): Size of the input tensor features.
            output_size (int): Size of the output tensor features.
            hidden_size (int): Size of the hidden dimension (often referred to as intermediate size).
        """
        super().__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size * 2, bias=False)
        self.fc2 = torch.nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.fc1(x)
        a1, a2 = a.chunk(2, dim=-1)
        b = torch.nn.functional.silu(a1) * a2
        out = self.fc2(b)

        return out

