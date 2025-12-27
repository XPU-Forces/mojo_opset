import torch

from mojo_opset.core import MojoGelu
from mojo_opset.core import MojoSilu
from mojo_opset.core import MojoSwiGLU


class RefGelu(MojoGelu):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # return torch.nn.GELU()(hidden_state)
        return torch.nn.functional.gelu(hidden_state)


class RefSilu(MojoSilu):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)
        # self._forward = torch.nn.SiLU()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(hidden_state)


class RefSwiGLU(MojoSwiGLU):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward(self, gate_out: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(gate_out) * up_out
