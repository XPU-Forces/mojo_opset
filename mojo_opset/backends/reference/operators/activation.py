import torch
import torch.nn.functional as F

from mojo_opset.core import MojoGelu
from mojo_opset.core import MojoSilu
from mojo_opset.core import MojoSwiGLU
from mojo_opset.utils.mode import LAST_PRIORITY


class RefGelu(MojoGelu, default_priority=LAST_PRIORITY):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)
        self._forward = torch.nn.GELU()

    def forward_std(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self._forward(hidden_state)


class RefSilu(MojoSilu, default_priority=LAST_PRIORITY):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)
        self._forward = torch.nn.SiLU()

    def forward_std(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self._forward(hidden_state)


class RefSwiGLU(MojoSwiGLU, default_priority=LAST_PRIORITY):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward_std(self, gate_out: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
        out = F.silu(gate_out) * up_out
        return out
