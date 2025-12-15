from typing import Any
from typing import Tuple
import torch.nn.functional as F
import torch

from ..mojo_operator import MojoOperator


class MojoSilu(MojoOperator):
    forward_ref = torch.nn.SiLU()

    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward_std(self, hidden_state: torch.Tensor) -> Tuple[Any]:
        pass

    def forward_analysis(self, hidden_state) -> Tuple[int, int, int]:
        pass


class MojoSiluQuant(MojoOperator):
    pass


class MojoSiluMul(MojoOperator):
    def __init__(
        self,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

    def forward_ref(self, gate_out: torch.Tensor, up_out: torch.Tensor) -> Tuple[Any]:
        out = F.silu(gate_out) * up_out
        return out

    def forward_std(self, gate_out: torch.Tensor, up_out: torch.Tensor) -> Tuple[Any]:
        pass

    def forward_analysis(self, gate_out: torch.Tensor, up_out: torch.Tensor) -> Tuple[int, int, int]:
        pass


class MojoSwiglu(MojoOperator):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward_ref(self, hidden_state: torch.Tensor) -> torch.Tensor:
        gate, up = hidden_state.chunk(2, dim=-1)
        return F.silu(gate) * up

    def forward_std(self, hidden_state: torch.Tensor) -> Tuple[Any]:
        pass

    def forward_analysis(self, hidden_state) -> Tuple[int, int, int]:
        pass