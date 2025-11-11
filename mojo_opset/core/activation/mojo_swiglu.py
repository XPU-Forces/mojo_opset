from typing import Any
from typing import Tuple
import torch.nn.functional as F
import torch

from ..mojo_operator import MojoOperator


class MojoSwiGLU(MojoOperator):
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
