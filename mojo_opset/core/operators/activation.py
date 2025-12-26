from typing import Any
from typing import Tuple

import torch

from ..mojo_operator import MojoOperator


class MojoGelu(MojoOperator):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward_std(self, hidden_state: torch.Tensor) -> Tuple[Any]:
        pass

    def forward_analysis(self, hidden_state) -> Tuple[int, int, int]:
        pass


class MojoGeluQuant(MojoOperator):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward_std(self, hidden_state: torch.Tensor) -> torch.Tensor:
        pass

    def forward_analysis(self, hidden_state) -> Tuple[int, int, int]:
        pass


class MojoSilu(MojoOperator):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward_std(self, hidden_state: torch.Tensor) -> Tuple[Any]:
        pass

    def forward_analysis(self, hidden_state) -> Tuple[int, int, int]:
        pass


class MojoSiluQuant(MojoOperator):
    pass


class MojoSwiGLU(MojoOperator):
    def __init__(
        self,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

    def forward_std(self, gate_out: torch.Tensor, up_out: torch.Tensor) -> Tuple[Any]:
        pass

    def forward_analysis(self, gate_out: torch.Tensor, up_out: torch.Tensor) -> Tuple[int, int, int]:
        pass

