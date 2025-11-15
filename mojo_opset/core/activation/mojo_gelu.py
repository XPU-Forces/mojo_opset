from typing import Any
from typing import Tuple

import torch

from ..mojo_operator import MojoOperator


class MojoGelu(MojoOperator):
    forward_ref = torch.nn.GELU()

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

    def forward_ref(self, hidden_state: torch.Tensor) -> Tuple[Any]:
        pass

    def forward_analysis(self, hidden_state) -> Tuple[int, int, int]:
        pass
