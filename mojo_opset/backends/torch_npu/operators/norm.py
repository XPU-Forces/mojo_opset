from typing import Optional
from typing import Tuple

import torch
import torch_npu

from mojo_opset.core import MojoResidualAddRMSNorm
from mojo_opset.core import MojoRMSNorm


class TorchNpuRMSNorm(MojoRMSNorm, default_priority=0):
    def __init__(
        self,
        weight: torch.Tensor,
        eps: float = 1e-05,
    ):
        super().__init__(weight, eps)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(hidden_state, self.weight, epsilon=self.variance_epsilon)[0]


class TorchNpuResidualAddRMSNorm(MojoResidualAddRMSNorm, default_priority=0):
    def __init__(
        self,
        weight: torch.Tensor,
        eps: float = 1e-05,
        norm_pos: str = "post",
    ):
        super().__init__(weight, eps, norm_pos)

    def forward(
        self,
        hidden_state: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state_out, _, residual_before_norm = torch_npu.npu_add_rms_norm(
            hidden_state, residual, self.weight, self.variance_epsilon
        )

        if self.norm_pos == "pre":
            return hidden_state_out, residual_before_norm
        else:
            return hidden_state_out, hidden_state_out
