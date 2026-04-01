from typing import Optional
from typing import Tuple

import torch
import torch_npu

from mojo_opset.core import MojoResidualAddRMSNorm
from mojo_opset.core import MojoRMSNorm
from mojo_opset.core import MojoResidualAddNormCast


class TorchNpuRMSNorm(MojoRMSNorm, default_priority=0):
    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-05,
        **kwargs,
    ):
        super().__init__(norm_size, eps, **kwargs)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(hidden_state, self.weight, epsilon=self.variance_epsilon)[0]


class TorchNpuResidualAddRMSNorm(MojoResidualAddRMSNorm, default_priority=0):
    def __init__(
        self,
        norm_size: int,
        eps: float = 1e-05,
        norm_pos: str = "post",
        **kwargs,
    ):
        super().__init__(norm_size, eps, norm_pos, **kwargs)

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

class TorchNpuResidualAddNormCast(MojoResidualAddNormCast):
    supported_platforms_list = ["npu"]

    def forward(
        self, 
        hidden_state: torch.Tensor, 
        residual: torch.Tensor, 
        weight: torch.Tensor=None
        ) -> torch.Tensor:
        """
        Forward pass with residual add normalization and cast to compute_dtype.

        Args:
            hidden_state (torch.Tensor): Input tensor of any shape.
            residual (torch.Tensor): Residual tensor of the same shape as hidden_state.
            weight (torch.Tensor, optional): Weight tensor for scaling. Defaults to None.

        Returns:
            torch.Tensor: Same shape as input with element-wise normalization applied.
        """
        if weight is None:
            weight = self.weight
        if hidden_state.dtype in (torch.float16, torch.bfloat16):
            compute_dtype = hidden_state.dtype
        else:
            raise ValueError(f"unsupported dtype {hidden_state.dtype}")

        residual = residual.to(compute_dtype)
        self.weight = self.weight.to(compute_dtype)
        hidden_state, _, _, residual = torch_npu.npu_add_rms_norm_cast(hidden_state, residual, weight, self.variance_epsilon)
        if self.norm_pos != "pre":
            residual = hidden_state
        return hidden_state, residual
    
