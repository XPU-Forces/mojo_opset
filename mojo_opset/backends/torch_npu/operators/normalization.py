import torch
import torch_npu

from mojo_opset.core import MojoLayerNorm
from mojo_opset.core import MojoRMSNorm
from mojo_opset.core import MojoNormQuant
from mojo_opset.core import MojoResidualAddRMSNorm
from mojo_opset.core import MojoResidualAddLayerNorm
from mojo_opset.core import MojoResidualAddNormCast



class TorchNpuLayerNorm(MojoLayerNorm):
    supported_platforms_list = ["npu"]


class TorchNpuRMSNorm(MojoRMSNorm):
    supported_platforms_list = ["npu"]
    
    def forward(self, hidden_state: torch.Tensor, weight: torch.Tensor=None) -> torch.Tensor:
        if weight is None:
            weight = self.weight
        return torch_npu.npu_rms_norm(hidden_state, weight, self.variance_epsilon)[0]

class TorchNpuResidualAddRMSNorm(MojoResidualAddRMSNorm):
    supported_platforms_list = ["npu"] 

    def forward(
        self, 
        hidden_state: torch.Tensor, 
        residual: torch.Tensor, 
        weight: torch.Tensor=None
        ) -> torch.Tensor:
        """
        Forward pass with residual add RMS normalization.

        Args:
            hidden_state (torch.Tensor): Input tensor of any shape.
            residual (torch.Tensor): Residual tensor of the same shape as hidden_state.
            weight (torch.Tensor, optional): Weight tensor for scaling. Defaults to None.

        Returns:
            torch.Tensor: Same shape as input with element-wise RMSNorm applied.
        """
        if weight is None:
            weight = self.weight
        hidden_state, _, residual = torch_npu.npu_add_rms_norm(hidden_state, residual, weight, self.variance_epsilon)
        if self.norm_pos != "pre":
            residual = hidden_state
        return hidden_state, residual
    

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