import torch

from mojo_opset.backends.ttx.kernels import layernorm_infer
from mojo_opset.backends.ttx.kernels import rmsnorm_infer
from mojo_opset.backends.ttx.kernels import fused_add_rmsnorm_infer
from mojo_opset.backends.ttx.kernels import fused_add_layernorm_infer
from mojo_opset.core import MojoLayerNorm
from mojo_opset.core import MojoResidualAddLayerNorm
from mojo_opset.core import MojoResidualAddRMSNorm
from mojo_opset.core import MojoRMSNorm


class TTXLayerNorm(MojoLayerNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return layernorm_infer(hidden_state, self.weight, self.bias, self.variance_epsilon)


class TTXRMSNorm(MojoRMSNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return rmsnorm_infer(hidden_state, self.weight, self.variance_epsilon)


class TTXResidualAddRMSNorm(MojoResidualAddRMSNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor = None):
        output, res = fused_add_rmsnorm_infer(
            hidden_state,
            residual,
            self.weight,
            self.norm_pos,
            self.variance_epsilon,
        )

        return output, res


class TTXResidualAddLayerNorm(MojoResidualAddLayerNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor = None):
        output, res = fused_add_layernorm_infer(
            hidden_state,
            residual,
            self.weight,
            self.bias,
            self.norm_pos,
            self.variance_epsilon,
        )

        return output, res
