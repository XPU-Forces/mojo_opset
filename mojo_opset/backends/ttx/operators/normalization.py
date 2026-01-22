import torch

from mojo_opset.backends.ttx.kernels import rmsnorm_infer
from mojo_opset.backends.ttx.kernels.npu.fused_add_layer_norm import ttx_fused_add_layer_norm
from mojo_opset.backends.ttx.kernels.npu.fused_add_rms_norm import ttx_fused_add_rms_norm
from mojo_opset.backends.ttx.kernels.npu.layernorm import ttx_layer_norm
from mojo_opset.core import MojoLayerNorm
from mojo_opset.core import MojoResidualAddLayerNorm
from mojo_opset.core import MojoResidualAddRMSNorm
from mojo_opset.core import MojoRMSNorm


class TTXLayerNorm(MojoLayerNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return ttx_layer_norm(hidden_state, self.weight, self.bias, self.variance_epsilon)


class TTXRMSNorm(MojoRMSNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return rmsnorm_infer(hidden_state, self.weight, self.variance_epsilon)


class TTXResidualAddRMSNorm(MojoResidualAddRMSNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor = None):
        output, res = ttx_fused_add_rms_norm(
            hidden_states=hidden_state,
            residual=residual,
            add_mode=self.norm_pos,
            eps=self.variance_epsilon,
            weight=self.weight,
        )

        return output, res


class TTXResidualAddLayerNorm(MojoResidualAddLayerNorm):
    supported_platforms_list = ["npu"]

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor = None):
        output, res = ttx_fused_add_layer_norm(
            hidden_states=hidden_state,
            residual=residual,
            add_mode=self.norm_pos,
            eps=self.variance_epsilon,
            weight=self.weight,
            bias=self.bias,
        )

        return output, res
