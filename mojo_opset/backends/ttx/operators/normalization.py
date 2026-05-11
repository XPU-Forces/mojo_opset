from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels import fused_add_layernorm_infer
from mojo_opset.backends.ttx.kernels import fused_add_rmsnorm_infer
from mojo_opset.backends.ttx.kernels import layernorm_infer
from mojo_opset.backends.ttx.kernels import rmsnorm_infer
from mojo_opset.backends.ttx.kernels import group_rmsnorm
from mojo_opset.backends.ttx.kernels import rmsnorm_quant_infer
from mojo_opset.core import MojoLayerNorm
from mojo_opset.core import MojoResidualAddLayerNorm
from mojo_opset.core import MojoResidualAddRMSNorm
from mojo_opset.core import MojoRMSNorm
from mojo_opset.core import MojoGroupRMSNorm
from mojo_opset.core import MojoRMSNormQuant

class TTXGroupRMSNorm(MojoGroupRMSNorm):
    supported_platforms_list = ["mlu"]

    def forward(self, input_groups) -> list[torch.tensor]:
        return group_rmsnorm(input_groups, weight = self.weight,
                             eps = self.variance_epsilon, output_like_input_stride = True)

class TTXLayerNorm(MojoLayerNorm):
    supported_platforms_list = ["npu", "ilu", "mlu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return layernorm_infer(hidden_state, self.weight, self.bias, self.variance_epsilon)


class TTXRMSNorm(MojoRMSNorm):
    supported_platforms_list = ["npu", "ilu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return rmsnorm_infer(hidden_state, self.weight, self.variance_epsilon)


class TTXResidualAddRMSNorm(MojoResidualAddRMSNorm):
    supported_platforms_list = ["npu", "ilu"]

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
    supported_platforms_list = ["npu", "ilu"]

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

class TTXRMSNormQuant(MojoRMSNormQuant):
    supported_platforms_list = ["ilu"]

    def forward(self, hidden_state: torch.Tensor, smooth_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        return rmsnorm_quant_infer(hidden_state, smooth_scale, self.weight, self.variance_epsilon, self.q_min, self.q_max, self.quant_dtype)
