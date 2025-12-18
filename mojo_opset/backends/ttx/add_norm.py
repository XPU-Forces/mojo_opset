import torch

from mojo_opset.backends.ttx.kernels.ascend.fused_add_layer_norm import ttx_fused_add_layer_norm
from mojo_opset.backends.ttx.kernels.ascend.fused_add_rms_norm import ttx_fused_add_rms_norm
from mojo_opset.core import MojoResidualAddNorm


class TTXResidualAddNorm(MojoResidualAddNorm, default_priority=2):
    supported_platforms_list = ["npu"]

    def forward_std(self, hidden_state: torch.Tensor, residual: torch.Tensor = None):
        if self.norm_type == "rmsnorm":
            norm_func = ttx_fused_add_rms_norm
            kwargs = dict(weight=self.gamma)
        elif self.norm_type == "layernorm":
            norm_func = ttx_fused_add_layer_norm
            kwargs = dict(weight=self.gamma, bias=self.beta)
        else:
            raise NotImplementedError(
                f"[TTXResidualAddNorm] Only support rmsnorm and layernorm, but got {self.norm_type}"
            )

        output, res = norm_func(
            hidden_states=hidden_state,
            residual=residual,
            add_mode=self.norm_pos,
            eps=self.epsilon,
            **kwargs,
        )

        return output, res
