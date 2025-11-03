import torch

from mojo_opset.backends.ttx_kernels.src.ascend.fused_add_layer_norm import ttx_fused_add_layer_norm
from mojo_opset.backends.ttx_kernels.src.ascend.fused_add_rms_norm import ttx_fused_add_rms_norm

from mojo_opset.core import MojoResidualAddNorm


class TTXResidualAddNorm(MojoResidualAddNorm, default_priority=2):
    def __init__(
        self,
        weight: torch.Tensor = None,
        bias: torch.Tensor = None,
        eps: float = 1e-05,
        norm_pos: str = "pre",
        norm_type: str = "rmsnorm",
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(weight, bias, eps, norm_pos, norm_type, op_name, layer_idx)
        self.norm_type = norm_type

    def forward_std(self, hidden_state: torch.Tensor, residual: torch.Tensor = None):
        if self.norm_type == "rmsnorm":
            norm_func = ttx_fused_add_rms_norm
            kwargs = dict(weight=self.weight)
        elif self.norm_type == "layernorm":
            norm_func = ttx_fused_add_layer_norm
            kwargs = dict(weight=self.weight, bias=self.bias)
        else:
            raise NotImplementedError(
                f"[TTXResidualAddNorm] Only support rmsnorm and layernorm, but got {self.norm_type}"
            )

        output, res = norm_func(
            hidden_states=hidden_state,
            residual=residual,
            add_mode=self.norm_pos,
            eps=self.eps,
            **kwargs,
        )

        return output, res
