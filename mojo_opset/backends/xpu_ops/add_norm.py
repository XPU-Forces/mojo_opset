import torch

from xpu_ops.modules import FusedNorm

from mojo_opset.core import MojoResidualAddNorm


class XOpsResidualAddNorm(MojoResidualAddNorm, default_priority=1):
    def __init__(
        self,
        weight: torch.Tensor = None,
        bias: torch.Tensor = None,
        eps: float = 1e-05,
        norm_pos: str = "post",
        norm_type: str = "rms",
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(weight, bias, eps, norm_pos, norm_type, op_name, layer_idx)
        self.norm_type = norm_type

        if norm_type == "layernorm" or norm_pos == "post":
            raise NotImplementedError("[XOpsResidualAddNorm] Only support rmsnorm with pre-norm residual output.")

        self.op = FusedNorm(norm_mode=norm_type, eps=eps, pre_norm=True)

    def forward_std(self, hidden_state: torch.Tensor, residual: torch.Tensor = None):
        if hidden_state.dtype not in [torch.half, torch.bfloat16]:
            raise NotImplementedError(
                f"[XOpsResidualAddNorm] Only support fp16/bf16 dtype input, but got {hidden_state.dtype}"
            )

        if self.norm_type == "rmsnorm":
            res, out = self.op(input=hidden_state, weight=self.weight, residual=residual)
            return out, res  # note that order is reversed!
        else:
            raise NotImplementedError(f"[XOpsResidualAddNorm] Only support rmsnorm, but got {self.norm_type}")
