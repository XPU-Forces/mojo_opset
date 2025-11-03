import torch

from mojo_opset.core import MojoNorm
from xpu_ops.modules import FusedNorm


class XOpsNorm(MojoNorm, default_priority=1):
    def __init__(
        self,
        hidden_size,
        eps: float = 1e-05,
        norm_type: str = "rmsnorm",
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(hidden_size=hidden_size, eps=eps, norm_type=norm_type, op_name=op_name, layer_idx=layer_idx)
        self.norm_type = norm_type
        self.op = FusedNorm(norm_mode=norm_type, eps=eps)

        # TODO(guoshuaishuai.x): layer norm requires weight and bias to be float32. Remove this constraint later.
        if self.norm_type == "layernorm":
            self.weight_fp32 = self.weight.float()
            if self.bias is not None:
                self.bias_fp32 = self.bias.float()

    def forward_std(self, hidden_state: torch.Tensor):
        if hidden_state.dtype not in [torch.half, torch.bfloat16]:
            raise NotImplementedError(f"[XOpsNorm] Only support fp16/bf16 dtype input, but got {hidden_state.dtype}")

        # xops RMSNorm has precision issues; temporarily disabled
        # if self.norm_type == "rmsnorm":
        #     return self.op(input=hidden_state, weight=self.weight)
        if self.norm_type == "layernorm":
            original_shape = hidden_state.shape
            hidden_state_bsd = hidden_state.view(1, -1, hidden_state.size(-1))
            return self.op(input=hidden_state_bsd, weight=self.weight_fp32, bias=self.bias_fp32).view(original_shape)
        else:
            raise NotImplementedError(f"[XOpsNorm] Only support layernorm, but got {self.norm_type}")
