import torch
from mojo_opset.core import MojoNorm
import torch_npu


class TorchNorm(MojoNorm, default_priority=0):
    def __init__(
        self,
        hidden_size,
        eps: float = 1e-05,
        norm_type: str = "rmsnorm",
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(hidden_size, eps, norm_type, op_name, layer_idx)

    def forward_std(self, hidden_state: torch.Tensor):
        if self.norm_type == "rmsnorm":
            return torch_npu.npu_rms_norm(hidden_state, self.weight, epsilon=self.variance_epsilon)[0]
        if self.norm_type == "layernorm":
            return torch.nn.functional.layer_norm(
                hidden_state, hidden_state.shape[-1:], weight=self.weight, bias=self.bias, eps=self.variance_epsilon
            )
        else:
            raise NotImplementedError(
                f"[NativeNorm] Only support rmsnorm or layernorm, but got {self.norm_type}")
