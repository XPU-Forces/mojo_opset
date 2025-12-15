import torch
from mojo_opset.core import MojoNorm, MojoResidualAddNorm
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


class TorchResidualAddNorm(MojoResidualAddNorm, default_priority=0):
    def __init__(
        self,
        weight: torch.Tensor = None,
        bias: torch.Tensor = None,
        eps: float = 1e-05,
        norm_pos: str = "post",
        norm_type: str = "rmsnorm",
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(weight, bias, eps, norm_pos, norm_type, op_name, layer_idx)

    def forward_std(self, hidden_state: torch.Tensor, residual: torch.Tensor = None):
        if self.norm_type != "rmsnorm" or residual is None:
            # Fallback to reference implementation if not rmsnorm or no residual is provided
            return self.forward_ref(hidden_state, residual)

        # Use the NPU fused kernel for the main computation
        hidden_state_out, _, residual_before_norm = torch_npu.npu_add_rms_norm(
            hidden_state, residual, self.weight, self.eps
        )

        if self.norm_pos == "pre":
            # For pre-norm, ref returns (norm_out, pre_norm_residual), which matches the kernel's output pattern.
            return hidden_state_out, residual_before_norm
        else:  # self.norm_pos == "post"
            # For post-norm, ref returns (norm_out, norm_out). We must match this contract.
            return hidden_state_out, hidden_state_out
