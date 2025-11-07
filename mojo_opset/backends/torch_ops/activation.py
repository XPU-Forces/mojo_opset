import torch
import torch_npu
from mojo_opset.core import MojoSilu, MojoSiluMul, MojoGelu


class NativeMojoSilu(MojoSilu, default_priority=0):
    def forward_std(self, hidden_state: torch.Tensor):
        return torch.nn.functional.silu(hidden_state)


class NativeMojoSiluMul(MojoSiluMul, default_priority=0):
    def forward_std(self, gate_out: torch.Tensor, up_out: torch.Tensor):
        return torch.nn.functional.silu(gate_out) * up_out


class NativeMojoGelu(MojoGelu, default_priority=0):
    def forward_std(self, hidden_state: torch.Tensor):
        return torch_npu.npu_gelu(hidden_state, approximate="tanh")
