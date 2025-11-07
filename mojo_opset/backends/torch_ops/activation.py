import torch
import torch_npu
from mojo_opset.core import MojoSilu, MojoSiluMul, MojoGelu


class TorchSilu(MojoSilu, default_priority=0):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward_std(self, hidden_state: torch.Tensor):
        return torch.nn.functional.silu(hidden_state)


class TorchSiluMul(MojoSiluMul, default_priority=0):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward_std(self, gate_out: torch.Tensor, up_out: torch.Tensor):
        return torch.nn.functional.silu(gate_out) * up_out


class TorchGelu(MojoGelu, default_priority=0):
    def __init__(self, op_name: str = "", layer_idx: int = 0):
        super().__init__(op_name, layer_idx)

    def forward_std(self, hidden_state: torch.Tensor):
        return torch_npu.npu_gelu(hidden_state, approximate="tanh")
