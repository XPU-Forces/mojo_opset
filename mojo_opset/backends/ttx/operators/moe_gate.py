from typing import Optional
import torch


from mojo_opset.backends.ttx.kernels.npu.moe_gate import (
    MoeGating_forward
)
from mojo_opset.core import MojoMoEGating

class TTxMoEGating(MojoMoEGating):
    def forward(
        self,
        hidden_states: torch.Tensor
    ):
        # _, S, D = hidden_states.shape
        # # 权重维度
        # E, _ = self.gate_weight.shape

        # if hidden_states.ndim != 3:
        #         raise ValueError(f"Expected BNSD when is_varlen=False; got shape {tuple(hidden_states.shape)}")
        #     #使用TD格式
        # else:
        #     hidden_states = hidden_states.view(-1, D) #[T,D]
        return MoeGating_forward(hidden_states, self.gate_weight, self.top_k)