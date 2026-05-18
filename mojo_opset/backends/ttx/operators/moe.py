import torch
from typing import Tuple

from mojo_opset.backends.ttx.kernels import moe_combine
from mojo_opset.backends.ttx.kernels import moe_dispatch
from mojo_opset.backends.ttx.kernels import moe_experts
from mojo_opset.backends.ttx.kernels import moe_gating
from mojo_opset.backends.ttx.kernels import quant_moe_experts
from mojo_opset.core import MojoExperts
from mojo_opset.core import MojoMoE
from mojo_opset.core import MojoMoECombine
from mojo_opset.core import MojoMoEDispatch
from mojo_opset.core import MojoMoEGating
from mojo_opset.core import MojoQuantExperts
from mojo_opset.core import MojoQuantMoE


class TTXMoEGating(MojoMoEGating):
    supported_platforms_list = ["npu", "ilu"]

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.gate_weight.dtype == torch.float32
        return moe_gating(hidden_states, self.gate_weight, self.top_k)


class TTXMoEDispatch(MojoMoEDispatch):
    supported_platforms_list = ["npu", "ilu"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_gates: torch.Tensor,
        top_k_indices: torch.Tensor,
    ):
        return moe_dispatch(hidden_states, top_k_gates, top_k_indices, self.num_experts)


class TTXExperts(MojoExperts):
    supported_platforms_list = ["npu", "ilu"]

    def forward(
        self,
        sorted_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ):
        return moe_experts(
            sorted_hidden_states,
            tokens_per_expert,
            self.up_proj_weight,
            self.down_proj_weight,
        )


class TTXMoECombine(MojoMoECombine):
    supported_platforms_list = ["npu", "ilu"]

    def forward(
        self,
        output_buffer: torch.Tensor,
        expert_outputs: torch.Tensor,
        sorted_gates: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        return moe_combine(
            output_buffer,
            expert_outputs,
            sorted_gates,
            token_indices,
            self.multiply_by_gates,
        )


class TTXMoE(MojoMoE):
    """Staged MoE forward (same control flow as :class:`IxformerQuantMoE` / :meth:`MojoMoE.forward`).

    ``gating`` → ``dispatch`` → ``experts`` → ``combine``. Each stage uses the platform-specific
    ``moe_*`` helpers from :mod:`mojo_opset.backends.ttx.kernels` (ILU dense MoE: Triton gating,
    dispatch on CUDA after ``bincount``/``argsort`` and fused gather, experts, and combine).
    """

    supported_platforms_list = ["npu", "ilu"]

    def forward(self, hidden_states):
        top_k_indices, top_k_gates = self.gating(hidden_states)
        sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = self.dispatch(
            hidden_states, top_k_gates, top_k_indices
        )
        expert_outputs = self.experts(sorted_hidden_states, tokens_per_expert)
        output_buffer = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
        return self.combine(output_buffer, expert_outputs, sorted_gates, token_indices)


class TTXQuantExperts(MojoQuantExperts):
    supported_platforms_list = ["npu", "ilu"]

    def load_state_dict(self, state_dict, strict=True):
        from mojo_opset.backends.ttx.kernels.ilu.moe_quant_experts import clear_quant_moe_weight_unpack_cache

        clear_quant_moe_weight_unpack_cache(self)
        return super().load_state_dict(state_dict, strict)

    def forward(self, sorted_hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor):
        return quant_moe_experts(self, sorted_hidden_states, tokens_per_expert)


class TTXQuantMoE(MojoQuantMoE):
    supported_platforms_list = ["npu", "ilu"]
