import torch
from typing import Tuple

from mojo_opset.backends.ttx.kernels import moe_combine
from mojo_opset.backends.ttx.kernels import moe_dispatch
from mojo_opset.backends.ttx.kernels import moe_experts
from mojo_opset.backends.ttx.kernels import moe_gating
from mojo_opset.backends.ttx.kernels import moe_gating_top_k_hash_infer
from mojo_opset.backends.ttx.kernels import quant_moe_experts
from mojo_opset.core import MojoExperts
from mojo_opset.core import MojoMoE
from mojo_opset.core import MojoMoECombine
from mojo_opset.core import MojoMoEDispatch
from mojo_opset.core import MojoMoEGating
from mojo_opset.core import MojoMoEGatingTopKHash
from mojo_opset.core import MojoQuantExperts
from mojo_opset.core import MojoQuantMoE


class TTXMoEGating(MojoMoEGating):
    supported_platforms_list = ["ilu"]

    def forward(
        self,
        hidden_states: torch.Tensor,  # (num_tokens, hidden_size)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (top_k_indices, top_k_gates).

        Args:
            hidden_states: (num_tokens, hidden_size), fp16/bf16/fp32.
        Returns:
            top_k_indices: (num_tokens, top_k), int32.
            top_k_gates:   (num_tokens, top_k), fp32.
        """
        assert self.gate_weight.dtype == torch.float32
        return moe_gating(hidden_states, self.gate_weight, self.top_k)


class TTXMoEGatingTopKHash(MojoMoEGatingTopKHash):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor = None,
        tid2eid: torch.Tensor = None,
    ):
        return moe_gating_top_k_hash_infer(
            x,
            self.k,
            input_ids=input_ids,
            tid2eid=tid2eid,
            routed_scaling_factor=self.routed_scaling_factor,
            eps=self.eps,
            norm_type=self.norm_type,
            out_flag=self.out_flag,
        )


class TTXMoEDispatch(MojoMoEDispatch):
    supported_platforms_list = ["ilu"]

    def forward(
        self,
        hidden_states: torch.Tensor,   # (num_tokens, hidden_size)
        top_k_gates: torch.Tensor,     # (num_tokens, top_k)
        top_k_indices: torch.Tensor,   # (num_tokens, top_k)
    ):
        """Sort tokens by expert id and gather into contiguous layout.

        Returns:
            sorted_hidden_states: (num_tokens * top_k, hidden_size).
            tokens_per_expert:    (num_experts,), int32.
            sorted_gates:         (num_tokens * top_k, 1).
            token_indices:        (num_tokens * top_k,), int32.
        """
        return moe_dispatch(hidden_states, top_k_gates, top_k_indices, self.num_experts)


class TTXExperts(MojoExperts):
    supported_platforms_list = ["ilu"]

    def forward(
        self,
        sorted_hidden_states: torch.Tensor,  # (num_tokens * top_k, hidden_size)
        tokens_per_expert: torch.Tensor,      # (num_experts,), int32
    ):
        """Grouped up_proj (fused SwiGLU) + down_proj.

        Returns:
            expert_outputs: (num_tokens * top_k, hidden_size).
        """
        return moe_experts(
            sorted_hidden_states,
            tokens_per_expert,
            self.up_proj_weight,   # (num_experts, 2 * intermediate_size, hidden_size)
            self.down_proj_weight, # (num_experts, hidden_size, intermediate_size)
        )


class TTXMoECombine(MojoMoECombine):
    supported_platforms_list = ["ilu"]

    def forward(
        self,
        output_buffer: torch.Tensor,   # (num_tokens, hidden_size)
        expert_outputs: torch.Tensor,   # (num_tokens * top_k, hidden_size)
        sorted_gates: torch.Tensor,     # (num_tokens * top_k, 1)
        token_indices: torch.Tensor,    # (num_tokens * top_k,), int32
    ) -> torch.Tensor:
        """Weighted scatter-add expert outputs back to token positions.

        Returns:
            output_buffer: (num_tokens, hidden_size), same dtype as input.
        """
        return moe_combine(
            output_buffer,
            expert_outputs,
            sorted_gates,
            token_indices,
            self.multiply_by_gates,
        )


class TTXQuantExperts(MojoQuantExperts):
    supported_platforms_list = ["ilu"]

    def load_state_dict(self, state_dict, strict=True):
        from mojo_opset.backends.ttx.kernels.ilu.moe_quant_experts import clear_quant_moe_weight_unpack_cache

        clear_quant_moe_weight_unpack_cache(self)
        return super().load_state_dict(state_dict, strict)

    def forward(
        self,
        sorted_hidden_states: torch.Tensor,  # (num_tokens * top_k, hidden_size)
        tokens_per_expert: torch.Tensor,      # (num_experts,), int32
    ):
        """Quantized grouped experts: smooth_quant → int8 GEMM + SwiGLU → smooth_quant → int8 GEMM.

        Returns:
            expert_outputs: (num_tokens * top_k, hidden_size).
        """
        return quant_moe_experts(self, sorted_hidden_states, tokens_per_expert)


class TTXMoE(MojoMoE):
    """Staged MoE forward: ``gating`` -> ``dispatch`` -> ``experts`` -> ``combine``."""
    supported_platforms_list = ["ilu"]
    _use_fused_moe = False


class TTXQuantMoE(MojoQuantMoE):
    supported_platforms_list = ["ilu"]
    _use_fused_moe = False
