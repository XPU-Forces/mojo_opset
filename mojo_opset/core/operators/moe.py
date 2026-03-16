from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..operator import MojoOperator


class MojoMoE(MojoOperator):
    def __init__(
        self,
        num_experts,
        # num_experts_share,
        top_k,
        hidden_size,
        intermediate_size=None,
        ffn_intermediate_size=None,
        activation: str = "swiglu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        for k in ("ep_rank", "ep_size"):
            if k in kwargs:
                raise ValueError(f"MojoMoE: {k} is not supported; use ParallelStyle to set expert partition.")

        if activation != "swiglu":
            raise NotImplementedError(f"MojoMoe: Activation {activation} is not supported.")

        # NOTE: in some cases, branches may have different expert num or topk
        self.num_experts = num_experts
        # self.num_experts_share = num_experts_share
        if intermediate_size is None:
            intermediate_size = ffn_intermediate_size
        if intermediate_size is None:
            raise ValueError("MojoMoE: intermediate_size must be provided.")

        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_hidden_size = intermediate_size

        self.gating = MojoMoEGating(hidden_size=self.hidden_size, num_experts=self.num_experts, top_k=self.top_k, **kwargs)
        self.dispatch = MojoMoEDispatch(num_experts=self.num_experts, **kwargs)
        self.experts = MojoExperts(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            activation=activation,
            **kwargs,
        )
        self.combine = MojoMoECombine(multiply_by_gates=True, **kwargs)

        if self.gating.gate_weight is not None:
            setattr(self.gating.gate_weight, "force_dtype", torch.float32)

    def forward(self, hidden_states, dp_rank_input_len=None):
        # hidden_states: [BS, H]
        top_k_indices, top_k_gates = self.gating(hidden_states)
        # top_k_indices, top_k_gates: [BS, top_k]
        sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = self.dispatch(hidden_states, top_k_gates, top_k_indices)
        # sorted_hidden_states: [local_toks, H]
        # tokens_per_expert: [num_experts]
        # sorted_gates: [local_toks]
        # token_indices: [local_toks]
        expert_outputs = self.experts(sorted_hidden_states, tokens_per_expert)
        # expert_outputs: [local_toks, H]
        # placeholder: shared_experts?
        # Note: do not use hidden_states.shape as output_buffer.shape, as it may change after dispatch
        output_buffer = torch.zeros(hidden_states.size(0), hidden_states.size(1), device=hidden_states.device, dtype=hidden_states.dtype)

        combined = self.combine(output_buffer, expert_outputs, sorted_gates, token_indices)
        # combined: [BS, H]
        return combined


class MojoMoEGating(MojoOperator):
    def __init__(
        self,
        hidden_size: Optional[int] = None,
        num_experts: Optional[int] = None,
        top_k: int = 2,
        in_features: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        if hidden_size is None:
            hidden_size = in_features
        if hidden_size is not None and num_experts is not None:
            self.gate_weight = torch.nn.Parameter(torch.empty(hidden_size, num_experts, **self.tensor_factory_kwargs))
        else:
            self.gate_weight = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if gating_weight is None:
            if self.gate_weight is None:
                raise ValueError("MojoMoEGating: gating_weight must be provided when gate_weight is not initialized.")
            gating_weight = self.gate_weight

        gate_logits = torch.matmul(hidden_states.float(), gating_weight.float())
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        expert_weights = torch.softmax(top_k_logits, dim=-1)
        return top_k_indices, expert_weights.to(hidden_states.dtype)


class MojoMoEDispatch(MojoOperator):
    def __init__(
        self,
        num_experts: int,
        **kwargs,
    ):
        """
        Common parameter definitions for MoE Dispatch operator.

        Init parameters:
        - num_experts (int): Number of experts.

        Scope: Only covers common semantics, does not involve backend communication implementation or core partitioning details.
        """
        super().__init__(**kwargs)
        self.num_experts = num_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_gates: torch.Tensor,
        top_k_indices: torch.Tensor,
    ):
        """
        Forward pass for MoE Dispatch operator.

        Input:
        - hidden_states (torch.Tensor): Input tensor.
        - top_k_gates (torch.Tensor): Top-k gating weights.
        - top_k_indices (torch.Tensor): Top-k expert indices.

        Output:
        - sorted_hidden_states: Sorted inputs for experts.
        - tokens_per_expert: Count of tokens for each expert.
        - sorted_gates: Packed gating weights.
        - token_indices: Indices for packing/unpacking.
        """
        batch_token_indices = (
            torch.arange(0, hidden_states.shape[0], device=hidden_states.device, dtype=top_k_indices.dtype)
            .unsqueeze(1)
            .repeat(1, top_k_indices.shape[-1])
            .flatten()
        )
        # batch_token_indices: [BS * top_k]
        flat_top_k_gates = top_k_gates.reshape(-1, 1)
        flat_top_k_indices = top_k_indices.flatten()
        sorted_experts, expert_sort_indices = flat_top_k_indices.sort()

        token_indices = batch_token_indices[expert_sort_indices]
        tokens_per_expert = torch.bincount(flat_top_k_indices, minlength=self.num_experts)

        sorted_gates = flat_top_k_gates[expert_sort_indices, :]
        sorted_hidden_states = hidden_states[token_indices].squeeze(1)
        return sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices


class MojoExperts(MojoOperator):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        ffn_hidden_size: int,
        activation: str = "swiglu",
        **kwargs,
    ):
        """
        Common parameter definitions for MoE Experts operator.

        Init parameters:
        - num_experts (int): Number of experts.
        - hidden_size (int): Hidden size of the model.
        - ffn_hidden_size (int): Hidden size of the feed-forward network within each expert.
        - activation (str): Activation function to use.

        Scope: Only covers common parameters, does not involve backend specialization.
        """
        super().__init__(**kwargs)
        if activation != "swiglu":
            raise NotImplementedError(f"MojoExperts: Activation {activation} is not supported.")
        self.activation = activation

        self.fc1 = nn.Parameter(torch.empty(num_experts, ffn_hidden_size * 2, hidden_size, **self.tensor_factory_kwargs))
        self.fc2 = nn.Parameter(torch.empty(num_experts, hidden_size, ffn_hidden_size, **self.tensor_factory_kwargs))

    def forward(
        self,
        sorted_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ):
        # Mocked GroupGemm
        expert_inputs = torch.split(sorted_hidden_states, tokens_per_expert.tolist(), dim=0)
        num_experts = len(expert_inputs)
        
        fc1_outs = [F.linear(expert_inputs[i], self.fc1[i]) for i in range(num_experts)]
        activated_outs = []
        for fc1_out in fc1_outs:
            gate_proj, up_proj = fc1_out.chunk(2, dim=-1)
            activated_outs.append(F.silu(gate_proj) * up_proj)
            
        fc2_outs = [F.linear(activated_outs[i], self.fc2[i]) for i in range(num_experts)]
        return torch.cat(fc2_outs, dim=0)


class MojoMoECombine(MojoOperator):
    def __init__(
        self,
        multiply_by_gates: bool = True,
        **kwargs,
    ):
        """
        Common parameter definitions for MoE Combine operator.

        Init parameters:
        - multiply_by_gates (bool): Whether to multiply the expert output by the gating weights.

        Scope: Only covers common semantics, does not involve backend communication or core partitioning details.
        """
        super().__init__(**kwargs)
        self.multiply_by_gates = multiply_by_gates  

    def forward(
        self,
        output_buffer: torch.Tensor,
        expert_outputs: torch.Tensor,
        sorted_gates: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for MoE Combine operator.

        Input:
        - output_buffer (torch.Tensor): Initial tensor to combine results into.
        - expert_outputs (torch.Tensor): Output from experts.
        - sorted_gates (torch.Tensor): Packed gating weights.
        - token_indices (torch.Tensor): Indices for packing/unpacking.

        Output:
        - combined: Combined output tensor.
        """
        dtype = expert_outputs.dtype
        combined_expert_outputs = expert_outputs.float()
        if self.multiply_by_gates:
            combined_expert_outputs = combined_expert_outputs.mul(sorted_gates).to(dtype=dtype)

        scatter_indices = token_indices.unsqueeze(-1).expand(-1, output_buffer.size(1))
        combined = output_buffer.scatter_reduce(0, scatter_indices, combined_expert_outputs, reduce="sum", include_self=True)
        return combined.to(dtype=dtype)
