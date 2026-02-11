from typing import Optional
from typing import Tuple

import torch

from ..operator import MojoOperator


class MojoMoE(MojoOperator):
    def __init__(
        self,
        hidden_size: int,
        ffn_intermediate_size: int,
        num_experts: int,
        top_k: int,
        activation: str = "swiglu",
        ep_size: Optional[int] = None,
        ep_rank: Optional[int] = None,
    ):
        if activation != "swiglu":
            raise NotImplementedError(f"MojoMoe: Activation {activation} is not supported.")

        self.hidden_size = hidden_size
        self.ffn_intermediate_size = ffn_intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_weights = torch.nn.Parameter(torch.empty(hidden_size, num_experts))
        # Ref: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
        # torch.nn.functional.linear: out = input @ weight.T + bias
        self.ffn1_weights = torch.nn.Parameter(torch.empty(num_experts, 2 * ffn_intermediate_size, hidden_size))
        self.ffn2_weights = torch.nn.Parameter(torch.empty(num_experts, hidden_size, ffn_intermediate_size))
        self.activation_func = lambda x: torch.nn.functional.silu((xc := x.chunk(2, dim=-1))[0]) * xc[1]

        self.expert_start = 0
        self.expert_end = num_experts
        if ep_size and ep_rank:
            self.experts_start_idx = ep_rank * num_experts // ep_size
            self.experts_end_idx = min((ep_rank + 1) * num_experts // ep_size, num_experts % ep_size)

    def _gating(self, hidden_states):
        gate_logits = torch.nn.functional.linear(hidden_states, self.expert_weights)
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        expert_weights = torch.softmax(top_k_logits, dim=-1)
        return top_k_indices, expert_weights

    def _dispatch(self, hidden_states, expert_weights, top_k_indices):
        token_idx = (
            torch.arange(0, hidden_states.shape[0], device=hidden_states.device, dtype=top_k_indices.dtype)
            .unsqueeze(1)
            .repeat(1, top_k_indices.shape[-1])
            .flatten()
        )
        top_k_gates_flatten = expert_weights.reshape(-1, 1)
        top_k_indices_flatten = top_k_indices.flatten()

        sorted_experts, index_sorted_experts = top_k_indices_flatten.sort()
        start_idx = torch.searchsorted(sorted_experts, self.experts_start_idx, side="left")
        end_idx = torch.searchsorted(sorted_experts, self.experts_end_idx, side="left")
        index_sorted_experts = index_sorted_experts[start_idx:end_idx]
        pack_index = token_idx[index_sorted_experts]

        counts = torch.bincount(top_k_indices.flatten(), minlength=self.config.model_config.moe_expert_num).tolist()
        counts = counts[self.experts_start_idx : self.experts_end_idx]

        pack_gates = top_k_gates_flatten[index_sorted_experts, :]

        inp_exp = hidden_states[pack_index].squeeze(1)

        return torch.split(inp_exp, counts, dim=0), pack_gates, pack_index

    def _experts(self, expert_inputs):
        fc1_out = [
            torch.nn.functional.linear(expert_inputs[i], self.ffn1_weights[i]) for i in range(len(expert_inputs))
        ]
        fc1_out = [self.activation_func(x) for x in fc1_out]
        fc2_out = [torch.nn.functional.linear(fc1_out[i], self.ffn2_weights[i]) for i in range(len(fc1_out))]
        return fc2_out

    def _combine(self, experts_out, x, pack_gates, pack_index):
        dtype = experts_out[0].dtype
        stitched = torch.cat(experts_out, 0).float()

        stitched = stitched.mul(pack_gates).to(dtype=dtype)

        combined = torch.zeros(x.size(0), experts_out[-1].size(1), device=stitched.device, dtype=stitched.dtype)
        # combine samples that have been processed by the same k experts
        scatter_indices = pack_index.unsqueeze(-1).expand(-1, combined.size(1))
        combined = combined.scatter_reduce(0, scatter_indices, stitched, reduce="sum", include_self=True)

        return combined

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        top_k_indices, top_k_gates = self._gating(hidden_states)
        expert_inputs, pack_gates, pack_index = self._dispatch(
            hidden_states,
            top_k_gates,
            top_k_indices,
        )
        experts_outputs = self._experts(expert_inputs)
        experts_output = self._combine(
            experts_outputs,
            hidden_states,
            pack_gates,
            pack_index,
        )

        return experts_output


class MojoMoEGating(MojoOperator):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        **kwargs,
    ):
        """
        Common parameter definitions for MoE Gating operator.

        Init parameters:
        - gate_weight (torch.Tensor): Gating weight, common shape [hidden_dim, num_experts].
        - top_k (int): Number of experts to select, positive integer.

        Scope: Only covers common parameters, does not involve backend specialization or quantization implementation.
        """
        super().__init__()
        self.gate_weight = torch.nn.Parameter(torch.empty(hidden_size, num_experts, **self.tensor_factory_kwargs))
        self.top_k = top_k

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for MoE Gating operator.

        Input:
        - hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].

        Output:
        - torch.Tensor: Output tensor of shape [batch_size, seq_len, num_experts].
        """
        gate_logits = torch.matmul(hidden_states, self.gate_weight)
        gate_logits = torch.softmax(gate_logits, dim=-1)
        top_k_logits, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        expert_weights = top_k_logits / torch.sum(top_k_logits, dim=-1, keepdim=True)
        return indices, expert_weights


class MojoMoEDispatch(MojoOperator):
    def __init__(
        self,
        ep_group: Optional[object] = None,
        tp_group: Optional[object] = None,
    ):
        """
        Common parameter definitions for MoE Dispatch operator.

        Init parameters:
        - ep_group: Expert parallel process group (torch.distributed.ProcessGroup placeholder), optional.
        - tp_group: Tensor parallel process group (torch.distributed.ProcessGroup placeholder), optional.

        Scope: Only covers common semantics, does not involve backend communication implementation or core partitioning details.
        """
        super().__init__()
        self.ep_group = ep_group
        self.tp_group = tp_group


class MojoMoECombine(MojoOperator):
    def __init__(
        self,
        ep_group: Optional[object] = None,
        tp_group: Optional[object] = None,
    ):
        """
        Common parameter definitions for MoE Combine operator.

        Init parameters:
        - ep_group: Expert parallel process group (torch.distributed.ProcessGroup placeholder), optional.
        - tp_group: Tensor parallel process group (torch.distributed.ProcessGroup placeholder), optional.
        - is_varlen (bool): When True, prioritize TND (per token) aggregation; when False, use BSND; default True.
        - op_name: Operator name placeholder.

        Scope: Only covers common semantics, does not involve backend communication or core partitioning details.
        """
        super().__init__()
        self.ep_group = ep_group
        self.tp_group = tp_group
