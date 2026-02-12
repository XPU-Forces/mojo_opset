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
        top_k,
        hidden_size,
        intermediate_size,
        ep_rank,
        ep_size,
        activation: str = "swiglu",
    ):
        super().__init__()
        if activation != "swiglu":
            raise NotImplementedError(f"MojoMoe: Activation {activation} is not supported.")

        self.ep_rank = ep_rank
        self.ep_size = ep_size

        # NOTE: in some cases, branches may have different expert num or topk
        self.num_experts = num_experts

        # support undivisible expert num
        base_num_experts = self.num_experts // self.ep_size
        remainder_experts = self.num_experts % self.ep_size

        if self.ep_rank < remainder_experts:
            self.num_experts_per_partion = base_num_experts + 1
        else:
            self.num_experts_per_partion = base_num_experts

        self.experts_start_idx = base_num_experts * self.ep_rank + min(self.ep_rank, remainder_experts)
        self.experts_end_idx = self.experts_start_idx + self.num_experts_per_partion

        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_hidden_size = intermediate_size
        self.activation_func = lambda x: torch.nn.functional.silu((xc := x.chunk(2, dim=-1))[0]) * xc[1]

        self.fc1 = nn.Parameter(torch.empty(self.num_experts_per_partion, self.ffn_hidden_size * 2, self.hidden_size))
        self.fc2 = nn.Parameter(torch.empty(self.num_experts_per_partion, self.hidden_size, self.ffn_hidden_size))
        self.gating_weight = nn.Parameter(torch.empty(self.hidden_size, self.num_experts))
        setattr(self.gating_weight, "force_dtype", torch.float32)

    def _gating(self, x):
        gate_logits = torch.matmul(x.float(), self.gating_weight.float())
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        expert_weights = torch.softmax(top_k_logits, dim=-1)
        return top_k_indices, expert_weights.to(x.dtype)

    def _dispatch_ep(self, inp, top_k_gates, top_k_indices):
        token_idx = (
            torch.arange(0, inp.shape[0], device=inp.device, dtype=top_k_indices.dtype)
            .unsqueeze(1)
            .repeat(1, top_k_indices.shape[-1])
            .flatten()
        )
        top_k_gates_flatten = top_k_gates.reshape(-1, 1)
        top_k_indices_flatten = top_k_indices.flatten()

        sorted_experts, index_sorted_experts = top_k_indices_flatten.sort()
        start_idx = torch.searchsorted(sorted_experts, self.experts_start_idx, side="left")
        end_idx = torch.searchsorted(sorted_experts, self.experts_end_idx, side="left")
        index_sorted_experts = index_sorted_experts[start_idx:end_idx]
        pack_index = token_idx[index_sorted_experts]

        counts = torch.bincount(top_k_indices.flatten(), minlength=self.num_experts).tolist()
        counts = counts[self.experts_start_idx : self.experts_end_idx]

        pack_gates = top_k_gates_flatten[index_sorted_experts, :]

        inp_exp = inp[pack_index].squeeze(1)

        return torch.split(inp_exp, counts, dim=0), pack_gates, pack_index

    def _experts(self, expert_inputs):
        fc1_out = [F.linear(expert_inputs[i], self.fc1[i]) for i in range(len(expert_inputs))]
        fc1_out = [self.activation_func(x) for x in fc1_out]
        fc2_out = [F.linear(fc1_out[i], self.fc2[i]) for i in range(len(fc1_out))]
        return fc2_out

    def _combine(self, expert_out, x, pack_gates, pack_index, multiply_by_gates=True):
        dtype = expert_out[0].dtype
        stitched = torch.cat(expert_out, 0).float()

        if multiply_by_gates:
            stitched = stitched.mul(pack_gates).to(dtype=dtype)

        combined = torch.zeros(x.size(0), expert_out[-1].size(1), device=stitched.device, dtype=stitched.dtype)
        # combine samples that have been processed by the same k experts
        scatter_indices = pack_index.unsqueeze(-1).expand(-1, combined.size(1))
        combined = combined.scatter_reduce(0, scatter_indices, stitched, reduce="sum", include_self=True)

        return combined.to(dtype=dtype)

    def forward(self, input, dp_rank_input_len=None):
        top_k_indices, top_k_gates = self._gating(input)

        expert_inputs, pack_gates, pack_index = self._dispatch_ep(input, top_k_gates, top_k_indices)

        experts_outputs = self._experts(expert_inputs)

        experts_output = self._combine(experts_outputs, input, pack_gates, pack_index)

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
