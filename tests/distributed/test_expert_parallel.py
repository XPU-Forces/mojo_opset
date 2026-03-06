# ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m pytest -q tests/distributed/test_expert_parallel.py 
import os

from typing import Optional

import pytest
import torch

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

from mojo_opset import MojoGroupGemm
from mojo_opset.distributed.parallel import MojoExpertParallel


def _get_world_size():
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size <= 0:
        pytest.skip("This test requires launching with torchrun (WORLD_SIZE must be set).")
    return world_size

def test_moe_parallel():
    world_size = _get_world_size()

    class MojoMoE(torch.nn.Module):
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
            super().__init__()

            if activation != "swiglu":
                raise NotImplementedError(f"MojoMoE: activation {activation} is not supported.")

            self.hidden_size = hidden_size
            self.ffn_intermediate_size = ffn_intermediate_size
            self.num_experts = num_experts
            self.top_k = top_k
            self.expert_weights = torch.nn.Parameter(torch.empty(num_experts, hidden_size))
            # Ref: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
            # torch.nn.functional.linear: out = input @ weight.T + bias
            self.ffn1_weight = torch.nn.Parameter(torch.rand(num_experts, 2 * ffn_intermediate_size, hidden_size))
            self.ffn2_weight = torch.nn.Parameter(torch.rand(num_experts, hidden_size, ffn_intermediate_size))
            self.ffn1 = MojoGroupGemm(self.ffn1_weight)
            self.ffn2 = MojoGroupGemm(self.ffn2_weight)

            self.activation_func = lambda x: torch.nn.functional.silu((xc := x.chunk(2, dim=-1))[0]) * xc[1]

            self.expert_start = 0
            self.expert_end = num_experts
            if ep_size is not None and ep_rank is not None:
                self.expert_start = ep_rank * num_experts // ep_size
                self.expert_end = (ep_rank + 1) * num_experts // ep_size
 
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

            start_idx = torch.searchsorted(sorted_experts, self.expert_start, side="left")
            end_idx = torch.searchsorted(sorted_experts, self.expert_end, side="left")
            index_sorted_experts = index_sorted_experts[start_idx:end_idx]
            pack_index = token_idx[index_sorted_experts]

            counts = torch.bincount(top_k_indices.flatten(), minlength=self.num_experts)
            counts = counts[self.expert_start : self.expert_end]
            pack_gates = top_k_gates_flatten[index_sorted_experts, :]

            inp_exp = hidden_states[pack_index].squeeze(1)

            return inp_exp, pack_gates, pack_index, counts

        def _experts(self, expert_inputs, group_list):
            fc1_out = self.ffn1(expert_inputs, group_list = group_list)
            fc1_out = self.activation_func(fc1_out)
            fc2_out = self.ffn2(fc1_out, group_list = group_list)
            return fc2_out

        def _combine(self, experts_out, x, pack_gates, pack_index):
            dtype = experts_out.dtype
            experts_out = experts_out.mul(pack_gates).to(dtype=dtype)

            combined = torch.zeros(x.size(0), experts_out.size(1), device=experts_out.device, dtype=experts_out.dtype)
            # Combine tokens processed by the same set of top-k experts.
            scatter_indices = pack_index.unsqueeze(-1).expand(-1, combined.size(1))
            combined = combined.scatter_reduce(0, scatter_indices, experts_out, reduce="sum", include_self=True)

            return combined

        def forward(
            self,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            top_k_indices, top_k_gates = self._gating(hidden_states)
            expert_inputs, pack_gates, pack_index, counts = self._dispatch(
                hidden_states,
                top_k_gates,
                top_k_indices,
            )

            experts_outputs = self._experts(expert_inputs, counts)

            experts_output = self._combine(
                experts_outputs,
                hidden_states,
                pack_gates,
                pack_index,
            )

            return experts_output
    
    in_feature = 512
    intermedia_feture = 128
    num_experts = 16 * world_size
    select_expert_num = 2

    device_mesh = init_device_mesh("npu", (world_size,))    
    full_moe = MojoMoE(
        in_feature, 
        intermedia_feture, 
        num_experts, 
        select_expert_num, 
        ep_size=device_mesh.shape[0], 
        ep_rank=device_mesh.get_rank()
    )
    full_moe = full_moe.npu()
    parallel_moe = parallelize_module(
        full_moe, 
        device_mesh=device_mesh,
        parallelize_plan={
            "ffn1": MojoExpertParallel(), 
            "ffn2": MojoExpertParallel()
        }
    )

    hidden_states = torch.zeros(16, in_feature).npu()
    parallel_res = parallel_moe(hidden_states)
    full_res = full_moe(hidden_states)
    assert torch.allclose(parallel_res, full_res)
    

if __name__ == "__main__":
    test_moe_parallel()
