from typing import Optional

import torch

from torch import nn
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.device_mesh import init_device_mesh

from mojo_opset.distributed.parallel import (
    MojoColwiseParallel,
    MojoRowwiseParallel,
    MojoExpertParallel
)

from mojo_opset import MojoGroupGemm

def test_colwise_parallel():
    class ColwiseGemm(torch.nn.Module):
        def __init__(self, in_feature, out_feature):
            super().__init__()
            self.linear = nn.Linear(in_features=in_feature, out_features=out_feature)

        @torch.inference_mode
        def forward(self, hidden_states):
            return self.linear(hidden_states)
    
    in_feature = 4096
    out_feature = 16384

    test_colwise_gemm = ColwiseGemm(in_feature, out_feature)

    device_mesh = init_device_mesh("npu", (4,))
    test_colwise_gemm = parallelize_module(
        test_colwise_gemm, 
        device_mesh=device_mesh,
        parallelize_plan={"linear": MojoColwiseParallel()}
    )

    hidden_states = torch.zeros(16,4096)
    hidden_states = test_colwise_gemm(hidden_states)
    assert hidden_states.shape[-1] == 16384 // 4

def test_rowwise_parallel():
    class RowwiseGemm(torch.nn.Module):
        def __init__(self, in_feature, out_feature):
            super().__init__()
            self.linear = nn.Linear(in_features=in_feature, out_features=out_feature)
        
        @torch.inference_mode
        def forward(self, hidden_states):
            return self.linear(hidden_states)
    
    in_feature = 16384
    out_feature = 4096

    test_rowwise_gemm = RowwiseGemm(in_feature, out_feature)
    device_mesh = init_device_mesh("npu", (4,))

    test_rowwise_gemm = parallelize_module(
        test_rowwise_gemm, 
        device_mesh=device_mesh,
        parallelize_plan={"linear": MojoRowwiseParallel()}
    )

    hidden_states = torch.zeros(16,in_feature // 4)
    assert hidden_states.shape[-1] == test_rowwise_gemm.linear.weight.to_local().shape[-1]
    hidden_states = test_rowwise_gemm(hidden_states)

def test_moe_parallel():
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
                raise NotImplementedError(f"MojoMoe: Activation {activation} is not supported.")

            self.hidden_size = hidden_size
            self.ffn_intermediate_size = ffn_intermediate_size
            self.num_experts = num_experts
            self.top_k = top_k
            self.expert_weights = torch.nn.Parameter(torch.empty(num_experts, hidden_size))
            # Ref: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
            # torch.nn.functional.linear: out = input @ weight.T + bias
            self.ffn1 = MojoGroupGemm(num_experts, hidden_size, 2 * ffn_intermediate_size)
            self.ffn2 = MojoGroupGemm(num_experts, ffn_intermediate_size, hidden_size)

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
            # combine samples that have been processed by the same k experts
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
    
    in_feature = 4096
    out_feature = 1024
    num_experts = 384
    select_expert_num = 8

    device_mesh = init_device_mesh("npu", (4,))    
    test_ep_moe = MojoMoE(
        in_feature, 
        out_feature, 
        num_experts, 
        select_expert_num, 
        ep_size=device_mesh.shape[0], 
        ep_rank=device_mesh.get_rank()
    )
    test_ep_moe = test_ep_moe.npu()
    test_ep_moe = parallelize_module(
        test_ep_moe, 
        device_mesh=device_mesh,
        parallelize_plan={
            "ffn1": MojoExpertParallel(), 
            "ffn2": MojoExpertParallel()
        }
    )

    hidden_states = torch.zeros(16, in_feature).npu()
    hidden_states = test_ep_moe(hidden_states)

if __name__ == "__main__":
    #test_colwise_parallel()
    #test_rowwise_parallel()
    test_moe_parallel()