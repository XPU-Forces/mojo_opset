import torch
from torch import nn
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import Shard
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle
import torch.distributed._functional_collectives as fc

from mojo_opset.core.operators.moe import MojoMoE, MojoExperts, MojoMoEDispatch, MojoMoECombine
from mojo_opset.distributed.parallel.utils import compute_local_shard_size_and_offset


class _MojoEPReplicateDispatchWrapper(nn.Module):
    def __init__(self, inner: nn.Module, start_expert_id: int, end_expert_id: int, group):
        super().__init__()
        assert isinstance(inner, MojoMoEDispatch)
        self.inner = inner
        self.group = group
        self.start_expert_id = start_expert_id
        self.end_expert_id = end_expert_id

    def forward(self, hidden_states, top_k_gates, top_k_indices):
        sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = self.inner(hidden_states, top_k_gates, top_k_indices)
        # shard to local experts
        sorted_start_idx = tokens_per_expert[:self.start_expert_id].sum().item()
        sorted_end_idx = tokens_per_expert[:self.end_expert_id].sum().item()
        return sorted_hidden_states[sorted_start_idx:sorted_end_idx], tokens_per_expert[self.start_expert_id:self.end_expert_id], sorted_gates[sorted_start_idx:sorted_end_idx], token_indices[sorted_start_idx:sorted_end_idx]
    
class _MojoEPReplicateCombineWrapper(nn.Module):
    def __init__(self, inner: nn.Module, start_expert_id: int, end_expert_id: int, group):
        super().__init__()
        assert isinstance(inner, MojoMoECombine)
        self.inner = inner
        self.group = group
        self.start_expert_id = start_expert_id
        self.end_expert_id = end_expert_id
    
    def forward(self, output_buffer, expert_outputs, sorted_gates, token_indices):
        outputs = self.inner(output_buffer, expert_outputs, sorted_gates, token_indices)
        outputs = fc.all_reduce(outputs, "sum", self.group)
        return outputs
    
class _MojoEPShardedDispatchWrapper(nn.Module):
    def __init__(self, inner: nn.Module, start_expert_id: int, end_expert_id: int, group):
        super().__init__()
        assert isinstance(inner, MojoMoEDispatch)
        self.start_expert_id = start_expert_id
        self.end_expert_id = end_expert_id
        self.inner = inner
        self.group = group

    def forward(self, hidden_states, top_k_gates, top_k_indices):
        hidden_states = fc.all_gather_tensor(hidden_states, 0, self.group)
        top_k_gates = fc.all_gather_tensor(top_k_gates, 0, self.group)
        top_k_indices = fc.all_gather_tensor(top_k_indices, 0, self.group)
        sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = self.inner(hidden_states, top_k_gates, top_k_indices)

        sorted_start_idx = tokens_per_expert[:self.start_expert_id].sum().item()
        sorted_end_idx = tokens_per_expert[:self.end_expert_id].sum().item()
        return sorted_hidden_states[sorted_start_idx:sorted_end_idx], tokens_per_expert[self.start_expert_id:self.end_expert_id], sorted_gates[sorted_start_idx:sorted_end_idx], token_indices[sorted_start_idx:sorted_end_idx]

class _MojoEPShardedCombineWrapper(nn.Module):
    def __init__(self, inner: nn.Module, start_expert_id: int, end_expert_id: int, group):
        super().__init__()
        assert isinstance(inner, MojoMoECombine)
        self.inner = inner
        self.group = group
        self.start_expert_id = start_expert_id
        self.end_expert_id = end_expert_id
    
    def forward(self, output_buffer, expert_outputs, sorted_gates, token_indices):
        output_buffer_all = torch.zeros(output_buffer.size(0) * self.group.size(), output_buffer.size(1), device=output_buffer.device, dtype = output_buffer.dtype)
        outputs = self.inner(output_buffer_all, expert_outputs, sorted_gates, token_indices)
        outputs = fc.reduce_scatter_tensor(outputs, reduceOp="sum", scatter_dim=0, group=self.group)
        return outputs + output_buffer

class MojoExpertParallel(ParallelStyle):
    def __init__(self, *, replicate_input: bool = True, src_data_rank: int = 0):
        super().__init__()
        # self._parallel_style_map = {
        #     MojoExperts: self._partition_experts,
        #     MojoMoEDispatch: _MojoEPShardedDispatchWrapper,
        #     MojoMoECombine: _MojoEPShardedCombineWrapper, 
        # }
        self.replicate_input = replicate_input
        self.src_data_rank = src_data_rank

    def _partition_experts(self, name: str, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        assert isinstance(module, MojoExperts)
        fc1_dt = distribute_tensor(module.fc1, device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)
        module.register_parameter("fc1", nn.Parameter(fc1_dt.to_local()))
        fc2_dt = distribute_tensor(module.fc2, device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)
        module.register_parameter("fc2", nn.Parameter(fc2_dt.to_local()))

        return module

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        ep_size = device_mesh.size()
        ep_rank = device_mesh.get_rank()
        assert isinstance(module, MojoMoE)
        num_experts = module.num_experts
        local_num_experts, expert_start_idx = compute_local_shard_size_and_offset(num_experts, ep_size, ep_rank)
        expert_end_idx = expert_start_idx + local_num_experts

        module.register_module("experts", self._partition_experts("", module.experts, device_mesh))
        if self.replicate_input:
            module.register_module("dispatch", _MojoEPReplicateDispatchWrapper(module.dispatch, expert_start_idx, expert_end_idx, device_mesh))
            module.register_module("combine", _MojoEPReplicateCombineWrapper(module.combine, expert_start_idx, expert_end_idx, device_mesh))
        else:
            module.register_module("dispatch", _MojoEPShardedDispatchWrapper(module.dispatch, expert_start_idx, expert_end_idx, device_mesh))
            module.register_module("combine", _MojoEPShardedCombineWrapper(module.combine, expert_start_idx, expert_end_idx, device_mesh))

        return module