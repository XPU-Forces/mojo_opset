import torch
from torch import nn
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import Shard
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle
import torch.distributed._functional_collectives as fc

from mojo_opset.core.operators.moe import MojoMoE


class _MojoEPAllReduceWrapper(nn.Module):
    def __init__(self, inner: nn.Module, group):
        super().__init__()
        self.inner = inner
        self.group = group

    def forward(self, x, *args, **kwargs):
        y = self.inner(x, *args, **kwargs)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            y = fc.all_reduce(y, "sum", self.group)
        return y
    
class _MojoEPAll2AllWrapper(nn.Module):
    def __init__(self, inner: nn.Module, group):
        super().__init__()
        assert isinstance(inner, MojoMoE)
        self.inner = inner
        self.group = group

    def forward(self, x, *args, **kwargs):
        x = fc.all_gather_tensor_autograd(x, 0, self.group)
        y = self.inner(x, *args, **kwargs)
        y = fc.reduce_scatter_tensor_autograd(y, reduceOp="sum", scatter_dim=0, group=self.group)
        # Step 1: gating on local tokens
        # top_k_indices, expert_weights = self.inner._gating(x)
        # Step 2: dispatch local tokens according to router
        # expert_inputs, pack_gates, pack_index = self.inner._dispatch_ep(x, expert_weights, top_k_indices)
        
        # Step 3: compute on local experts
        # Step 4: combine tokens according to router
        return y


class MojoExpertParallel(ParallelStyle):
    def __init__(self, *, replicate_input: bool = True):
        super().__init__()
        self._parallel_style_map = {(MojoMoE): self._partition_fn}
        self.replicate_input = replicate_input

    def _partition_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        ep_size = device_mesh.size()
        ep_rank = device_mesh.get_rank()

        def get_local_num_experts(ep_id: int):
            base_num_experts = module.num_experts // ep_size
            remainder_experts = module.num_experts % ep_size

            if ep_id < remainder_experts:
                local_num_experts = base_num_experts + 1
            else:
                local_num_experts = base_num_experts
            return local_num_experts
        
        expert_map = [get_local_num_experts(ep_id) for ep_id in range(ep_size)]
        
        experts_start_idx = sum(expert_map[:ep_rank])
        experts_end_idx = experts_start_idx + expert_map[ep_rank]

        fc1_dt = distribute_tensor(module.fc1, device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)
        module.register_parameter("fc1", nn.Parameter(fc1_dt.to_local()))
        fc2_dt = distribute_tensor(module.fc2, device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)
        module.register_parameter("fc2", nn.Parameter(fc2_dt.to_local()))

        module.expert_map = expert_map
        module.experts_start_idx = experts_start_idx
        module.experts_end_idx = experts_end_idx

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        self._partition_fn("", module, device_mesh)
        if self.replicate_input:
            return _MojoEPAllReduceWrapper(module, (device_mesh, 0))
        else:
            return _MojoEPAll2AllWrapper(module, (device_mesh, 0))
