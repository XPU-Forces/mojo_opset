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


class MojoExpertParallel(ParallelStyle):
    def __init__(self):
        super().__init__()
        self._parallel_style_map = {(MojoMoE): self._partition_fn}

    def _partition_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        ep_size = device_mesh.size()
        ep_rank = device_mesh.get_rank()

        base_num_experts = module.num_experts // ep_size
        remainder_experts = module.num_experts % ep_size

        if ep_rank < remainder_experts:
            local_num_experts = base_num_experts + 1
        else:
            local_num_experts = base_num_experts

        experts_start_idx = base_num_experts * ep_rank + min(ep_rank, remainder_experts)
        experts_end_idx = experts_start_idx + local_num_experts

        fc1_dt = distribute_tensor(module.fc1, device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)
        module.register_parameter("fc1", nn.Parameter(fc1_dt.to_local()))
        fc2_dt = distribute_tensor(module.fc2, device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)
        module.register_parameter("fc2", nn.Parameter(fc2_dt.to_local()))

        module.num_experts_per_partion = local_num_experts
        module.experts_start_idx = experts_start_idx
        module.experts_end_idx = experts_end_idx

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        self._partition_fn("", module, device_mesh)
        return _MojoEPAllReduceWrapper(module, (device_mesh, 0))
