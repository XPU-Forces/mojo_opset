from functools import partial

import torch
from torch import nn
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import Shard
import torch.distributed._functional_collectives as fc

from mojo_opset.core.operators.moe import MojoMoE
from mojo_opset.distributed.parallel.mojo_parallel import MojoDistributedModule
from mojo_opset.distributed.parallel.mojo_parallel import MojoRegisterableParallelStyle
from mojo_opset.distributed.parallel.utils import shard_tensor


def _ep_partition_fn(src_data_rank, name, module, device_mesh):
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

    module.register_parameter(
        "fc1", nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.fc1))
    )
    module.register_parameter(
        "fc2", nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.fc2))
    )

    module.num_experts_per_partion = local_num_experts
    module.experts_start_idx = experts_start_idx
    module.experts_end_idx = experts_end_idx


def _ep_prepare_output_fn(device_mesh, output):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return fc.all_reduce(output, "sum", (device_mesh, 0))
    return output


class MojoExpertParallel(MojoRegisterableParallelStyle):
    def __init__(self):
        super().__init__()

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        partition_fn, _, prepare_output_fn, _, _ = self.get_dist_info(module)

        return MojoDistributedModule(
            module,
            device_mesh,
            partial(partition_fn, self.src_data_rank) if partition_fn else None,
            None,
            prepare_output_fn,
            parallel_style_name=self.__class__.__name__,
        )


MojoExpertParallel.register_dist_info(
    MojoMoE,
    partiton_fn=_ep_partition_fn,
    prepare_output_fn=_ep_prepare_output_fn,
)
