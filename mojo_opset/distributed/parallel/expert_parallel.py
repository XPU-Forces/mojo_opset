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


class _EPDispatchWrapper(nn.Module):
    """Wraps MojoMoEDispatch to slice dispatch output to a local expert partition."""

    def __init__(self, dispatch: nn.Module, ep_start: int, ep_end: int):
        super().__init__()
        self._dispatch = dispatch
        self.ep_start = ep_start
        self.ep_end = ep_end

    def forward(self, hidden_states, top_k_gates, top_k_indices):
        sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = (
            self._dispatch(hidden_states, top_k_gates, top_k_indices)
        )
        cumsum = tokens_per_expert.cumsum(0)
        tok_start = 0 if self.ep_start == 0 else cumsum[self.ep_start - 1].item()
        tok_end = cumsum[self.ep_end - 1].item()
        return (
            sorted_hidden_states[tok_start:tok_end],
            tokens_per_expert[self.ep_start : self.ep_end],
            sorted_gates[tok_start:tok_end],
            token_indices[tok_start:tok_end],
        )


def _ep_partition_fn(src_data_rank, name, module, device_mesh):
    from mojo_opset.core.operators.moe import MojoMoE, MojoExperts

    ep_size = device_mesh.size()
    ep_rank = device_mesh.get_rank()

    if isinstance(module, MojoMoE):
        base = module.num_experts // ep_size
        rem = module.num_experts % ep_size
        local = base + 1 if ep_rank < rem else base
        start = base * ep_rank + min(ep_rank, rem)
        end = start + local
        module.dispatch = _EPDispatchWrapper(module.dispatch, start, end)

    elif isinstance(module, MojoExperts):
        module.register_parameter(
            "up_proj_weight",
            nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.up_proj_weight)),
        )
        module.register_parameter(
            "down_proj_weight",
            nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.down_proj_weight)),
        )


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
