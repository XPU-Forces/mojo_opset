from functools import partial

import torch
from torch import nn
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import Shard
import torch.distributed._functional_collectives as fc

from mojo_opset.core.operators.moe import MojoMoE, MojoMoECombine, MojoMoEDispatch
from mojo_opset.distributed.parallel.mojo_parallel import MojoDistributedModule
from mojo_opset.distributed.parallel.mojo_parallel import MojoRegisterableParallelStyle
from mojo_opset.distributed.parallel.utils import shard_tensor


class _EPDispatchWrapper(nn.Module):
    """Wraps MojoMoEDispatch to slice dispatch output to a local expert partition."""

    def __init__(self, dispatch: nn.Module, ep_mesh: DeviceMesh):
        super().__init__()
        assert isinstance(dispatch, MojoMoEDispatch)
        self._dispatch = dispatch
        ep_size = ep_mesh.size()
        ep_rank = ep_mesh.get_rank()
        base = dispatch.num_experts // ep_size
        rem = dispatch.num_experts % ep_size
        local = base + 1 if ep_rank < rem else base
        self.ep_start = base * ep_rank + min(ep_rank, rem)
        self.ep_end = self.ep_start + local

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


class _EPCombineWrapper(nn.Module):
    """Wraps MojoMoECombine to combine partial output of each local expert partition."""

    def __init__(self, combine: nn.Module, ep_mesh: DeviceMesh):
        super().__init__()
        assert isinstance(combine, MojoMoECombine)
        self._combine = combine
        self.ep_mesh = ep_mesh

    def forward(self, output_buffer, expert_outputs, sorted_gates, token_indices):
        output = self._combine(output_buffer, expert_outputs, sorted_gates, token_indices)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return fc.all_reduce(output, "sum", (self.ep_mesh, 0))
        return output


def _ep_partition_fn(src_data_rank, name, module, device_mesh):
    from mojo_opset.core.operators.moe import MojoMoE, MojoExperts

    if isinstance(module, MojoMoE):
        module.dispatch = _EPDispatchWrapper(module.dispatch, device_mesh)
        module.combine = _EPCombineWrapper(module.combine, device_mesh)

    elif isinstance(module, MojoExperts):
        module.register_parameter(
            "up_proj_weight",
            nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.up_proj_weight)),
        )
        module.register_parameter(
            "down_proj_weight",
            nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.down_proj_weight)),
        )


class MojoExpertParallel(MojoRegisterableParallelStyle):
    def __init__(self):
        super().__init__()

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        partition_fn, _, _, _, _ = self.get_dist_info(module)

        return MojoDistributedModule(
            module,
            device_mesh,
            partial(partition_fn, self.src_data_rank) if partition_fn else None,
            None,
            None,
            parallel_style_name=self.__class__.__name__,
        )


MojoExpertParallel.register_dist_info(
    MojoMoE,
    partiton_fn=_ep_partition_fn,
)
