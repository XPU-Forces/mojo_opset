from functools import partial

import torch
import torch.distributed._functional_collectives as fc

from torch import nn
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import Shard

from mojo_opset.core.operators.moe import MojoMoE
from mojo_opset.core.operators.moe import MojoMoECombine
from mojo_opset.core.operators.moe import MojoMoEDispatch
from mojo_opset.core.operators.moe import MojoQuantMoE
from mojo_opset.distributed.parallel.mojo_parallel import MojoDistributedModule
from mojo_opset.distributed.parallel.mojo_parallel import MojoRegisterableParallelStyle
from mojo_opset.distributed.parallel.utils import shard_tensor
from mojo_opset.distributed.parallel.utils import stat_dict_rename_hook


class _EPDispatchWrapper(nn.Module):
    """Wraps MojoMoEDispatch to slice dispatch output to a local expert partition."""

    def __init__(self, dispatch: nn.Module, ep_mesh: DeviceMesh):
        super().__init__()
        assert isinstance(dispatch, MojoMoEDispatch)
        self._dispatch = dispatch
        ep_size = ep_mesh.size()
        ep_rank = ep_mesh.get_local_rank()
        base = dispatch.num_experts // ep_size
        rem = dispatch.num_experts % ep_size
        local = base + 1 if ep_rank < rem else base
        self.ep_start = base * ep_rank + min(ep_rank, rem)
        self.ep_end = self.ep_start + local

    def forward(self, hidden_states, top_k_gates, top_k_indices):
        sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = self._dispatch(
            hidden_states, top_k_gates, top_k_indices
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
    from mojo_opset.core.operators.moe import MojoExperts
    from mojo_opset.core.operators.moe import MojoMoE
    from mojo_opset.core.operators.moe import MojoQuantExperts
    from mojo_opset.core.operators.moe import MojoQuantMoE

    if isinstance(module, (MojoMoE, MojoQuantMoE)):
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
        module.register_state_dict_post_hook(
            partial(stat_dict_rename_hook, ("up_proj_weight", "down_proj_weight"), device_mesh)
        )
    elif isinstance(module, MojoQuantExperts):
        module.register_buffer(
            "up_proj_weight",
            shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.up_proj_weight),
        )
        module.register_buffer(
            "down_proj_weight",
            shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.down_proj_weight),
        )
        module.register_parameter(
            "up_proj_weight_scale",
            nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.up_proj_weight_scale)),
        )
        module.register_parameter(
            "down_proj_weight_scale",
            nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.down_proj_weight_scale)),
        )
        module.up_proj_quantize.register_parameter(
            "inv_smooth_scale",
            nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.up_proj_quantize.inv_smooth_scale)),
        )
        module.down_proj_quantize.register_parameter(
            "inv_smooth_scale",
            nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.down_proj_quantize.inv_smooth_scale)),
        )
        module.register_state_dict_post_hook(
            partial(
                stat_dict_rename_hook,
                (
                    "up_proj_weight",
                    "down_proj_weight",
                    "up_proj_weight_scale",
                    "down_proj_weight_scale",
                    "up_proj_quantize.inv_smooth_scale",
                    "down_proj_quantize.inv_smooth_scale",
                ),
                device_mesh,
            )
        )


class MojoExpertParallel(MojoRegisterableParallelStyle):
    def __init__(
        self,
        *,
        hccl_comm_dict: dict | None = None,
        ep_size: int | None = None,
        ep_rank: int | None = None,
        global_rank: int | None = None,
    ):
        super().__init__()
        self.hccl_comm_dict = hccl_comm_dict
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.global_rank = global_rank

    def _deepseek_v4_moe_partition_fn(self, src_data_rank, name, module, device_mesh):
        del src_data_rank
        del device_mesh
        if name or module.__class__.__name__ != "DeepseekV4MoE":
            return
        if self.hccl_comm_dict is not None:
            module.hccl_comm_dict = self.hccl_comm_dict
            module.ep_group = self.hccl_comm_dict.get("moe_ep_group")
        if self.ep_size is not None:
            module.ep_size = self.ep_size
        if self.ep_rank is not None:
            module.ep_rank = self.ep_rank
        if self.ep_size is not None and self.ep_rank is not None:
            module.experts_per_rank = module.n_routed_experts // module.ep_size
            module.ep_start = module.ep_rank * module.experts_per_rank
            module.ep_end = module.ep_start + module.experts_per_rank
        module.dispatch_kwargs = None
        module.combine_kwargs = None

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        partition_fn, _, _, _, _ = self.get_dist_info(module)
        if partition_fn is None and module.__class__.__name__ == "DeepseekV4MoE":
            partition_fn = self._deepseek_v4_moe_partition_fn

        return MojoDistributedModule(
            module,
            device_mesh,
            partial(partition_fn, self.src_data_rank) if partition_fn else None,
            None,
            None,
            parallel_style_name=self.__class__.__name__,
        )


MojoExpertParallel.register_dist_info(
    (MojoMoE, MojoQuantMoE),
    partiton_fn=_ep_partition_fn,
)
