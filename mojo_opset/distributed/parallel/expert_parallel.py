from functools import partial
from typing import Optional

from torch import nn
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import Shard
from torch.distributed.tensor import distribute_module
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement

from mojo_opset import MojoGroupGemm
from mojo_opset.core.operators.moe import MojoMoE
import torch


class MojoExpertParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Replicate(),)
        self.output_layouts = (output_layouts or Replicate(),)
        self.use_local_output = use_local_output
        self.desired_input_layouts = (Replicate(),)

        self._parallel_style_map = {
            (MojoGroupGemm): self._partition_fn,
        }

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, input_layouts, run_check=False)

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(placements=desired_input_layouts, async_op=True)
        return input_tensor

    def _partition_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        module.register_parameter(
            "weight",
            nn.Parameter(
                distribute_tensor(
                    module.weight,
                    device_mesh,
                    [Shard(0)],
                    src_data_rank=self.src_data_rank,
                )
            ),
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        for _module_types, _partition_fn in self._parallel_style_map.items():
            if isinstance(module, _module_types):
                partition_fn = _partition_fn
                break

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),
        )


class MojoMoEExpertParallel(ParallelStyle):
    def __init__(
        self,
        *,
        shard_expert_params: bool = True,
    ):
        super().__init__()
        self.shard_expert_params = shard_expert_params
        self._parallel_style_map = {
            (MojoMoE): self._partition_fn,
        }

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

        if self.shard_expert_params:
            if module.fc1.size(0) == module.num_experts:
                module.register_parameter("fc1", nn.Parameter(module.fc1[experts_start_idx:experts_end_idx].contiguous()))
            if module.fc2.size(0) == module.num_experts:
                module.register_parameter("fc2", nn.Parameter(module.fc2[experts_start_idx:experts_end_idx].contiguous()))

        module.num_experts_per_partion = local_num_experts
        module.experts_start_idx = experts_start_idx
        module.experts_end_idx = experts_end_idx

        if module.fc1.size(0) != module.num_experts_per_partion or module.fc2.size(0) != module.num_experts_per_partion:
            raise ValueError(
                f"MojoMoEExpertParallel: unexpected expert param shapes: "
                f"fc1[0]={module.fc1.size(0)}, fc2[0]={module.fc2.size(0)}, "
                f"expected {module.num_experts_per_partion}."
            )

    @staticmethod
    def _output_fn(mod: nn.Module, outputs, device_mesh: DeviceMesh):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(outputs, op=torch.distributed.ReduceOp.SUM, group=device_mesh.get_group(0))
        return outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        for _module_types, _partition_fn in self._parallel_style_map.items():
            if isinstance(module, _module_types):
                partition_fn = _partition_fn
                break

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            None,
            self._output_fn,
        )
