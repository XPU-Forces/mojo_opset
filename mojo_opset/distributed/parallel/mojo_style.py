from functools import partial
from typing import Optional

from torch import nn

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    ParallelStyle,
)

from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    Replicate,
    Shard,
)

from torch.distributed.tensor.placement_types import Placement

from mojo_opset import MojoBatchLinear
from mojo_opset import MojoEmbedding
from mojo_opset import MojoGroupGemm

class MojoColwiseParallel(ColwiseParallel):
    def __init__(
        self, 
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True
    ):
        super().__init__(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output
        )

        self._parallel_style_map = {
            (nn.Linear, MojoBatchLinear): self._partition_linear_fn,
            (MojoEmbedding): self._partition_embedding_fn,
        }

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        find_parallel_style = False

        for _module_types, _partition_fn in self._parallel_style_map.items():
            if isinstance(module, _module_types):
                partition_fn = _partition_fn
                find_parallel_style = True
                break
        
        if not find_parallel_style:
            raise NotImplementedError(
                f"RowwiseParallel currently is currently not supported on {type(module)}!"
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )
    
class MojoRowwiseParallel(RowwiseParallel):
    def __init__(
        self, 
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True
    ):
        super().__init__(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output
        )

        self._parallel_style_map = {
            (nn.Linear, MojoBatchLinear): (
                self._partition_linear_fn,
                (Shard(-1),),
            ),
            (MojoEmbedding): (
                self._partition_embedding_fn,
                (Replicate(),),
            ),
        }
    
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        find_parallel_style = False

        for _module_types, (_partition_fn, _desired_input_layouts) in self._parallel_style_map.items():
            if isinstance(module, _module_types):
                self.desired_input_layouts = _desired_input_layouts
                partition_fn = _partition_fn
                find_parallel_style = True
                break
        
        if not find_parallel_style:
            raise NotImplementedError(
                f"RowwiseParallel currently is currently not supported on {type(module)}!"
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )
    
class MojoExpertParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Shard(-1),)
        self.output_layouts = (output_layouts or Replicate(),)
        self.use_local_output = use_local_output

        self._parallel_style_map = {
            (MojoGroupGemm): self._partition_fn,
        }

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
                find_parallel_style = True
                break

        if not find_parallel_style:
            raise NotImplementedError(
                f"ExpertParallel currently is currently not supported on {type(module)}!"
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
        )
