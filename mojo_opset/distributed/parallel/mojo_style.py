from functools import partial
from typing import Optional

from torch import nn

from torch.distributed.tensor.parallel import ColwiseParallel
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    Replicate,
    Shard,
)

from torch.distributed.tensor.placement_types import Placement

from mojo_opset import MojoLinear
from mojo_opset import MojoBatchLinear
from mojo_opset import MojoEmbedding

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
            (MojoLinear, MojoBatchLinear): (
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

        for module_types, (_partition_fn, _desired_input_layouts) in self._parallel_style_map.items():
            if isinstance(module, module_types):
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