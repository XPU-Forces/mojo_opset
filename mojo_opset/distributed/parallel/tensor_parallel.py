import inspect
from typing import List
import torch
from torch.distributed.tensor import DeviceMesh
from functools import partial
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate

from mojo_opset.distributed.parallel.mojo_parallel import MojoRegisterableParallelStyle, MojoDistributedModule

class MojoTensorParallel(MojoRegisterableParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Placement,
        output_layouts: Placement,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = input_layouts
        self.output_layouts = output_layouts
        self.use_local_output = use_local_output

    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh):
        (
            partition_fn,
            prepare_input_fn,
            prepare_output_fn,
            desired_input_layouts,
            desired_output_layouts,
        ) = self.get_dist_info(module)

        prepare_input_fn = prepare_input_fn if prepare_input_fn  else self.prepare_input_fn
        prepare_output_fn = prepare_output_fn if prepare_output_fn  else self.prepare_output_fn


        if desired_input_layouts:
            prepare_input_fn = partial(prepare_input_fn, desired_input_layouts)
        else:
            try:
                if inspect.signature(prepare_input_fn).parameters['desired_input_layouts'].default == inspect._empty:
                    prepare_input_fn = partial(prepare_input_fn, None)
            except KeyError:
                ...
        if desired_output_layouts:
            prepare_output_fn = partial(prepare_output_fn, desired_output_layouts)
        else:
            try:
                if inspect.signature(prepare_output_fn).parameters['desired_output_layouts'].default == inspect._empty:
                    prepare_output_fn = partial(prepare_output_fn, None)
            except KeyError:
                ...

        # WARNING(liuyuan): we should follow the positional parameter order.
        prepare_input_fn = partial(prepare_input_fn, self.input_layouts)
        prepare_output_fn = partial(
            prepare_output_fn,
            self.output_layouts,
            self.use_local_output
        )

        return MojoDistributedModule(
            module,
            device_mesh,
            partial(partition_fn, self.src_data_rank) if partition_fn else None,
            prepare_input_fn,
            prepare_output_fn,
        )


class MojoRowwiseParallel(MojoTensorParallel):
    def __init__(
        self,
        *,
        input_layouts: List[Placement] | None = None,
        output_layouts: List[Placement] | None = None,
        use_local_output: bool = True,
    ):
        super().__init__(
            input_layouts=input_layouts or (Shard(-1),),
            output_layouts=output_layouts or (Replicate(),),
            use_local_output=use_local_output,
        )


class MojoColwiseParallel(MojoTensorParallel):
    def __init__(
        self,
        *,
        input_layouts: List[Placement] | None = None,
        output_layouts: List[Placement] | None = None,
        use_local_output: bool = True,
    ):
        super().__init__(
            input_layouts=input_layouts or (Replicate(),),
            output_layouts=output_layouts or (Shard(-1),),
            use_local_output=use_local_output,
        )
