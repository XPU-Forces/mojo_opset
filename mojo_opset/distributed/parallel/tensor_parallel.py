import inspect

from functools import partial
from typing import List

import torch

from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.placement_types import Placement
from torch.distributed.tensor.placement_types import Replicate
from torch.distributed.tensor.placement_types import Shard

from mojo_opset.distributed.parallel.mojo_parallel import MojoDistributedModule
from mojo_opset.distributed.parallel.mojo_parallel import MojoRegisterableParallelStyle


class MojoTensorParallel(MojoRegisterableParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Placement,  # Requires the user to provide the distributed semantics information of the local tensor to reconstruct the DTensor
        output_layouts: Placement,  # Requires the user to provide the distributed semantics information of the local tensor to reconstruct the DTensor
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

        prepare_input_fn = prepare_input_fn if prepare_input_fn else self.prepare_input_fn
        prepare_output_fn = prepare_output_fn if prepare_output_fn else self.prepare_output_fn

        if desired_input_layouts:
            prepare_input_fn = partial(prepare_input_fn, desired_input_layouts)
        else:
            try:
                if inspect.signature(prepare_input_fn).parameters["desired_input_layouts"].default == inspect._empty:
                    prepare_input_fn = partial(prepare_input_fn, None)
            except KeyError:
                ...
        if desired_output_layouts:
            prepare_output_fn = partial(prepare_output_fn, desired_output_layouts)
        else:
            try:
                if inspect.signature(prepare_output_fn).parameters["desired_output_layouts"].default == inspect._empty:
                    prepare_output_fn = partial(prepare_output_fn, None)
            except KeyError:
                ...

        # WARNING(liuyuan): we should follow the positional parameter order.
        prepare_input_fn = partial(prepare_input_fn, self.input_layouts)
        prepare_output_fn = partial(prepare_output_fn, self.output_layouts, self.use_local_output)

        return MojoDistributedModule(
            module,
            device_mesh,
            partial(partition_fn, self.src_data_rank) if partition_fn else None,
            prepare_input_fn,
            prepare_output_fn,
            parallel_style_name=self.__class__.__name__,
        )


class MojoRowwiseParallel(MojoTensorParallel):
    def __init__(
        self,
        *,
        input_layouts: List[Placement]
        | None = None,  # Requires the user to provide the distributed semantics information of the local tensor to reconstruct the DTensor
        output_layouts: List[Placement]
        | None = None,  # Requires the user to provide the distributed semantics information of the local tensor to reconstruct the DTensor
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
        input_layouts: List[Placement]
        | None = None,  # Requires the user to provide the distributed semantics information of the local tensor to reconstruct the DTensor
        output_layouts: List[Placement]
        | None = None,  # Requires the user to provide the distributed semantics information of the local tensor to reconstruct the DTensor
        use_local_output: bool = True,
    ):
        super().__init__(
            input_layouts=input_layouts or (Replicate(),),
            output_layouts=output_layouts or (Shard(-1),),
            use_local_output=use_local_output,
        )


class MojoSwiGLUParallel(MojoTensorParallel):
    def __init__(self, **kwargs):
        super().__init__(
            input_layouts=kwargs.get("input_layouts") or (Replicate(),),
            output_layouts=kwargs.get("output_layouts") or (Replicate(),),
            use_local_output=kwargs.get("use_local_output", True),
        )


class MojoQKVColwiseParallel(MojoColwiseParallel):
    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh):
        partition_fn, _, _, desired_input_layouts, desired_output_layouts = self.get_dist_info(module)

        if partition_fn is not None:
            partition_fn = partial(
                partition_fn,
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )

        prepare_input_fn = self.prepare_input_fn
        prepare_output_fn = self.prepare_output_fn
        if desired_input_layouts:
            prepare_input_fn = partial(prepare_input_fn, desired_input_layouts)
        else:
            prepare_input_fn = partial(prepare_input_fn, None)
        if desired_output_layouts:
            prepare_output_fn = partial(prepare_output_fn, desired_output_layouts)
        else:
            prepare_output_fn = partial(prepare_output_fn, None)
        prepare_input_fn = partial(prepare_input_fn, self.input_layouts)
        prepare_output_fn = partial(prepare_output_fn, self.output_layouts, self.use_local_output)

        return MojoDistributedModule(
            module,
            device_mesh,
            partial(partition_fn, self.src_data_rank) if partition_fn else None,
            prepare_input_fn,
            prepare_output_fn,
            parallel_style_name=self.__class__.__name__,
        )
