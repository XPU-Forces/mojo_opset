import warnings
from fnmatch import fnmatch
from typing import Callable, Any, Tuple, List, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.tensor.placement_types import Placement
from torch.distributed.tensor.parallel._utils import _validate_tp_mesh_dim


class MojoRegisterableParallelStyle(ParallelStyle):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.dist_info_map = {}

    @classmethod
    def register_dist_info(
        cls,
        module_clses: torch.nn.Module | Tuple[torch.nn.Module],
        partiton_fn: Callable[[torch.nn.Module, Any, DeviceMesh], None] = None,
        prepare_input_fn: Callable[
            [List[Placement], List[Placement], DeviceMesh, Tuple, Dict[str, Any]],
            Tuple[tuple, Dict[str, Any]],
        ] = None,
        prepare_output_fn: Callable[
            [List[Placement], List[Placement], bool, DeviceMesh, Any], Any
        ] = None,
        desired_input_layouts: Tuple = None,
        desired_output_layouts: Tuple = None,
    ):
        module_clses = (module_clses,) if not isinstance(module_clses, tuple) else module_clses
        for module_cls in module_clses:
            cls.dist_info_map[module_cls] = (
                partiton_fn,
                prepare_input_fn,
                prepare_output_fn,
                desired_input_layouts,
                desired_output_layouts,
            )

    @classmethod
    def get_dist_info(cls, module: torch.nn.Module):
        return cls.dist_info_map.get(type(module), (None,) * 5)

    @staticmethod
    def prepare_input_fn(
        desired_input_layouts, input_layouts, device_mesh, *args, **kwargs
    ):
        def mapping(tensor):
            if not isinstance(tensor, torch.Tensor):
                return tensor
            if not isinstance(tensor, DTensor):
                tensor = DTensor.from_local(
                    tensor, device_mesh, input_layouts, run_check=False
                )

            if desired_input_layouts and tensor.placements != desired_input_layouts:
                tensor = tensor.redistribute(
                    placements=desired_input_layouts, async_op=True
                )
            return tensor.to_local()

        args = list(args)
        for idx, input_tensor in enumerate(args):
            args[idx] = mapping(input_tensor)
        for key, input_tensor in kwargs.items():
            kwargs[key] = mapping(input_tensor)

        return (tuple(args), kwargs)

    @staticmethod
    def prepare_output_fn(
        desired_output_layouts, output_layouts, use_local_output, device_mesh, outputs
    ):
        is_single = False
        is_tuple = False
        if isinstance(outputs, (list, tuple)): 
            is_tuple = isinstance(outputs, tuple)
            outputs=list(outputs)
        else:
            outputs = [outputs]
            is_single=True

        for idx, output_tensor in enumerate(outputs):
            if not isinstance(output_tensor, torch.Tensor):
                continue
            if not isinstance(output_tensor, DTensor):
                output_tensor = DTensor.from_local(
                    output_tensor, device_mesh, desired_output_layouts, run_check=False
                )
            if output_tensor.placements != output_layouts:
                output_tensor = output_tensor.redistribute(placements=output_layouts, async_op=True)
            outputs[idx] = output_tensor.to_local() if use_local_output else output_tensor
            if isinstance(outputs[idx], AsyncCollectiveTensor):
                outputs[idx] = outputs[idx].wait()

        return outputs[0] if is_single else tuple(outputs) if is_tuple else outputs

    def __call__(self, module: torch.nn.Module, device_mesh: DeviceMesh):
        return self._apply(module, device_mesh)

class MojoDistributedModule(torch.nn.Module):

    def __init__(
        self,
        mod: torch.nn.Module,
        device_mesh: DeviceMesh | None = None,
        partition_fn: Callable[[str, torch.nn.Module, DeviceMesh], None] | None = None,
        prepare_input_fn: Callable[[torch.nn.Module, Any, DeviceMesh], Any] | None = None,
        prepare_output_fn: Callable[[torch.nn.Module, Any, DeviceMesh], Any] | None = None,
    ):
        super().__init__()
        self._mod = mod
        self._device_mesh = device_mesh
        self._prepare_input_fn = prepare_input_fn
        self._prepare_output_fn = prepare_output_fn
        if partition_fn is not None:
            for name, submod in self._mod.named_modules():
                partition_fn(name, submod, self._device_mesh)

    def forward(self, *args, **kwargs):
        if self._prepare_input_fn:
            args, kwargs = self._prepare_input_fn(self._device_mesh, *args, **kwargs)

        output = self._mod(*args, **kwargs)

        if self._prepare_output_fn:
            output = self._prepare_output_fn(self._device_mesh, output)
        return output


# NOTE(liuyuan): MojoDistributedModule is a wrapper around nn.Module without using forward_hook that
# MojoRegisterableParallelStyle can apply to but cannot modify the original module in-place.
# NOTE(liuyuan): ported from torch.distributed.tensor.parallel.parallelize_module
def mojo_parallelize_module(  # type: ignore[return]
    module: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    parallelize_plan: Optional[Union[ParallelStyle, dict[str, ParallelStyle]]] = None,
    *,
    src_data_rank: Optional[int] = 0,
) -> nn.Module:
    device_mesh = device_mesh or _mesh_resources.get_current_mesh()
    _validate_tp_mesh_dim(device_mesh)

    if parallelize_plan is None:
        warnings.warn(
            "No parallelize_plan is provided and auto-parallel is not supported "
            "at the moment, so this parallelize_module call will do nothing."
        )
        return module

    # note: The RNG tracker will be initialized in distribute_tensor() call if it hasn't
    # been initialized.

    if isinstance(parallelize_plan, ParallelStyle):
        parallelize_plan.src_data_rank = src_data_rank
        return parallelize_plan._apply(module, device_mesh)
    elif isinstance(parallelize_plan, dict):
        for module_path, parallelize_style in parallelize_plan.items():
            path_splits = module_path.split(".")
            if len(path_splits) == 0:
                raise ValueError(
                    "Expect module path to be non-empty, but got empty string!"
                )
            while path_splits:
                atom = path_splits.pop(0)
                matched_children = filter(
                    # `t[0]` is child name
                    lambda t: fnmatch(t[0], atom),
                    module.named_children(),
                )
                # apply the plan to all matched submodules
                for name, submodule in matched_children:
                    if path_splits:
                        # we haven't reached the leaf, apply in dict style
                        leaf_path = ".".join(
                            path_splits
                        )  # rest of the path after `atom`
                        mojo_parallelize_module(
                            submodule,
                            device_mesh,
                            {leaf_path: parallelize_style},
                            src_data_rank=src_data_rank,
                        )
                    else:
                        # otherwise, directly apply style to this submodule
                        # NOTE(liuyuan): key change here.
                        module.set_submodule(
                            name,
                            mojo_parallelize_module(
                                submodule,
                                device_mesh,
                                parallelize_style,
                                src_data_rank=src_data_rank,
                            ),
                        )
        return module
    else:
        raise TypeError(  # pyre-ignore[7]
            "Expect Union[ParallelStyle, Dict[str, ParallelStyle]] for"
            f" parallelize_plan, {type(parallelize_plan)} found!"
        )
