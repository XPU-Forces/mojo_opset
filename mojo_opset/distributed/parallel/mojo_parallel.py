import warnings
from fnmatch import fnmatch

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.device_mesh import _mesh_resources
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.parallel._utils import _validate_tp_mesh_dim
from torch.distributed.tensor.placement_types import Placement

# class ShortcutDispatcher(type):
#     def __call__(cls, *args, **kwargs):
#         obj = cls.__new__(*args, **kwargs)
#         if getattr(obj, "_short_cut_initialized", False):
#             return obj
#         else:
#             obj.__init__(*args, **kwargs)
#             return obj

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
        desired_input_layouts: Tuple = None,  # This layout is the placement used to convert the local tensor output
        # by the operator into the corresponding DTensor according to the distributed semantics required by the operator
        desired_output_layouts: Tuple = None,  # This layout is the placement used to convert the local tensor output
        # by the operator into the corresponding DTensor according to the distributed semantics required by the operator
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
        module_type = type(module)

        if module_type not in cls.dist_info_map:
            # NOTE(liuyuan): fallback to parent class, specially designed for TorchXXX.
            module_type = module_type.__base__
        
        return cls.dist_info_map.get(module_type, (None,) * 5)

    @staticmethod
    def prepare_input_fn(
        desired_input_layouts,  # This layout is the placement used to convert the local tensor output by the operator
        # into the corresponding DTensor according to the distributed semantics required by the operator
        input_layouts,  # Requires the user to provide the distributed semantics information of the local tensor to reconstruct the DTensor
        device_mesh,
        *args,
        **kwargs,
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
        desired_output_layouts,  # This layout is the placement used to convert the local tensor output by the operator
        # into the corresponding DTensor according to the distributed semantics required by the operator
        output_layouts,  # Requires the user to provide the distributed semantics information of the local tensor to reconstruct the DTensor
        use_local_output,
        device_mesh,
        outputs,
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
    
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        if 'module' in kwargs and 'device_mesh' in kwargs:
            module = kwargs.pop('module')
            device_mesh = kwargs.pop('device_mesh')
            obj.__init__(*args, **kwargs)
            # NOTE(liuyuan): Maybe we should use ShortcutDispatcher as metaclass, but it seems like Python has already
            # known the MojoDistributedModule object is initialized.
            # So we can just return the MojoDistributedModule object here.
            return obj._apply(module, device_mesh)
        return obj

class MojoDistributedModule(torch.nn.Module):
    def __init__(
        self,
        mod: torch.nn.Module,
        device_mesh: DeviceMesh | None = None,
        partition_fn: Callable[[str, torch.nn.Module, DeviceMesh], None] | None = None,
        prepare_input_fn: Callable[[torch.nn.Module, Any, DeviceMesh], Any] | None = None,
        prepare_output_fn: Callable[[torch.nn.Module, Any, DeviceMesh], Any] | None = None,
        parallel_style_name: str | None = None,
    ):
        super().__init__()
        self._mod = mod
        self._device_mesh = device_mesh
        self._prepare_input_fn = prepare_input_fn
        self._prepare_output_fn = prepare_output_fn
        self._parallel_style_name = parallel_style_name
        if partition_fn is not None:
            for name, submod in self._mod.named_modules():
                partition_fn(name, submod, self._device_mesh)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._mod, name)

    def forward(self, *args, **kwargs):
        if self._prepare_input_fn:
            args, kwargs = self._prepare_input_fn(self._device_mesh, *args, **kwargs)

        output = self._mod(*args, **kwargs)

        if self._prepare_output_fn:
            output = self._prepare_output_fn(self._device_mesh, output)
        return output

    def extra_repr(self):
        if self._parallel_style_name:
            return f"parallel_style_name={self._parallel_style_name}"

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
