from numpy import isin
import torch
from typing import Callable, Any, Tuple, List, Dict
from torch.distributed.tensor import (
    DeviceMesh,
    DTensor,
)
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.tensor.placement_types import Placement


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

            if desired_input_layouts and input_layouts != desired_input_layouts:
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
