from typing import Dict
from typing import List

import torch

from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Placement

from mojo_opset.distributed.parallel.mojo_parallel import MojoDistributedModule
from mojo_opset.distributed.parallel.mojo_parallel import MojoRegisterableParallelStyle


class MojoDataParallel(MojoRegisterableParallelStyle):
    def __init__(
        self,
        *,
        desired_args_input_layouts: List[Placement] = [],  # Layouts used only for non-DTensor inputs
        desired_kwargs_input_layouts: Dict[str, Placement] = {},  # Layouts used only for non-DTensor inputs
        desired_output_layouts: List[Placement] = [],  # This layout is the placement used to convert the local tensor
        # output by the operator into the corresponding DTensor according to the distributed semantics required by the operator
        use_local_output: bool = True,
    ):
        super().__init__()
        assert desired_args_input_layouts or desired_kwargs_input_layouts or desired_output_layouts
        self.desired_args_input_layouts = desired_args_input_layouts
        self.desired_kwargs_input_layouts = desired_kwargs_input_layouts
        self.desired_output_layouts = desired_output_layouts
        self.use_local_output = use_local_output

    def prepare_input_fn(
        self,
        device_mesh,
        *args,
        **kwargs,
    ):
        def mapping(tensor, desired_input_layout):  # desired_input_layout is used only for non-DTensor inputs
            if not isinstance(tensor, torch.Tensor):
                return tensor
            desired_input_layout = (
                [desired_input_layout] if isinstance(desired_input_layout, Placement) else desired_input_layout
            )

            if not isinstance(tensor, DTensor):
                tensor = DTensor.from_local(tensor, device_mesh, desired_input_layout, run_check=False)

            if tensor.placements != desired_input_layout:
                tensor = tensor.redistribute(placements=desired_input_layout, async_op=True)
            return tensor.to_local()

        args = list(args)
        for (idx, input_tensor), desired_input_layout in zip(enumerate(args), self.desired_args_input_layouts):
            args[idx] = mapping(input_tensor, desired_input_layout)
        for key, input_tensor in kwargs.items():
            if key in self.desired_kwargs_input_layouts:
                kwargs[key] = mapping(input_tensor, self.desired_kwargs_input_layouts[key])

        return (tuple(args), kwargs)

    def prepare_output_fn(self, device_mesh, outputs):
        is_single = False
        is_tuple = False
        if isinstance(outputs, (list, tuple)):
            is_tuple = isinstance(outputs, tuple)
            outputs = list(outputs)
        else:
            outputs = [outputs]
            is_single = True

        # desired_output_layout is used only for non-DTensor outputs
        for (idx, output_tensor), desired_output_layout in zip(
            enumerate(outputs),
            self.desired_output_layouts,
        ):
            if not isinstance(output_tensor, torch.Tensor):
                continue
            desired_output_layout = (
                [desired_output_layout] if isinstance(desired_output_layout, Placement) else desired_output_layout
            )

            if not isinstance(output_tensor, DTensor):
                output_tensor = DTensor.from_local(
                    output_tensor,
                    device_mesh,
                    desired_output_layout,
                    run_check=False,
                )
            if output_tensor.placements != desired_output_layout:
                output_tensor = output_tensor.redistribute(
                    placements=desired_output_layout,
                    async_op=True,
                )
            outputs[idx] = output_tensor.to_local() if self.use_local_output else output_tensor
            if isinstance(outputs[idx], AsyncCollectiveTensor):
                outputs[idx] = outputs[idx].wait()

        return outputs[0] if is_single else tuple(outputs) if is_tuple else outputs

    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh):

        return MojoDistributedModule(
            module,
            device_mesh,
            None,
            self.prepare_input_fn,
            self.prepare_output_fn,
            parallel_style_name=self.__class__.__name__,
        )
