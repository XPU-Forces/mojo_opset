from functools import partial
import torch
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate, Partial
from torch.distributed.tensor import (
    DeviceMesh,
    DTensor,
)
from torch.distributed._functional_collectives import AsyncCollectiveTensor

from mojo_opset.distributed.parallel.utils import stat_dict_rename_hook, shard_tensor
from mojo_opset.distributed.parallel.tensor_parallel import MojoRowwiseParallel, MojoColwiseParallel

__DUMMY_NODE__ = "this is the partitions file."

def _partition_torch_nn_linear(src_data_rank, module_name, module: torch.nn.Module, device_mesh: DeviceMesh, sharding_dim=-1):
    module.register_parameter(
        "weight",
        torch.nn.Parameter(
            shard_tensor(device_mesh, [Shard(sharding_dim)], src_data_rank, module.weight)
        ),
    )
    module.register_state_dict_post_hook(partial(stat_dict_rename_hook, ("weight",), device_mesh))

MojoRowwiseParallel.register_dist_info(
    torch.nn.Linear,
    _partition_torch_nn_linear,
    desired_input_layouts=[Shard(-1)],
    desired_output_layouts=[Partial()],
)

MojoColwiseParallel.register_dist_info(
    torch.nn.Linear,
    partial(_partition_torch_nn_linear, sharding_dim=0),
    desired_input_layouts=[Replicate()],
    desired_output_layouts=[Shard(-1)]
)


def __attn_prepare_input_fn(
    args_desired_input_layouts,
    kwargs_desired_input_layouts,
    input_layouts,
    device_mesh,
    *args,
    **kwargs,
):
    def mapping(tensor, desired_input_layout):
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if not isinstance(tensor, DTensor):
            tensor = DTensor.from_local(
                tensor, device_mesh, input_layouts, run_check=False
            )

        if input_layouts != desired_input_layout:
            tensor = tensor.redistribute(
                placements=desired_input_layout, async_op=True
            )
        return tensor.to_local()

    args = list(args)
    for (idx, input_tensor), desired_input_layout in zip(enumerate(args), args_desired_input_layouts):
        args[idx] = mapping(input_tensor, desired_input_layout)
    for key, input_tensor in kwargs.items():
        if key in kwargs_desired_input_layouts:
            kwargs[key] = mapping(input_tensor, kwargs_desired_input_layouts[key])

    return (tuple(args), kwargs)


# def __attn_prepare_output_fn(
#     desired_output_layouts, output_layouts, use_local_output, device_mesh, outputs
# ):
#     is_single = False
#     is_tuple = False
#     if isinstance(outputs, (list, tuple)): 
#         is_tuple = isinstance(outputs, tuple)
#         outputs=list(outputs)
#     else:
#         outputs = [outputs]
#         is_single=True

#     for idx, output_tensor in enumerate(outputs):
#         if not isinstance(output_tensor, torch.Tensor):
#             continue
#         if not isinstance(output_tensor, DTensor):
#             output_tensor = DTensor.from_local(
#                 output_tensor, device_mesh, desired_output_layouts, run_check=False
#             )
#         if output_tensor.placements != output_layouts:
#             output_tensor = output_tensor.redistribute(placements=output_layouts, async_op=True)
#         outputs[idx] = output_tensor.to_local() if use_local_output else output_tensor
#         if isinstance(outputs[idx], AsyncCollectiveTensor):
#             outputs[idx] = outputs[idx].wait()

#     return outputs[0] if is_single else tuple(outputs) if is_tuple else outputs


from mojo_opset.core.operators.attention import MojoPagedDecodeGQA, MojoPagedPrefillGQA
MojoRowwiseParallel.register_dist_info(
    (MojoPagedPrefillGQA, MojoPagedDecodeGQA),
    prepare_input_fn=partial(
        __attn_prepare_input_fn, [[Shard(-2)], [Shard(-3)], [Shard(-3)]], {}
    ),
    desired_output_layouts=[Shard(-2)],
)
