from functools import partial
import torch
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate, Partial
from torch.distributed.tensor import (
    DeviceMesh,
    DTensor,
)
from torch.distributed._functional_collectives import AsyncCollectiveTensor

from mojo_opset.distributed.parallel.utils import stat_dict_rename_hook, shard_tensor
from mojo_opset.distributed.parallel.tensor_parallel import (
    MojoRowwiseParallel,
    MojoColwiseParallel,
    MojoTensorParallel,
)

__DUMMY_NODE__ = "this is the partitions file."

def _simple_partition(src_data_rank, module_name, module: torch.nn.Module, device_mesh: DeviceMesh, sharding_dim=-1):
    module.register_parameter(
        "weight",
        torch.nn.Parameter(
            shard_tensor(device_mesh, [Shard(sharding_dim)], src_data_rank, module.weight)
        ),
    )
    if module.bias is not None and sharding_dim == -1:
        module.register_parameter(
            "bias",
            torch.nn.Parameter(
                shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.bias)
            ),
        )

    module.register_state_dict_post_hook(partial(stat_dict_rename_hook, ("weight", "bias"), device_mesh))

MojoRowwiseParallel.register_dist_info(
    torch.nn.Linear,
    _simple_partition,
    desired_input_layouts=[Shard(-1)],
    desired_output_layouts=[Partial()],
)

MojoColwiseParallel.register_dist_info(
    torch.nn.Linear,
    partial(_simple_partition, sharding_dim=0),
    desired_input_layouts=[Replicate()],
    desired_output_layouts=[Shard(-1)]
)

from mojo_opset.core.operators.normalization import MojoLayerNorm
MojoTensorParallel.register_dist_info(
    MojoLayerNorm,
    partial(_simple_partition),
    desired_input_layouts=[Shard(-1)],
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

        if tensor.placements != desired_input_layout:
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


from mojo_opset.core.operators.attention import MojoPagedDecodeGQA, MojoPagedPrefillGQA
MojoTensorParallel.register_dist_info(
    (MojoPagedPrefillGQA, MojoPagedDecodeGQA),
    prepare_input_fn=partial(
        __attn_prepare_input_fn, [[Shard(-2)], [Shard(-3)], [Shard(-3)]], {}
    ),
    desired_output_layouts=[Shard(-2)],
)
