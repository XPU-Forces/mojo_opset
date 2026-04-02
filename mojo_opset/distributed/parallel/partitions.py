from functools import partial

import torch
import torch.nn as nn

from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial
from torch.distributed.tensor.placement_types import Replicate
from torch.distributed.tensor.placement_types import Shard

from mojo_opset.core.operators.attention import MojoPagedDecodeGQA
from mojo_opset.core.operators.attention import MojoPagedPrefillGQA
from mojo_opset.core.operators.mlp import MojoSwiGLUMLP
from mojo_opset.distributed.parallel.tensor_parallel import MojoColwiseParallel
from mojo_opset.distributed.parallel.tensor_parallel import MojoQKVColwiseParallel
from mojo_opset.distributed.parallel.tensor_parallel import MojoRowwiseParallel
from mojo_opset.distributed.parallel.tensor_parallel import MojoSwiGLUParallel
from mojo_opset.distributed.parallel.tensor_parallel import MojoTensorParallel
from mojo_opset.distributed.parallel.utils import shard_tensor
from mojo_opset.distributed.parallel.utils import stat_dict_rename_hook

__DUMMY_NODE__ = "this is the partitions file."


def __torch_nn_linear_partition(
    src_data_rank, module_name, module: torch.nn.Module, device_mesh: DeviceMesh, sharding_dim=-1
):
    module.register_parameter(
        "weight",
        torch.nn.Parameter(shard_tensor(device_mesh, [Shard(sharding_dim)], src_data_rank, module.weight)),
    )
    if module.bias is not None and sharding_dim == -1:
        module.register_parameter(
            "bias",
            torch.nn.Parameter(shard_tensor(device_mesh, [Shard(0)], src_data_rank, module.bias)),
        )

    module.register_state_dict_post_hook(partial(stat_dict_rename_hook, ("weight", "bias"), device_mesh))


MojoRowwiseParallel.register_dist_info(
    torch.nn.Linear,
    __torch_nn_linear_partition,
    desired_input_layouts=[Shard(-1)],
    desired_output_layouts=[Partial()],
)

MojoColwiseParallel.register_dist_info(
    torch.nn.Linear,
    partial(__torch_nn_linear_partition, sharding_dim=0),
    desired_input_layouts=[Replicate()],
    desired_output_layouts=[Shard(-1)],
)


def __attn_prepare_input_fn(
    args_desired_input_layouts,  # Layouts used only for non-DTensor inputs
    kwargs_desired_input_layouts,  # Layouts used only for non-DTensor inputs
    input_layouts,  # Requires the user to provide the distributed semantics information of the local tensor to reconstruct the DTensor
    device_mesh,
    *args,
    **kwargs,
):
    def mapping(tensor, desired_input_layout):
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if not isinstance(tensor, DTensor):
            tensor = DTensor.from_local(tensor, device_mesh, input_layouts, run_check=False)

        if tensor.placements != desired_input_layout:
            tensor = tensor.redistribute(placements=desired_input_layout, async_op=True)
        return tensor.to_local()

    args = list(args)
    for (idx, input_tensor), desired_input_layout in zip(enumerate(args), args_desired_input_layouts):
        args[idx] = mapping(input_tensor, desired_input_layout)
    for key, input_tensor in kwargs.items():
        if key in kwargs_desired_input_layouts:
            kwargs[key] = mapping(input_tensor, kwargs_desired_input_layouts[key])

    return (tuple(args), kwargs)


MojoTensorParallel.register_dist_info(
    (MojoPagedPrefillGQA, MojoPagedDecodeGQA),
    prepare_input_fn=partial(__attn_prepare_input_fn, [[Shard(-2)], [Shard(-3)], [Shard(-3)]], {}),
    desired_output_layouts=[Shard(-2)],
)


def __swiglu_partition_fn(src_data_rank, name, mod, mesh):
    if not isinstance(mod, nn.Linear):
        return
    rank = mesh.get_local_rank()
    size = mesh.size()

    if name == "fc1":
        weight = shard_tensor(mesh, [Replicate()], src_data_rank, mod.weight)
        half = weight.shape[0] // 2
        per_tp = half // size
        gate_shard = weight[rank * per_tp : (rank + 1) * per_tp]
        up_shard = weight[half + rank * per_tp : half + (rank + 1) * per_tp]
        new_weight = torch.cat([gate_shard, up_shard], dim=0)
        mod.register_parameter("weight", nn.Parameter(new_weight))
    elif name == "fc2":
        new_weight = shard_tensor(mesh, [Shard(1)], src_data_rank, mod.weight)
        mod.register_parameter("weight", nn.Parameter(new_weight))
    else:
        return

    mod.register_state_dict_post_hook(partial(stat_dict_rename_hook, ("weight", "bias"), mesh))


MojoSwiGLUParallel.register_dist_info(
    MojoSwiGLUMLP,
    partiton_fn=__swiglu_partition_fn,
    desired_input_layouts=[Replicate()],
    desired_output_layouts=[Partial()],
)


def __qkv_partition_fn(src_data_rank, name, mod, mesh, *, num_q_heads, num_kv_heads, head_dim):
    if not isinstance(mod, nn.Linear):
        return
    rank = mesh.get_local_rank()
    size = mesh.size()

    q_total_dim = num_q_heads * head_dim
    kv_total_dim = num_kv_heads * head_dim

    weight = shard_tensor(mesh, [Replicate()], src_data_rank, mod.weight)

    q_per_rank = num_q_heads // size
    q_start = rank * q_per_rank * head_dim
    q_end = q_start + q_per_rank * head_dim
    local_q = weight[q_start:q_end, :]

    replicate = max(1, size // num_kv_heads)
    kv_idx = rank // replicate
    k_offset = q_total_dim
    k_start = k_offset + kv_idx * head_dim
    local_k = weight[k_start : k_start + head_dim, :]

    v_offset = q_total_dim + kv_total_dim
    v_start = v_offset + kv_idx * head_dim
    local_v = weight[v_start : v_start + head_dim, :]

    new_weight = torch.cat([local_q, local_k, local_v], dim=0)
    mod.register_parameter("weight", nn.Parameter(new_weight))

    if mod.bias is not None:
        bias = shard_tensor(mesh, [Replicate()], src_data_rank, mod.bias)
        local_q_bias = bias[q_start:q_end]
        local_k_bias = bias[k_offset + kv_idx * head_dim : k_offset + kv_idx * head_dim + head_dim]
        local_v_bias = bias[v_offset + kv_idx * head_dim : v_offset + kv_idx * head_dim + head_dim]
        new_bias = torch.cat([local_q_bias, local_k_bias, local_v_bias], dim=0)
        mod.register_parameter("bias", nn.Parameter(new_bias))

    mod.register_state_dict_post_hook(partial(stat_dict_rename_hook, ("weight", "bias"), mesh))


MojoQKVColwiseParallel.register_dist_info(
    nn.Linear,
    partiton_fn=__qkv_partition_fn,
    desired_input_layouts=[Replicate()],
    desired_output_layouts=[Shard(-1)],
)
