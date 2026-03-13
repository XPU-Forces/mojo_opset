from typing import List, Tuple
from numpy import isin
import torch
from torch.distributed.tensor import DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Placement
from torch.distributed._functional_collectives import AsyncCollectiveTensor

def get_coordinate_str_with_dim_names(mesh: DeviceMesh):
    coordinate_str = []
    for dim in mesh.mesh_dim_names:
        coordinate_str.append(f"{dim}{mesh.get_local_rank(dim)}")
    return "_".join(coordinate_str)

def shard_tensor(device_mesh, placements:List[Placement], src_data_rank, tensor:torch.Tensor):
    new_tensor = distribute_tensor(
        tensor,
        device_mesh,
        placements,
        src_data_rank=src_data_rank,
    ).to_local()

    if isinstance(new_tensor, AsyncCollectiveTensor):
        new_tensor = new_tensor.wait()
    return new_tensor


def stat_dict_rename_hook(
    name_list: List[str] | Tuple[str],
    device_mesh: DeviceMesh,
    module: torch.nn.Module,
    state_dict,
    prefix,
    local_metadata,
):
    for state in (module.named_parameters(), module.named_buffers()):
        for n, _ in state:
            if n in name_list:
                new_key = get_coordinate_str_with_dim_names(device_mesh) + n
                state_dict[prefix + new_key] = state_dict.pop(prefix + n)
