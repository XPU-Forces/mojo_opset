import io
from typing import List, Tuple
from functools import reduce
from numpy import isin
import torch
import torch.distributed as dist
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

def mojo_parallel_save_state_dict_naive(module: torch.nn.Module, f: str | io.BytesIO):
    state_dict = module.state_dict()
    fname = [f if isinstance(f, str) else f.name]
    if dist.get_rank() == 0:
        gather_list = [{}] * dist.get_world_size()
        dist.gather_object(state_dict, object_gather_list=gather_list)
        dist_state_dict = reduce(lambda x, y: x.update(y) or x, gather_list)
        torch.save(dist_state_dict, f)
        dist.broadcast_object_list(fname, src=0)
    else:
        dist.gather_object(state_dict)
        dist.broadcast_object_list(fname, src=0)
    return fname[0]


def mojo_parallel_load_state_dict_naive(
    module: torch.nn.Module, f: str | io.BytesIO, device_mesh: DeviceMesh
):
    # NOTE(liuyuan): mmap is necessary for us to avoid loading the entire state_dict into memory.
    dist_state_dict = torch.load(f, mmap=True)
    named_rank = get_coordinate_str_with_dim_names(device_mesh)
    dist_state_dict = {k.replace(named_rank, ""): v for k, v in dist_state_dict.items()}
    result = module.load_state_dict(dist_state_dict, strict=False)
    assert result.missing_keys == [], f"{result.missing_keys=}"
