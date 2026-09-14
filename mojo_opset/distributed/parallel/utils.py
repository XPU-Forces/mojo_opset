import io
import os

from functools import reduce
from typing import List
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Placement

from mojo_opset.runtime.config import MojoParallelConfig
from mojo_opset.utils.hf_utils import convert_hf_state_dict
from mojo_opset.utils.platform import get_torch_device

from .mojo_parallel import get_unmanaged_params
from .mojo_parallel import mojo_parallelize_module


def get_coordinate_str_with_dim_names(mesh: DeviceMesh):
    coordinate_str = []
    for dim in mesh.mesh_dim_names:
        coordinate_str.append(f"{dim}{mesh.get_local_rank(dim)}")
    return "_".join(coordinate_str)


def shard_tensor(device_mesh, placements: List[Placement], src_data_rank, tensor: torch.Tensor):
    if tensor.is_meta:
        from torch.distributed.tensor.placement_types import Shard

        rank = device_mesh.get_local_rank()
        size = device_mesh.size()
        for p in placements:
            if isinstance(p, Shard):
                chunk_size = tensor.shape[p.dim] // size
                tensor = tensor.narrow(p.dim, rank * chunk_size, chunk_size).contiguous()
        return tensor

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


def mojo_parallel_load_state_dict_naive(module: torch.nn.Module, f: str | io.BytesIO, device_mesh: DeviceMesh):
    # NOTE(liuyuan): mmap is necessary for us to avoid loading the entire state_dict into memory.
    dist_state_dict = torch.load(f, mmap=True)
    named_rank = get_coordinate_str_with_dim_names(device_mesh)
    dist_state_dict = {k.replace(named_rank, ""): v for k, v in dist_state_dict.items()}
    result = module.load_state_dict(dist_state_dict, strict=False)
    assert result.missing_keys == [], f"{result.missing_keys=}"


def _restore_forced_parameter_dtypes(module: nn.Module, forced_param_dtypes: dict[str, torch.dtype]) -> None:
    if not forced_param_dtypes:
        return
    named_parameters = dict(module.named_parameters())
    with torch.no_grad():
        for name, dtype in forced_param_dtypes.items():
            param = named_parameters[name]
            if param.dtype == dtype:
                continue
            param.data = param.data.to(dtype=dtype)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(dtype=dtype)


def _record_forced_dtypes(module: nn.Module) -> dict[str, torch.dtype]:
    return {name: param.force_dtype for name, param in module.named_parameters() if hasattr(param, "force_dtype")}


def build_distributed_model(
    model: nn.Module,
    par: MojoParallelConfig,
    model_root: str,
    mp_mesh: DeviceMesh,
    plan: dict,
    weight_renaming_callback: callable,
    create_ref_model: callable,
    after_device_move_hook: callable,
):
    local_rank = dist.get_rank()
    torch_device = get_torch_device()
    device = f"{torch_device}:{local_rank}"

    ckpt_tag = (
        f"dist_tp{par.ATTN_TP_SIZE}_ep{par.FFN_EP_SIZE}_sp{par.ATTN_SP_SIZE}_dp{par.ATTN_DP_SIZE}_pp{par.PP_SIZE}"
    )
    ckpt_prefix = os.path.join(model_root, ckpt_tag)
    rank_ckpt = f"{ckpt_prefix}.rank{local_rank}.pt"
    checkpoint_exists = os.path.exists(f"{ckpt_prefix}.rank0.pt")

    if checkpoint_exists:
        mojo_parallelize_module(model, mp_mesh, plan)

        forced = _record_forced_dtypes(model)
        model.to_empty(device=device)
        model = model.to(torch.bfloat16)
        _restore_forced_parameter_dtypes(model, forced)

        if after_device_move_hook:
            after_device_move_hook(model, device)

        named_rank = get_coordinate_str_with_dim_names(mp_mesh)
        rank_state = torch.load(rank_ckpt, map_location=device)
        rank_state = {k.replace(named_rank, ""): v for k, v in rank_state.items()}
        result = model.load_state_dict(rank_state, strict=False)
        assert not result.missing_keys, f"Missing keys: {result.missing_keys}"
        del rank_state

    else:
        mp_group = mp_mesh.get_group()
        stage_src_global_rank = dist.get_process_group_ranks(mp_group)[0]
        is_stage_leader = local_rank == stage_src_global_rank

        full_sd = None
        if is_stage_leader and create_ref_model:
            with torch.device("meta"):
                ref_model = create_ref_model()
                ref_model.create_layers()
            native_keys = set(ref_model.state_dict().keys())
            del ref_model
            pt_files = [f for f in os.listdir(model_root) if f.endswith(".pt") and not f.startswith("dist_")]
            assert len(pt_files) == 1, f"Expected one .pt: {pt_files}"
            hf_sd = torch.load(
                os.path.join(model_root, pt_files[0]),
                map_location="cpu",
                mmap=True,
            )
            renamings, converters = weight_renaming_callback(native_keys)
            full_sd = convert_hf_state_dict(hf_sd, native_keys, renamings, converters)
            del hf_sd
        dist.barrier()

        layer_plan = {k.replace("layers.*.", ""): v for k, v in plan.items()}

        for layer in model.layers:
            layer_id = layer.layer_id
            forced = _record_forced_dtypes(layer)
            layer.to_empty(device=device)
            layer.to(torch.bfloat16)
            _restore_forced_parameter_dtypes(layer, forced)

            if is_stage_leader:
                prefix = f"layers.{layer_id}."
                layer_sd = {k[len(prefix) :]: v for k, v in full_sd.items() if k.startswith(prefix)}
                layer.load_state_dict(layer_sd, strict=True)
                del layer_sd
            dist.barrier(group=mp_group)

            mojo_parallelize_module(layer, mp_mesh, layer_plan)

            with torch.no_grad():
                for _, p in get_unmanaged_params(layer):
                    dist.broadcast(p.data, src=stage_src_global_rank, group=mp_group)

        for mod_name, mod in model.named_children():
            if mod_name == "layers":
                continue
            mod.to_empty(device=device)
            mod.to(torch.bfloat16)
            if is_stage_leader:
                prefix = f"{mod_name}."
                mod_sd = {k[len(prefix) :]: v for k, v in full_sd.items() if k.startswith(prefix)}
                mod.load_state_dict(mod_sd, strict=True)
                del mod_sd
            with torch.no_grad():
                for p in mod.parameters():
                    dist.broadcast(p.data, src=stage_src_global_rank, group=mp_group)

        if after_device_move_hook:
            after_device_move_hook(model, device)

        if is_stage_leader:
            del full_sd
        dist.barrier()

        state_dict = model.state_dict()
        cpu_sd = {k: v.cpu() for k, v in state_dict.items()}
        del state_dict
        torch.save(cpu_sd, rank_ckpt)
        del cpu_sd
        dist.barrier()
