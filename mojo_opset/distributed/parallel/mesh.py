import os

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from torch.distributed.device_mesh import DeviceMesh


@dataclass(frozen=True)
class LLMParallelConfig:
    """Common LLM parallel settings shared by examples and model code."""

    ep_size: int = 1
    cp_size: int = 1
    attn_tp_size: int = 1
    lmhead_tp_size: int = 1
    o_proj_tp_size: int = 1

    def validate(self, world_size: int, *, require_pure_ep: bool = False) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        for name, value in (
            ("ep_size", self.ep_size),
            ("cp_size", self.cp_size),
            ("attn_tp_size", self.attn_tp_size),
            ("lmhead_tp_size", self.lmhead_tp_size),
            ("o_proj_tp_size", self.o_proj_tp_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")

        if world_size > 1 and world_size % self.ep_size != 0:
            raise ValueError(f"WORLD_SIZE={world_size} must be divisible by EP_SIZE={self.ep_size}")
        if require_pure_ep and world_size > 1 and self.ep_size != world_size:
            raise ValueError(
                "DeepSeek-V4 EP migration currently supports pure EP only, "
                f"got EP_SIZE={self.ep_size}, WORLD_SIZE={world_size}"
            )
        if world_size > 1 and world_size % self.attn_tp_size != 0:
            raise ValueError(
                "WORLD_SIZE must be divisible by ATTN_TP_SIZE, got "
                f"WORLD_SIZE={world_size}, ATTN_TP_SIZE={self.attn_tp_size}"
            )
        if world_size > 1 and self.lmhead_tp_size > 1 and world_size % self.lmhead_tp_size != 0:
            raise ValueError(f"WORLD_SIZE={world_size} must be divisible by LMHEAD_TP_SIZE={self.lmhead_tp_size}")
        if world_size > 1 and self.o_proj_tp_size > 1 and world_size % self.o_proj_tp_size != 0:
            raise ValueError(f"WORLD_SIZE={world_size} must be divisible by O_PROJ_TP_SIZE={self.o_proj_tp_size}")
        if self.cp_size > 1 and self.cp_size != world_size:
            raise ValueError(
                "Golden-style CP requires CP_SIZE == WORLD_SIZE when CP is enabled, got "
                f"CP_SIZE={self.cp_size}, WORLD_SIZE={world_size}"
            )

        prefill_dp_size = self.prefill_dp_size(world_size)
        if self.cp_size > 1 and prefill_dp_size < 1:
            raise ValueError(
                "Invalid Golden-style CP prefill parallel config: "
                f"prefill_dp_size={prefill_dp_size}, WORLD_SIZE={world_size}, "
                f"CP_SIZE={self.cp_size}, ATTN_TP_SIZE={self.attn_tp_size}"
            )

    def ep_rank(self, global_rank: int) -> int:
        return global_rank % self.ep_size if self.ep_size > 1 else 0

    def attn_dp_size(self, world_size: int) -> int:
        return world_size // self.attn_tp_size

    def prefill_dp_size(self, world_size: int) -> int:
        return world_size // max(self.cp_size, 1) // self.attn_tp_size

    def as_model_parallel_config(self, world_size: int) -> dict[str, int]:
        return {
            "cp_size": self.cp_size,
            "attn_dp_size": self.attn_dp_size(world_size),
            "prefill_dp_size": self.prefill_dp_size(world_size),
            "attn_tp_size": self.attn_tp_size,
            "lmhead_tp_size": self.lmhead_tp_size,
            "o_proj_tp_size": self.o_proj_tp_size,
        }


def create_contiguous_subgroup(
    subgroup_size: int,
    global_rank: int,
    world_size: int,
    *,
    pg_options: Any = None,
):
    if subgroup_size <= 1 or world_size <= 1:
        return None
    if world_size % subgroup_size != 0:
        raise ValueError(f"world_size={world_size} must be divisible by subgroup_size={subgroup_size}")
    my_group = None
    for start in range(0, world_size, subgroup_size):
        ranks = list(range(start, start + subgroup_size))
        group = dist.new_group(ranks=ranks, pg_options=pg_options)
        if global_rank in ranks:
            my_group = group
    return my_group


def _hccl_options(buffer_size: int):
    import torch_npu

    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.hccl_config = {"hccl_buffer_size": int(buffer_size)}
    return options


def init_llm_parallel_groups(
    parallel_config: LLMParallelConfig,
    *,
    moe_intermediate_size: int | None = None,
    hidden_size: int | None = None,
    world_size: int | None = None,
    global_rank: int | None = None,
) -> dict[str, Any]:
    """Create LLM process groups in one canonical place under distributed.parallel."""

    if not dist.is_initialized():
        return {
            "cp_group": None,
            "attn_tp_group": None,
            "lmhead_tp_group": None,
            "o_proj_tp_group": None,
            "moe_ep_group": None,
            "moe_ep_group_mc2": None,
            "moe_ep_group_mc2_name": None,
        }

    world_size = dist.get_world_size() if world_size is None else world_size
    global_rank = dist.get_rank() if global_rank is None else global_rank
    parallel_config.validate(world_size)

    hccl_comm_dict = {
        "cp_group": create_contiguous_subgroup(parallel_config.cp_size, global_rank, world_size)
        if parallel_config.cp_size > 1
        else None,
        "attn_tp_group": create_contiguous_subgroup(parallel_config.attn_tp_size, global_rank, world_size)
        if parallel_config.attn_tp_size > 1
        else None,
        "lmhead_tp_group": create_contiguous_subgroup(parallel_config.lmhead_tp_size, global_rank, world_size)
        if parallel_config.lmhead_tp_size > 1
        else None,
        "o_proj_tp_group": create_contiguous_subgroup(parallel_config.o_proj_tp_size, global_rank, world_size)
        if parallel_config.o_proj_tp_size > 1
        else None,
        "moe_ep_group": None,
        "moe_ep_group_mc2": None,
        "moe_ep_group_mc2_name": None,
    }

    if parallel_config.ep_size > 1:
        hccl_buffer_size = int(os.environ.get("HCCL_BUFFSIZE", "200"))
        hccl_comm_dict["moe_ep_group"] = create_contiguous_subgroup(
            parallel_config.ep_size,
            global_rank,
            world_size,
            pg_options=_hccl_options(hccl_buffer_size),
        )

        if moe_intermediate_size is not None and hidden_size is not None:
            default_mc2 = max(
                200,
                moe_intermediate_size * hidden_size * parallel_config.ep_size // (1024 * 1024) + 100,
            )
        else:
            default_mc2 = 200
        mc2_buffer_size = int(os.environ.get("MC2_BUFFSIZE", str(default_mc2)))
        moe_ep_group_mc2 = create_contiguous_subgroup(
            parallel_config.ep_size,
            global_rank,
            world_size,
            pg_options=_hccl_options(mc2_buffer_size),
        )
        hccl_comm_dict["moe_ep_group_mc2"] = moe_ep_group_mc2
        if moe_ep_group_mc2 is not None:
            hccl_comm_dict["moe_ep_group_mc2_name"] = moe_ep_group_mc2._get_backend(
                torch.device("npu")
            ).get_hccl_comm_name(global_rank)

    return hccl_comm_dict


def build_contiguous_submesh(
    parallel_size: int,
    *,
    device_type: str = "npu",
    mesh_dim_name: str = "tp",
    world_size: int | None = None,
) -> DeviceMesh | None:
    """Build a DeviceMesh whose mesh_dim_name groups contiguous ranks.

    The layout matches create_contiguous_subgroup(): ranks [0..tp-1],
    [tp..2*tp-1], ... are grouped together along the requested mesh dim.
    """

    if not dist.is_initialized() or parallel_size <= 1:
        return None
    world_size = dist.get_world_size() if world_size is None else world_size
    if world_size % parallel_size != 0:
        raise ValueError(f"world_size={world_size} must be divisible by parallel_size={parallel_size}")
    dp_size = world_size // parallel_size
    mesh = torch.arange(world_size, dtype=torch.int).reshape(dp_size, parallel_size)
    return DeviceMesh(device_type, mesh, mesh_dim_names=(f"{mesh_dim_name}_dp", mesh_dim_name))[mesh_dim_name]
