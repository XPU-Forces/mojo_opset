import os
from abc import ABC
from datetime import timedelta

import torch
import torch.distributed as dist

import logger

DEFAULT_TIMEOUT = timedelta(minutes=30)


class MojoDistBackend(ABC):
    """
    NOTICE: a certain backend need to replace device_lib with the corresponding device lib.
    """

    dist_backend = torch.cuda


mojo_dist_backend = MojoDistBackend()
ori_init_process_group = dist.init_process_group


def mojo_init_process_group(
    backend=None,
    init_method=None,
    timeout=DEFAULT_TIMEOUT,
    world_size=None,
    rank=None,
    group_name="",
    pg_options=None,
    device_id=None,
):
    if os.getenv("XPU_INTER_MACHINE_MASTER_ADDR", "") != "":
        os.environ["MASTER_ADDR"] = os.getenv("XPU_INTER_MACHINE_MASTER_ADDR", "")
    if os.getenv("XPU_INTER_MACHINE_MASTER_PORT", "") != "":
        os.environ["MASTER_PORT"] = os.getenv("XPU_INTER_MACHINE_MASTER_PORT", "")

    if world_size == 1:
        return

    if backend is not None:
        logger.info(f"backend will be replaced with {mojo_dist_backend.dist_backend}")

    logger.info(f"RANK:{rank}/{world_size} init_process_group...")

    ori_init_process_group(
        backend=mojo_dist_backend.dist_backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        group_name=group_name,
        timeout=timeout,
        pg_options=pg_options,
        device_id=device_id,
    )


dist.init_process_group = mojo_init_process_group
