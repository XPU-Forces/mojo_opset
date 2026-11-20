"""Capability-gated NPU integration for vLLM Qwen3-VL/Omni position tables."""

import importlib.metadata
import os
import platform

from functools import lru_cache
from pathlib import Path

import torch

from packaging.version import Version

from mojo_opset.functions import vision_pos_embed_interp_batch
from mojo_opset.utils.target import detect_target


@lru_cache(maxsize=1)
def _software_supported():
    try:
        if Version(importlib.metadata.version("triton-ascend")) < Version("3.6.0"):
            return False
        if Version(importlib.metadata.version("torch-npu")) < Version("2.10.0"):
            return False
        home = os.environ.get("ASCEND_HOME_PATH")
        if not home:
            return False
        info = Path(home) / (platform.machine() + "-linux") / "ascend_toolkit_install.info"
        fields = dict(line.split("=", 1) for line in info.read_text().splitlines() if "=" in line)
        return Version(fields["version"]) >= Version("9.2.0")
    except (importlib.metadata.PackageNotFoundError, OSError, KeyError, ValueError):
        return False


def can_use_npu_triton(embed_weight, has_triton):
    if not has_triton or embed_weight.device.type != "npu":
        return False
    if embed_weight.dtype not in (torch.float32, torch.bfloat16) or not _software_supported():
        return False
    try:
        target = detect_target()
    except (RuntimeError, ValueError):
        return False
    return target.arch_key in ("npu.a2", "npu.a5")


def interpolate_vllm_grid(embed_weight, grid_thw, num_grid_per_side, m_size, dtype, *, has_triton, native):
    """Use one NPU launch when supported; otherwise preserve native semantics.

    Only the capability decision falls back. Kernel/compilation failures are
    propagated to the caller rather than being hidden behind native execution.
    """
    if can_use_npu_triton(embed_weight, has_triton):
        return vision_pos_embed_interp_batch(embed_weight, grid_thw, num_grid_per_side, m_size, dtype)
    if not grid_thw:
        return embed_weight.new_empty((0, embed_weight.shape[1]), dtype=dtype)
    return torch.cat([native(embed_weight, t, h, w, num_grid_per_side, m_size, dtype) for t, h, w in grid_thw], dim=0)
