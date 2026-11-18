# SPDX-License-Identifier: Apache-2.0
"""Parameterized fast_pos_embed_interpolate module wrapper.

Implements the multi-image merged single-launch path of
``fast_pos_embed_interpolate`` (qwen3_vl.py:744-763) on top of the
vision_pos_embed_interp function interface, and the vllm-ascend dispatch
that removes the forced-native monkey-patch.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def triton_ascend_available() -> bool:
    """Capability detection, cached once per process."""
    try:
        import triton  # noqa: F401
        import torch_npu  # noqa: F401

        import torch

        return torch.npu.is_available()
    except ImportError:
        return False


def fast_pos_embed_interpolate(
    embed_weight,
    grid_thw_list,
    num_grid_per_side,
    m_size,
    dtype,
):
    """Interpolate position embeddings for a list of (t, h, w) grids.

    Single kernel launch for all images; output is bitwise identical to
    per-image launch + torch.cat.
    """
    from mojo_opset.functions.vision_pos_embed_interp import (
        pos_embed_interpolate_native,
        triton_pos_embed_interpolate_merged,
    )

    grids = [tuple(g) for g in grid_thw_list]
    if triton_ascend_available():
        return triton_pos_embed_interpolate_merged(
            embed_weight, grids, num_grid_per_side, m_size, dtype
        )
    import torch

    outputs = []
    for t, h, w in grids:
        outputs.append(
            pos_embed_interpolate_native(
                embed_weight, t, h, w, num_grid_per_side, m_size, dtype
            )
        )
    return torch.cat(outputs, dim=0)


def _fast_pos_embed_interpolate_triton_merged(self, grid_thw):
    return fast_pos_embed_interpolate(
        self.pos_embed.weight,
        [tuple(g) for g in grid_thw],
        self.num_grid_per_side,
        self.spatial_merge_size,
        self.dtype,
    )


def _fast_pos_embed_interpolate_native(self, grid_thw):
    from mojo_opset.functions.vision_pos_embed_interp import pos_embed_interpolate_native

    import torch

    outputs = []
    for t, h, w in grid_thw:
        outputs.append(
            pos_embed_interpolate_native(
                self.pos_embed.weight,
                t,
                h,
                w,
                self.num_grid_per_side,
                self.spatial_merge_size,
                self.dtype,
            )
        )
    return torch.cat(outputs, dim=0)


def patch_qwen3vl_fast_pos_embed() -> bool:
    """Capability dispatch replacing patch_qwen3vl.py:72-89 monkey-patch.

    Returns True when the fused Triton path is enabled; False when the
    native fallback (identical behaviour to the removed patch) is kept.
    """
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

    if triton_ascend_available():
        Qwen3_VisionTransformer.fast_pos_embed_interpolate = (
            _fast_pos_embed_interpolate_triton_merged
        )
        return True
    Qwen3_VisionTransformer.fast_pos_embed_interpolate = (
        _fast_pos_embed_interpolate_native
    )
    return False
