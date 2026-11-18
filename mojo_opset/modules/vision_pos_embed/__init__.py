# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL fast_pos_embed_interpolate module (vllm-ascend dispatch)."""

from .fast_pos_embed_interpolate import fast_pos_embed_interpolate
from .fast_pos_embed_interpolate import patch_qwen3vl_fast_pos_embed

__all__ = ["fast_pos_embed_interpolate", "patch_qwen3vl_fast_pos_embed"]
