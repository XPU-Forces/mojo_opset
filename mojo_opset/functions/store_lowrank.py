"""In-place storage of low-rank key vectors in a paged cache."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def store_lowrank(
    label_cache: torch.Tensor,
    key_lr: torch.Tensor,
    block_idxs: torch.Tensor,
    token_idxs: torch.Tensor,
    token_num: int,
    *,
    implementation: Optional[str] = None,
):
    _require_inference("store_lowrank", label_cache, key_lr)
    if block_idxs.dtype != torch.int32 or token_idxs.dtype != torch.int32:
        raise ValueError("block_idxs and token_idxs must be int32")
    if label_cache.ndim != 4 or key_lr.ndim != 3:
        raise ValueError("label_cache must be BNSD and key_lr must be SND")
    forward, _ = load_impl("store_lowrank", implementation)
    forward(label_cache, key_lr, block_idxs, token_idxs, token_num)
    return label_cache
