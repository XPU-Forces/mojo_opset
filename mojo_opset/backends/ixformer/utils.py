"""Shared helpers for the ixformer backend (Iluvatar)."""

from __future__ import annotations

import torch


def _get_ixf_and_check_device(tensor: torch.Tensor, class_name: str):
    """Import ixformer ``functions`` and require a CUDA tensor."""
    if not tensor.is_cuda:
        raise RuntimeError(f"{class_name} expects CUDA tensors on Iluvatar.")
    from ixformer import functions as ixf_f

    return ixf_f
