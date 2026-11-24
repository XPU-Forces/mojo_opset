from typing import Optional

import torch

from ._dispatch import load_impl


def rotate_activation(x: torch.Tensor, *, implementation: Optional[str] = None):
    forward, _ = load_impl("rotate_activation", implementation)
    return forward(x)
