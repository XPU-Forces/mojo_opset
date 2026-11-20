"""Collapse hyper-connection residual streams with FP32 pre-mix coefficients."""

from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def hc_collapse(
    x: torch.Tensor,
    pre_mix: torch.Tensor,
    *,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Weighted FP32 sum over ``hc_mult`` residual streams (DeepSeek mHC).

    ``x`` is ``[num_tokens, hc_mult, hidden_size]`` and ``pre_mix`` is
    ``[num_tokens, hc_mult]`` FP32. Returns a contiguous
    ``[num_tokens, hidden_size]`` tensor with ``x``'s dtype, equal to
    ``(pre_mix.unsqueeze(-1) * x.float()).sum(dim=1).to(x.dtype)``.
    Inference only; there is no backward.
    """

    _require_inference("hc_collapse", x, pre_mix)
    if x.ndim != 3:
        raise ValueError(
            f"hc_collapse expects [num_tokens, hc_mult, hidden_size], got {tuple(x.shape)}"
        )
    if pre_mix.shape != x.shape[:2]:
        raise ValueError(
            f"hc_collapse pre_mix shape {tuple(pre_mix.shape)} must be {tuple(x.shape[:2])}"
        )
    forward, _ = load_impl("hc_collapse", implementation)
    return forward(x, pre_mix)
