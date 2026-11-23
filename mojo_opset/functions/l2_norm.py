"""L2 normalization over the last dimension; forward only."""

import math
from typing import Optional

import torch

from ._checks import _require_inference
from ._dispatch import load_impl

_OUTPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def l2_norm(
    x: torch.Tensor,
    eps: float = 1e-6,
    *,
    output_dtype: Optional[torch.dtype] = None,
    implementation: Optional[str] = None,
) -> torch.Tensor:
    """Normalize each last-dimension slice to unit L2 norm.

    Returns the input shape, and the input dtype unless ``output_dtype`` selects
    a cast on store.
    """
    if x.ndim < 1 or x.shape[-1] <= 0 or not x.is_floating_point():
        raise ValueError("l2_norm requires floating input with a nonempty last dimension")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive")
    if output_dtype is not None and output_dtype not in _OUTPUT_DTYPES:
        raise NotImplementedError("l2_norm output must be float16, bfloat16 or float32")
    _require_inference("l2_norm", x)
    forward, _ = load_impl("l2_norm", implementation)
    return forward(x, float(eps), output_dtype)
