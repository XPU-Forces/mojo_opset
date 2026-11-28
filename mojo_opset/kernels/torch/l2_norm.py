"""Device-independent L2 normalization reference semantics."""

from typing import Optional

import torch


def l2_norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Row-wise L2 normalization: x * rsqrt(sum(x**2) + eps), over the last dim."""
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    values = x.to(compute_dtype)
    rstd = torch.rsqrt(values.pow(2).sum(dim=-1, keepdim=True) + eps)
    dtype = x.dtype if output_dtype is None else output_dtype
    return (values * rstd).to(dtype).contiguous()
