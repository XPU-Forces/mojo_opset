from typing import Optional

import torch
import torch.nn.functional as torch_f


def causal_conv1d_update_fwd(
    x: torch.Tensor,
    cache: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    activation: Optional[str],
    head: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation-plane reference: cache is [W, N, D], head names the plane
    that receives x. Returns (y, cache) with cache updated in place."""
    shape = x.shape
    x2 = x.reshape(-1, x.shape[-1])
    N, D = x2.shape
    W = cache.shape[0]

    # window in natural order: oldest .. newest(x)
    cols = [cache[(head + 1 + j) % W] for j in range(W - 1)] + [x2]
    window = torch.stack(cols, dim=-1)  # [N, D, W]

    if weight is not None:
        y = (window.float() * weight.float().unsqueeze(0)).sum(-1)
    else:
        y = window.float().sum(-1)
    if bias is not None:
        y = y + bias.float()
    if activation in ("silu", "swish"):
        y = torch_f.silu(y)
    if residual is not None:
        y = y + residual.reshape(N, D).float()

    cache[head] = x2.to(cache.dtype)
    return y.to(x.dtype).reshape(shape), cache
