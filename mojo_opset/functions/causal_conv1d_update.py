from typing import Optional

import torch

from ._dispatch import load_impl


def causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    head: int = 0,
    *,
    implementation: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-token depthwise causal convolution for decode, updating ``cache`` in place.

    ``x`` is one token per sequence, either ``[N, D]``, ``[1, N, D]`` (packed) or
    ``[N, 1, D]``. ``cache`` is ``[W, N, D]``: W rotation planes, where
    ``cache[(head + k) % W]`` holds the k-th oldest token and ``head`` names the
    plane that receives ``x``. Advance it with ``head = (head + 1) % W`` per step.
    ``weight`` is ``[D, W]``; when omitted the window is summed without weights
    and ``W`` is taken from ``cache``.

    Returns ``(y, cache)`` where ``y`` keeps the input shape and dtype and
    ``cache`` is the same storage, updated in place.
    """
    if weight is not None and weight.ndim != 2:
        raise ValueError("weight must be [channels, width]")
    if x.ndim not in (2, 3):
        raise ValueError("x must be [N, D], [1, N, D] or [N, 1, D]")
    D = x.shape[-1]
    N = x.numel() // D
    W = weight.shape[1] if weight is not None else cache.shape[0]
    if cache.ndim != 3 or cache.shape != (W, N, D):
        raise ValueError("cache must be [width, N, channels] matching weight and x")
    if not (0 <= int(head) < W):
        raise ValueError("head must be in [0, width)")
    if activation not in (None, "silu", "swish"):
        raise ValueError("activation must be None, 'silu', or 'swish'")
    if bias is not None and (bias.shape != (D,) or (weight is not None and bias.dtype != weight.dtype)):
        raise ValueError("bias must have shape [channels] and the weight dtype")
    if residual is not None and residual.numel() != x.numel():
        raise ValueError("residual must have the same number of elements as x")
    if not cache.is_contiguous():
        raise ValueError("cache must be contiguous because it is updated in place")
    for tensor in (x, cache, weight, bias, residual):
        if tensor is None:
            continue
        if tensor.device != x.device or tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("convolution tensors must be floating and on the input device")
        if torch.is_grad_enabled() and tensor.requires_grad:
            raise RuntimeError("causal_conv1d_update does not support autograd; use torch.no_grad()")

    forward, _ = load_impl("causal_conv1d_update", implementation)
    return forward(x, cache, residual, weight, bias, activation, int(head))
