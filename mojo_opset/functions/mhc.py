"""Forward-only manifold-constrained Hyper-Connections."""

import torch

from ._checks import _require_inference
from ._dispatch import load_impl


def mhc_coefficients(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor,
    *,
    implementation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collapse four GLM-5.3-Flash streams and return the post and residual weights.

    x is BF16 or FP16 with shape [B, S, 4, H], where H is 4096 or 7168
    and B*S is at most 8192. phi, alpha and bias are FP32 with shapes
    [24, 4*H], [3] and [24]. All inputs are contiguous on the same device.

    Returns (hin, h_post, h_res) with shapes [B, S, H], [B, S, 4]
    and [B, S, 16]. hin has x's dtype and the two weight tensors are
    FP32. h_res stores the 4-by-4 matrix in row-major order. RMS epsilon
    is 1e-5. Sinkhorn uses epsilon 1e-6 and 20 iterations.
    """
    if x.ndim != 4 or x.shape[2] != 4 or x.shape[3] not in (4096, 7168):
        raise ValueError("x must have shape [B, S, 4, H] with H equal to 4096 or 7168")
    if x.shape[0] * x.shape[1] > 8192:
        raise ValueError("mhc_coefficients supports at most 8192 tokens")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("x must be BF16 or FP16")
    if phi.shape != (24, 4 * x.shape[-1]) or alpha.shape != (3,) or bias.shape != (24,):
        raise ValueError("phi, alpha and bias must have shapes [24, 4*H], [3] and [24]")
    for tensor in (phi, alpha, bias):
        if tensor.dtype != torch.float32:
            raise ValueError("phi, alpha and bias must be FP32")
        if tensor.device != x.device:
            raise ValueError("all inputs must be on the same device")
    if not all(tensor.is_contiguous() for tensor in (x, phi, alpha, bias)):
        raise ValueError("all inputs must be contiguous")
    _require_inference("mhc_coefficients", x, phi, alpha, bias)
    forward, _ = load_impl("mhc_coefficients", implementation)
    return forward(x, phi, alpha, bias)
