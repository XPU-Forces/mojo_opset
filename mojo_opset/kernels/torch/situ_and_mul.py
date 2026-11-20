from typing import Optional

import torch


def situ_and_mul_fwd(
    x: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
) -> torch.Tensor:
    """Reference forward mirroring Kimi-K3 SituAndMul on [gate | up] rows."""
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    d = x.shape[-1] // 2
    gate = x[..., :d].to(compute_dtype)
    up = x[..., d:].to(compute_dtype)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype).contiguous()


def situ_and_mul_bwd(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
) -> torch.Tensor:
    """Analytic gradient w.r.t. the concatenated [gate | up] input."""
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    d = x.shape[-1] // 2
    gate = x[..., :d].to(compute_dtype)
    up = x[..., d:].to(compute_dtype)
    grad = grad_output.to(compute_dtype)

    tanh_gate = torch.tanh(gate / beta)
    sigmoid_gate = torch.sigmoid(gate)
    situ_a = beta * tanh_gate * sigmoid_gate
    d_situ = (1 - tanh_gate * tanh_gate) * sigmoid_gate + beta * tanh_gate * sigmoid_gate * (1 - sigmoid_gate)

    if linear_beta is not None:
        tanh_up = torch.tanh(up / linear_beta)
        up_transformed = linear_beta * tanh_up
        d_up = 1 - tanh_up * tanh_up
    else:
        up_transformed = up
        d_up = torch.ones_like(up)

    grad_gate = grad * d_situ * up_transformed
    grad_up = grad * situ_a * d_up
    return torch.cat((grad_gate, grad_up), dim=-1).to(x.dtype).contiguous()
