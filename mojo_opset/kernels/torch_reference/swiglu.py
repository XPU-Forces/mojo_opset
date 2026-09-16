from typing import Optional

import torch
import torch.nn.functional as F

def swiglu_fwd(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    compute_dtype = torch.float32 if x1.dtype in (torch.float16, torch.bfloat16) else x1.dtype
    output = F.silu(x1.to(compute_dtype)) * x2.to(compute_dtype)
    if scales is not None:
        output = output * scales.to(compute_dtype).reshape(*x1.shape[:-1], 1)
    return output.to(x1.dtype).contiguous()


def swiglu_bwd(
    grad_output: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    scales: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    compute_dtype = torch.float32 if x1.dtype in (torch.float16, torch.bfloat16) else x1.dtype
    x1_compute = x1.to(compute_dtype)
    x2_compute = x2.to(compute_dtype)
    grad_compute = grad_output.to(compute_dtype)
    sigmoid = torch.sigmoid(x1_compute)
    silu = x1_compute * sigmoid
    silu_grad = sigmoid * (1 + x1_compute * (1 - sigmoid))

    grad_base = grad_compute
    grad_scales = None
    if scales is not None:
        grad_base = grad_base * scales.to(compute_dtype).reshape(*x1.shape[:-1], 1)
        grad_scales = (grad_compute * silu * x2_compute).sum(dim=-1)
        grad_scales = grad_scales.reshape(scales.shape).to(scales.dtype)

    grad_x1 = grad_base * x2_compute * silu_grad
    grad_x2 = grad_base * silu
    return grad_x1.to(x1.dtype).contiguous(), grad_x2.to(x2.dtype).contiguous(), grad_scales
