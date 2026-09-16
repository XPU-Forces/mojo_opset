import torch
import torch.nn.functional as F

def silu_fwd(x: torch.Tensor) -> torch.Tensor:
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    return F.silu(x.to(compute_dtype)).to(x.dtype)


def silu_bwd(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_compute = x.to(compute_dtype)
    sigmoid = torch.sigmoid(x_compute)
    grad = grad_output.to(compute_dtype) * sigmoid * (1 + x_compute * (1 - sigmoid))
    return grad.to(x.dtype)
