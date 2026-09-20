import torch

def rms_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_2d = x.to(compute_dtype).reshape(-1, x.shape[-1]).contiguous()
    rstd = torch.rsqrt(x_2d.square().mean(dim=-1) + eps)
    output = (x_2d * rstd.unsqueeze(-1) * weight.to(compute_dtype)).to(x.dtype)
    return output.reshape(x.shape), rstd


def rms_norm_bwd(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_2d = x.to(compute_dtype).reshape(-1, x.shape[-1]).contiguous()
    grad_2d = grad_output.to(compute_dtype).reshape_as(x_2d).contiguous()
    rstd_2d = rstd.to(compute_dtype).reshape(-1, 1).contiguous()
    weighted_grad = grad_2d * weight.to(compute_dtype)
    projection = (weighted_grad * x_2d).mean(dim=-1, keepdim=True)
    grad_x = rstd_2d * weighted_grad - x_2d * rstd_2d.pow(3) * projection
    grad_weight = (grad_2d * x_2d * rstd_2d).sum(dim=0)
    return grad_x.to(x.dtype).reshape(x.shape), grad_weight.to(weight.dtype)
