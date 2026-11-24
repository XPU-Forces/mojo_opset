import torch


def rms_norm_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    values = x.to(compute_dtype)
    rstd = torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (weight * (values * rstd)).to(x.dtype).contiguous(), rstd


def rms_norm_bwd(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    values = x.to(compute_dtype)
    grad = grad_output.to(torch.promote_types(compute_dtype, weight.dtype))
    normalized = values * rstd
    weighted_grad = (grad * weight).to(compute_dtype)
    grad_rstd = (weighted_grad * values).sum(dim=-1, keepdim=True)
    grad_variance = grad_rstd * -0.5 * rstd.pow(3)
    # Preserve PowBackward0's pow(1): CANN can round it differently from values.
    grad_x = grad_variance.expand_as(values) / x.shape[-1] * (2 * values.pow(1)) + weighted_grad * rstd
    grad_weight = (grad * normalized).sum_to_size(weight.shape)
    return grad_x.to(x.dtype).contiguous(), grad_weight.to(weight.dtype)
