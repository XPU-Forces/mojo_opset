import torch


def kimi_rms_norm_fwd(hidden_states, weight, eps):
    dtype = hidden_states.dtype
    x = hidden_states.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * x.to(dtype)).contiguous()
