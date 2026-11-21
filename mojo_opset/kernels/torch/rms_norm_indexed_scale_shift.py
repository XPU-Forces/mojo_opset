"""PyTorch reference for indexed RMSNorm and affine modulation."""

import torch


def rms_norm_indexed_scale_shift_fwd(x, weight, shift, scale, indices, eps):
    values = x.float()
    variance = (values * values).mean(dim=-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + eps) * weight.float()
    gain = 1.0 + scale.index_select(0, indices).float()
    bias = shift.index_select(0, indices).float()
    return (normalized * gain + bias).to(x.dtype).contiguous()
