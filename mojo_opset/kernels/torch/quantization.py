"""Original Mojo quantization math, including its near-zero scale convention."""

import torch


def static_quant_fwd(input, scale, quant_dtype):
    low, high = (-128, 127) if quant_dtype == torch.int8 else (-448, 448)
    return (input.float() / scale.float()).round().clamp(low, high).to(quant_dtype)


def dequant_fwd(input, scale, output_dtype):
    return (input.float() * scale.float()).to(output_dtype)


def dynamic_quant_fwd(input, inv_smooth_scale=None):
    values = input.float()
    if inv_smooth_scale is not None:
        values = values * inv_smooth_scale.float()
    scale = values.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127
    scale = torch.where(scale < 1e-6, 1.0, scale)
    return (values / scale).round().clamp(-128, 127).to(torch.int8), scale
