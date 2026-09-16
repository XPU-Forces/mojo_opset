"""FP32 reference math preserving Mojo master's fused quantization contracts."""

import torch
import torch.nn.functional as F


def _quant(input, smooth_scale=None, quant_dtype=torch.int8, symmetric=True, *, tiny_rows=False):
    input = input.float()
    if smooth_scale is not None:
        input = input * smooth_scale.float()
    qmax = 127 if quant_dtype == torch.int8 else torch.finfo(quant_dtype).max
    qmin = (-128 if symmetric else 0) if quant_dtype == torch.int8 else -qmax
    scale = input.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax
    if tiny_rows:
        scale = torch.where(scale < 1e-6, 1.0, scale)
    return (input / scale).round().clamp(qmin, qmax).to(quant_dtype), scale


def rms_norm_quant_fwd(x, weight, smooth_scale, eps, quant_dtype, symmetric):
    normed = F.rms_norm(x.float(), [x.shape[-1]], weight.float(), eps)
    return _quant(normed, smooth_scale, quant_dtype, symmetric)


def layer_norm_quant_fwd(x, weight, bias, smooth_scale, eps, quant_dtype, symmetric):
    normed = F.layer_norm(
        x.float(),
        [x.shape[-1]],
        None if weight is None else weight.float(),
        None if bias is None else bias.float(),
        eps,
    )
    return _quant(normed, smooth_scale, quant_dtype, symmetric)


def residual_add_rms_norm_quant_fwd(x, residual, weight, smooth_scale, eps, norm_pos, quant_dtype, symmetric):
    summed = x + residual
    normed = F.rms_norm(summed.float(), [x.shape[-1]], weight.float(), eps)
    output, scale = _quant(normed, smooth_scale, quant_dtype, symmetric)
    return output, summed if norm_pos == "pre" else normed, scale


def residual_add_layer_norm_quant_fwd(x, residual, weight, bias, smooth_scale, eps, norm_pos, quant_dtype, symmetric):
    summed = x + residual
    output, scale = layer_norm_quant_fwd(summed, weight, bias, smooth_scale, eps, quant_dtype, symmetric)
    return output, summed, scale


def _expand(scale, token_count, rows):
    return scale.float().repeat_interleave(token_count, dim=0, output_size=rows)


def moe_dynamic_quant_fwd(input, token_count, inv_smooth_scale):
    flat = input.reshape(-1, input.shape[-1]).float()
    smooth_scale = _expand(inv_smooth_scale, token_count, flat.shape[0])
    output, scale = _quant(flat, smooth_scale, tiny_rows=True)
    return output.view(input.shape), scale.view(*input.shape[:-1], 1)


def dequant_swiglu_quant_fwd(
    x, weight_scale, quant_scale, activation_scale, bias, token_count, activate_left, quant_mode
):
    if token_count is not None:
        weight_scale = _expand(weight_scale, token_count, x.shape[0])
        quant_scale = _expand(quant_scale, token_count, x.shape[0])
        if bias is not None and bias.ndim == 2:
            bias = _expand(bias, token_count, x.shape[0])
    value = x.float() * weight_scale.float()
    if activation_scale is not None:
        value = value * activation_scale.float().unsqueeze(-1)
    if bias is not None:
        value = value + bias.float()
    left, right = value.chunk(2, dim=-1)
    output = F.silu(left) * right if activate_left else F.silu(right) * left
    return _quant(output, quant_scale)
