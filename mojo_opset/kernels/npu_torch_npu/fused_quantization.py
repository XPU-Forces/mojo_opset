"""Original torch_npu fused quantization, with independently visible leaves."""

from typing import Optional

import torch
import torch.nn.functional as F
import torch_npu

from .normalization import residual_add_rms_norm_infer_fwd
from .normalization import rms_norm_infer_fwd


@torch.library.custom_op("mojo_npu_torch_npu::norm_dynamic_quant", mutates_args=())
def _dynamic_quant(
    x: torch.Tensor, smooth_scale: Optional[torch.Tensor], quant_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    output, scale = torch_npu.npu_dynamic_quant(
        x,
        smooth_scales=None if smooth_scale is None else smooth_scale.to(x.dtype),
        dst_type=quant_dtype,
    )
    return output, scale.unsqueeze(-1)


@_dynamic_quant.register_fake
def _dynamic_quant_fake(x, smooth_scale, quant_dtype):
    return torch.empty_like(x, dtype=quant_dtype), x.new_empty((*x.shape[:-1], 1), dtype=torch.float32)


def _quant(x, smooth_scale, quant_dtype, symmetric):
    output, scale = _dynamic_quant(x, smooth_scale, quant_dtype)
    if not symmetric and quant_dtype == torch.int8:
        output = output.clamp_min(0)
    return output, scale


@torch.library.custom_op("mojo_npu_torch_npu::layer_norm_quant_normalize", mutates_args=())
def _layer_norm(
    x: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor], eps: float
) -> torch.Tensor:
    return F.layer_norm(
        x.float(),
        [x.shape[-1]],
        None if weight is None else weight.float(),
        None if bias is None else bias.float(),
        eps,
    ).to(x.dtype)


@_layer_norm.register_fake
def _layer_norm_fake(x, weight, bias, eps):
    return torch.empty_like(x)


def rms_norm_quant_fwd(x, weight, smooth_scale, eps, quant_dtype, symmetric):
    normed = rms_norm_infer_fwd(x, weight.to(x.dtype), eps)
    return _quant(normed, smooth_scale, quant_dtype, symmetric)


def layer_norm_quant_fwd(x, weight, bias, smooth_scale, eps, quant_dtype, symmetric):
    normed = _layer_norm(x, weight, bias, eps)
    return _quant(normed, smooth_scale, quant_dtype, symmetric)


def residual_add_rms_norm_quant_fwd(x, residual, weight, smooth_scale, eps, norm_pos, quant_dtype, symmetric):
    normed, summed = residual_add_rms_norm_infer_fwd(x, residual, weight.to(x.dtype), eps)
    output, scale = _quant(normed, smooth_scale, quant_dtype, symmetric)
    return output, summed if norm_pos == "pre" else normed.float(), scale


def residual_add_layer_norm_quant_fwd(x, residual, weight, bias, smooth_scale, eps, norm_pos, quant_dtype, symmetric):
    summed = x + residual
    output, scale = layer_norm_quant_fwd(summed, weight, bias, smooth_scale, eps, quant_dtype, symmetric)
    return output, summed, scale


@torch.library.custom_op("mojo_npu_torch_npu::moe_dynamic_quant", mutates_args=())
def moe_dynamic_quant_fwd(
    input: torch.Tensor, token_count: torch.Tensor, inv_smooth_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output, scale = torch_npu.npu_dynamic_quant(
        input.reshape(-1, input.shape[-1]),
        dst_type=torch.int8,
        group_index=torch.cumsum(token_count.to(dtype=torch.int32), dim=0, dtype=torch.int32),
        smooth_scales=inv_smooth_scale.to(input.dtype),
    )
    output = output.view(input.shape)
    scale = scale.view(*input.shape[:-1], 1)
    small = scale < 1e-6
    return torch.where(small, 0, output), torch.where(small, 1.0, scale)


@moe_dynamic_quant_fwd.register_fake
def _moe_dynamic_quant_fake(input, token_count, inv_smooth_scale):
    return torch.empty_like(input, dtype=torch.int8), input.new_empty((*input.shape[:-1], 1), dtype=torch.float32)


@torch.library.custom_op("mojo_npu_torch_npu::dequant_swiglu_quant", mutates_args=())
def dequant_swiglu_quant_fwd(
    x: torch.Tensor,
    weight_scale: torch.Tensor,
    quant_scale: torch.Tensor,
    activation_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    token_count: Optional[torch.Tensor],
    activate_left: bool,
    quant_mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, scale = torch_npu.npu_dequant_swiglu_quant(
        x,
        weight_scale=weight_scale,
        activation_scale=activation_scale,
        bias=bias,
        quant_scale=quant_scale,
        quant_offset=None,
        group_index=token_count,
        activate_left=activate_left,
        quant_mode=quant_mode,
    )
    return output, scale.unsqueeze(-1)


@dequant_swiglu_quant_fwd.register_fake
def _dequant_swiglu_quant_fake(
    x, weight_scale, quant_scale, activation_scale, bias, token_count, activate_left, quant_mode
):
    return (
        x.new_empty((x.shape[0], x.shape[1] // 2), dtype=torch.int8),
        x.new_empty((x.shape[0], 1), dtype=torch.float32),
    )
