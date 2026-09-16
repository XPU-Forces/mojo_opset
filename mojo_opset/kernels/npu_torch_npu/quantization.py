"""torch_npu dynamic quantization with public scale/layout normalization."""

from typing import Optional

import torch
import torch_npu


@torch.library.custom_op("mojo_npu_torch_npu::dynamic_quant_fwd", mutates_args=())
def dynamic_quant_fwd(input: torch.Tensor, inv_smooth_scale: Optional[torch.Tensor] = None
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    if input.numel() == 0:
        return (torch.empty_like(input, dtype=torch.int8),
                torch.empty((*input.shape[:-1], 1), device=input.device, dtype=torch.float32))
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("torch_npu dynamic_quant requires float16 or bfloat16 input")
    smooth = None if inv_smooth_scale is None else inv_smooth_scale.to(input.dtype).contiguous()
    quantized, scale = torch_npu.npu_dynamic_quant(
        input.reshape(-1, input.shape[-1]).contiguous(), smooth_scales=smooth, dst_type=torch.int8
    )
    quantized = quantized.reshape(input.shape)
    scale = scale.reshape(*input.shape[:-1], 1)
    small = scale < 1e-6
    return torch.where(small, 0, quantized), torch.where(small, 1.0, scale)


@dynamic_quant_fwd.register_fake
def _dynamic_quant_fake(input, inv_smooth_scale=None):
    if input.numel() and input.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("torch_npu dynamic_quant requires float16 or bfloat16 input")
    return (torch.empty_like(input, dtype=torch.int8, memory_format=torch.contiguous_format),
            torch.empty((*input.shape[:-1], 1), device=input.device, dtype=torch.float32))
