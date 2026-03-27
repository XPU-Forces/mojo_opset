from typing import Optional

import torch
import torch_npu

from mojo_opset.core import MojoDequantSwiGLUQuant
from mojo_opset.core import MojoDynamicQuant


class TorchNpuDynamicQuant(MojoDynamicQuant):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        input: torch.Tensor,
        smooth_scale: Optional[torch.Tensor] = None,
        token_count: Optional[torch.Tensor] = None,
    ):
        kwargs = {"dst_type": self.quant_dtype}
        if smooth_scale is not None:
            kwargs["smooth_scales"] = smooth_scale.to(dtype=input.dtype)
        if token_count is not None:
            kwargs["group_index"] = torch.cumsum(
                token_count.to(dtype=torch.int32, device=input.device),
                dim=0,
                dtype=torch.int32,
            )
        return torch_npu.npu_dynamic_quant(input, **kwargs)


class TorchNpuDequantSwiGLUQuant(MojoDequantSwiGLUQuant):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        weight_scale: Optional[torch.Tensor] = None,
        activation_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_scale: Optional[torch.Tensor] = None,
        quant_offset: Optional[torch.Tensor] = None,
        token_count: Optional[torch.Tensor] = None,
    ):
        return torch_npu.npu_dequant_swiglu_quant(
            x,
            weight_scale=weight_scale,
            activation_scale=activation_scale,
            bias=bias,
            quant_scale=quant_scale,
            quant_offset=quant_offset,
            group_index=token_count,
            activate_left=self.activate_left,
            quant_mode=self.quant_mode,
        )
