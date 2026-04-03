from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels import dynamic_quant
from mojo_opset.core import MojoDynamicQuant
from mojo_opset.core import MojoQuant
from mojo_opset.core.operators.quantize import _apply_smooth_scale


class TTXQuant(MojoQuant):
    pass


class TTXDynamicQuant(MojoDynamicQuant):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        input: torch.Tensor,
        smooth_scale: Optional[torch.Tensor] = None,
        token_count: Optional[torch.Tensor] = None,
    ):
        if self.smooth_input and smooth_scale is None:
            raise ValueError("smooth_scale is required when smooth_input=True")
        if self.moe_mode and token_count is None:
            raise ValueError("token_count is required when moe_mode=True")
        if token_count is not None and not self.moe_mode:
            raise ValueError("token_count is only supported when moe_mode=True")

        if smooth_scale is not None and (smooth_scale.dim() > 1 or token_count is not None):
            input_fp = _apply_smooth_scale(input.float(), smooth_scale, token_count)
            scale_tensor = torch.ones(
                input_fp.shape[-1],
                device=input_fp.device,
                dtype=torch.float32,
            )
            return dynamic_quant(input_fp, scale_tensor)

        if smooth_scale is not None:
            scale_tensor = smooth_scale.float()
        else:
            scale_tensor = torch.ones(
                input.shape[-1],
                device=input.device,
                dtype=torch.float32,
            )
        return dynamic_quant(input, scale_tensor)
