"""
Copyright (c) 2026 Bytedance. All Rights Reserved.
"""

import logging

import torch
import torch.nn.functional as F
import torch_npu

from mojo_opset.core import MojoGemm

from .mxfp8 import mxfp8_linear

logger = logging.getLogger(__name__)


class TorchNpuGemm(MojoGemm):
    supported_platforms_list = ["npu"]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype == torch.int8:
            raise TypeError("TorchNpuGemm: int8 weight requires MojoQuantGemm / TorchNpuQuantGemm")
        if (
            hasattr(self, "per_group_scales")
            and self.weight.dtype == torch.float8_e4m3fn
        ):
            inv = getattr(self, "input_smooth_inv", None)
            if inv is not None:
                input = input * inv.to(input.dtype)
            return mxfp8_linear(input, self.weight, self.per_group_scales, self.bias)
        if input.device.type == "npu":
            try:
                return torch_npu.npu_linear(input, self.weight, self.bias)
            except Exception as exc:
                logger.warning("TorchNpuGemm: npu_linear failed (%s), fallback to F.linear", exc)
        return F.linear(input, self.weight, self.bias)
