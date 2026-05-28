from __future__ import annotations

from typing import List

import torch

from mojo_opset.core import MojoInplacePartialRotaryMul


class AscendcInplacePartialRotaryMul(MojoInplacePartialRotaryMul):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        r1: torch.Tensor,
        r2: torch.Tensor,
        *,
        rotary_mode: str = "interleave",
        partial_slice: List[int] = None,
    ) -> torch.Tensor:
        torch.ops.custom.inplace_partial_rotary_mul(
            x,
            r1,
            r2,
            rotary_mode=rotary_mode,
            partial_slice=partial_slice,
        )
        return x

