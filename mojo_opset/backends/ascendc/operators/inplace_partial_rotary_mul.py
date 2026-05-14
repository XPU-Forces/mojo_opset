from __future__ import annotations

from typing import List

import torch

from mojo_opset.core import MojoInplacePartialRotaryMul
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


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
        try:
            try:
                import custom_ops  # noqa: F401
            except Exception:
                pass

            torch.ops.custom.inplace_partial_rotary_mul(
                x,
                r1,
                r2,
                rotary_mode=rotary_mode,
                partial_slice=partial_slice,
            )
            return x
        except Exception:
            logger.warning(
                "AscendC inplace_partial_rotary_mul kernel not available, falling back to reference implementation."
            )
            return super().forward(x, r1, r2, rotary_mode=rotary_mode, partial_slice=partial_slice)

