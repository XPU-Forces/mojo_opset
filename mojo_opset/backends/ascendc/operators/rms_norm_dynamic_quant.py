from __future__ import annotations

from typing import Optional, Tuple

import torch

from mojo_opset.core import MojoRMSNormDynamicQuant


class AscendcRMSNormDynamicQuant(MojoRMSNormDynamicQuant):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        *,
        smooth_scale: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        epsilon: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.custom.npu_rms_norm_dynamic_quant(
            x,
            gamma,
            smooth_scale=smooth_scale,
            beta=beta,
            epsilon=epsilon,
        )

