from __future__ import annotations

from typing import Optional, Tuple

import torch

from mojo_opset.core import MojoRMSNormDynamicQuant
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


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
        try:
            try:
                import custom_ops  # noqa: F401
            except Exception:
                pass

            return torch.ops.custom.npu_rms_norm_dynamic_quant(
                x,
                gamma,
                smooth_scale=smooth_scale,
                beta=beta,
                epsilon=epsilon,
            )
        except Exception:
            logger.warning(
                "AscendC RMSNormDynamicQuant kernel not available, falling back to reference implementation."
            )
            return super().forward(x, gamma, smooth_scale=smooth_scale, beta=beta, epsilon=epsilon)

