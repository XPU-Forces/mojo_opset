from __future__ import annotations

import torch

from mojo_opset.core import MojoScatterNdUpdateAsc
from mojo_opset.utils.logging import get_logger

logger = get_logger(__name__)


class AscendcScatterNdUpdateAsc(MojoScatterNdUpdateAsc):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        var: torch.Tensor,
        indices: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        try:
            try:
                import custom_ops  # noqa: F401
            except Exception:
                pass

            torch.ops.custom.scatter_nd_update_asc(var, indices, update)
            return var
        except Exception:
            logger.warning(
                "AscendC scatter_nd_update_asc kernel not available, falling back to reference implementation."
            )
            return super().forward(var, indices, update)

