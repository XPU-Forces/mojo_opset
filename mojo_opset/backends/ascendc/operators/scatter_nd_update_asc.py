from __future__ import annotations

import torch

from mojo_opset.core import MojoScatterNdUpdateAsc


class AscendcScatterNdUpdateAsc(MojoScatterNdUpdateAsc):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        var: torch.Tensor,
        indices: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        torch.ops.custom.scatter_nd_update_asc(var, indices, update)
        return var

