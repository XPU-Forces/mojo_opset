from typing import Optional
import torch

from mojo_opset.backends.ttx.kernels.ascend.lightning_index import (
    lightning_index_forward,
)
from mojo_opset.core import MojoLightningIndexer


class TTXLightningIndex(MojoLightningIndexer, default_priority=0):
    def forward_std(
        self,
        query: torch.Tensor,
        query_scale: torch.Tensor,
        key: torch.Tensor,
        key_scale: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        return lightning_index_forward(query, query_scale, key, self.top_k, key_scale, mask)
