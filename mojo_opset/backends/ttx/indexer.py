from typing import Optional
import torch


from mojo_opset.backends.ttx.kernels.ascend.indexer import (
    indexer_forward,
    TritonConfig
)
from mojo_opset.core import MojoIndexer

class TTXIndexer(MojoIndexer, default_priority=0):
    def forward_std(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        config = TritonConfig(
            self.dim,
            self.n_heads,
            self.n_local_heads,
            self.head_dim,
            self.rope_head_dim,
            self.topk,
            self.q_lora_rank
        )
        return indexer_forward(x, qr, start_pos, freqs_cis, config, self.k_cache, self.k_scale_cache, mask)