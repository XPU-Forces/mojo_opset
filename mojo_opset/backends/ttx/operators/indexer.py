from typing import Optional

import torch

from mojo_opset.backends.ttx.kernels import lightning_indexer_impl
from mojo_opset.core import MojoLightningIndexer
from mojo_opset.experimental.operators.indexer import MojoIndexer


class TTXLightningIndexer(MojoLightningIndexer):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        query: torch.Tensor,
        query_scale: torch.Tensor,
        key: torch.Tensor,
        key_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, q_seq_len, head_num, head_dim = query.shape
        k_seq_len = key.shape[1]

        assert query_scale.size() == (
            batch_size,
            q_seq_len,
            head_num,
        ), f"query_scale must be [B, M, H], got {query_scale.size()}"

        if key_scale is None:
            key_scale = torch.ones(
                (batch_size, k_seq_len),
                dtype=torch.float32,
                device=query.device,
            )
        else:
            key_scale_shape = key_scale.shape
            if len(key_scale_shape) == 1:
                assert key_scale_shape[0] == k_seq_len, (
                    f"key_scale [N] must have N={k_seq_len}, got {key_scale_shape[0]}"
                )
                key_scale = key_scale.to(torch.float32).unsqueeze(0).expand(batch_size, -1)
            elif len(key_scale_shape) == 2:
                assert key_scale_shape == (batch_size, k_seq_len), f"key_scale must be [B, N], got {key_scale_shape}"
            else:
                raise ValueError(f"Invalid key_scale shape {key_scale_shape}")

        query = query.contiguous()
        query_scale = query_scale.contiguous()
        key = key.contiguous()

        return lightning_indexer_impl(query, query_scale, key, key_scale)


class TTXIndexer(MojoIndexer):
    """TTX backend for MojoIndexer.

    All sub-components (LayerNorm, RoPE, Quant, LightningIndexer) are
    auto-dispatched to their TTX implementations through the MojoOperator
    registry mechanism.  No manual replacement is needed.
    """

    supported_platforms_list = ["npu"]
