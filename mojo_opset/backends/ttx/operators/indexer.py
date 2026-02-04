import torch
from typing import Optional

from mojo_opset.backends.ttx.kernels import lightning_index_impl
from mojo_opset.core import MojoLightningIndex


class TTXLightningIndex(MojoLightningIndex):
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

        # Create index score tensor
        index_score = torch.zeros(
            (batch_size, q_seq_len, k_seq_len), dtype=torch.float32, device=query.device
        )

        # Handle key_scale: validate and broadcast if needed
        if key_scale is None:
            # If no key_scale provided, use all ones
            key_scale = torch.ones(
                (batch_size, k_seq_len, head_dim),
                dtype=torch.float32,
                device=query.device,
            )
        else:
            key_scale_shape = key_scale.shape
            if len(key_scale_shape) == 1:
                # [N] -> expand to [B, N, K]
                assert (
                    key_scale_shape[0] == k_seq_len
                ), f"key_scale [N] must have N={k_seq_len}, got {key_scale_shape[0]}"
                key_scale = (
                    key_scale.to(torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(batch_size, -1, head_dim)
                )
            elif len(key_scale_shape) == 2:
                # [B,N] -> expand to [B, N, K]
                assert key_scale_shape == (
                    batch_size,
                    k_seq_len,
                ), f"key_scale must be [B, N], got {key_scale_shape}"
                key_scale = (
                    key_scale.to(torch.float32).unsqueeze(-1).expand(-1, -1, head_dim)
                )
            elif len(key_scale_shape) == 3:
                assert key_scale_shape == (
                    batch_size,
                    k_seq_len,
                    head_dim,
                ), f"key_scale must be [B, N, K], got {key_scale_shape}"
            else:
                raise ValueError(f"Invalid key_scale shape {key_scale_shape}")

        query = query.contiguous()
        query_scale = query_scale.contiguous()
        key = key.contiguous()

        index_score = lightning_index_impl(query, query_scale, key, key_scale)

        return index_score
