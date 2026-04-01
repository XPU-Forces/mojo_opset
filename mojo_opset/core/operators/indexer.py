from typing import Optional

import torch

from ..operator import MojoOperator


class MojoLightningIndexer(MojoOperator):
    def forward(
        self,
        query: torch.Tensor,
        query_scale: torch.Tensor,
        key: torch.Tensor,
        key_scale: Optional[torch.Tensor] = None,
    ):
        """
        Lightning index calculation with query and optional key scaling.

        Args:
            query: Query tensor. Shape ``[B, M, H, K]``, where B is batch size,
                M is the sequence length of query, H is head number, K is head dimension.
            query_scale: Query scaling factors. Shape ``[B, M, H]``.
            key: Key tensor. Shape ``[B, N, K]``, where N is the sequence length of key.
            key_scale: Optional scaling factors for key. Shape can be ``[B, N]`` or ``[N]``.

        Returns:
            index_score: Index score tensor. Shape ``[B, M, N]``.
        """
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

        index_score = torch.zeros(
            (batch_size, q_seq_len, k_seq_len),
            dtype=torch.float32,
            device=query.device,
        )

        for batch_id in range(batch_size):
            key_batch = key[batch_id].to(torch.float32)  # [N, K]
            key_scale_batch = key_scale[batch_id].unsqueeze(-1)  # [N, 1]
            key_scaled = key_batch * key_scale_batch  # [N, K]

            for i in range(q_seq_len):
                q_slice = query[batch_id, i].to(torch.float32)  # [H, K]
                dot_product = torch.matmul(q_slice, key_scaled.transpose(0, 1))  # [H, N]
                relu_out = torch.maximum(dot_product, torch.tensor(0.0))
                q_scale_slice = query_scale[batch_id, i].unsqueeze(-1)  # [H, 1]
                scaled_out = relu_out * q_scale_slice
                index_score[batch_id, i] = torch.sum(scaled_out, dim=0)

        return index_score
