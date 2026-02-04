import torch
from typing import Optional

from ..operator import MojoOperator


class MojoLightningIndex(MojoOperator):

    def __init__(self):
        pass

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
            query: Query tensor. Shape '[B, M, H, K]', where B is batch size, M is the sequence length of query,
                H is head number, K is head dimension.
            query_scale: Query scaling factors. Shape '[B, M, H]'.
            key: Key tensor. Shape '[B, N, K]', where N is the sequence length of key.
            key_scale: Optional scaling factors for key. Shape can be '[B, N, K]', '[B, N]', '[N, K]', or '[N]'.
            
        Returns:
            index_score: Index score tensor. Shape '[B, M, N]'.
        """
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

        # Process each batch and sequence position
        for batch_id in range(batch_size):
            # Get batch slices
            key_batch = key[batch_id].to(torch.float32)  # [N, K]
            key_scale_batch = key_scale[batch_id]  # [N, K]

            # Apply key scaling: K_scaled = K * key_scale
            key_scaled = key_batch * key_scale_batch

            for i in range(q_seq_len):
                # Get query slice: [H, K]
                q_slice = query[batch_id, i].to(torch.float32)

                # Calculate dot product: Q @ K_scaled^T = [H, N]
                dot_product = torch.matmul(q_slice, key_scaled.transpose(0, 1))

                # Apply ReLU
                relu_out = torch.maximum(dot_product, torch.tensor(0.0))

                # Get query scale for this position: [H] -> [H, 1]
                q_scale_slice = query_scale[batch_id, i].unsqueeze(-1)  # [H, 1]

                # Apply query scaling: [H, N] * [H, 1] = [H, N] (broadcast along N dimension)
                scaled_out = relu_out * q_scale_slice

                # Sum over heads: [N]
                reduce_out = torch.sum(scaled_out, dim=0)

                # Store result
                index_score[batch_id, i] = reduce_out
        return index_score
