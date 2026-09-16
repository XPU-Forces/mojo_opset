from typing import Optional

import torch


def lightning_indexer_fwd(
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
            assert key_scale_shape[0] == k_seq_len, f"key_scale [N] must have N={k_seq_len}, got {key_scale_shape[0]}"
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

    # Chunked over M with a single reused workspace for the [Mc, H, N] fp32
    # matmul result. Everything runs under no_grad: inputs may carry grad
    # (e.g. scales computed from trainable projections), and an autograd
    # graph over B*M chunks would pin every intermediate (OOM at 32x4096).
    chunk = max(1, min(q_seq_len, 2**28 // max(1, head_num * k_seq_len)))
    with torch.no_grad():
        dot_buf = torch.empty((chunk, head_num, k_seq_len), dtype=torch.float32, device=query.device)
        for batch_id in range(batch_size):
            key_batch = key[batch_id].to(torch.float32)  # [N, K]
            key_scale_batch = key_scale[batch_id]  # [N]
            q_batch = query[batch_id].to(torch.float32)  # [M, H, K]

            for m0 in range(0, q_seq_len, chunk):
                m1 = min(m0 + chunk, q_seq_len)
                dot_product = torch.matmul(
                    q_batch[m0:m1], key_batch.transpose(0, 1), out=dot_buf[: m1 - m0]
                )  # [Mc, H, N]
                dot_product.relu_()
                dot_product.mul_(query_scale[batch_id, m0:m1].unsqueeze(-1))  # [Mc, H, 1]
                index_score[batch_id, m0:m1] = dot_product.sum(dim=1).mul_(key_scale_batch)

    return index_score
