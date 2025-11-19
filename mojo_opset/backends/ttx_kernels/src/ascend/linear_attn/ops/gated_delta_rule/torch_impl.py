import torch
import torch_npu


def torch_rnn_gated_delta_rule(
    q,  # (1, T, H, K)
    k,  # (1, T, H, K)
    v,  # (1, T, H, V)
    g,  # (1, T, H)
    beta,  # (1, T, H)
    cu_seqlens=None,
    use_qk_l2norm_in_kernel=True,
):
    """
    Computes the Gated Delta Rule recurrently over chunks of a sequence.

    This function implements the Gated Delta Rule, a state update mechanism,
    by processing sequences in a recurrent, token-by-token fashion.

    Args:
        q (torch.Tensor): Query tensor of shape `(1, T, H, K)`.
        k (torch.Tensor): Key tensor of shape `(1, T, H, K)`.
        v (torch.Tensor): Value tensor of shape `(1, T, H, V)`.
        g (torch.Tensor): Log of the state decay gate `alpha`, shape `(1, T, H)`.
        beta (torch.Tensor): Delta update gate, shape `(1, T, H)`.
        cu_seqlens (list): Cumulative sequence lengths for batch processing.
        use_qk_l2norm_in_kernel (bool): If True, applies L2 normalization to q and k.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Output tensor `o` of shape `(1, T, H, V)`.
            - Final states tensor of shape `(num_sequences, H, K, V)`.
    """
    if use_qk_l2norm_in_kernel:
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)

    _, T, H, K, V = *q.shape, v.shape[-1]
    final_states = torch.zeros(len(cu_seqlens) - 1, H, K, V, dtype=torch.float32, device="npu")
    g = g.exp()
    o = torch.zeros(1, T, H, V, dtype=torch.bfloat16, device="npu")

    for batch in range(len(cu_seqlens) - 1):
        seq_start = cu_seqlens[batch]
        seq_end = cu_seqlens[batch + 1]
        identity = torch.eye(K, device=k.device, dtype=k.dtype).unsqueeze(0).expand(H, -1, -1)  # Shape: [H, K, K]

        # now we do the rnn for a easy impl
        initial_state = torch.zeros(H, K, V, dtype=torch.bfloat16, device="npu")
        for token_i in range(seq_start, seq_end):
            qi = q[0, token_i, :, :]  # [H, K]
            ki = k[0, token_i, :, :]  # [H, K]
            vi = v[0, token_i, :, :]  # [H, V]
            gi = g[0, token_i, :]  # [H]
            betai = beta[0, token_i, :]  # [H]

            initial_state = torch.einsum(
                "hkK, hKv ->hkv",
                (identity - betai[:, None, None] * torch.einsum("hk, hK->hkK", ki, ki)),
                initial_state * gi[:, None, None],
            ) + betai[:, None, None] * torch.einsum("hK, hv->hKv", ki, vi)
            o[0, token_i, :, :] = torch.einsum("hk, hkv -> hv", qi, initial_state)

        final_states[batch] = initial_state

    return o, final_states


def torch_chunk_gated_delta_rule(
    q,  # (1, T, H, K)
    k,  # (1, T, H, K)
    v,  # (1, T, H, V)
    g,  # (1, T, H)
    beta,  # (1, T, H)
    cu_seqlens=None,
    use_qk_l2norm_in_kernel=True,
    scale=None,
    chunk_size=16,
):
    """
    Computes the Gated Delta Rule recurrently over chunks of a sequence.

    This function implements the Gated Delta Rule, a state update mechanism,
    by processing sequences in a recurrent, token-by-token fashion.

    Args:
        q (torch.Tensor): Query tensor of shape `(1, T, H, K)`.
        k (torch.Tensor): Key tensor of shape `(1, T, H, K)`.
        v (torch.Tensor): Value tensor of shape `(1, T, H, V)`.
        g (torch.Tensor): Log of the state decay gate `alpha`, shape `(1, T, H)`.
        beta (torch.Tensor): Delta update gate, shape `(1, T, H)`.
        cu_seqlens (list): Cumulative sequence lengths for batch processing.
        use_qk_l2norm_in_kernel (bool): If True, applies L2 normalization to q and k.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Output tensor `o` of shape `(1, T, H, V)`.
            - Final states tensor of shape `(num_sequences, H, K, V)`.
    """
    if use_qk_l2norm_in_kernel:
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)

    _, T, H, K, V = *q.shape, v.shape[-1]
    Hk = k.shape[2]
    if scale is None:
        scale = K**-0.5
    final_states = torch.zeros(len(cu_seqlens) - 1, Hk, K, V, dtype=torch.float32, device="npu")
    o = torch.zeros(1, T, H, V, dtype=torch.bfloat16, device="npu")

    for batch in range(len(cu_seqlens) - 1):
        seq_start = cu_seqlens[batch]
        seq_end = cu_seqlens[batch + 1]
        prev_state = torch.zeros(Hk, K, V, dtype=torch.float32, device="npu")
        for chunk_i in range(seq_start, seq_end, chunk_size):
            chunk_end = min(chunk_i + chunk_size, seq_end)
            b_q = q[0, chunk_i:chunk_end, :, :].transpose(0, 1)  # [Hq, chunk_size, K]
            b_q_g = b_q.view(Hk, H // Hk, b_q.shape[-2], b_q.shape[-1])  # -> [h, g,  c, K], aabb
            b_k = k[0, chunk_i:chunk_end, :, :].transpose(0, 1)  # [Hk, chunk_size, K]
            b_v = v[0, chunk_i:chunk_end, :, :].transpose(0, 1)  # [Hk, chunk_size, V]
            b_g = g[0, chunk_i:chunk_end, :].transpose(0, 1)  # [Hk, chunk_size]
            b_beta = beta[0, chunk_i:chunk_end, :].transpose(0, 1)  # [Hk, chunk_size]
            identity = (
                torch.eye(chunk_end - chunk_i, device=k.device, dtype=torch.float32).unsqueeze(0).expand(Hk, -1, -1)
            )
            b_gamma = b_g.cumsum(dim=-1, dtype=torch.float32)
            b_gamma_diff = torch.tril((b_gamma[:, :, None] - b_gamma[:, None, :]).exp(), diagonal=0)
            b_A = identity + b_gamma_diff * torch.tril(
                torch.einsum("hcK, hCK->hcC", b_k * b_beta[:, :, None], b_k),
                diagonal=-1,
            )
            b_A_inv = torch.linalg.inv(b_A)
            b_u = torch.einsum(
                "hcC, hCV -> hcV",
                b_A_inv.to(b_k.dtype),
                (
                    b_v * b_beta[:, :, None]
                    - torch.einsum(
                        "hcK, hKV->hcV",
                        (b_k * (b_beta * b_gamma.exp())[:, :, None]).to(b_k.dtype),
                        prev_state.to(b_k.dtype),
                    )
                ).to(b_k.dtype),
            )

            b_o = torch.einsum(
                "hgcK, hKV->hgcV",
                (b_q_g * scale * b_gamma.exp()[:, None, :, None]).to(b_q.dtype),
                prev_state.to(b_q.dtype),
            )
            b_o += torch.einsum(
                "hgcC, hCV -> hgcV",
                b_gamma_diff.to(b_q.dtype)[:, None, :, :]
                * torch.tril(
                    torch.einsum("hgcK, hCK->hgcC", (b_q_g * scale).to(b_q.dtype), b_k),
                    diagonal=0,
                ),
                b_u,
            )
            prev_state = b_gamma[:, -1].exp()[:, None, None] * prev_state.to(b_q.dtype) + torch.einsum(
                "hcK, hcV->hKV",
                ((b_gamma[:, -1][:, None] - b_gamma).exp()[:, :, None] * b_k).to(b_q.dtype),
                b_u,
            )
            o[0, chunk_i:chunk_end, :, :] = b_o.reshape(H, chunk_end - chunk_i, V).transpose(0, 1)

        final_states[batch] = prev_state

    return o, final_states
