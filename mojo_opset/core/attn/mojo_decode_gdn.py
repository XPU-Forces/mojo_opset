from typing import Tuple

import torch

from ..mojo_operator import MojoOperator


class MojoDecodeGDN(MojoOperator):
    def __init__(
        self,
        use_qk_l2norm_in_kernel: bool = True,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel

    def forward_std(
        self,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        scale=None,
    ):
        raise NotImplementedError

    def forward_ref(
        self,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        scale=None,
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

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output tensor `o` of shape `(1, T, H, V)`.
                - Final states tensor of shape `(num_sequences, H, K, V)`.
        """
        B, T, H, K, V = *q.shape, v.shape[-1]
        Hk = k.shape[-2]
        num_repeats = H // Hk
        k = k
        v = v
        g = g
        beta = beta
        q = q.float()
        k = k.float()
        v = v.float()
        g = g.float()
        beta = beta.float()

        if scale is None:
            scale = K**-0.5

        if self.use_qk_l2norm_in_kernel:
            q = q / q.norm(dim=-1, keepdim=True)
            k = k / k.norm(dim=-1, keepdim=True)

        if cu_seqlens is None:
            batch_size = B
        else:
            batch_size = len(cu_seqlens) - 1

        final_states = torch.zeros(batch_size, Hk, K, V, dtype=torch.float32, device=q.device)
        g = g.exp()
        if cu_seqlens is None:
            o = torch.zeros(B, 1, H, V, dtype=q.dtype, device=q.device)
        else:
            o = torch.zeros(1, T, H, V, dtype=q.dtype, device=q.device)

        for batch in range(batch_size):
            if cu_seqlens is None:
                seq_start = 0
                seq_end = T
                batch_i = batch
            else:
                seq_start = cu_seqlens[batch]
                seq_end = cu_seqlens[batch + 1]
                batch_i = 0
            identity = torch.eye(K, device=k.device, dtype=k.dtype).unsqueeze(0).expand(Hk, -1, -1)  # Shape: [Hk, K, K]

            # now we do the rnn for a easy impl
            if initial_state is None:
                initial_state_batch = torch.zeros(Hk, K, V, dtype=q.dtype, device=q.device)
            else:
                initial_state_batch = initial_state[batch].float()

            for token_i in range(seq_start, seq_end):
                qi = q[batch_i, token_i, :, :]  # [H, K]
                ki = k[batch_i, token_i, :, :]  # [Hk, K]
                vi = v[batch_i, token_i, :, :]  # [Hk, V]
                gi = g[batch_i, token_i, :]  # [Hk]
                betai = beta[batch_i, token_i, :]  # [Hk]

                initial_state_batch = torch.einsum(
                    "hkK, hKv ->hkv",
                    (identity - betai[:, None, None] * torch.einsum("hk, hK->hkK", ki, ki)),
                    initial_state_batch.float() * gi[:, None, None],
                ) + betai[:, None, None] * torch.einsum("hK, hv->hKv", ki, vi)
                o[batch_i, token_i, :, :] = (
                    torch.einsum("hk, hkv -> hv", qi, initial_state_batch.repeat_interleave(num_repeats, dim=0)) * scale
                ).to(q.dtype)
                initial_state_batch = initial_state_batch.to(q.dtype)

            final_states[batch] = initial_state_batch

        return o, final_states

    def forward_analysis(
        self,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        scale=None,
    ) -> Tuple[int, int, int]:
        pass
