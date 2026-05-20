from typing import Optional

import torch
import torch.nn as nn

from mojo_opset.core import MojoApplyRoPE
from mojo_opset.core import MojoDynamicQuant
from mojo_opset.core import MojoLayerNorm
from mojo_opset.core.operator import MojoOperator
from mojo_opset.experimental.operators.activation import MojoRotateActivation

__all__ = [
    "MojoIndexer",
    "MojoLightningIndexer",
]


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


class MojoIndexer(MojoOperator):
    def __init__(
        self,
        dim: int = 7168,
        n_heads: int = 128,
        head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        topk: int = 2048,
        q_lora_rank: int = 1536,
        max_batch_size: int = 128,
        max_seq_len: int = 32768,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_head_dim = qk_rope_head_dim
        self.topk = topk
        self.q_lora_rank = q_lora_rank
        self.softmax_scale = self.head_dim**-0.5

        self.wq_b = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = MojoLayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)

        self.register_buffer(
            "k_cache",
            torch.zeros(max_batch_size, max_seq_len, self.head_dim, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "k_scale_cache",
            torch.zeros(max_batch_size, max_seq_len, dtype=torch.float32),
            persistent=False,
        )

        self.rope = MojoApplyRoPE()
        self.activation = MojoRotateActivation()
        self.quant = MojoDynamicQuant()
        self.lightning_indexer = MojoLightningIndexer()

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)

        with torch.no_grad():
            k = self.k_norm(self.wk(x.detach()))

        cos_half, sin_half = freqs_cis.real, freqs_cis.imag
        cos = torch.cat((cos_half, cos_half), dim=-1)
        sin = torch.cat((sin_half, sin_half), dim=-1)
        k = k.unsqueeze(2)

        q, k = self.rope.forward(
            q,
            k,
            cos,
            sin,
            head_first=False,
        )
        k = k.squeeze(2)

        q = self.activation(q)
        k = self.activation(k)

        q_quant, q_scale = self.quant(q, None)
        k_quant, k_scale = self.quant(k, None)
        if k_scale.dim() == 3:
            k_scale = k_scale.amax(dim=-1)

        self.k_cache[:bsz, start_pos:end_pos] = k_quant
        self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale

        weights = self.weights_proj(x.float()) * self.n_heads**-0.5
        weights = weights * q_scale * self.softmax_scale

        index_score = self.lightning_indexer(
            q_quant.contiguous(),
            weights.contiguous(),
            key=self.k_cache[:bsz, :end_pos].contiguous(),
            key_scale=self.k_scale_cache[:bsz, :end_pos].contiguous(),
        )

        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(min(self.topk, end_pos), dim=-1)[1]

        return topk_indices, index_score

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, n_heads={self.n_heads}, head_dim={self.head_dim}, "
            f"rope_head_dim={self.rope_head_dim}, topk={self.topk}, "
            f"q_lora_rank={self.q_lora_rank}"
        )
