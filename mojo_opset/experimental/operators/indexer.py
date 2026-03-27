import math
from typing import Optional

import torch
import torch.nn as nn

from mojo_opset.core import (
    MojoIndexerRotateActivation,
    MojoLayerNorm,
    MojoLightningIndexer,
    MojoQuantIndexer,
    MojoRoPE,
)
from mojo_opset.core.operator import MojoOperator


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
        block_size: int = 128,
        scale_fmt: str = "fp32",
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_local_heads = n_heads // 1  # world size
        self.head_dim = head_dim
        self.rope_head_dim = qk_rope_head_dim
        self.topk = topk
        self.q_lora_rank = q_lora_rank
        self.wq_b = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)

        self.k_norm = MojoLayerNorm(self.head_dim)
        # weights_proj in the checkpoint is stored in bf16, while the parameters here are stored in fp32 for convenient.
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = scale_fmt

        self.register_buffer("k_cache", torch.zeros(max_batch_size, max_seq_len, self.head_dim, dtype=torch.int8), persistent=False)
        self.register_buffer(
            "k_scale_cache",
            torch.zeros(max_batch_size, max_seq_len, dtype=torch.float32),
            persistent=False,
        )

        self.rope = MojoRoPE()
        self.activation = MojoIndexerRotateActivation()
        self.quant = MojoQuantIndexer()

        self.lightning_indexer = MojoLightningIndexer()

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        q = self.wq_b(qr)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)

        with torch.no_grad():
            k = self.k_norm(self.wk(x.detach()))

        cos_half, sin_half = freqs_cis.real, freqs_cis.imag
        cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(0).expand(bsz, -1, -1)
        sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(0).expand(bsz, -1, -1)
        k = k.unsqueeze(2)

        q, k = self.rope.forward(
            q,
            k,
            cos,
            sin,
            head_first=False,
            rope_percentage=self.rope_head_dim / self.head_dim,
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
