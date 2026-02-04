import math

from typing import Any
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.linalg import hadamard
from mojo_opset.core.operator import MojoOperator
from mojo_opset.core import MojoLinear, MojoLayerNorm
from mojo_opset.core import MojoIndexerRoPE
from mojo_opset.core import MojoIndexerRotateActivation
from mojo_opset.core import MojoQuantInt8, MojoQuant
from mojo_opset.core import MojoLightningIndexer
from mojo_opset.utils.platform import get_platform


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
        self.wq_b = MojoLinear(weight=nn.Parameter(torch.empty(n_heads * head_dim, q_lora_rank)))
        self.wk = MojoLinear(weight=nn.Parameter(torch.empty(self.head_dim, self.dim)))

        self.k_norm = MojoLayerNorm(self.head_dim)
        # weights_proj in the checkpoint is stored in bf16, while the parameters here are stored in fp32 for convenient.
        self.weights_proj = MojoLinear(weight=nn.Parameter(torch.empty((self.n_heads, self.dim))))
        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = scale_fmt

        self.register_buffer("k_cache", torch.zeros(max_batch_size, max_seq_len, self.head_dim, dtype=torch.int8), persistent=False)
        self.register_buffer(
            "k_scale_cache",
            torch.zeros(max_batch_size, max_seq_len, self.head_dim // block_size, dtype=torch.float32),
            persistent=False,
        )

        self.rope = MojoIndexerRoPE()
        self.activation = MojoIndexerRotateActivation()
        if get_platform() == "npu":
            self.quant = MojoQuantInt8()
        else:
            self.quant = MojoQuant()
        self.lightning_indexer = MojoLightningIndexer()

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], k_w=None, q_w=None, weights_w=None):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)

        k = self.k_norm(self.wk(x))

        q, k = self.rope(q, k, freqs_cis, freqs_cis, rope_head_dim=self.rope_head_dim)

        q = self.activation(q)
        k = self.activation(k)

        q_quant, q_scale = self.quant(q, self.scale_fmt)
        k_quant, k_scale = self.quant(k, self.scale_fmt)

        # return q_int8
        self.k_cache[:bsz, start_pos:end_pos] = k_quant
        self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale

        weights = self.weights_proj(x.float()) * self.n_heads**-0.5

        index_score = self.lightning_indexer(
            q_quant,
            weights,
            key=k_quant,
            key_scale=k_scale.squeeze(-1),
        )

        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(min(self.topk, end_pos), dim=-1)[1]
        return topk_indices, index_score
