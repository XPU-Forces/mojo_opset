from typing import Optional

import torch
from torch import nn

from mojo_opset.backends.ttx.kernels import lightning_indexer_impl
from mojo_opset.backends.ttx.operators.activation import TTXIndexerRotateActivation

from mojo_opset.backends.ttx.operators.normalization import TTXLayerNorm
from mojo_opset.core import MojoLightningIndexer, MojoQuantIndexer, MojoRoPE
from mojo_opset.experimental.operators.indexer import MojoIndexer


class TTXLightningIndexer(MojoLightningIndexer):
    supported_platforms_list = ["npu"]

    def __init__(self):
        super().__init__()

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
        index_score = torch.zeros((batch_size, q_seq_len, k_seq_len), dtype=torch.float32, device=query.device)

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
                # [N] -> expand to [B, N]
                assert key_scale_shape[0] == k_seq_len, f"key_scale [N] must have N={k_seq_len}, got {key_scale_shape[0]}"
                key_scale = key_scale.to(torch.float32).unsqueeze(0).expand(batch_size, -1)
            elif len(key_scale_shape) == 2:
                assert key_scale_shape == (
                    batch_size,
                    k_seq_len,
                ), f"key_scale must be [B, N], got {key_scale_shape}"
            else:
                raise ValueError(f"Invalid key_scale shape {key_scale_shape}")

        query = query.contiguous()
        query_scale = query_scale.contiguous()
        key = key.contiguous()

        index_score = lightning_indexer_impl(query, query_scale, key, key_scale)

        return index_score


class TTXIndexer(MojoIndexer):
    supported_platforms_list = ["npu"]

    def __init__(
        self,
        parent_instance: MojoIndexer,
        max_batch_size: int = 128,
        max_seq_len: int = 32768,
    ):
        # Copy all attributes from parent_instance
        self.__dict__.update(parent_instance.__dict__)

        # Save original k_norm weights before replacing
        original_norm = self.k_norm

        # Get weights from parent_instance's nn.Linear modules
        wq_b_weight = self.wq_b.weight.data
        wk_weight = self.wk.weight.data
        weights_proj_weight = self.weights_proj.weight.data

        # Replace with TTX implementations
        self.wq_b = nn.Linear(wq_b_weight.size(1), wq_b_weight.size(0), bias=False)
        self.wq_b.weight.data = wq_b_weight
        self.wk = nn.Linear(wk_weight.size(1), wk_weight.size(0), bias=False)
        self.wk.weight.data = wk_weight
        self.k_norm = TTXLayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(weights_proj_weight.size(1), weights_proj_weight.size(0), bias=False)
        self.weights_proj.weight.data = weights_proj_weight

        # Copy norm weights
        self.k_norm.weight.data = original_norm.weight.data
        if original_norm.bias is not None:
            self.k_norm.bias.data = original_norm.bias.data
        self.k_norm.variance_epsilon = original_norm.variance_epsilon

        # Register TTX-specific buffers
        self.register_buffer("k_cache_ttx", torch.zeros(max_batch_size, max_seq_len, self.head_dim, dtype=torch.int8), persistent=False)
        self.register_buffer(
            "k_scale_cache_ttx",
            torch.zeros(max_batch_size, max_seq_len, dtype=torch.float32),
            persistent=False,
        )

        # Replace with TTX-specific components
        self.rope = MojoRoPE()
        self.activation = TTXIndexerRotateActivation()

        self.quant = MojoQuantIndexer()
        self.lightning_indexer = TTXLightningIndexer()

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)

        with torch.no_grad():
            k = self.k_norm(self.wk(x.detach()))

        cos = freqs_cis.real.unsqueeze(0).expand(bsz, -1, -1)
        sin = freqs_cis.imag.unsqueeze(0).expand(bsz, -1, -1)

        q, k = self.rope(q, k, cos, sin, rope_head_dim=self.rope_head_dim)

        q = self.activation(q)
        k = self.activation(k)

        q_quant, q_scale = self.quant(q, None)
        k_quant, k_scale = self.quant(k, None)
        if k_scale.dim() == 3:
            k_scale = k_scale.amax(dim=-1)

        self.k_cache_ttx[:bsz, start_pos:end_pos] = k_quant
        self.k_scale_cache_ttx[:bsz, start_pos:end_pos] = k_scale

        weights = self.weights_proj(x.float()) * self.n_heads**-0.5
        weights = weights * q_scale * self.softmax_scale

        index_score = self.lightning_indexer(
            q_quant.contiguous(),
            weights.contiguous(),
            key=self.k_cache_ttx[:bsz, :end_pos].contiguous(),
            key_scale=self.k_scale_cache_ttx[:bsz, :end_pos].contiguous(),
        )

        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(min(self.topk, end_pos), dim=-1)[1]

        return topk_indices, index_score
