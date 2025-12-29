import math

from typing import Any
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.linalg import hadamard

from ..mojo_operator import MojoOperator


class MojoLightningIndexer(MojoOperator):
    def __init__(
        self,
        top_k: int = 10,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)
        self.top_k = top_k

    def forward_std(
        self, query, query_scale, key, key_scale: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> Tuple[Any]:
        raise NotImplementedError

    def forward_ref(
        self, query, query_scale, key, key_scale: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> Tuple[Any]:
        batch_size, q_seq_len, _, _ = query.shape
        k_seq_len = key.shape[1]
        index_score = torch.zeros((batch_size, q_seq_len, k_seq_len), dtype=torch.float32, device=query.device)

        for batch_id in range(batch_size):
            for i in range(q_seq_len):
                q_slice = query[batch_id, i].to(torch.float32)
                k_slice = key[batch_id].to(torch.float32)
                relu_out = torch.maximum(
                    torch.matmul(q_slice.to(torch.float32), k_slice.to(torch.float32).transpose(0, 1)),
                    torch.tensor(0),
                )
                weight_out = relu_out * query_scale[batch_id, i].unsqueeze(-1)
                reduce_out = torch.sum(weight_out, dim=0)
                index_score[batch_id, i] = reduce_out

        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(self.top_k, dim=-1)[1]

        return topk_indices, index_score

    def forward_analysis(
        self, query, query_scale, key, key_scale: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> Tuple[int, int, int]:
        pass

def lightindex(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: Optional[torch.Tensor] = None,
) -> Tuple[Any]:
    """
    Light index calculation with query and optional key scaling.
    
    Args:
        query: [B, M, H, K] query tensor
        query_scale: [B, M, H] query scaling factors
        key: [B, N, K] key tensor
        key_scale: Optional scaling factors for key.
                   Can be [B, N, K], [B, N], [N, K], or [N]
    Returns:
        index_score: [B, M, N]
    """
    batch_size, q_seq_len, num_heads, head_dim = query.shape
    k_seq_len = key.shape[1]
    topk = q_seq_len // 2
    
    # Create index score tensor
    index_score = torch.zeros((batch_size, q_seq_len, k_seq_len), 
                              dtype=torch.float32, device=query.device)
    
    # Handle key_scale: validate and broadcast if needed
    if key_scale is None:
        # If no key_scale provided, use all ones
        key_scale_expanded = torch.ones((batch_size, k_seq_len, head_dim), 
                                        dtype=torch.float32, device=query.device)
    else:
        key_scale_shape = key_scale.shape         
        if len(key_scale_shape) == 1:
            # [N] -> expand to [B, N, K]
            assert key_scale_shape[0] == k_seq_len, \
            f"key_scale [N] must have N={k_seq_len}, got {key_scale_shape[0]}"
            key_scale_expanded = key_scale.to(torch.float32).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, head_dim)
        else:
            raise ValueError(f"Invalid key_scale shape {key_scale_shape}")
    
    # Process each batch and sequence position
    for batch_id in range(batch_size):
        # Get batch slices
        key_batch = key[batch_id].to(torch.float32)  # [N, K]
        key_scale_batch = key_scale_expanded[batch_id]  # [N, K]
        
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

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    hidden_size = x.size(-1)
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device))
    out = out * hidden_size**-0.5
    return out[..., :dim].reshape(*x_shape)


def int8_index(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
) -> Tuple[Any, Any]:
    batch_size, q_seq_len, num_heads, head_dim = query.shape
    k_seq_len = key.shape[1]

    index_score = torch.zeros((batch_size, q_seq_len, k_seq_len), dtype=torch.float32, device=query.device)

    for batch_id in range(batch_size):
        for i in range(q_seq_len):
            q_slice = query[batch_id, i]

            k_slice = key[batch_id]

            q_slice_fp32 = q_slice.to(torch.float32)
            k_slice_fp32 = k_slice.to(torch.float32)

            logits = torch.matmul(q_slice_fp32, k_slice_fp32.transpose(0, 1))
            relu_out = torch.relu(logits)
            current_q_scale = query_scale[batch_id, i].unsqueeze(1)

            weighted_logits = relu_out * current_q_scale
            logits_sum = torch.sum(weighted_logits, dim=0)

            if key_scale is not None:
                current_k_scale = key_scale[batch_id]
                final_score_slice = logits_sum * current_k_scale
            else:
                final_score_slice = logits_sum

            index_score[batch_id, i] = final_score_slice

    return index_score


def act_quant(x: torch.Tensor, scale_fmt: str = "fp32"):
    max_abs = x.abs().amax(dim=-1, keepdim=True) + 1e-12

    scale = max_abs / 127.0

    q = torch.round(x / scale).clamp(-127, 127)
    q_int8 = q.to(torch.int8)

    return q_int8, scale


class MojoIndexer(MojoOperator):
    def __init__(
        self,
        dim: int = 7168,
        n_heads: int = 128,
        n_local_heads: int = 128,
        head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        topk: int = 2048,
        q_lora_rank: int = 1536,
        max_batch_size: int = 128,
        max_seq_len: int = 32768,
        block_size: int = 128,
        scale_fmt: str = "fp32",
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)
        torch.set_default_device('npu:0')
        self.dim = dim
        self.n_heads = n_heads
        self.n_local_heads = n_heads // 1  # world size
        self.head_dim = head_dim
        self.rope_head_dim = qk_rope_head_dim
        self.topk = topk
        self.q_lora_rank = q_lora_rank
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wk = nn.Linear(self.dim, self.head_dim)
        self.k_norm = LayerNorm(self.head_dim)
        # weights_proj in the checkpoint is stored in bf16, while the parameters here are stored in fp32 for convenient.
        self.weights_proj = nn.Linear(self.dim, self.n_heads)
        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = scale_fmt

        self.register_buffer(
            "k_cache", torch.zeros(max_batch_size, max_seq_len, self.head_dim, dtype=torch.int8), persistent=False
        )
        self.register_buffer(
            "k_scale_cache",
            torch.zeros(max_batch_size, max_seq_len, self.head_dim // block_size, dtype=torch.float32),
            persistent=False,
        )

    def forward_std(
        self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[Any]:
        raise NotImplementedError

    def forward_ref(
        self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]
    ):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        # rope in indexer is not interleaved
        q_pe = apply_rotary_emb(q_pe, freqs_cis, False)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        # rope in indexer is not interleaved
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        q = rotate_activation(q)
        k = rotate_activation(k)
        q_int8, q_scale = act_quant(q, self.scale_fmt)
        k_int8, k_scale = act_quant(k, self.scale_fmt)
        self.k_cache[:bsz, start_pos:end_pos] = k_int8
        self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale
        weights = self.weights_proj(x.float()) * self.n_heads**-0.5
        #weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        # index_score = int8_index(
        #     q_int8.contiguous(),
        #     weights,
        #     key=self.k_cache[:bsz, :end_pos].contiguous(),
        #     key_scale=self.k_scale_cache[:bsz, :end_pos].contiguous(),
        # )
        index_score = lightindex(
            q_int8.contiguous(),
            weights,
            key=self.k_cache[:bsz, :end_pos].contiguous(),
            key_scale=self.k_scale_cache[:bsz, :end_pos].contiguous(),
        )
        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
        topk_indices_ = topk_indices.clone()
        dist.broadcast(topk_indices_, src=0)
        assert torch.all(topk_indices == topk_indices_), f"{topk_indices=} {topk_indices_=}"
        return topk_indices

    def forward_analysis(
        self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[int, int, int]:
        pass