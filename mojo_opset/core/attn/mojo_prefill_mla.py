import os
import torch
from torch import nn

from ..mojo_operator import MojoOperator


class MojoPrefillMLA(MojoOperator):
    """
    MLA attention operator for LLM Prefill.
    """

    def __init__(self, is_causal, softmax_scale, window_size, alibi_slope):
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query,  # [B, H, S, E]
        key,  # [B, H, S, E]
        value,  # [B, H, S, E]
        out,
        cu_seqlens_q,
        cu_seq_lens_k,
        max_seq_len_q,
        max_seq_len_kv,
        workspace=None,
    ):
        batch_size, num_attn_heads, seq_len, head_dim = query.size()

        v_head_dim = value.shape(3)

        attn_weights = torch.matmul(query, key.transpose(2, 3))

        if self.softmax_scale is None:
            attn_weights *= 1 / (head_dim**0.5)
        else:
            attn_weights *= self.softmax_scale

        if self.is_causal:
            attention_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.uint8))
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, num_attn_heads, v_head_dim)

        return attn_output


class MojoPagedPrefillMLA(MojoOperator):
    """
    Paged MLA attention operator for LLM Prefill.
    """

    def __init__(self, is_causal, softmax_scale, window_size, alibi_slope):
        self.is_causal = is_causal
        self.softmax_scale = softmax_scale
