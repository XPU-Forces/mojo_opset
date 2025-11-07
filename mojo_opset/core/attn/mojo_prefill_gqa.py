import os
import torch
from torch.nn import functional as F
from typing import Optional, Tuple, Any
import math

from ..mojo_operator import MojoOperator


class MojoPrefillGQA(MojoOperator):
    """
    GQA attention operator.
    Args:
        is_causal (bool): Whether to apply causal masking.
        is_prefill (bool): Whether running in prefill mode.
        softmax_scale (float): Scaling factor for the softmax operation.
        gqa_layout (str): Layout for GQA attention.
        rm_padding (bool): Whether to remove padding from attention computation.
        window_size (int): Window size for attention computation, -1 means full attention.
        op_name (str): Name of the operator.
    """

    def __init__(
        self,
        is_causal: bool = True,
        is_prefill: bool = True,
        softmax_scale: float = None,
        gqa_layout: str = "ABAB",
        rm_padding: bool = False,
        window_size: int = -1,
        op_name: str = "",
    ):
        super().__init__(op_name)

        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.rm_padding = rm_padding
        self.softmax_scale = softmax_scale
        self.window_size = window_size

    """
    Forward pass of the Mojo GQA attention operator, reference for backend.
    Args:
        query (torch.Tensor): Query tensor, in shape [B, Q_H, S, D].
        key (torch.Tensor): Key tensor, in shape [B, K_H, S, D].
        value (torch.Tensor): Value tensor, inshape [B, V_H, S, D].

    Returns:
        torch.Tensor: Output tensor.
    """

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if self.window_size != -1:
            raise NotImplementedError

        batch_size, num_attn_heads, seq_len, head_dim = query.size()

        num_kv_heads = key.shape(1)

        group = num_attn_heads // num_kv_heads

        query = query.reshape(-1, seq_len, head_dim)
        key = torch.transpose(key, -2, -1)

        if self.gqa_layout == "ABAB":
            key = torch.cat([key] * group, axis=1).reshape(-1, head_dim, seq_len)
            value = torch.cat([value] * group, axis=1).reshape(-1, seq_len, head_dim)
        else:
            raise NotImplementedError

        score = torch.bmm(query, key).float()

        if self.softmax_scale is None:
            score *= 1 / (head_dim**0.5)
        else:
            score *= self.softmax_scale

        if self.is_causal:
            mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.uint8))
            reverse_mask = (1 - mask).float() * -100000.0
            score += reverse_mask
        else:
            raise NotImplementedError

        score = torch.softmax(score, -1).to(query.dtype)

        attn_output = torch.bmm(score, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, num_attn_heads, head_dim)

        return attn_output


class MojoPagedPrefillGQA(MojoOperator):
    def __init__(
        self,
        page_size: int,
        num_q_heads: int,
        num_kv_heads: int, 
        head_dim: int,
        is_causal: bool = True,
        is_prefill: bool = True,
        gqa_layout: str = "ABAB",
        window_size: int = -1,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

        self.page_size = page_size
        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.window_size = window_size

    def forward_std(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_tables: torch.Tensor,
        sm_scale: Optional[float] = None,
    ) -> Tuple[Any]:
        raise NotImplementedError

    def forward_ref(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_tables: torch.Tensor,
        sm_scale: Optional[float] = None,
    ) -> torch.Tensor:
        total_q_tokens, num_q_heads, head_dim = query.shape
        num_total_blocks, num_kv_heads, block_size, _ = k_cache.shape

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(head_dim)

        total_kv_tokens = total_q_tokens

        k_unpadded = torch.zeros(total_kv_tokens, num_kv_heads, head_dim, dtype=query.dtype, device=query.device)
        v_unpadded = torch.zeros(total_kv_tokens, num_kv_heads, head_dim, dtype=query.dtype, device=query.device)

        q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        batch_size = len(q_lens)

        for i in range(batch_size):
            seq_len = q_lens[i].item()
            start_loc = cu_seqlens_q[i].item()
            end_loc = cu_seqlens_q[i + 1].item()

            num_blocks_for_seq = (seq_len + block_size - 1) // block_size

            for j in range(num_blocks_for_seq):
                physical_block_id = block_tables[i, j].item()

                start_pos_in_seq = j * block_size
                tokens_in_block = min(block_size, seq_len - start_pos_in_seq)

                start_loc_in_batch = start_loc + start_pos_in_seq
                end_loc_in_batch = start_loc_in_batch + tokens_in_block

                k_slice = k_cache[physical_block_id, :, :tokens_in_block, :]

                k_unpadded[start_loc_in_batch:end_loc_in_batch, :, :] = k_slice.permute(1, 0, 2)

                v_slice = v_cache[physical_block_id, :, :tokens_in_block, :]
                v_unpadded[start_loc_in_batch:end_loc_in_batch, :, :] = v_slice.permute(1, 0, 2)

        if num_q_heads != num_kv_heads:
            k_expanded = k_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
            v_expanded = v_unpadded.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        else:
            k_expanded = k_unpadded
            v_expanded = v_unpadded

        attn_mask = torch.ones(total_q_tokens, total_q_tokens, device=query.device, dtype=torch.bool).tril(diagonal=0)

        tok_to_seq = torch.repeat_interleave(torch.arange(batch_size, device=query.device), q_lens)

        seq_mask = tok_to_seq[:, None] == tok_to_seq[None, :]
        final_mask = attn_mask & seq_mask

        attn_scores = torch.einsum("thd,khd->thk", query, k_expanded) * sm_scale
        attn_scores.masked_fill_(~final_mask.unsqueeze(1), -torch.inf)

        attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)

        output = torch.einsum("thk,khd->thd", attn_probs, v_expanded)
        return output

    def forward_analysis(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_tables: torch.Tensor,
        sm_scale: Optional[float] = None,
    ) -> Tuple[int, int, int]:
        pass
