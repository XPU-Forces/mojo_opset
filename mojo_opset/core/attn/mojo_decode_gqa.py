import os
import torch
from torch.nn import functional as F
import math
from typing import Optional, Tuple, Any

from ..mojo_operator import MojoOperator


class MojoDecodeGQA(MojoOperator):
    """
    Paged GQA attention operator.
    Args:
        is_causal (bool): Whether to apply causal masking.
        is_prefill (bool): Whether running in prefill mode.
        page_size (int): Page size for attention computation.
        softmax_scale (float): Scaling factor for the softmax operation.
        gqa_layout (str): Layout for GQA attention.
        window_size (int): Window size for attention computation, -1 means full attention.
        op_name (str): Name of the operator.
    """

    def __init__(self, is_causal, is_prefill, page_size, softmax_scale, gqa_layout, window_size, op_name):
        super().__init__(op_name)
        self.is_causal = is_causal
        self.is_prefill = is_prefill
        self.page_size = page_size


class MojoPagedDecodeGQA(MojoOperator):
    """
    Paged GQA attention operator.
    Args:
        is_causal (bool): Whether to apply causal masking.
        is_prefill (bool): Whether running in prefill mode.
        page_size (int): Page size for attention computation.
        softmax_scale (float): Scaling factor for the softmax operation.
        gqa_layout (str): Layout for GQA attention.
        window_size (int): Window size for attention computation, -1 means full attention.
        op_name (str): Name of the operator.
    """

    def __init__(
        self,
        is_causal: bool = True,
        is_prefill: bool = False,
        gqa_layout: str = "ABAB",
        window_size: int = -1,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(op_name, layer_idx)

        self.is_causal = is_causal
        self.gqa_layout = gqa_layout
        self.window_size = window_size

    def forward_std(self, q, k_cache, v_cache, seqlens, block_tables, input_layout, sm_scale) -> Tuple[Any]:
        raise NotImplementedError

    def forward_ref(self, q, k_cache, v_cache, seqlens, block_tables, input_layout, sm_scale):
        batch_size, num_q_heads, head_dim = q.shape
        num_kv_heads, block_size, head_dim = k_cache.shape[1], k_cache.shape[2], k_cache.shape[3]
        max_len_in_batch = seqlens.max().item()

        k_ref = torch.zeros(batch_size, max_len_in_batch, num_kv_heads, head_dim, device=q.device, dtype=q.dtype)
        v_ref = torch.zeros(batch_size, max_len_in_batch, num_kv_heads, head_dim, device=q.device, dtype=q.dtype)

        for i in range(batch_size):
            seq_len = seqlens[i].item()
            num_blocks_for_seq = (seq_len + block_size - 1) // block_size

            for j in range(num_blocks_for_seq):
                physical_block_id = block_tables[i, j].item()

                start_pos = j * block_size
                tokens_in_block = min(block_size, seq_len - start_pos)

                k_slice = k_cache[physical_block_id, :, :tokens_in_block, :]
                v_slice = v_cache[physical_block_id, :, :tokens_in_block, :]

                k_ref[i, start_pos : start_pos + tokens_in_block, :, :] = k_slice.permute(1, 0, 2)
                v_ref[i, start_pos : start_pos + tokens_in_block, :, :] = v_slice.permute(1, 0, 2)

        _, k_len, num_k_heads, _ = k_ref.shape
        num_share_q_heads = num_q_heads // num_k_heads
        if sm_scale is None:
            sm_scale = 1 / math.sqrt(head_dim)

        if num_share_q_heads > 1:
            k_ref = k_ref.repeat_interleave(num_share_q_heads, dim=2)
            v_ref = v_ref.repeat_interleave(num_share_q_heads, dim=2)

        attn = torch.einsum("bhd,bkhd->bhk", q, k_ref) * sm_scale

        mask = torch.arange(k_len, device=q.device)[None, :] >= seqlens[:, None]
        attn.masked_fill_(mask[:, None, :], -torch.inf)

        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.einsum("bhk,bkhd->bhd", attn, v_ref)
        return out

    def forward_analysis(self, q, k_cache, v_cache, seqlens, block_tables, sm_scale) -> Tuple[int, int, int]:
        pass
