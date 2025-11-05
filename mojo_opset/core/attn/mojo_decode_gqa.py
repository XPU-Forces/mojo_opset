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
    def __init__(
        self,
        is_causal: bool = True,
        q_scale_factor: int = 1,
        gqa_layout: str = "ABAB",
        window_size: int = -1,
        kv_layout: str = "ND",
        tp_size: int = 1,
        is_varlen: bool = True,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        """
        初始化通用参数层面的 Paged Decode GQA 注意力算子。
        参数说明：
        - q_scale_factor (int)：q head 的倍数（整数，默认 1），不对 query 进行缩放。
        - gqa_layout (str)：GQA 头分组布局，取值 {"ABAB","AABB"}，默认 "ABAB"。
        - is_causal (bool)：是否启用因果掩码，默认 True。
        - window_size (int)：注意力窗口长度；-1 表示全窗口，或 >=1 表示滑窗长度，默认 -1。
        - softmax_scale (Optional[float])：注意力 score 的缩放系数，需 >0；默认 None。
        - kv_layout (str)：KV 存储布局指示，取值 {"ND","NZ","CB"}，默认 "ND"。
        - tp_size (int)：张量并行大小，默认 1。
        - is_varlen (bool)：为 True 时走 TND（变长）优先路径；为 False 时走 BSND；默认 True。
        - op_name (str)：算子名称占位，用于注册与诊断。
        """
        super().__init__(op_name, layer_idx)

        # 输入参数校验
        if not isinstance(q_scale_factor, int) or q_scale_factor <= 0:
            raise ValueError(f"q_scale_factor must be a positive integer, got {q_scale_factor}")
        
        if gqa_layout not in ["ABAB", "AABB"]:
            raise ValueError(f"gqa_layout must be one of ['ABAB', 'AABB'], got {gqa_layout}")
        
        if not isinstance(window_size, int) or (window_size != -1 and window_size < 1):
            raise ValueError(f"window_size must be -1 or >= 1, got {window_size}")
        
        if kv_layout not in ["ND", "NZ", "CB"]:
            raise ValueError(f"kv_layout must be one of ['ND', 'NZ', 'CB'], got {kv_layout}")
        
        if not isinstance(tp_size, int) or tp_size <= 0:
            raise ValueError(f"tp_size must be a positive integer, got {tp_size}")
        
        if not isinstance(is_varlen, bool):
            raise ValueError(f"is_varlen must be a boolean, got {is_varlen}")

        # 成员变量赋值
        self.is_causal = is_causal
        self.q_scale_factor = q_scale_factor
        self.gqa_layout = gqa_layout
        self.window_size = window_size
        self.kv_layout = kv_layout
        self.tp_size = tp_size
        self.is_varlen = is_varlen

    def forward_std(self, q, k_cache, v_cache, seqlens, block_tables, softmax_scale: Optional[float] = None) -> Tuple[Any]:
        raise NotImplementedError

    def forward_ref(self, q, k_cache, v_cache, seqlens, block_tables, softmax_scale: Optional[float] = None):
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
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        if num_share_q_heads > 1:
            k_ref = k_ref.repeat_interleave(num_share_q_heads, dim=2)
            v_ref = v_ref.repeat_interleave(num_share_q_heads, dim=2)

        attn = torch.einsum("bhd,bkhd->bhk", q, k_ref) * softmax_scale

        mask = torch.arange(k_len, device=q.device)[None, :] >= seqlens[:, None]
        attn.masked_fill_(mask[:, None, :], -torch.inf)

        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.einsum("bhk,bkhd->bhd", attn, v_ref)
        return out

    def forward_analysis(self, q, k_cache, v_cache, seqlens, block_tables, softmax_scale: Optional[float] = None) -> Tuple[int, int, int]:
        pass
