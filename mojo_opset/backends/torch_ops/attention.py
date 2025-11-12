from typing import Optional, Tuple, Any
from mojo_opset.core import MojoPagedDecodeGQA, MojoPagedPrefillGQA
import torch
import torch_npu
import numpy as np


class TorchPagedPrefillGQA(MojoPagedPrefillGQA, default_priority=0):
    def __init__(self,
                 is_causal: bool = True,
                 is_prefill: bool = True,
                 gqa_layout: str = "ABAB",
                 window_size: int = -1,
                 op_name: str = "",
                 layer_idx: int = 0,):
        super().__init__(is_causal, is_prefill, gqa_layout, window_size, op_name, layer_idx)

    def forward_std(self, query, k_cache, v_cache, cu_seqlens_q, block_tables, sm_scale=None):
        _, num_q_heads, head_dim = query.shape
        _, num_kv_heads, block_size, _ = k_cache.shape

        if block_size % 128 != 0 or block_size > 512:
            # high performance attention kernel only supports block_size % 128 == 0 and block_size <= 512
            return self.forward_ref(query, k_cache, v_cache, cu_seqlens_q, block_tables, sm_scale)

        if sm_scale is None:
            sm_scale = head_dim ** -0.5
        compress_mask = torch.from_numpy(
            np.triu(np.ones((2048, 2048), dtype=np.float16), k=1) * -1
        ).to(dtype=torch.bool).to("npu")
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=k_cache,
            value=v_cache,
            atten_mask=compress_mask,
            block_table=block_tables.to(torch.int32),
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=cu_seqlens_q[1:],
            actual_seq_lengths_kv=cu_seqlens_q[1:]-cu_seqlens_q[:-1],
            num_key_value_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale=sm_scale,
            sparse_mode=3
        )
        return out


class TorchPagedDecodeGQA(MojoPagedDecodeGQA, default_priority=0):
    def __init__(self,
                 is_causal: bool = True,
                 is_prefill: bool = True,
                 gqa_layout: str = "ABAB",
                 window_size: int = -1,
                 op_name: str = "",
                 layer_idx: int = 0,):
        super().__init__(is_causal, is_prefill, gqa_layout, window_size, op_name, layer_idx)

    def forward_std(self,
                    query: torch.Tensor,
                    k_cache: torch.Tensor,
                    v_cache: torch.Tensor,
                    cu_seqlens_q: torch.Tensor,
                    block_tables: torch.Tensor,
                    sm_scale: Optional[float] = None) -> Tuple[Any]:
        batch_size, num_q_heads, head_dim = query.shape
        _, head_nums, block_size, _ = k_cache.shape

        if block_size % 128 != 0 or block_size > 512:
            # high performance attention kernel only supports block_size % 128 == 0 and block_size <= 512
            return self.forward_ref(query, k_cache, v_cache, cu_seqlens_q, block_tables, sm_scale)

        if sm_scale is None:
            sm_scale = 1.0 / (head_dim ** 0.5)

        actual_seq_lengths_q = torch.arange(
            1, batch_size + 1, dtype=torch.int32, device="npu")
        out, _ = torch_npu.npu_fused_infer_attention_score(query,
                                                           k_cache,
                                                           v_cache,
                                                           input_layout="TND",
                                                           block_table=block_tables.to(
                                                               torch.int32),
                                                           block_size=block_size,
                                                           num_heads=num_q_heads,
                                                           num_key_value_heads=head_nums,
                                                           actual_seq_lengths=actual_seq_lengths_q,
                                                           actual_seq_lengths_kv=cu_seqlens_q,
                                                           scale=sm_scale)
        return out
