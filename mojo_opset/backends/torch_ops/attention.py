from typing import Optional, Tuple, Any
from mojo_opset.core import MojoPagedDecodeGQA, MojoPrefillGQA, MojoPagedPrefillGQA
import torch
import torch_npu
import numpy as np


class TorchPrefillGQA(MojoPrefillGQA, default_priority=0):
    def __init__(self,
                 is_causal: bool = True,
                 is_prefill: bool = True,
                 gqa_layout: str = "ABAB",
                 window_size: int = -1,
                 op_name: str = "",
                 layer_idx: int = 0,):
        super().__init__(is_causal, is_prefill, gqa_layout, False, window_size, op_name, layer_idx)

    def forward_std(self, query, k_cache, v_cache, cu_seqlens_q, sm_scale=None):
        batch_size, num_q_heads, seq_len, head_dim = query.shape
        _, num_kv_heads, block_size, _ = k_cache.shape

        if block_size % 128 != 0 or block_size > 512:
            # high performance attention kernel only supports block_size % 128 == 0 and block_size <= 512
            return self.forward_ref(query, k_cache, v_cache, cu_seqlens_q, sm_scale)

        if sm_scale is None:
            sm_scale = head_dim ** -0.5
        atten_mask = torch.triu(
            torch.ones([seq_len, seq_len], dtype=torch.bool, device=query.device), diagonal=1
        )
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=k_cache,
            value=v_cache,
            actual_seq_lengths=cu_seqlens_q,
            num_heads=num_q_heads,
            input_layout="BSND",
            scale=sm_scale,
            pre_tokens=65535,
            next_tokens=0,
            sparse_mode=2,
            num_key_value_heads=num_kv_heads,
            atten_mask=atten_mask,
        )
        return out

    def forward_analysis(self, query, k_cache, v_cache, cu_seqlens_q, sm_scale=None) -> Tuple[int, int, int]:
        pass


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
                    input_layout: str,
                    sm_scale: Optional[float] = None) -> Tuple[Any]:
        batch_size, num_q_heads, head_dim = query.shape
        _, head_nums, block_size, _ = k_cache.shape

        if block_size % 128 != 0 or block_size > 512:
            # high performance attention kernel only supports block_size % 128 == 0 and block_size <= 512
            return self.forward_ref(query, k_cache, v_cache, cu_seqlens_q, block_tables, input_layout, sm_scale)

        if sm_scale is None:
            sm_scale = 1.0 / (head_dim ** 0.5)

        actual_seq_lengths_q = torch.arange(
            1, batch_size + 1, dtype=torch.int32, device="npu")
        out, _ = torch_npu.npu_fused_infer_attention_score(query,
                                                           k_cache,
                                                           v_cache,
                                                           input_layout=input_layout,
                                                           block_table=block_tables.to(
                                                               torch.int32),
                                                           block_size=block_size,
                                                           num_heads=num_q_heads,
                                                           num_key_value_heads=head_nums,
                                                           actual_seq_lengths=actual_seq_lengths_q,
                                                           actual_seq_lengths_kv=cu_seqlens_q,
                                                           scale=sm_scale)
        return out
