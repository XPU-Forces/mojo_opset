from typing import Any
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch_npu

from mojo_opset.core import MojoPagedDecodeGQA
from mojo_opset.core import MojoPagedPrefillGQA
from mojo_opset.core import MojoPrefillGQA
from mojo_opset.core.operators.attention import assert_paged_decode_contract
from mojo_opset.core.operators.attention import assert_paged_prefill_contract


class TorchNpuPrefillGQA(MojoPrefillGQA, default_priority=0):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "ABAB",
        window_size: int = -1,
    ):
        super().__init__(is_causal=is_causal, gqa_layout=gqa_layout, rm_padding=False, window_size=window_size)

    def forward(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        
        batch_size, num_q_heads, seq_len, head_dim = query.shape
        _, num_kv_heads, block_size, _ = k_cache.shape


        if head_dim % 128 != 0:
            raise NotImplementedError(f"NPU kernel requires head_dim % 128 == 0, got {query.shape[-1]}")

        if block_size % 128 != 0 or block_size > 512:
            # high performance attention kernel only supports block_size % 128 == 0 and block_size <= 512
            return super().forward(query, k_cache, v_cache, cu_seqlens_q, softmax_scale)

        if softmax_scale is None:
            softmax_scale = head_dim**-0.5
        atten_mask = torch.triu(torch.ones([seq_len, seq_len], dtype=torch.bool, device=query.device), diagonal=1)
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=k_cache,
            value=v_cache,
            actual_seq_lengths=cu_seqlens_q,
            num_heads=num_q_heads,
            input_layout="BSND",
            scale=softmax_scale,
            pre_tokens=65535,
            next_tokens=0,
            sparse_mode=2,
            num_key_value_heads=num_kv_heads,
            atten_mask=atten_mask,
        )
        return out


class TorchNpuPagedPrefillGQA(MojoPagedPrefillGQA, default_priority=0):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "ABAB",
        window_size: int = -1,
    ):
        super().__init__(is_causal=is_causal, gqa_layout=gqa_layout, window_size=window_size)

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_total_seqlens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_total_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        assert_paged_prefill_contract(cu_seqlens_q, block_tables, cu_total_seqlens)
        _, num_q_heads, head_dim = query.shape
        _, num_kv_heads, block_size, _ = key_cache.shape
        total_seq_lens = (
            cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            if cu_total_seqlens is None
            else cu_total_seqlens[1:] - cu_total_seqlens[:-1]
        )
        
        if head_dim % 128 != 0:
            raise NotImplementedError(f"NPU kernel npu_fused_infer_attention_score currently produces incorrect results for head_dim={head_dim} (not a multiple of 128)")
        if cu_total_seqlens is not None:
            raise NotImplementedError("NPU kernel npu_fused_infer_attention_score currently does not support TND layout with sparse_mode=3 (Page Attention), raising RuntimeError: call aclnnFusedInferAttentionScoreV3 failed.")


        if block_size % 128 != 0 or block_size > 512:
            # high performance attention kernel only supports block_size % 128 == 0 and block_size <= 512
            return super().forward(
                query,
                key_cache,
                value_cache,
                cu_seqlens_q,
                block_tables,
                softmax_scale,
                mask,
                cu_total_seqlens=cu_total_seqlens,
                max_seqlen_q=max_seqlen_q,
                max_total_seqlen=max_total_seqlen,
            )

        if softmax_scale is None:
            softmax_scale = head_dim**-0.5

        compress_mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device=query.device), diagonal=1)
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key_cache,
            value=value_cache,
            atten_mask=compress_mask,
            block_table=block_tables,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=cu_seqlens_q[1:],
            actual_seq_lengths_kv=total_seq_lens,
            num_key_value_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale=softmax_scale,
            sparse_mode=3,
        )
        return out


class TorchNpuPagedDecodeGQA(MojoPagedDecodeGQA, default_priority=0):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "ABAB",
        window_size: int = -1,
    ):
        super().__init__(is_causal=is_causal, gqa_layout=gqa_layout, window_size=window_size)

    def forward(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        total_seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        input_layout: Optional[str] = None,
        cu_q_lens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        *,
        max_total_seq_len: Optional[int] = None,
    ) -> Tuple[Any]:
        batch_size, num_q_heads, head_dim = query.shape
        _, head_nums, block_size, _ = k_cache.shape
        assert_paged_decode_contract(block_tables, total_seq_lens)
        if head_dim % 128 != 0:
            raise NotImplementedError(f"NPU kernel npu_fused_infer_attention_score currently produces incorrect results for head_dim={head_dim} (not a multiple of 128)")

        if block_size % 128 != 0 or block_size > 512:
            return super().forward(
                query,
                k_cache,
                v_cache,
                total_seq_lens,
                block_tables,
                softmax_scale=softmax_scale,
                mask=mask,
                max_total_seq_len=max_total_seq_len,
            )

        if softmax_scale is None:
            softmax_scale = 1.0 / (head_dim**0.5)

        is_unsqueezed = False
        if input_layout is None:
            if query.dim() == 3:
                query = query.unsqueeze(2)
                input_layout = "BNSD"
                is_unsqueezed = True
            else:
                input_layout = "BNSD"

        actual_seq_lengths_q = torch.arange(1, batch_size + 1, dtype=torch.int32, device=query.device)
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query,
            k_cache,
            v_cache,
            input_layout=input_layout,
            block_table=block_tables,
            block_size=block_size,
            num_heads=num_q_heads,
            num_key_value_heads=head_nums,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=total_seq_lens,
            scale=softmax_scale,
        )

        if is_unsqueezed:
            out = out.squeeze(2)
        return out
