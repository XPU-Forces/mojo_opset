from typing import Any
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch_npu

from mojo_opset.core import MojoPagedDecodeGQA
from mojo_opset.core import MojoPagedPrefillGQA
from mojo_opset.core import MojoPrefillGQA
from mojo_opset.core import MojoFusedInferAttentionScore
from mojo_opset.core import MojoFusionAttention


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
        seqlens_kv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, num_q_heads, head_dim = query.shape
        _, num_kv_heads, block_size, _ = key_cache.shape

        if block_size % 128 != 0 or block_size > 512:
            # high performance attention kernel only supports block_size % 128 == 0 and block_size <= 512
            return super().forward(query, key_cache, value_cache, cu_seqlens_q, block_tables, softmax_scale, seqlens_kv, mask)

        if softmax_scale is None:
            softmax_scale = head_dim**-0.5

        compress_mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device=query.device), diagonal=1)
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key_cache,
            value=value_cache,
            atten_mask=compress_mask,
            block_table=block_tables.to(torch.int32),
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=cu_seqlens_q[1:],
            actual_seq_lengths_kv=seqlens_kv if seqlens_kv is not None else (cu_seqlens_q[1:] - cu_seqlens_q[:-1]),
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
        seqlens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        input_layout: Optional[str] = None,
        cu_seq_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[Any]:
        batch_size, num_q_heads, head_dim = query.shape
        _, head_nums, block_size, _ = k_cache.shape

        if block_size % 128 != 0 or block_size > 512:
            return super().forward(query, k_cache, v_cache, seqlens, block_tables, softmax_scale, cu_seq_lens)

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
        kv_seq_lens = cu_seq_lens if cu_seq_lens is not None else seqlens
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query,
            k_cache,
            v_cache,
            input_layout=input_layout,
            block_table=block_tables.to(torch.int32),
            block_size=block_size,
            num_heads=num_q_heads,
            num_key_value_heads=head_nums,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=kv_seq_lens,
            scale=softmax_scale,
        )

        if is_unsqueezed:
            out = out.squeeze(2)
        return out

class TorchNpuFusionAttention(MojoFusionAttention):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        query,
        key,
        value,
        actual_seq_qlen=None,
        actual_seq_kvlen=None,
        pre_tockens=65535,
        next_tockens=65536,
        sparse_mode=0,
        is_varlen=False,
        is_causal=False,
        input_layout: str = "BNSD",
        **kwargs,
    ):
        input_layout = "TND" if is_varlen else input_layout

        # Prepare causal mask if needed
        if is_causal and self.mask is None:
            if is_varlen:
                if actual_seq_qlen is None or actual_seq_kvlen is None:
                    raise ValueError("actual_seq_qlen/kvlen required for varlen mode")
                seq_q = actual_seq_qlen if isinstance(actual_seq_qlen, torch.Tensor) else torch.tensor(actual_seq_qlen)
                seq_kv = actual_seq_kvlen if isinstance(actual_seq_kvlen, torch.Tensor) else torch.tensor(actual_seq_kvlen)
                max_q = torch.diff(seq_q, prepend=torch.tensor([0])).max().item()
                max_kv = torch.diff(seq_kv, prepend=torch.tensor([0])).max().item()
            else:
                max_q, max_kv = query.shape[-2], key.shape[-2]
            self.mask = torch.triu(
                torch.ones(max_q, max_kv, dtype=torch.bool, device=query.device),
                diagonal=1,
            )

        # Convert to list for NPU API
        if isinstance(actual_seq_qlen, torch.Tensor):
            actual_seq_qlen = actual_seq_qlen.tolist()
        if isinstance(actual_seq_kvlen, torch.Tensor):
            actual_seq_kvlen = actual_seq_kvlen.tolist()

        # Forward
        attn_out = \
            torch_npu.npu_fusion_attention(
                query=query, 
                key=key, 
                value=value, 
                head_num=self.head_num,
                input_layout=input_layout,
                pre_tockens=pre_tockens,
                next_tockens=next_tockens,
                sparse_mode=sparse_mode,
                atten_mask=self.mask,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                scale=self.scale,
                **kwargs,
            )[0]

        return attn_out


class TorchNpuFusedInferAttentionScore(MojoFusedInferAttentionScore):
    supported_platforms_list = ["npu"]

    def forward(
        self,
        query,
        key,
        value,
        actual_seq_lengths=None,
        actual_seq_lengths_kv=None,
        num_kv_heads=0,
        block_table=None,
        sparse_mode=0,
        is_varlen=False,
        input_layout: str = "BNSD",
        block_size: int = 0,
        **kwargs,   
    ):
        input_layout = "TND" if is_varlen else input_layout
        
        # assert query.dtype == key.dtype == value.dtype == torch.float16, \
        #      "query, key, value must have the same dtype, \
        #     got {} {} {}".format(query.dtype, key.dtype, value.dtype)
        
        attn_out = \
            torch_npu.npu_fused_infer_attention_score(
                query,
                key,
                value,
                atten_mask=self.mask,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                block_table=block_table,
                num_heads=self.head_num,
                scale=self.scale,
                input_layout=input_layout,
                num_key_value_heads=num_kv_heads,
                block_size=block_size,
            )[0]
        return attn_out