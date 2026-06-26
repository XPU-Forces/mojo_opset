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
    ):
        super().__init__(is_causal=is_causal, gqa_layout=gqa_layout)

    def forward(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_q_lens: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        
        batch_size, num_q_heads, seq_len, head_dim = query.shape
        _, num_kv_heads, block_size, _ = k_cache.shape


        if head_dim % 128 != 0:
            raise NotImplementedError(f"NPU kernel requires head_dim % 128 == 0, got {query.shape[-1]}")

        if block_size % 128 != 0 or block_size > 512:
            # high performance attention kernel only supports block_size % 128 == 0 and block_size <= 512
            return super().forward(query, k_cache, v_cache, cu_q_lens, softmax_scale)

        if softmax_scale is None:
            softmax_scale = head_dim**-0.5
        atten_mask = torch.triu(torch.ones([seq_len, seq_len], dtype=torch.bool, device=query.device), diagonal=1)
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=k_cache,
            value=v_cache,
            actual_seq_lengths=cu_q_lens,
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
    _ATTN_MASK_SIZE = 2048

    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "ABAB",
    ):
        super().__init__(is_causal=is_causal, gqa_layout=gqa_layout)
        self._attn_mask: Optional[torch.Tensor] = None
        self._attn_mask_device: Optional[torch.device] = None

    def _get_attn_mask(self, device: torch.device) -> torch.Tensor:
        if self._attn_mask is None or self._attn_mask_device != device:
            self._attn_mask = torch.triu(
                torch.ones(
                    (self._ATTN_MASK_SIZE, self._ATTN_MASK_SIZE),
                    dtype=torch.bool,
                    device=device,
                ),
                diagonal=1,
            )
            self._attn_mask_device = device
        return self._attn_mask

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_q_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_total_seq_lens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
        max_total_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        assert_paged_prefill_contract(cu_q_lens, block_tables, cu_total_seq_lens)
        _, num_q_heads, head_dim = query.shape
        _, num_kv_heads, block_size, _ = key_cache.shape

        if head_dim % 128 != 0:
            raise NotImplementedError(
                f"NPU kernel requires head_dim % 128 == 0, got {head_dim}"
            )

        if (
            mask is not None
            or not self.is_causal
            or block_size % 128 != 0
            or block_size > 512
        ):
            return super().forward(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                cu_q_lens=cu_q_lens,
                block_tables=block_tables,
                softmax_scale=softmax_scale,
                cu_total_seq_lens=cu_total_seq_lens,
                mask=mask,
                max_q_len=max_q_len,
                max_total_seq_len=max_total_seq_len,
            )

        if softmax_scale is None:
            softmax_scale = head_dim**-0.5

        actual_seq_qlen = cu_q_lens[1:] - cu_q_lens[:-1]
        if cu_total_seq_lens is not None:
            actual_seq_kvlen = cu_total_seq_lens[1:] - cu_total_seq_lens[:-1]
        else:
            actual_seq_kvlen = actual_seq_qlen

        out, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query,
            key_cache,
            value_cache,
            atten_mask=self._get_attn_mask(query.device),
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            num_query_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            block_size=block_size,
            block_table=block_tables,
            softmax_scale=softmax_scale,
            input_layout="TND",
            sparse_mode=3,
            query_dtype=query.dtype,
            key_dtype=key_cache.dtype,
            value_dtype=value_cache.dtype,
        )
        return out


class TorchNpuPagedDecodeGQA(MojoPagedDecodeGQA, default_priority=0):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "ABAB",
    ):
        super().__init__(is_causal=is_causal, gqa_layout=gqa_layout)

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
