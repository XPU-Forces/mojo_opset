from typing import Any
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch_npu

from mojo_opset.core import MojoPagedDecodeGQA
from mojo_opset.core import MojoPagedPrefillGQA
from mojo_opset.core import MojoPrefillGQA


class TorchNpuPrefillGQA(MojoPrefillGQA, default_priority=0):
    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "ABAB",
        window_size: int = -1,
        op_name: str = "",
        layer_idx: int = 0,
    ):
        super().__init__(is_causal, gqa_layout, False, window_size, op_name, layer_idx)

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
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        _, num_q_heads, head_dim = query.shape
        _, num_kv_heads, block_size, _ = k_cache.shape

        if block_size % 128 != 0 or block_size > 512:
            # high performance attention kernel only supports block_size % 128 == 0 and block_size <= 512
            return super().forward(query, k_cache, v_cache, cu_seqlens_q, block_tables, softmax_scale)

        if softmax_scale is None:
            softmax_scale = head_dim**-0.5
        compress_mask = (
            torch.from_numpy(np.triu(np.ones((2048, 2048), dtype=np.float16), k=1) * -1).to(dtype=torch.bool).to("npu")
        )
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=k_cache,
            value=v_cache,
            atten_mask=compress_mask,
            block_table=block_tables.to(torch.int32),
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=cu_seqlens_q[1:],
            actual_seq_lengths_kv=cu_seqlens_q[1:] - cu_seqlens_q[:-1],
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

        if input_layout is None:
            input_layout = "BND" if query.dim() == 3 else "BSND"

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

        return out
