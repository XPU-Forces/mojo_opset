from __future__ import annotations

import math

from typing import Any, Optional, Tuple

import torch

from ixformer import functions as ixf_f
from ixformer.contrib import vllm_flash_attn as ix_fa

from mojo_opset.core import MojoPagedPrefillGQA
from mojo_opset.core import MojoPagedDecodeGQA
from mojo_opset.core.operators.attention import assert_paged_decode_contract
from mojo_opset.core.operators.attention import assert_paged_prefill_contract

_IXFORMER_PAGED_DECODE_SUPPORTED_HEAD_DIMS = frozenset(
    {32, 64, 80, 96, 128, 160, 192, 224, 256}
)


class IxformerPagedPrefillGQA(MojoPagedPrefillGQA):
    """Ixformer implementation for paged prefill GQA."""

    supported_platforms_list = ["ilu"]

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
        max_q_lens: Optional[int] = None,
        max_total_seq_lens: Optional[int] = None,
    ) -> Tuple[Any]:
        """
        Paged prefill attention with grouped query heads (GQA) using a blocked KV cache.

        Args:
            query (torch.Tensor): Query tokens of shape (T, Hq, D).
                Hq->q head num , D->head dim.dtype must be fp16 or bf16.
            key_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D). 
                HKv-> kv head num , D->head dim.dtype must be fp16 or bf16.
                block_size must be <= 256 and % 16 == 0.
            value_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D). 
                HKv-> kv head num , D->head dim.dtype must be fp16 or bf16.
                block_size must be <= 256 and % 16 == 0.
            cu_q_lens (torch.Tensor): Cumulative query lengths, shape (B+1,);
                `cu_q_lens[i]` is the start offset for query at batch i; `cu_q_lens[-1] == T`.
                dtype must be int32.
            cu_total_seq_lens (torch.Tensor): Cumulative total KV lengths, shape (B+1,);
                `cu_total_seq_lens[i]` is the start offset for batch i; `cu_total_seq_lens[-1] == K`.
                dtype must be int32.
            max_q_lens (int): Maximum query sequence length across the batch. >0 and must be int.
            max_total_seq_lens (int): Maximum total visible key/value sequence length across the batch. >0 and must be int.
            block_tables (torch.Tensor): Logical-to-physical block IDs per batch,
                shape (B, num_blocks).
                dtype must be int32.
            softmax_scale (Optional[float]): Attention scaling factor; defaults to 1/sqrt(D).
            mask (Optional[torch.Tensor]): Attention mask; defaults to None.
                Not support custom mask yet.
        Returns:
            torch.Tensor: Attention output of shape (T, Hq, D).

        Notes:
            - If Hq != Hkv, expands K/V heads to match Hq via repeat_interleave.
            - Applies a causal lower-triangular mask and restricts attention within each sequence.
            - Softmax is computed in float32 and cast back to the input dtype.
            - Despite the type annotation Tuple[Any], this implementation returns a single tensor.
        """

        assert_paged_prefill_contract(cu_q_lens, block_tables, cu_total_seq_lens)
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(f"query dtype must be fp16 or bf16, got {query.dtype}.")
        if mask is not None:
            raise NotImplementedError("IxformerPagedPrefillGQA does not support custom mask yet.")
        if self.gqa_layout != "AABB":
            raise NotImplementedError("IxformerPagedPrefillGQA only supports AABB layout.")
        page_block_size = key_cache.shape[2]
        if page_block_size > 256 or page_block_size % 16 != 0:
            raise NotImplementedError(
                f"IxformerPagedPrefillGQA only supports page_block_size <= 256 and % 16 == 0, got {page_block_size}."
            )

        if cu_q_lens.dtype != torch.int32:
            raise ValueError(f"cu_q_lens must be int32, got {cu_q_lens.dtype}.")
        if block_tables.dtype != torch.int32:
            raise ValueError(f"block_tables must be int32, got {block_tables.dtype}.")

        q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
        if q_lens.dtype != torch.int32:
            raise ValueError(f"q_lens must be int32, got {q_lens.dtype}.")
        if (q_lens <= 0).any():
            raise NotImplementedError(
                "IxformerPagedPrefillGQA does not support zero/negative q_lens. "
                "Please provide cu_q_lens such that all sequence lengths are >= 1."
            )
        if not isinstance(cu_total_seq_lens, torch.Tensor):
            raise ValueError(f"cu_total_seq_lens must be torch.Tensor, got {type(cu_total_seq_lens)}.")
        if cu_total_seq_lens.dtype != torch.int32:
            raise ValueError(f"cu_total_seq_lens must be int32, got {cu_total_seq_lens.dtype}.")
        if cu_total_seq_lens.ndim != 1:
            raise ValueError(f"cu_total_seq_lens must be 1D, got shape {tuple(cu_total_seq_lens.shape)}.")
        if not isinstance(max_q_lens, int) or max_q_lens <= 0:
            raise ValueError(
                f"max_q_lens must be a positive int, got {max_q_lens} ({type(max_q_lens)})."
            )
        if not isinstance(max_total_seq_lens, int) or max_total_seq_lens <= 0:
            raise ValueError(
                f"max_total_seq_lens must be a positive int, got {max_total_seq_lens} ({type(max_total_seq_lens)})."
            )

        if softmax_scale is None:
            head_dim = query.shape[-1]
            softmax_scale = 1.0 / math.sqrt(head_dim)

        return ix_fa.flash_attn_varlen_func(
            query,
            key_cache,
            value_cache,
            cu_q_lens,
            cu_total_seq_lens,
            max_q_lens,
            max_total_seq_lens,
            causal=self.is_causal,
            softmax_scale=softmax_scale,
            block_table=block_tables,
        )

class IxformerPagedDecodeGQA(MojoPagedDecodeGQA):
    """Ixformer implementation for paged decode GQA."""

    supported_platforms_list = ["ilu"]

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        total_seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_q_lens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        *,
        max_total_seq_len: Optional[int] = None,
    ):
        """
        Paged decode attention with grouped query heads (GQA) using a blocked KV cache.

        Args:
            query (torch.Tensor): Query of shape (B, Hq, D).
            key_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D).
            value_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D).
            total_seq_lens (torch.Tensor): Total visible sequence lengths (B).
            block_tables (torch.Tensor): (B, num_blocks) mapping logical blocks to physical IDs.
            max_total_seq_len (int): Max sequence length, should be equal to total_seq_lens.max().
            softmax_scale (Optional[float]): Scale factor; defaults to 1/sqrt(D).

        Returns:
            torch.Tensor: Attention output of shape (B, Hq, D).

        Notes:
            - Head dim D only support [32, 64, 80, 96, 128, 160, 192, 224, 256].
            - Block size must be a multiple of 16.
            - If Hq > Hkv, K/V heads are repeated to match query heads.
            - Softmax is computed in float32 and cast back to the input dtype.
            - This implementation references variables `query` and `total_seq_lens`; ensure they
              correspond to `query` and the sequence-lengths tensor in the caller.
        """
        assert_paged_decode_contract(block_tables, total_seq_lens)

        if mask is not None:
            raise NotImplementedError("IxformerPagedDecodeGQA does not support custom mask yet.")

        if self.gqa_layout == "ABAB":
            raise NotImplementedError("IxformerPagedDecodeGQA does not support ABAB layout.")

        batch_size, num_q_heads, head_dim_q = query.shape
        _, num_kv_heads, block_size, head_dim_kv = key_cache.shape

        if head_dim_q != head_dim_kv:
            raise ValueError(
                f"query head_dim ({head_dim_q}) must match key_cache head_dim ({head_dim_kv})."
            )
        head_dim = head_dim_q

        if head_dim not in _IXFORMER_PAGED_DECODE_SUPPORTED_HEAD_DIMS:
            raise NotImplementedError(
                "IxformerPagedDecodeGQA only supports head_dim in "
                f"{sorted(_IXFORMER_PAGED_DECODE_SUPPORTED_HEAD_DIMS)}, got {head_dim}."
            )

        if query.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                f"IxformerPagedDecodeGQA only supports fp16/bf16 query, got {query.dtype}."
            )

        if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
            raise NotImplementedError(
                "IxformerPagedDecodeGQA requires key_cache/value_cache dtype to match query dtype."
            )

        if not self.is_causal:
            raise NotImplementedError("IxformerPagedDecodeGQA only supports causal attention.")

        # vllm_paged_attention expects a positive KV length for every batch row.
        if bool((total_seq_lens < 1).any().item()):
            raise NotImplementedError(
                "IxformerPagedDecodeGQA requires total_seq_lens >= 1 for every batch row."
            )

        # GQA + small page size is not supported by the fused kernel vs. eager reference parity.
        if num_q_heads != num_kv_heads and block_size < 128:
            raise NotImplementedError(
                "IxformerPagedDecodeGQA does not support GQA (Hq != Hkv) when block_size < 128."
            )

        output = torch.empty(batch_size, num_q_heads, head_dim, dtype=query.dtype, device=query.device)

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        if block_size % 16 != 0:
            raise NotImplementedError("Block size must be a multiple of 16.")

        if block_tables.dtype != torch.int32:
            raise NotImplementedError("IxformerPagedDecodeGQA only support block_tables dtype = torch.int32.")

        if cu_q_lens is not None:
            raise NotImplementedError("IxformerPagedDecodeGQA does not support varlen decode.")
        if total_seq_lens.dtype != torch.int32:
            raise ValueError(f"total_seq_lens must be int32, got {total_seq_lens.dtype}.")
        if not isinstance(max_total_seq_len, int) or max_total_seq_len <= 0:
            raise ValueError(f"max_total_seq_len must be > 0 and int, got {max_total_seq_len}.")
        if (total_seq_lens <= 0).any():
            raise NotImplementedError("IxformerPagedDecodeGQA does not support zero/negative total_seq_lens.")
        if max_total_seq_len <= 1:
            raise NotImplementedError("IxformerPagedDecodeGQA only supports max_total_seq_len > 1.")

        ixf_f.vllm_paged_attention(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            scale=softmax_scale,
            block_tables=block_tables,
            context_lens=total_seq_lens,
            block_size=block_size,
            max_context_len=max_total_seq_len,
            causal=self.is_causal,
        )

        return output
