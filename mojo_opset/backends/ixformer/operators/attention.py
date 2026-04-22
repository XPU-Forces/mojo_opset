from __future__ import annotations

import math

from typing import Optional

import torch

from ixformer import functions as ixf_f
from ixformer.contrib import vllm_flash_attn as ix_fa

from mojo_opset.core import MojoPagedPrefillGQA
from mojo_opset.core import MojoPagedDecodeGQA

_IXFORMER_PAGED_DECODE_SUPPORTED_HEAD_DIMS = frozenset(
    {32, 64, 80, 96, 128, 160, 192, 224, 256}
)


class IxformerPagedPrefillGQA(MojoPagedPrefillGQA):
    """Ixformer implementation for paged prefill GQA."""

    supported_platforms_list = ["ilu"]

    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "AABB",
        window_size: int = -1,
    ):
        super().__init__(is_causal=is_causal, gqa_layout=gqa_layout, window_size=window_size)

        if self.window_size != -1:
            raise NotImplementedError("IxformerPagedPrefillGQA only supports window_size == -1")

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
        *,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ) -> torch.Tensor:
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
            cu_seqlens_q (torch.Tensor): Cumulative query lengths, shape (B+1,);
                `cu_seqlens_q[i]` is the start offset for query at batch i; `cu_seqlens_q[-1] == T`.
                dtype must be int32.
            cu_seqlens_kv (torch.Tensor): Cumulative key/value lengths, shape (B+1,);
                `cu_seqlens_kv[i]` is the start offset for key/value at batch i; `cu_seqlens_kv[-1] == K`.
                dtype must be int32.
            max_seqlen_q (int): Maximum query sequence length across the batch. >0 and must be int.
            max_seqlen_k (int): Maximum key/value sequence length across the batch. >0 and must be int.
            block_tables (torch.Tensor): Logical-to-physical block IDs per batch,
                shape (B, num_blocks).
                dtype must be int32.
            softmax_scale (Optional[float]): Attention scaling factor; defaults to 1/sqrt(D).
            seqlens_kv (Optional[torch.Tensor]): key/value lengths, shape (B,). Used to derive
                cu_seqlens_kv when the cumulative lengths are not provided.
            mask (Optional[torch.Tensor]): Attention mask; defaults to None.
                we do not support custom mask yet.
        Returns:
            torch.Tensor: Attention output of shape (T, Hq, D).

        Notes:
            - If Hq != Hkv, expands K/V heads to match Hq via repeat_interleave.
            - Applies a causal lower-triangular mask and restricts attention within each sequence.
            - Softmax is computed in float32 and cast back to the input dtype.
            - Despite the type annotation Tuple[Any], this implementation returns a single tensor.
        """

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

        if cu_seqlens_q.dtype != torch.int32:
            raise ValueError(f"cu_seqlens_q must be int32, got {cu_seqlens_q.dtype}.")
        if block_tables.dtype != torch.int32:
            raise ValueError(f"block_tables must be int32, got {block_tables.dtype}.")

        q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        if cu_seqlens_kv is None:
            kv_lens = seqlens_kv if seqlens_kv is not None else q_lens
            if kv_lens.dtype != torch.int32:
                raise ValueError(f"seqlens_kv must be int32, got {kv_lens.dtype}.")
            cu_seqlens_kv = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=kv_lens.device),
                    torch.cumsum(kv_lens, dim=0).to(torch.int32),
                ]
            )
        if cu_seqlens_kv.dtype != torch.int32:
            raise ValueError(f"cu_seqlens_kv must be int32, got {cu_seqlens_kv.dtype}.")
        if max_seqlen_q is None:
            max_seqlen_q = int(q_lens.max().item())
        if max_seqlen_k is None:
            max_seqlen_k = int((cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).max().item())
        if max_seqlen_q <= 0 or not isinstance(max_seqlen_q, int):
            raise ValueError(f"max_seqlen_q must be >0 and int, got {max_seqlen_q}.")
        if max_seqlen_k <= 0 or not isinstance(max_seqlen_k, int):
            raise ValueError(f"max_seqlen_k must be >0 and int, got {max_seqlen_k}.")

        return ix_fa.flash_attn_varlen_func(
            query,
            key_cache,
            value_cache,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_k,
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
        seqlens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_seq_lens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        *,
        max_context_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Paged decode attention with grouped query heads (GQA) using a blocked KV cache.

        Args:
            query (torch.Tensor): Query of shape (B, Hq, D).
            key_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D).
            value_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D).
            seqlens (torch.Tensor): Sequence lengths (B).
            block_tables (torch.Tensor): (B, num_blocks) mapping logical blocks to physical IDs.
            max_context_len (int): Max sequence length, should be equal to seqlens.max().
            softmax_scale (Optional[float]): Scale factor; defaults to 1/sqrt(D).

        Returns:
            torch.Tensor: Attention output of shape (B, Hq, D).

        Notes:
            - Head dim D only support [32, 64, 80, 96, 128, 160, 192, 224, 256].
            - Block size must be a multiple of 16.
            - If Hq > Hkv, K/V heads are repeated to match query heads.
            - Softmax is computed in float32 and cast back to the input dtype.
            - This implementation references variables `query` and `seqlens`; ensure they
              correspond to `query` and the sequence-lengths tensor in the caller.
        """

        if mask is not None:
            raise NotImplementedError("IxformerPagedDecodeGQA does not support custom mask yet.")

        if self.gqa_layout == "ABAB":
            raise NotImplementedError("IxformerPagedDecodeGQA does not support ABAB layout.")

        if self.window_size != -1:
            raise NotImplementedError(
                "IxformerPagedDecodeGQA only supports window_size == -1 (no sliding window)."
            )

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
        if bool((seqlens < 1).any().item()):
            raise NotImplementedError(
                "IxformerPagedDecodeGQA requires context_lens >= 1 for every batch row."
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

        if cu_seq_lens is not None:
            raise NotImplementedError("IxformerPagedDecodeGQA does not support varlen decode.")
        if seqlens.dtype != torch.int32:
            raise ValueError(f"seqlens must be int32, got {seqlens.dtype}.")
        if max_context_len is None:
            max_context_len = int(seqlens.max().item())
        if max_context_len <= 1:
            raise NotImplementedError("IxformerPagedDecodeGQA only support max_context_len > 1.")

        ixf_f.vllm_paged_attention(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            scale=softmax_scale,
            block_tables=block_tables,
            context_lens=seqlens,
            block_size=block_size,
            max_context_len=max_context_len,
            causal=self.is_causal,
            window_left=self.window_size,
        )

        return output
