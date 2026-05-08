from __future__ import annotations

import math

from typing import Any, Optional, Tuple

import torch

from ixformer import functions as ixf_f
from ixformer.contrib import vllm_flash_attn as ix_fa

from mojo_opset.core import MojoPagedDecodeGQA
from mojo_opset.core import MojoPagedDecodeSWA
from mojo_opset.core import MojoPagedPrefillGQA
from mojo_opset.core import MojoPagedPrefillSWA
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
        cu_kv_lens = cu_total_seq_lens if cu_total_seq_lens is not None else cu_q_lens
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

        if not isinstance(cu_kv_lens, torch.Tensor):
            raise ValueError(f"cu_total_seq_lens must be torch.Tensor, got {type(cu_total_seq_lens)}.")
        if cu_kv_lens.dtype != torch.int32:
            raise ValueError(f"cu_total_seq_lens must be int32, got {cu_total_seq_lens.dtype}.")
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
            cu_kv_lens,
            max_q_lens,
            max_total_seq_lens,
            causal=self.is_causal,
            softmax_scale=softmax_scale,
            block_table=block_tables,
        )

class IxformerPagedPrefillSWA(MojoPagedPrefillSWA):
    """Ixformer implementation for paged prefill SWA (Sliding Window Attention)."""

    supported_platforms_list = ["ilu"]

    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "AABB",
        global_window_size: Optional[int] = None,
        local_window_size: Optional[int] = None,
    ):
        
        super().__init__(
            is_causal=is_causal,
            gqa_layout=gqa_layout,
            global_window_size=global_window_size,
            local_window_size=local_window_size,
        )
        
        if global_window_size is not None and global_window_size < 0:
            raise ValueError(f"global_window_size must be >= 0, got {global_window_size}.")
        if local_window_size is not None and local_window_size < 0:
            raise ValueError(f"local_window_size must be >= 0, got {local_window_size}.")
        

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_q_lens: torch.Tensor,
        block_table: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_total_seq_lens: Optional[torch.Tensor] = None,
        *,
        max_q_lens: Optional[int] = None,
        max_total_seq_lens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Paged prefill attention with sliding window (SWA) using a blocked KV cache.

        Combines local sliding window attention with optional global window attention.
        When local_window_size is set, each query token only attends to the nearest
        local_window_size key/value tokens (causal direction). When global_window_size
        is also set, the first global_window_size tokens of each sequence are always
        visible to all query positions, regardless of the sliding window.

        The attention mask (when is_causal=True) is constructed as:
            mask[q_pos, kv_pos] = causal_mask AND (local_window OR global_window)
        where:
            - causal_mask: kv_pos <= q_pos + (kv_seq_len - q_seq_len)
            - local_window: q_pos + (kv_seq_len - q_seq_len) - kv_pos < local_window_size
            - global_window: kv_pos < global_window_size

        Args:
            q (torch.Tensor): Query tokens of shape (T, Hq, D).
                Hq -> query head count, D -> head dimension.
                dtype must be fp16 or bf16.
            k_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D).
                Hkv -> kv head count, D -> head dimension.
                dtype must be fp16 or bf16.
                block_size must be <= 256 and % 16 == 0.
            v_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D).
                Hkv -> kv head count, D -> head dimension.
                dtype must be fp16 or bf16.
                block_size must be <= 256 and % 16 == 0.
            cu_q_lens (torch.Tensor): Cumulative query lengths, shape (B+1,);
                `cu_q_lens[i]` is the start offset for query at batch i;
                `cu_q_lens[-1] == T`.
                dtype must be int32.
            block_table (torch.Tensor): Logical-to-physical block IDs per batch,
                shape (B, num_blocks). Maps logical block indices to physical block
                locations in k_cache / v_cache.
                dtype must be int32.
            softmax_scale (Optional[float]): Attention scaling factor;
                defaults to 1/sqrt(D).
            cu_total_seq_lens (Optional[torch.Tensor]): Cumulative total visible KV lengths, shape (B+1,).
                If None, defaults to cu_q_lens (same as the reference MojoPagedPrefillSWA).
            max_q_lens (Optional[int]): Ixformer kernel hint; if None, derived from cu_q_lens.
            max_total_seq_lens (Optional[int]): Ixformer kernel hint; if None, derived from cu_kv.

        Returns:
            torch.Tensor: Attention output of shape (T, Hq, D).

        Notes:
            - Only AABB GQA layout is supported.
            - If Hq != Hkv, K/V heads are expanded to match Hq via repeat_interleave.
            - local_window_size and global_window_size are configured via __init__,
              not passed to forward directly.
            - window_size is derived as (local_window_size, 0) when local_window_size
              is set, or (-1, -1) for full attention.
            - global_window_size is not supported together with alibi_slopes in the
              underlying kernel.
        """

        if q.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(f"query dtype must be fp16 or bf16, got {q.dtype}.")
        if self.gqa_interleave:
            raise NotImplementedError("IxformerPagedPrefillSWA only supports AABB layout.")
        page_block_size = k_cache.shape[2]
        if page_block_size > 256 or page_block_size % 16 != 0:
            raise NotImplementedError(
                f"IxformerPagedPrefillSWA only supports page_block_size <= 256 and % 16 == 0, got {page_block_size}."
            )

        assert_paged_prefill_contract(cu_q_lens, block_table, cu_total_seq_lens)
        cu_kv_lens = cu_total_seq_lens if cu_total_seq_lens is not None else cu_q_lens
        if cu_q_lens.dtype != torch.int32:
            raise ValueError(f"cu_q_lens must be int32, got {cu_q_lens.dtype}.")
        if block_table.dtype != torch.int32:
            raise ValueError(f"block_table must be int32, got {block_table.dtype}.")

        if not isinstance(cu_kv_lens, torch.Tensor):
            raise ValueError(
                f"cu_total_seq_lens must be torch.Tensor, got {type(cu_total_seq_lens)}."
            )
        if cu_kv_lens.dtype != torch.int32:
            raise ValueError(
                f"cu_total_seq_lens must be int32, got {cu_kv_lens.dtype}."
            )
            
        if not isinstance(max_q_lens, int) or max_q_lens <= 0:
            raise ValueError(
                f"max_q_lens must be a positive int, got {max_q_lens} ({type(max_q_lens)})."
            )
        if not isinstance(max_total_seq_lens, int) or max_total_seq_lens <= 0:
            raise ValueError(
                f"max_total_seq_lens must be a positive int, got {max_total_seq_lens} ({type(max_total_seq_lens)})."
            )

        window_size = (self.local_window_size, 0) if self.local_window_size is not None else (-1, -1)

        if window_size == (-1, -1) and self.global_window_size is not None and self.global_window_size > 0:
            raise ValueError(
                "global_window_size > 0 requires a valid local_window_size (sliding window)."
            )

        return ix_fa.flash_attn_varlen_func(
            q,
            k_cache,
            v_cache,
            cu_q_lens,
            cu_kv_lens,
            max_q_lens,
            max_total_seq_lens,
            causal=self.is_causal,
            softmax_scale=softmax_scale,
            block_table=block_table,
            window_size=window_size,
            global_window_size=self.global_window_size,
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
            - Hq: the number of query head; Hkv: the number of kv head.
            - If Hq > Hkv, K/V heads are repeated to match query heads.
            - Softmax is computed in float32 and cast back to the input dtype.
            - This implementation references variables `query` and `total_seq_lens`; ensure they
              correspond to `query` and the sequence-lengths tensor in the caller.
        """
        assert_paged_decode_contract(block_tables, total_seq_lens)

        if bool((total_seq_lens <= 0).any().item()):
            raise NotImplementedError(
                "IxformerPagedDecodeGQA does not support batch rows with zero KV length (PADSEQ)."
            )

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

        output = torch.empty(batch_size, num_q_heads, head_dim, dtype=query.dtype, device=query.device)

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        if block_size % 16 != 0:
            raise NotImplementedError("Block size must be a multiple of 16.")

        if block_tables.dtype != torch.int32:
            raise NotImplementedError("IxformerPagedDecodeGQA only support block_tables dtype = torch.int32.")

        if total_seq_lens.dtype != torch.int32:
            raise NotImplementedError(f"total_seq_lens must be int32, got {total_seq_lens.dtype}.")
        if not isinstance(max_total_seq_len, int) or max_total_seq_len <= 0:
            raise NotImplementedError(f"max_total_seq_len must be > 0 and int, got {max_total_seq_len}.")

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


class IxformerPagedDecodeSWA(MojoPagedDecodeSWA):
    """Ixformer implementation for paged decode SWA."""

    supported_platforms_list = ["ilu"]

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        total_seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        softmax_scale: Optional[float] = None,
        *,
        max_total_seq_len: Optional[int] = None,
    ) -> torch.Tensor:

        """
        Paged decode attention with sliding window (SWA) using a blocked KV cache.

        Args:
            q (torch.Tensor): Query of shape (B, Hq, D).
            k_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D).
            v_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D).
            total_seq_lens (torch.Tensor): Total visible sequence lengths (B).
            block_table (torch.Tensor): (B, num_blocks) mapping logical blocks to physical IDs.
            max_total_seq_len (int): Max sequence length, should be equal to total_seq_lens.max().
            softmax_scale (Optional[float]): Scale factor; defaults to 1/sqrt(D).

        Returns:
            torch.Tensor: Attention output of shape (B, Hq, D).

        Notes:
            - Head dim D only support [32, 64, 80, 96, 128, 160, 192, 224, 256].
            - Block size must be a multiple of 16.
            - Hq: the number of query head; Hkv: the number of kv head.
            - If Hq > Hkv, K/V heads are repeated to match query heads.
            - Softmax is computed in float32 and cast back to the input dtype.
            - This implementation references variables `q` and `total_seq_lens`; ensure they
              correspond to `q` and the total-sequence-lengths tensor in the caller.
        """
        assert_paged_decode_contract(block_table, total_seq_lens)

        if bool((total_seq_lens <= 0).any().item()):
            raise NotImplementedError(
                "IxformerPagedDecodeSWA does not support batch rows with zero KV length (PADSEQ)."
            )

        if self.gqa_interleave:
            raise NotImplementedError("IxformerPagedDecodeSWA does not support ABAB layout.")

        batch_size, num_q_heads, head_dim = q.shape
        _, num_kv_heads, block_size, _ = k_cache.shape

        output = torch.empty(batch_size, num_q_heads, head_dim, dtype=q.dtype, device=q.device)

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        if block_table.dtype != torch.int32:
            raise NotImplementedError("IxformerPagedDecodeSWA only support block_table dtype = torch.int32.")

        if not isinstance(max_total_seq_len, int) or max_total_seq_len <= 0:
            raise NotImplementedError(
                f"max_total_seq_len must be > 0 and int, got {max_total_seq_len}."
            )

        ixf_f.vllm_paged_attention(
            output=output,
            query=q,
            key_cache=k_cache,
            value_cache=v_cache,
            num_kv_heads=num_kv_heads,
            scale=softmax_scale,
            block_tables=block_table,
            context_lens=total_seq_lens,
            block_size=block_size,
            max_context_len=max_total_seq_len,
            causal=self.is_causal,
            window_left=self.local_window_size,
            global_window_size=self.global_window_size
        )

        return output
