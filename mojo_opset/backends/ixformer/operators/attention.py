from __future__ import annotations

from typing import Optional

import math

import torch

from ixformer import functions as ixf_f
from ixformer.contrib import vllm_flash_attn as ix_fa

from mojo_opset.core import MojoPagedPrefillGQA
from mojo_opset.core import MojoPagedPrefillSWA
from mojo_opset.core import MojoPagedDecodeGQA
from mojo_opset.core import MojoPagedDecodeSWA

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
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
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
            seqlens_kv (Optional[torch.Tensor]): key/value lengths, shape (B,);
                `seqlens_kv[i]` is the length for key/value in key/value cache at batch i.
                If None, defaults to `cu_seqlens_q[i+1] - cu_seqlens_q[i]` for each batch i.
                we do not use this parameter yet.we use cu_seqlens_kv instead.
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
        if cu_seqlens_kv.dtype != torch.int32:
            raise ValueError(f"cu_seqlens_kv must be int32, got {cu_seqlens_kv.dtype}.")
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
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        block_table: torch.Tensor,
        softmax_scale: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
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
            query (torch.Tensor): Query tokens of shape (T, Hq, D).
                Hq -> query head count, D -> head dimension.
                dtype must be fp16 or bf16.
            key_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D).
                Hkv -> kv head count, D -> head dimension.
                dtype must be fp16 or bf16.
                block_size must be <= 256 and % 16 == 0.
            value_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D).
                Hkv -> kv head count, D -> head dimension.
                dtype must be fp16 or bf16.
                block_size must be <= 256 and % 16 == 0.
            cu_seqlens_q (torch.Tensor): Cumulative query lengths, shape (B+1,);
                `cu_seqlens_q[i]` is the start offset for query at batch i;
                `cu_seqlens_q[-1] == T`.
                dtype must be int32.
            cu_seqlens_kv (torch.Tensor): Cumulative key/value lengths, shape (B+1,);
                `cu_seqlens_kv[i]` is the start offset for key/value at batch i;
                `cu_seqlens_kv[-1] == K`.
                dtype must be int32.
            max_seqlen_q (int): Maximum query sequence length across the batch.
                Must be > 0 and int.
            max_seqlen_k (int): Maximum key/value sequence length across the batch.
                Must be > 0 and int.
            block_table (torch.Tensor): Logical-to-physical block IDs per batch,
                shape (B, num_blocks). Maps logical block indices to physical block
                locations in key_cache / value_cache.
                dtype must be int32.
            softmax_scale (Optional[float]): Attention scaling factor;
                defaults to 1/sqrt(D).
            mask (Optional[torch.Tensor]): Attention mask; defaults to None.
                Custom mask is not supported; must be None.

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

        if query.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(f"query dtype must be fp16 or bf16, got {query.dtype}.")
        if mask is not None:
            raise NotImplementedError("IxformerPagedPrefillSWA does not support custom mask.")
        if self.gqa_interleave:
            raise NotImplementedError("IxformerPagedPrefillSWA only supports AABB layout.")
        page_block_size = key_cache.shape[2]
        if page_block_size > 256 or page_block_size % 16 != 0:
            raise NotImplementedError(
                f"IxformerPagedPrefillSWA only supports page_block_size <= 256 and % 16 == 0, got {page_block_size}."
            )

        if cu_seqlens_q.dtype != torch.int32:
            raise ValueError(f"cu_seqlens_q must be int32, got {cu_seqlens_q.dtype}.")
        if cu_seqlens_kv.dtype != torch.int32:
            raise ValueError(f"cu_seqlens_kv must be int32, got {cu_seqlens_kv.dtype}.")
        if block_table.dtype != torch.int32:
            raise ValueError(f"block_table must be int32, got {block_table.dtype}.")
        if max_seqlen_q <= 0 or not isinstance(max_seqlen_q, int):
            raise ValueError(f"max_seqlen_q must be >0 and int, got {max_seqlen_q}.")
        if max_seqlen_k <= 0 or not isinstance(max_seqlen_k, int):
            raise ValueError(f"max_seqlen_k must be >0 and int, got {max_seqlen_k}.")

        window_size = (self.local_window_size, 0) if self.local_window_size is not None else (-1, -1)

        if window_size == (-1, -1) and self.global_window_size is not None and self.global_window_size > 0:
            raise NotImplementedError(
                "global_window_size > 0 requires a valid local_window_size (sliding window)."
            )

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
        seqlens: torch.Tensor,
        block_tables: torch.Tensor,
        max_context_len: int,
        softmax_scale: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
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
            - Hq: the number of query head; Hkv: the number of kv head.
            - If Hq > Hkv, K/V heads are repeated to match query heads.
            - Softmax is computed in float32 and cast back to the input dtype.
            - This implementation references variables `query` and `seqlens`; ensure they
              correspond to `query` and the sequence-lengths tensor in the caller.
        """

        if mask is not None:
            raise NotImplementedError("IxformerPagedDecodeGQA does not support custom mask yet.")

        if self.gqa_layout == "ABAB":
            raise NotImplementedError("IxformerPagedDecodeGQA does not support ABAB layout.")

        batch_size, num_q_heads, head_dim = query.shape
        _, num_kv_heads, block_size, head_dim = key_cache.shape

        output = torch.empty(batch_size, num_q_heads, head_dim, dtype=query.dtype, device=query.device)

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        if block_size % 16 != 0:
            raise NotImplementedError("Block size must be a multiple of 16.")

        if block_tables.dtype != torch.int32:
            raise NotImplementedError("IxformerPagedDecodeGQA only support block_tables dtype = torch.int32.")

        if max_context_len <= 0:
            raise NotImplementedError("IxformerPagedDecodeGQA only support max_context_len > 0.")

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
            window_left=self.window_size
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
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        max_context_len: int,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:

        """
        Paged decode attention with sliding window (SWA) using a blocked KV cache.

        Args:
            q (torch.Tensor): Query of shape (B, Hq, D).
            k_cache (torch.Tensor): Key cache of shape (N_blocks, Hkv, block_size, D).
            v_cache (torch.Tensor): Value cache of shape (N_blocks, Hkv, block_size, D).
            seq_lens (torch.Tensor): Sequence lengths (B).
            block_table (torch.Tensor): (B, num_blocks) mapping logical blocks to physical IDs.
            max_context_len (int): Max sequence length, should be equal to seqlens.max().
            softmax_scale (Optional[float]): Scale factor; defaults to 1/sqrt(D).

        Returns:
            torch.Tensor: Attention output of shape (B, Hq, D).

        Notes:
            - Head dim D only support [32, 64, 80, 96, 128, 160, 192, 224, 256].
            - Block size must be a multiple of 16.
            - Hq: the number of query head; Hkv: the number of kv head.
            - If Hq > Hkv, K/V heads are repeated to match query heads.
            - Softmax is computed in float32 and cast back to the input dtype.
            - This implementation references variables `q` and `seq_lens`; ensure they
              correspond to `q` and the sequence-lengths tensor in the caller.
        """

        if self.gqa_interleave:
            raise NotImplementedError("IxformerPagedDecodeSWA does not support ABAB layout.")

        batch_size, num_q_heads, head_dim = q.shape
        _, num_kv_heads, block_size, head_dim = k_cache.shape

        output = torch.empty(batch_size, num_q_heads, head_dim, dtype=q.dtype, device=q.device)

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        if block_table.dtype != torch.int32:
            raise NotImplementedError("IxformerPagedDecodeSWA only support block_table dtype = torch.int32.")

        if max_context_len <= 0:
            raise NotImplementedError("IxformerPagedDecodeSWA only support max_context_len > 0.")

        ixf_f.vllm_paged_attention(
            output=output,
            query=q,
            key_cache=k_cache,
            value_cache=v_cache,
            num_kv_heads=num_kv_heads,
            scale=softmax_scale,
            block_tables=block_table,
            context_lens=seq_lens,
            block_size=block_size,
            max_context_len=max_context_len,
            causal=self.is_causal,
            window_left=self.local_window_size,
            global_window_size=self.global_window_size
        )

        return output