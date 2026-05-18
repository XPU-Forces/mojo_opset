from __future__ import annotations

import math

from typing import Any, Optional, Tuple

import torch

from ixformer import functions as ixf_f
from ixformer.contrib import vllm_flash_attn as ix_fa

from mojo_opset.core import MojoPagedDecodeGQA
from mojo_opset.core import MojoPagedDecodeSWA
from mojo_opset.core import MojoPagedPrefillGQA
from mojo_opset.core import MojoPagedPrefillGQAWithKVDequant
from mojo_opset.core import MojoPagedPrefillSWAWithKVDequant
from mojo_opset.core import MojoPagedPrefillSWA
from mojo_opset.core.operators.attention import assert_paged_decode_contract
from mojo_opset.core.operators.attention import assert_paged_prefill_contract

_IXFORMER_PAGED_DECODE_SUPPORTED_HEAD_DIMS = frozenset(
    {32, 64, 80, 96, 128, 160, 192, 224, 256}
)

def _remap_block_tables(
    block_tables: torch.Tensor,
    num_cache_blocks: int,
    used_block_ids: torch.Tensor,
    *,
    dummy_idx: Optional[int],
) -> torch.Tensor:
    fill_value = -1 if dummy_idx is None else dummy_idx
    remap = torch.full(
        (num_cache_blocks,), fill_value, dtype=block_tables.dtype, device=block_tables.device
    )
    remap[used_block_ids.long()] = torch.arange(
        len(used_block_ids), dtype=block_tables.dtype, device=block_tables.device
    )

    flat_ids = block_tables.reshape(-1)
    valid = flat_ids >= 0
    remapped = torch.full_like(flat_ids, fill_value)
    remapped[valid] = remap[flat_ids[valid].long()]
    return remapped.reshape(block_tables.shape)


def _dequantize_int8_paged_cache_by_block_ids(
    cache: torch.Tensor,
    scale: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    target_dtype: torch.dtype,
    name: str,
    append_dummy_block: bool = False,
) -> torch.Tensor:
    if cache.dtype != torch.int8:
        raise NotImplementedError(f"{name} must be int8, got {cache.dtype}.")
    if scale.shape != (cache.shape[1], cache.shape[-1]):
        raise ValueError(
            f"{name}_scale must have shape {(cache.shape[1], cache.shape[-1])}, got {tuple(scale.shape)}."
        )

    scale_expanded = scale.view(1, cache.shape[1], 1, cache.shape[-1])
    compact_cache = (
        cache[block_ids.long()].to(scale.dtype) * scale_expanded
    ).to(target_dtype)

    if append_dummy_block:
        dummy_block = torch.zeros(
            1, cache.shape[1], cache.shape[2], cache.shape[3],
            dtype=target_dtype, device=cache.device,
        )
        compact_cache = torch.cat([compact_cache, dummy_block], dim=0)

    return compact_cache


def _dequantize_int8_paged_kv_cache_compact(
    key_cache: torch.Tensor,
    key_scale: torch.Tensor,
    value_cache: torch.Tensor,
    value_scale: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    target_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    used_block_ids = block_tables[block_tables >= 0].unique()
    new_block_tables = _remap_block_tables(
        block_tables,
        key_cache.shape[0],
        used_block_ids,
        dummy_idx=None,
    )
    key_cache_f16 = _dequantize_int8_paged_cache_by_block_ids(
        key_cache,
        key_scale,
        used_block_ids,
        target_dtype=target_dtype,
        name="key_cache",
    )
    value_cache_f16 = _dequantize_int8_paged_cache_by_block_ids(
        value_cache,
        value_scale,
        used_block_ids,
        target_dtype=target_dtype,
        name="value_cache",
    )
    return key_cache_f16, value_cache_f16, new_block_tables


def _swa_window_block_ids(
    block_table: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_kv_lens: torch.Tensor,
    block_size: int,
    local_window_size: Optional[int],
    global_window_size: Optional[int],
) -> torch.Tensor:
    """Return the unique physical block IDs that fall inside the sliding window
    (local + global).  Block IDs < 0 are excluded.

    In prefill every query position has its own window, so the needed KV range
    is the **union** of all per-query windows:
        [max(0, kv_len - q_len - local_window_size), kv_len - 1]
    """
    batch_size, max_num_blocks = block_table.shape
    device = block_table.device
    kv_lens = cu_kv_lens[1:] - cu_kv_lens[:-1]
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    total_blocks = (kv_lens + block_size - 1) // block_size

    indices = torch.arange(max_num_blocks, device=device).view(1, -1)
    if local_window_size is not None and local_window_size >= 0:
        earliest_kv_pos = torch.clamp(kv_lens - q_lens - local_window_size, min=0)
        start_block = earliest_kv_pos // block_size
        keep = (indices >= start_block.unsqueeze(1)) & (indices < total_blocks.unsqueeze(1))
    else:
        keep = indices < total_blocks.unsqueeze(1)

    if global_window_size is not None and global_window_size > 0:
        global_blocks = (global_window_size + block_size - 1) // block_size
        keep |= (indices < global_blocks) & (indices < total_blocks.unsqueeze(1))

    ids = block_table[keep]
    ids = ids[ids >= 0].unique()
    return ids


def _dequantize_int8_paged_kv_cache_compact_swa(
    key_cache: torch.Tensor,
    key_scale: torch.Tensor,
    value_cache: torch.Tensor,
    value_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_kv_lens: torch.Tensor,
    *,
    target_dtype: torch.dtype,
    local_window_size: Optional[int],
    global_window_size: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dequantize only K/V blocks within the sliding window and remap
    *block_table* into a compact cache.  Blocks outside the window are mapped
    to a trailing dummy block filled with zeros so that the kernel can still
    dereference every ``block_table`` entry without accessing invalid memory.

    Returns:
        key_cache_f16, value_cache_f16: ``(num_window_blocks + 1, Hkv, block_size, D)``
        new_block_table: remapped table with the same shape as *block_table*.
    """
    block_size = key_cache.shape[2]

    window_ids = _swa_window_block_ids(
        block_table, cu_q_lens, cu_kv_lens, block_size,
        local_window_size, global_window_size,
    )

    num_window = len(window_ids)
    dummy_idx = num_window  # index of the trailing zero block

    new_block_table = _remap_block_tables(
        block_table,
        key_cache.shape[0],
        window_ids,
        dummy_idx=dummy_idx,
    )
    key_cache_f16 = _dequantize_int8_paged_cache_by_block_ids(
        key_cache,
        key_scale,
        window_ids,
        target_dtype=target_dtype,
        name="key_cache",
        append_dummy_block=True,
    )
    value_cache_f16 = _dequantize_int8_paged_cache_by_block_ids(
        value_cache,
        value_scale,
        window_ids,
        target_dtype=target_dtype,
        name="value_cache",
        append_dummy_block=True,
    )

    return key_cache_f16, value_cache_f16, new_block_table


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
            block_tables (torch.Tensor): Logical-to-physical block IDs per batch,
                shape (B, num_blocks).
                dtype must be int32.
            softmax_scale (Optional[float]): Attention scaling factor; defaults to 1/sqrt(D).
            mask (Optional[torch.Tensor]): Attention mask; defaults to None.
                Not support custom mask yet.
            max_q_lens (int): Maximum query sequence length across the batch. >0 and must be int.
            max_total_seq_lens (int): Maximum total visible key/value sequence length across the batch. >0 and must be int.
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


class IxformerPagedPrefillGQAWithKVDequant(MojoPagedPrefillGQAWithKVDequant):
    """Ixformer implementation for int8-KV paged prefill GQA."""

    supported_platforms_list = ["ilu"]

    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "AABB",
        query_dtype: torch.dtype = torch.bfloat16,
        context_dtype: torch.dtype = torch.int8,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(
            is_causal=is_causal,
            gqa_layout=gqa_layout,
            query_dtype=query_dtype,
            context_dtype=context_dtype,
            compute_dtype=compute_dtype,
        )
        if self.gqa_layout == "ABAB":
            raise NotImplementedError("IxformerPagedPrefillGQAWithKVDequant only supports AABB layout.")

    def forward(
        self,
        query: torch.Tensor,
        query_scale: Optional[torch.Tensor],
        key_cache: torch.Tensor,
        key_scale: torch.Tensor,
        value_cache: torch.Tensor,
        value_scale: torch.Tensor,
        cu_q_lens: torch.Tensor,
        block_tables: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_total_seq_lens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        max_q_lens: Optional[int] = None,
        max_total_seq_lens: Optional[int] = None,
    ) -> torch.Tensor:
        assert_paged_prefill_contract(cu_q_lens, block_tables, cu_total_seq_lens)
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(f"query dtype must be fp16 or bf16, got {query.dtype}.")
        if self.query_dtype == torch.int8:
            raise NotImplementedError("IxformerPagedPrefillGQAWithKVDequant does not support quantized query yet.")
        if query_scale is not None:
            raise ValueError("query_scale must be None for non-quantized query.")
        if self.context_dtype != torch.int8:
            raise NotImplementedError(
                f"IxformerPagedPrefillGQAWithKVDequant only supports int8 KV cache, got {self.context_dtype}."
            )
        if self.compute_dtype == torch.int8:
            raise NotImplementedError(
                "IxformerPagedPrefillGQAWithKVDequant uses float16 compute and does not support int8 compute."
            )
        if key_cache.dtype != torch.int8 or value_cache.dtype != torch.int8:
            raise NotImplementedError(
                "IxformerPagedPrefillGQAWithKVDequant requires int8 key_cache/value_cache, "
                f"got {key_cache.dtype}/{value_cache.dtype}."
            )
        if mask is not None:
            raise NotImplementedError("IxformerPagedPrefillGQAWithKVDequant does not support custom mask yet.")

        page_block_size = key_cache.shape[2]
        if page_block_size > 256 or page_block_size % 16 != 0:
            raise NotImplementedError(
                f"IxformerPagedPrefillGQAWithKVDequant only supports page_block_size <= 256 and % 16 == 0, got {page_block_size}."
            )

        cu_kv_lens = cu_total_seq_lens if cu_total_seq_lens is not None else cu_q_lens
        if not isinstance(max_q_lens, int) or max_q_lens <= 0:
            raise ValueError(
                f"max_q_lens must be a positive int, got {max_q_lens} ({type(max_q_lens)})."
            )
        if not isinstance(max_total_seq_lens, int) or max_total_seq_lens <= 0:
            raise ValueError(
                f"max_total_seq_lens must be a positive int, got {max_total_seq_lens} ({type(max_total_seq_lens)})."
            )

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(query.shape[-1])

        key_cache_f16, value_cache_f16, new_block_tables = _dequantize_int8_paged_kv_cache_compact(
            key_cache,
            key_scale,
            value_cache,
            value_scale,
            block_tables,
            target_dtype=self.compute_dtype,
        )

        output = ix_fa.flash_attn_varlen_func(
            query,
            key_cache_f16,
            value_cache_f16,
            cu_q_lens,
            cu_kv_lens,
            max_q_lens,
            max_total_seq_lens,
            causal=self.is_causal,
            softmax_scale=softmax_scale,
            block_table=new_block_tables,
        )
        return output


class IxformerPagedPrefillSWAWithKVDequant(MojoPagedPrefillSWAWithKVDequant):
    """Ixformer implementation for int8-KV paged prefill SWA."""

    supported_platforms_list = ["ilu"]

    def __init__(
        self,
        is_causal: bool = True,
        gqa_layout: str = "AABB",
        global_window_size: Optional[int] = None,
        local_window_size: Optional[int] = None,
        query_dtype: torch.dtype = torch.bfloat16,
        context_dtype: torch.dtype = torch.int8,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(
            is_causal=is_causal,
            gqa_layout=gqa_layout,
            global_window_size=global_window_size,
            local_window_size=local_window_size,
            query_dtype=query_dtype,
            context_dtype=context_dtype,
            compute_dtype=compute_dtype,
        )
        if self.gqa_interleave:
            raise NotImplementedError("IxformerPagedPrefillSWAWithKVDequant only supports AABB layout.")
        if global_window_size is not None and global_window_size < 0:
            raise ValueError(f"global_window_size must be >= 0, got {global_window_size}.")
        if local_window_size is not None and local_window_size < 0:
            raise ValueError(f"local_window_size must be >= 0, got {local_window_size}.")

    def forward(
        self,
        query: torch.Tensor,
        query_scale: Optional[torch.Tensor],
        key_cache: torch.Tensor,
        key_scale: torch.Tensor,
        value_cache: torch.Tensor,
        value_scale: torch.Tensor,
        cu_q_lens: torch.Tensor,
        block_table: torch.Tensor,
        softmax_scale: Optional[float] = None,
        cu_total_seq_lens: Optional[torch.Tensor] = None,
        max_q_lens: Optional[int] = None,
        max_total_seq_lens: Optional[int] = None,
    ) -> torch.Tensor:
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(f"query dtype must be fp16 or bf16, got {query.dtype}.")
        if self.query_dtype == torch.int8:
            raise NotImplementedError("IxformerPagedPrefillSWAWithKVDequant does not support quantized query yet.")
        if query_scale is not None:
            raise ValueError("query_scale must be None for non-quantized query.")
        if self.context_dtype != torch.int8:
            raise NotImplementedError(
                f"IxformerPagedPrefillSWAWithKVDequant only supports int8 KV cache, got {self.context_dtype}."
            )
        if self.compute_dtype == torch.int8:
            raise NotImplementedError(
                "IxformerPagedPrefillSWAWithKVDequant uses float16 compute and does not support int8 compute."
            )
        if key_cache.dtype != torch.int8 or value_cache.dtype != torch.int8:
            raise NotImplementedError(
                "IxformerPagedPrefillSWAWithKVDequant requires int8 key_cache/value_cache, "
                f"got {key_cache.dtype}/{value_cache.dtype}."
            )
        if self.gqa_interleave:
            raise NotImplementedError("IxformerPagedPrefillSWAWithKVDequant only supports AABB layout.")
        page_block_size = key_cache.shape[2]
        if page_block_size > 256 or page_block_size % 16 != 0:
            raise NotImplementedError(
                f"IxformerPagedPrefillSWAWithKVDequant only supports page_block_size <= 256 and % 16 == 0, got {page_block_size}."
            )

        assert_paged_prefill_contract(cu_q_lens, block_table, cu_total_seq_lens)
        cu_kv_lens = cu_total_seq_lens if cu_total_seq_lens is not None else cu_q_lens
        if not isinstance(max_q_lens, int) or max_q_lens <= 0:
            raise ValueError(
                f"max_q_lens must be a positive int, got {max_q_lens} ({type(max_q_lens)})."
            )
        if not isinstance(max_total_seq_lens, int) or max_total_seq_lens <= 0:
            raise ValueError(
                f"max_total_seq_lens must be a positive int, got {max_total_seq_lens} ({type(max_total_seq_lens)})."
            )

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(query.shape[-1])

        window_size = (self.local_window_size, 0) if self.local_window_size is not None else (-1, -1)
        if window_size == (-1, -1) and self.global_window_size is not None and self.global_window_size > 0:
            raise ValueError(
                "global_window_size > 0 requires a valid local_window_size (sliding window)."
            )

        key_cache_f16, value_cache_f16, new_block_table = _dequantize_int8_paged_kv_cache_compact_swa(
            key_cache,
            key_scale,
            value_cache,
            value_scale,
            block_table,
            cu_q_lens,
            cu_kv_lens,
            target_dtype=self.compute_dtype,
            local_window_size=self.local_window_size,
            global_window_size=self.global_window_size,
        )

        output = ix_fa.flash_attn_varlen_func(
            query,
            key_cache_f16,
            value_cache_f16,
            cu_q_lens,
            cu_kv_lens,
            max_q_lens,
            max_total_seq_lens,
            causal=self.is_causal,
            softmax_scale=softmax_scale,
            block_table=new_block_table,
            window_size=window_size,
            global_window_size=self.global_window_size,
        )
        return output


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
