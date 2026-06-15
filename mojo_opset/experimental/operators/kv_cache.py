from typing import Optional
from typing import Tuple

import torch

from mojo_opset.core.operators.kv_cache import assert_paged_kv_layout_contract
from mojo_opset.core.operator import MojoOperator


class MojoStorePagedMLAKVCache(MojoOperator):
    """Append new MLA compressed-KV and positional-key tokens into paged caches.

    MLA (Multi-head Latent Attention) stores a low-rank compressed latent
    ``compressed_kv`` and a positional key ``k_pe`` instead of full K/V per
    head.  This operator writes incoming tokens into the block-based paged
    caches following the same block-table scheme as
    :class:`MojoStorePagedKVCache`.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        compressed_kv_states: torch.Tensor,
        k_pe_states: torch.Tensor,
        compressed_kv_cache: torch.Tensor,
        k_pe_cache: torch.Tensor,
        block_table: torch.Tensor,
        cu_q_lens: torch.Tensor,
        context_kv_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            compressed_kv_states: ``(token_num, kv_lora_rank)`` new compressed
                KV latent tokens.
            k_pe_states: ``(token_num, qk_rope_head_dim)`` new positional key
                tokens.
            compressed_kv_cache: ``(N_blocks, 1, block_size, kv_lora_rank)``
                paged compressed-KV cache (modified in-place).
            k_pe_cache: ``(N_blocks, 1, block_size, qk_rope_head_dim)``
                paged positional-key cache (modified in-place).
            block_table: ``(B, max_blocks_per_seq)`` logical-to-physical block
                mapping.
            cu_q_lens: ``(B+1,)`` cumulative query lengths for prefill.
                ``None`` indicates decode mode (1 token per batch).
            context_kv_lens: ``(B,)`` history sequence lengths before
                storing the current tokens. Padding entries use -1.

        Returns:
            ``(compressed_kv_cache, k_pe_cache)`` after in-place writes.
        """
        assert_paged_kv_layout_contract(block_table, cu_q_lens, context_kv_lens)
        block_size = compressed_kv_cache.shape[2]
        num_batches = len(context_kv_lens) if context_kv_lens is not None else 0
        is_decode = cu_q_lens is None

        for batch_id in range(num_batches):
            if not is_decode:
                t_start = cu_q_lens[batch_id].item()
                t_end = cu_q_lens[batch_id + 1].item()
                seq_len = t_end - t_start
            else:
                t_start = batch_id
                t_end = batch_id + 1
                seq_len = 1

            if seq_len <= 0:
                continue

            ckv_slice = compressed_kv_states[t_start:t_end]  # (seq_len, kv_lora_rank)
            kpe_slice = k_pe_states[t_start:t_end]  # (seq_len, qk_rope_head_dim)

            write_start = context_kv_lens[batch_id].item()
            if write_start < 0:
                continue
            bt = block_table[batch_id]
            if bt.numel() == 0 or bt[0].item() < 0:
                continue

            blk_idx = write_start // block_size
            blk_off = write_start % block_size
            src = 0
            remain = seq_len

            while remain > 0:
                if blk_idx >= bt.shape[0]:
                    break
                phys_id = bt[blk_idx].item()
                if phys_id < 0:
                    break

                cap = block_size - blk_off
                n = min(remain, cap)

                compressed_kv_cache[phys_id, 0, blk_off : blk_off + n, :] = ckv_slice[src : src + n]
                k_pe_cache[phys_id, 0, blk_off : blk_off + n, :] = kpe_slice[src : src + n]

                src += n
                remain -= n
                blk_idx += 1
                blk_off = 0

        return compressed_kv_cache, k_pe_cache


class MojoGatherRopeStore(MojoOperator):
    """Gather paged key cache blocks, apply RoPE, and optionally store the result."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _reshape_scale(scale: torch.Tensor, kv_head_num: int, head_dim: int) -> torch.Tensor:
        if scale.dim() == 1:
            if kv_head_num != 1 or scale.size(0) != head_dim:
                raise ValueError(
                    f"1D scale requires kv_head_num=1 and shape [{head_dim}], got kv_head_num={kv_head_num}, shape={tuple(scale.shape)}."
                )
            return scale.reshape(1, 1, 1, head_dim)
        if scale.dim() == 2:
            if scale.shape != (kv_head_num, head_dim):
                raise ValueError(f"2D scale must have shape [{kv_head_num}, {head_dim}], got {tuple(scale.shape)}.")
            return scale.reshape(1, kv_head_num, 1, head_dim)
        raise ValueError(f"scale must be 1D or 2D, got shape {tuple(scale.shape)}.")

    def forward(
        self,
        key: torch.Tensor,
        rope_cache: Optional[torch.Tensor],
        dequant_scale: Optional[torch.Tensor],
        kv_idx: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        quant_scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            key: Paged key cache with shape ``[num_blocks, kv_head_num, page_size, head_dim]``.
                Dtype must be ``torch.int8`` or ``torch.bfloat16``.
            rope_cache: Optional cache updated in-place with the RoPE result. When
                present, shape and dtype must match ``key``.
            dequant_scale: Scale used to dequantize int8 ``key``. Shape is
                ``[kv_head_num, head_dim]`` or ``[head_dim]`` when ``kv_head_num == 1``.
            kv_idx: Page index table with shape ``[batch_size, max_block_nums]``.
                Invalid entries are ``-1``.
            sin: RoPE sine table with shape ``[max_block_nums * page_size, rope_head_dim]``.
            cos: RoPE cosine table with the same shape as ``sin``.
            quant_scale: Scale used to quantize int8 ``rope_cache``.

        Returns:
            BF16 tensor with the same shape as ``key``. Entries corresponding to
            invalid ``kv_idx`` pages are unspecified.
        """
        if key.dim() != 4:
            raise ValueError(f"key must be 4D [num_blocks, kv_head_num, page_size, head_dim], got {tuple(key.shape)}.")
        if key.dtype not in (torch.int8, torch.bfloat16):
            raise TypeError(f"key must be torch.int8 or torch.bfloat16, got {key.dtype}.")
        if rope_cache is not None:
            if rope_cache.shape != key.shape:
                raise ValueError(f"rope_cache shape must match key, got {tuple(rope_cache.shape)} and {tuple(key.shape)}.")
            if rope_cache.dtype != key.dtype:
                raise TypeError(f"rope_cache dtype must match key, got {rope_cache.dtype} and {key.dtype}.")
        if kv_idx.dim() != 2:
            raise ValueError(f"kv_idx must be 2D [batch_size, max_block_nums], got {tuple(kv_idx.shape)}.")
        if kv_idx.dtype != torch.int64:
            raise TypeError(f"kv_idx must be torch.int64, got {kv_idx.dtype}.")
        if sin.shape != cos.shape or sin.dim() != 2:
            raise ValueError(f"sin and cos must be matching 2D tensors, got {tuple(sin.shape)} and {tuple(cos.shape)}.")
        if sin.dtype != torch.bfloat16 or cos.dtype != torch.bfloat16:
            raise TypeError(f"sin and cos must be torch.bfloat16, got {sin.dtype} and {cos.dtype}.")

        _, kv_head_num, page_size, head_dim = key.shape
        max_block_nums = kv_idx.size(1)
        rope_head_dim = sin.size(1)
        if rope_head_dim >= head_dim:
            raise ValueError(f"rope_head_dim must be smaller than head_dim, got {rope_head_dim} and {head_dim}.")
        if rope_head_dim % 2 != 0:
            raise ValueError(f"rope_head_dim must be even, got {rope_head_dim}.")

        nope_head_dim = head_dim - rope_head_dim
        output = torch.empty(key.shape, dtype=torch.bfloat16, device=key.device)
        valid_idx = kv_idx.reshape(-1)
        valid_idx = valid_idx[valid_idx != -1]

        if key.dtype == torch.int8:
            if dequant_scale is None:
                raise ValueError("dequant_scale is required when key dtype is torch.int8.")
            key_work = key[valid_idx].float() * self._reshape_scale(dequant_scale, kv_head_num, head_dim)
        else:
            key_work = key[valid_idx].to(torch.bfloat16)

        global_block_id = 0
        for batch_id in range(kv_idx.size(0)):
            for block_id in range(max_block_nums):
                page_id = int(kv_idx[batch_id, block_id].item())
                if page_id < 0:
                    continue

                cur_block = key_work[global_block_id]
                rope_block = cur_block[:, :, nope_head_dim:]
                block_sin = sin[block_id * page_size : (block_id + 1) * page_size].float().reshape(1, page_size, -1)
                block_cos = cos[block_id * page_size : (block_id + 1) * page_size].float().reshape(1, page_size, -1)

                out_block = torch.empty(kv_head_num, page_size, head_dim, dtype=torch.bfloat16, device=key.device)
                out_block[:, :, :nope_head_dim] = cur_block[:, :, :nope_head_dim].to(torch.bfloat16)
                out_block[:, :, nope_head_dim:] = (
                    self._rotate(rope_block.float()) * block_sin + rope_block.float() * block_cos
                ).to(torch.bfloat16)
                output[page_id] = out_block

                if rope_cache is not None:
                    if rope_cache.dtype == torch.int8:
                        if quant_scale is None:
                            raise ValueError("quant_scale is required when rope_cache dtype is torch.int8.")
                        quant = torch.round(out_block.float() * self._reshape_scale(quant_scale, kv_head_num, head_dim).squeeze(0))
                        rope_cache[page_id] = torch.clamp(quant, -128, 127).to(torch.int8)
                    else:
                        rope_cache[page_id] = out_block

                global_block_id += 1

        return output


class MojoPagedAttentionStoreKvCache(MojoOperator):
    """Store packed paged-attention K/V states into paged KV cache."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        qkv: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: Optional[torch.Tensor],
        block_table: torch.Tensor,
        seq_len: torch.Tensor,
        kv_len: torch.Tensor,
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
        query_head_num: int,
        kv_head_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            qkv: Packed Q/K/V tensor with shape
                ``[sum(seq_len), (query_head_num + kv_head_num + maybe value heads) * head_dim]``.
            key_cache: Paged key cache, shape ``[num_blocks, kv_head_num, block_size, head_dim]``.
            value_cache: Optional paged value cache with the same shape as ``key_cache``.
            block_table: Logical-to-physical block table, shape ``[batch_size, max_blocks]``.
                Negative block ids stop storing for that sequence.
            seq_len: Current sequence lengths, shape ``[batch_size]``.
            kv_len: Existing KV lengths before storing, shape ``[batch_size]``.
            k_scale: Optional int8 key quantization scale, shape ``[kv_head_num, head_dim]``.
            v_scale: Optional int8 value quantization scale, shape ``[kv_head_num, head_dim]``.
            query_head_num: Number of query heads packed before key heads.
            kv_head_num: Number of key/value heads.

        Returns:
            Tuple of updated ``(key_cache, value_cache)``. If ``value_cache`` is
            ``None``, the second return value is ``key_cache`` for API compatibility.
        """
        if qkv.dim() != 2:
            raise ValueError(f"qkv must be 2D, got {tuple(qkv.shape)}.")
        if key_cache.dim() != 4:
            raise ValueError(f"key_cache must be 4D, got {tuple(key_cache.shape)}.")
        if value_cache is not None and value_cache.shape != key_cache.shape:
            raise ValueError(
                f"value_cache shape must match key_cache, got {tuple(value_cache.shape)} and {tuple(key_cache.shape)}."
            )
        if block_table.dim() != 2:
            raise ValueError(f"block_table must be 2D, got {tuple(block_table.shape)}.")
        if seq_len.dim() != 1 or kv_len.dim() != 1 or seq_len.numel() != kv_len.numel():
            raise ValueError(
                f"seq_len and kv_len must be 1D tensors with same length, got {tuple(seq_len.shape)} and {tuple(kv_len.shape)}."
            )

        _, _, block_size, head_dim = key_cache.shape
        has_value = value_cache is not None
        query_head_num = int(query_head_num)
        kv_head_num = int(kv_head_num)
        process_seq_len = 0

        for batch_id in range(seq_len.numel()):
            now_seq_len = int(seq_len[batch_id].item())
            now_kv_len = int(kv_len[batch_id].item())
            now_block_table = block_table[batch_id]

            key_start = query_head_num * head_dim
            key_end = (query_head_num + kv_head_num) * head_dim
            now_key = qkv[process_seq_len : process_seq_len + now_seq_len, key_start:key_end]
            now_key = now_key.reshape(-1, kv_head_num, head_dim).transpose(1, 0)

            if has_value:
                value_start = key_end
                now_value = qkv[process_seq_len : process_seq_len + now_seq_len, value_start:]
                now_value = now_value.reshape(-1, kv_head_num, head_dim).transpose(1, 0)

            if key_cache.dtype == torch.int8:
                if k_scale is None:
                    raise ValueError("k_scale is required when key_cache dtype is torch.int8.")
                now_key = torch.clamp(torch.round(now_key.to(torch.float32) * k_scale.unsqueeze(1)), -128, 127).to(
                    torch.int8
                )
                if has_value:
                    if v_scale is None:
                        raise ValueError("v_scale is required when value_cache dtype is torch.int8.")
                    now_value = torch.clamp(
                        torch.round(now_value.to(torch.float32) * v_scale.unsqueeze(1)), -128, 127
                    ).to(torch.int8)

            start_block_table_idx = now_kv_len // block_size
            block_offset = now_kv_len % block_size
            remain_seq_len = now_seq_len
            kv_offset = 0

            for block_id in now_block_table[start_block_table_idx:]:
                block_id = int(block_id.item())
                if block_id < 0:
                    break
                store_kv_len = min(block_size - block_offset, remain_seq_len)
                key_cache[block_id, :, block_offset : block_offset + store_kv_len, :] = now_key[
                    :, kv_offset : kv_offset + store_kv_len, :
                ]
                if has_value:
                    value_cache[block_id, :, block_offset : block_offset + store_kv_len, :] = now_value[
                        :, kv_offset : kv_offset + store_kv_len, :
                    ]
                block_offset = 0
                kv_offset += store_kv_len
                remain_seq_len -= store_kv_len
                if remain_seq_len <= 0:
                    break

            process_seq_len += now_seq_len

        return key_cache, value_cache if has_value else key_cache


class MojoPagedCacheDequant(MojoOperator):
    """Dequantize an int8 paged KV cache."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _reshape_scale(scale: torch.Tensor, head_num: int, head_dim: int) -> torch.Tensor:
        if scale.dim() == 1:
            if head_num != 1 or scale.size(0) != head_dim:
                raise ValueError(
                    f"1D dequant_scale requires head_num=1 and shape [{head_dim}], got head_num={head_num}, shape={tuple(scale.shape)}."
                )
            return scale.reshape(1, 1, 1, head_dim)
        if scale.dim() == 2:
            if scale.shape != (head_num, head_dim):
                raise ValueError(f"2D dequant_scale must have shape [{head_num}, {head_dim}], got {tuple(scale.shape)}.")
            return scale.reshape(1, head_num, 1, head_dim)
        raise ValueError(f"dequant_scale must be 1D or 2D, got shape {tuple(scale.shape)}.")

    def forward(
        self,
        quantized_cache: torch.Tensor,
        dequant_scale: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            quantized_cache: Int8 paged cache with shape
                ``[num_blocks, head_num, block_size, head_dim]``.
            dequant_scale: Dequantization scale with shape ``[head_num, head_dim]``
                or ``[head_dim]`` when ``head_num == 1``.
            block_table: Logical-to-physical block table with shape
                ``[batch_size, max_blocks]``. Entries with ``-1`` are ignored by
                paged cache implementations.

        Returns:
            BF16 dequantized cache with the same shape as ``quantized_cache``.
        """
        if quantized_cache.dim() != 4:
            raise ValueError(
                f"quantized_cache must be 4D [num_blocks, head_num, block_size, head_dim], got {tuple(quantized_cache.shape)}."
            )
        if quantized_cache.dtype != torch.int8:
            raise TypeError(f"quantized_cache must be torch.int8, got {quantized_cache.dtype}.")
        if block_table.dim() != 2:
            raise ValueError(f"block_table must be 2D, got {tuple(block_table.shape)}.")
        if block_table.dtype != torch.int64:
            raise TypeError(f"block_table must be torch.int64, got {block_table.dtype}.")

        num_blocks, head_num, _, head_dim = quantized_cache.shape
        scale = self._reshape_scale(dequant_scale, head_num, head_dim)
        output = torch.empty(quantized_cache.shape, dtype=torch.bfloat16, device=quantized_cache.device)
        valid_block_ids = block_table[block_table != -1]
        if valid_block_ids.numel() == 0:
            return output
        if int(valid_block_ids.min().item()) < 0 or int(valid_block_ids.max().item()) >= num_blocks:
            raise ValueError(f"block_table contains block ids outside [-1, {num_blocks}).")

        output[valid_block_ids] = (quantized_cache[valid_block_ids].float() * scale).to(torch.bfloat16)
        return output


__all__ = [
    "MojoGatherRopeStore",
    "MojoPagedCacheDequant",
    "MojoPagedAttentionStoreKvCache",
    "MojoStorePagedMLAKVCache",
]
