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


__all__ = [
    "MojoStorePagedMLAKVCache",
]
