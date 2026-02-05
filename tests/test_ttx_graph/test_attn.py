import math

import pytest
import torch

from tests.utils import auto_switch_platform


def generate_paged_prefill_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_computed_len: int,
    block_size: int,
    dtype: torch.dtype,
):
    q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32)
    q_lens = torch.clamp(q_lens, min=1)
    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0)])

    if max_kv_computed_len <= 0:
        kv_cache_lens = torch.zeros_like(q_lens)
    else:
        kv_cache_lens = torch.randint(max_kv_computed_len // 2, max_kv_computed_len, (batch_size,), dtype=torch.int32)
    kv_lens = q_lens + kv_cache_lens
    cu_seqlens_kv = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0)])

    total_q_tokens = cu_seqlens_q[-1].item()
    total_kv_tokens = cu_seqlens_kv[-1].item()

    query = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)
    k_unpadded = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    v_unpadded = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)

    max_num_blocks_per_seq = (kv_lens.max().item() + block_size - 1) // block_size
    total_blocks_needed = int(torch.div(kv_lens + block_size - 1, block_size, rounding_mode="floor").sum().item())

    if total_blocks_needed == 0:
        total_blocks_needed = batch_size * max_num_blocks_per_seq

    num_total_blocks = total_blocks_needed + 10

    k_cache = torch.zeros(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.zeros(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)

    block_tables = torch.zeros(batch_size, max_num_blocks_per_seq, dtype=torch.long)
    free_blocks = torch.randperm(num_total_blocks)

    current_block_offset = 0
    for i in range(batch_size):
        seq_len = kv_lens[i].item()
        start_loc = cu_seqlens_kv[i].item()

        num_blocks_for_seq = (seq_len + block_size - 1) // block_size
        assigned_blocks = free_blocks[current_block_offset : current_block_offset + num_blocks_for_seq]
        block_tables[i, :num_blocks_for_seq] = assigned_blocks
        current_block_offset += num_blocks_for_seq

        k_seq = k_unpadded[start_loc : start_loc + seq_len]
        v_seq = v_unpadded[start_loc : start_loc + seq_len]
        for j in range(num_blocks_for_seq):
            physical_block_id = assigned_blocks[j]
            start_pos_in_seq = j * block_size
            tokens_in_block = min(block_size, seq_len - start_pos_in_seq)

            k_slice = k_seq[start_pos_in_seq : start_pos_in_seq + tokens_in_block].permute(1, 0, 2)
            v_slice = v_seq[start_pos_in_seq : start_pos_in_seq + tokens_in_block].permute(1, 0, 2)

            k_cache[physical_block_id, :, :tokens_in_block, :] = k_slice
            v_cache[physical_block_id, :, :tokens_in_block, :] = v_slice

    return query, k_cache, v_cache, cu_seqlens_q, kv_lens, block_tables


test_configs = [
    (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "M_BF16"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_seqlens_q, seqlens_kv, block_tables",
    [
        pytest.param(
            *generate_paged_prefill_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                max_kv_computed_len=KV_COMPUTED_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, BLK_S, dtype, ID in test_configs
    ],
)
@auto_switch_platform()
def test_paged_attention_prefill(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqlens_kv: torch.Tensor,
    block_tables: torch.Tensor,
):
    head_dim = query.shape[-1]
    sm_scale = 1.0 / math.sqrt(head_dim)

    torch.library.opcheck(
        torch.ops.ttx.paged_attention_prefill, (query, k_cache, v_cache, cu_seqlens_q, seqlens_kv, block_tables, sm_scale)
    )


def generate_paged_decode_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    block_size: int,
    dtype: torch.dtype,
):
    query = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype)

    seqlens = torch.randint(1, max_seq_len, (batch_size,), dtype=torch.int32)

    max_num_blocks_per_seq = (seqlens.max().item() + block_size - 1) // block_size
    total_blocks_needed = int(torch.div(seqlens + block_size - 1, block_size, rounding_mode="floor").sum().item())

    if total_blocks_needed == 0:
        total_blocks_needed = batch_size * max_num_blocks_per_seq

    num_total_blocks = total_blocks_needed + 10

    k_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)

    block_tables = torch.zeros(batch_size, max_num_blocks_per_seq, dtype=torch.long)
    free_blocks = torch.randperm(num_total_blocks)

    current_block_offset = 0
    for i in range(batch_size):
        seq_len = seqlens[i].item()
        num_blocks_for_seq = (seq_len + block_size - 1) // block_size

        if current_block_offset + num_blocks_for_seq > num_total_blocks:
            raise ValueError("Not enough blocks to generate test data.")

        assigned_blocks = free_blocks[current_block_offset : current_block_offset + num_blocks_for_seq]
        block_tables[i, :num_blocks_for_seq] = assigned_blocks
        current_block_offset += num_blocks_for_seq

    return query, k_cache, v_cache, seqlens, block_tables


test_configs_decode = [
    (8, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, seqlens, block_tables",
    [
        pytest.param(
            *generate_paged_decode_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_seq_len=S_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, S_LEN, BLK_S, dtype, ID in test_configs_decode
    ],
)
@auto_switch_platform()
def test_paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
):
    head_dim = query.shape[-1]
    sm_scale = 1.0 / math.sqrt(head_dim)

    torch.library.opcheck(
        torch.ops.ttx.paged_attention_decode,
        (
            query,
            k_cache,
            v_cache,
            seqlens,
            block_tables,
            sm_scale,
        ),
    )
