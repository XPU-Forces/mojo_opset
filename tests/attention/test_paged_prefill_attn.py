import pytest
import torch
import math

from tests.utils import auto_switch_platform, bypass_not_implemented

from mojo_opset import MojoPagedPrefillGQA


def generate_paged_prefill_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    block_size: int,
    dtype: torch.dtype,
):
    q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32)
    q_lens = torch.clamp(q_lens, min=1)

    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0)])
    total_tokens = cu_seqlens_q[-1].item()

    query = torch.randn(total_tokens, num_q_heads, head_dim, dtype=dtype)
    k_unpadded = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype)
    v_unpadded = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype)

    max_num_blocks_per_seq = (q_lens.max().item() + block_size - 1) // block_size
    total_blocks_needed = int(torch.div(q_lens + block_size - 1, block_size, rounding_mode="floor").sum().item())

    if total_blocks_needed == 0:
        total_blocks_needed = batch_size * max_num_blocks_per_seq

    num_total_blocks = total_blocks_needed + 10

    k_cache = torch.zeros(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.zeros(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)

    block_tables = torch.zeros(batch_size, max_num_blocks_per_seq, dtype=torch.long)
    free_blocks = torch.randperm(num_total_blocks)

    current_block_offset = 0
    for i in range(batch_size):
        seq_len = q_lens[i].item()
        start_loc = cu_seqlens_q[i].item()

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

    return query, k_cache, v_cache, cu_seqlens_q, block_tables


test_configs = [
    (2, 16, 4, 128, 1024, 128, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 128, 1024, 256, torch.bfloat16, "M_BF16"),
    (2, 32, 8, 128, 1024, 512, torch.bfloat16, "M_BF16"),

]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_seqlens_q, block_tables, atol, rtol",
    [
        pytest.param(
            *generate_paged_prefill_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            2e-2 if dtype != torch.float32 else 1e-5,
            2e-3 if dtype != torch.float32 else 1e-6,
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, BLK_S, dtype, ID in test_configs
    ],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB"])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_prefill_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_tables: torch.Tensor,
    atol: float,
    rtol: float,
    gqa_layout: str,
):
    _, num_kv_heads, block_size, _ = k_cache.shape
    _, num_q_heads, head_dim = query.shape
    op = MojoPagedPrefillGQA(
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        is_causal=True,
        is_prefill=True,
        gqa_layout=gqa_layout,
    )

    op.forward_diff(
        query,
        k_cache,
        v_cache,
        cu_seqlens_q,
        block_tables,
        atol=atol,
        rtol=rtol,
    )
