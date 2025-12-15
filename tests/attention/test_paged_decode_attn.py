import pytest
import torch
import math

from tests.utils import auto_switch_platform, bypass_not_implemented

from mojo_opset import MojoPagedDecodeGQA


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

    input_layout = "TND"

    return query, k_cache, v_cache, seqlens, block_tables, input_layout


test_configs_decode = [
    (8, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, seqlens, block_tables, input_layout, atol, rtol",
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
            3e-2 if dtype != torch.float32 else 1e-5,
            1e-3 if dtype != torch.float32 else 1e-6,
            id=ID,
        )
        for B, Q_H, KV_H, D, S_LEN, BLK_S, dtype, ID in test_configs_decode
    ],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB"])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_decode_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    input_layout: str,
    atol: float,
    rtol: float,
    gqa_layout: str,
):
    head_dim = query.shape[-1]
    sm_scale = 1.0 / math.sqrt(head_dim)

    op = MojoPagedDecodeGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    op.forward_diff(
        query,
        k_cache,
        v_cache,
        seqlens,
        block_tables,
        input_layout,
        sm_scale=sm_scale,
        atol=atol,
        rtol=rtol,
    )
