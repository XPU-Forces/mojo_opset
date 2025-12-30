import functools
import math

import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoPagedDecodeGQA
from mojo_opset import MojoPagedPrefillGQA
from mojo_opset import MojoSdpa
from mojo_opset.backends.ref.operators.attention import RefPagedDecodeGQA
from mojo_opset.backends.ref.operators.attention import RefPagedPrefillGQA
from mojo_opset.backends.ref.operators.attention import RefSdpa


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
@pytest.mark.parametrize("gqa_layout", ["ABAB"])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_paged_decode_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_layout: str,
):
    head_dim = query.shape[-1]
    sm_scale = 1.0 / math.sqrt(head_dim)

    paged_attn_decode = MojoPagedDecodeGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
    )
    paged_attn_decode_ref = RefPagedDecodeGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    perf(  # noqa: F821
        lambda: paged_attn_decode_ref(
            query,
            k_cache,
            v_cache,
            seqlens,
            block_tables,
            softmax_scale=sm_scale,
        )
    )
    perf(  # noqa: F821
        lambda: paged_attn_decode(
            query,
            k_cache,
            v_cache,
            seqlens,
            block_tables,
            softmax_scale=sm_scale,
        )
    )


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
    (2, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_seqlens_q, block_tables",
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
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, BLK_S, dtype, ID in test_configs
    ],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB"])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_paged_prefill_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_layout: str,
):
    paged_attn_prefill = MojoPagedPrefillGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
    )
    paged_attn_prefill_ref = RefPagedPrefillGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    head_dim = query.shape[-1]
    sm_scale = 1.0 / math.sqrt(head_dim)

    perf(  # noqa: F821
        lambda: paged_attn_prefill_ref(
            query,
            k_cache,
            v_cache,
            cu_seqlens_q,
            block_tables,
            softmax_scale=sm_scale,
        )
    )
    perf(  # noqa: F821
        lambda: paged_attn_prefill(
            query,
            k_cache,
            v_cache,
            cu_seqlens_q,
            block_tables,
            softmax_scale=sm_scale,
        )
    )


@functools.lru_cache()
def generate_diffusion_attention_mask(
    seq_length: int,
    block_size: int,
) -> torch.Tensor:
    print("generate diffusion attention mask...")
    total_length = seq_length * 2
    attn_mask = torch.zeros(total_length, total_length, dtype=torch.int8)

    for i in range(total_length):
        for j in range(total_length):
            block_i = i // block_size
            block_j = j // block_size
            if block_i == block_j:
                attn_mask[i, j] = 1

            if j >= seq_length and i < seq_length and ((j - seq_length) // block_size) < block_i:
                attn_mask[i, j] = 1

            if i >= seq_length and j >= seq_length and block_j < block_i:
                attn_mask[i, j] = 1

    return attn_mask.to(torch.bool)


def generate_test_data(
    bsz: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    seq_length: int,
    block_size: int,
):
    query = torch.randn(bsz, q_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16)
    key = torch.randn(bsz, kv_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16)
    value = torch.randn(bsz, kv_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16)
    # blockwise_diffusion_attn_mask = generate_diffusion_attention_mask(seq_length, block_size)
    blockwise_diffusion_attn_mask = torch.ones(seq_length * 2, seq_length * 2, dtype=torch.bool)
    return query, key, value, blockwise_diffusion_attn_mask, q_head_num != kv_head_num


@pytest.mark.parametrize(
    "query, key, value, blockwise_diffusion_attn_mask, enable_gqa",
    [
        pytest.param(
            *generate_test_data(
                bsz=1,
                q_head_num=8,
                kv_head_num=1,
                head_dim=128,
                seq_length=8192,
                block_size=32,
            )
        ),
    ],
)
@auto_switch_platform(set_perf=True)
def test_diffusion_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    blockwise_diffusion_attn_mask: torch.Tensor,
    enable_gqa: bool,
):
    diffusion_attn_ref = RefSdpa(
        mask=blockwise_diffusion_attn_mask, scale=1.0 / math.sqrt(query.shape[-1]), enable_gqa=enable_gqa
    )
    diffusion_attn = MojoSdpa(
        mask=blockwise_diffusion_attn_mask, scale=1.0 / math.sqrt(query.shape[-1]), enable_gqa=enable_gqa
    )
    perf(lambda: diffusion_attn_ref(query, key, value))
    perf(lambda: diffusion_attn(query, key, value))
