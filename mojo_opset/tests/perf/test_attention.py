import math

from typing import Optional

import pytest
import torch

from mojo_opset import MojoConformerChunkAttention
from mojo_opset import MojoConformerSlidingWindowAttention
from mojo_opset import MojoPagedDecodeGQA
from mojo_opset import MojoPagedDecodeSWA
from mojo_opset import MojoPagedPrefillGQA
from mojo_opset import MojoPagedPrefillSWA
from mojo_opset import MojoSdpa
from mojo_opset import MojoSWA
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


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

    total_seq_lens = torch.randint(1, max_seq_len, (batch_size,), dtype=torch.int32)

    max_num_blocks_per_seq = (total_seq_lens.max().item() + block_size - 1) // block_size
    total_blocks_needed = int(
        torch.div(total_seq_lens + block_size - 1, block_size, rounding_mode="floor").sum().item()
    )

    if total_blocks_needed == 0:
        total_blocks_needed = batch_size * max_num_blocks_per_seq

    num_total_blocks = total_blocks_needed + 10

    k_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)

    block_tables = torch.full((batch_size, max_num_blocks_per_seq), -1, dtype=torch.int32)
    free_blocks = torch.randperm(num_total_blocks, dtype=torch.int32)

    current_block_offset = 0
    for i in range(batch_size):
        seq_len = total_seq_lens[i].item()
        num_blocks_for_seq = (seq_len + block_size - 1) // block_size

        if current_block_offset + num_blocks_for_seq > num_total_blocks:
            raise ValueError("Not enough blocks to generate test data.")

        assigned_blocks = free_blocks[current_block_offset : current_block_offset + num_blocks_for_seq]
        block_tables[i, :num_blocks_for_seq] = assigned_blocks
        current_block_offset += num_blocks_for_seq

    return query, k_cache, v_cache, total_seq_lens, block_tables


test_configs_decode = [
    (8, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
    (8, 16, 4, 96, 1024, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (8, 8, 1, 128, 8192, 128, torch.bfloat16, "M_BF16_LONG"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, total_seq_lens, block_tables",
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
@pytest.mark.parametrize("gqa_layout", ["ABAB", "AABB"])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_paged_decode_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    total_seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_layout: str,
):
    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    paged_attn_decode = MojoPagedDecodeGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    perf(  # noqa: F821
        lambda: paged_attn_decode(
            query,
            k_cache,
            v_cache,
            total_seq_lens,
            block_tables,
            softmax_scale=softmax_scale,
        )
    )


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
    cu_q_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)])

    if max_kv_computed_len <= 0:
        kv_cache_lens = None
        kv_lens = q_lens
    else:
        kv_cache_lens = torch.randint(max_kv_computed_len // 2, max_kv_computed_len, (batch_size,), dtype=torch.int32)
        kv_lens = q_lens + kv_cache_lens
    cu_total_seq_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0, dtype=torch.int32)])

    total_q_tokens = cu_q_lens[-1].item()
    total_kv_tokens = cu_total_seq_lens[-1].item()

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

    block_tables = torch.full((batch_size, max_num_blocks_per_seq), -1, dtype=torch.int32)
    free_blocks = torch.randperm(num_total_blocks, dtype=torch.int32)

    current_block_offset = 0
    for i in range(batch_size):
        seq_len = kv_lens[i].item()
        start_loc = cu_total_seq_lens[i].item()

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

    cu_total_seq_lens = (
        None
        if kv_cache_lens is None
        else torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0).to(torch.int32)])
    )
    return query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens


test_configs_prefill = [
    (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 1024, 0, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 8, 1, 128, 4096, 8192, 128, torch.bfloat16, "M_BF16_WITH_CACHE"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens",
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
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, BLK_S, dtype, ID in test_configs_prefill
    ],
)
@pytest.mark.parametrize("gqa_layout", ["ABAB", "AABB"])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_paged_prefill_gqa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_layout: str,
    cu_total_seq_lens: Optional[torch.Tensor],
):
    paged_attn_prefill = MojoPagedPrefillGQA(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    perf(  # noqa: F821
        lambda: paged_attn_prefill(
            query,
            k_cache,
            v_cache,
            cu_q_lens,
            block_tables,
            softmax_scale=softmax_scale,
            cu_total_seq_lens=cu_total_seq_lens,
        )
    )


def _make_cu_lens(lengths: list[int], device: torch.device | str) -> torch.Tensor:
    return torch.tensor(
        [0] + torch.cumsum(torch.tensor(lengths, dtype=torch.int32), 0).tolist(), dtype=torch.int32, device=device
    )


@pytest.mark.parametrize(
    "q_lens, cache_lens, num_heads, head_dim, left_window, right_window, dtype",
    [
        ([512, 384], [0, 0], 8, 64, 128, 0, torch.bfloat16),
        ([1024, 768], [0, 0], 8, 128, 256, 8, torch.bfloat16),
        ([512, 256, 128], [512, 256, 128], 8, 128, 256, 0, torch.bfloat16),
        ([128, 65, 33, 17], [64, 32, 16, 8], 4, 64, 64, 4, torch.bfloat16),
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_conformer_sliding_window_attention(
    q_lens: list[int],
    cache_lens: list[int],
    num_heads: int,
    head_dim: int,
    left_window: int,
    right_window: int,
    dtype: torch.dtype,
):
    device = get_torch_device()
    kv_lens = [q_len + cache_len for q_len, cache_len in zip(q_lens, cache_lens)]
    cu_q_lens = _make_cu_lens(q_lens, device)
    cu_total_seq_lens = _make_cu_lens(kv_lens, device)
    query = torch.randn(cu_q_lens[-1].item(), num_heads, head_dim, dtype=dtype, device=device)
    key = torch.randn(cu_total_seq_lens[-1].item(), num_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn_like(key)

    op = MojoConformerSlidingWindowAttention(left_window=left_window, right_window=right_window)
    perf(lambda: op(query, key, value, cu_q_lens, cu_total_seq_lens))  # noqa: F821


@pytest.mark.parametrize(
    "q_lens, cache_lens, num_heads, head_dim, chunk_size, left_context_chunks, dtype",
    [
        ([512, 384], [0, 0], 8, 64, 128, -1, torch.bfloat16),
        ([1024, 768], [0, 0], 8, 128, 256, 1, torch.bfloat16),
        ([512, 256, 128], [512, 256, 128], 8, 128, 64, 2, torch.bfloat16),
        ([128, 65, 33, 17], [64, 32, 16, 8], 4, 64, 32, 0, torch.bfloat16),
        # varied (chunk_size, left_context_chunks)
        ([256, 192, 128], [0, 0, 0], 8, 64, 8, -1, torch.bfloat16),   # small chunk, unlimited left
        ([512, 384], [0, 0], 4, 64, 32, -1, torch.bfloat16),          # medium chunk, unlimited left
        ([384, 256, 128], [0, 0, 0], 8, 96, 8, 8, torch.bfloat16),    # left_ctx > chunk
        ([256, 192], [0, 0], 8, 96, 16, 0, torch.bfloat16),           # no left context
        ([512, 384, 256], [0, 0, 0], 8, 128, 16, 4, torch.bfloat16),  # left_ctx = 4 chunks
        # longer sequences, larger batch
        ([1024, 768, 512], [0, 0, 0], 8, 64, 32, -1, torch.bfloat16),
        ([768, 512, 384, 256], [0, 0, 0, 0], 4, 96, 64, -1, torch.bfloat16),
        ([512, 384, 256, 128], [256, 128, 64, 32], 8, 128, 32, 0, torch.bfloat16),
        # multi-head
        ([512, 384], [0, 0], 16, 64, 32, 2, torch.bfloat16),
        ([384, 256, 128], [128, 64, 32], 12, 96, 16, 3, torch.bfloat16),
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_conformer_chunk_attention(
    q_lens: list[int],
    cache_lens: list[int],
    num_heads: int,
    head_dim: int,
    chunk_size: int,
    left_context_chunks: int,
    dtype: torch.dtype,
):
    device = get_torch_device()
    kv_lens = [q_len + cache_len for q_len, cache_len in zip(q_lens, cache_lens)]
    cu_q_lens = _make_cu_lens(q_lens, device)
    cu_total_seq_lens = _make_cu_lens(kv_lens, device)
    query = torch.randn(cu_q_lens[-1].item(), num_heads, head_dim, dtype=dtype, device=device)
    key = torch.randn(cu_total_seq_lens[-1].item(), num_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn_like(key)

    op = MojoConformerChunkAttention(chunk_size=chunk_size, left_context_chunks=left_context_chunks)
    perf(lambda: op(query, key, value, cu_q_lens, cu_total_seq_lens))  # noqa: F821


def generate_test_data(
    bsz: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    seq_length: int,
):
    query = torch.randn(bsz, q_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16, requires_grad=False)
    key = torch.randn(bsz, kv_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16, requires_grad=False)
    value = torch.randn(bsz, kv_head_num, seq_length * 2, head_dim, dtype=torch.bfloat16, requires_grad=False)
    blockwise_diffusion_attn_mask = torch.ones(seq_length * 2, seq_length * 2, dtype=torch.bool, requires_grad=False)
    return query, key, value, blockwise_diffusion_attn_mask, q_head_num != kv_head_num


@pytest.mark.parametrize(
    "query, key, value, blockwise_diffusion_attn_mask, enable_gqa",
    [
        pytest.param(
            *generate_test_data(
                bsz=1,
                q_head_num=8,
                kv_head_num=2,
                head_dim=128,
                seq_length=8192,
            )
        ),
    ],
)
@auto_switch_platform(set_perf=True)
def test_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    blockwise_diffusion_attn_mask: torch.Tensor,
    enable_gqa: bool,
):
    diffusion_attn = MojoSdpa(scale=1.0 / math.sqrt(query.shape[-1]), enable_gqa=enable_gqa)
    perf(lambda: diffusion_attn(query, key, value, blockwise_diffusion_attn_mask))  # noqa: F821


test_configs_swa_prefill = [
    (2, 16, 4, 128, 1024, 0, 32, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 1024, 0, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 16, 4, 128, 1024, 8192, 128, torch.bfloat16, "M_BF16_WITH_CACHE"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens",
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
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, BLK_S, dtype, ID in test_configs_swa_prefill
    ],
)
@pytest.mark.parametrize(
    "gqa_layout, global_window, local_window",
    [
        ("AABB", 4, 1023),
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_paged_prefill_swa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_layout: str,
    cu_total_seq_lens: Optional[torch.Tensor],
    global_window: int,
    local_window: int,
):
    swa_prefill = MojoPagedPrefillSWA(
        is_causal=True,
        gqa_layout=gqa_layout,
        global_window_size=global_window,
        local_window_size=local_window,
    )

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    perf(  # noqa: F821
        lambda: swa_prefill(
            query,
            k_cache,
            v_cache,
            cu_q_lens,
            block_tables,
            softmax_scale=softmax_scale,
            cu_total_seq_lens=cu_total_seq_lens,
        )
    )


test_configs_swa_decode = [
    (8, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
    (8, 16, 4, 96, 1024, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (8, 16, 4, 128, 8192, 128, torch.bfloat16, "M_BF16_LONG"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, total_seq_lens, block_tables",
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
        for B, Q_H, KV_H, D, S_LEN, BLK_S, dtype, ID in test_configs_swa_decode
    ],
)
@pytest.mark.parametrize(
    "gqa_layout, global_window, local_window",
    [
        ("AABB", 4, 1023),
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_paged_decode_swa(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    total_seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    gqa_layout: str,
    global_window: int,
    local_window: int,
):
    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    swa_decode = MojoPagedDecodeSWA(
        is_causal=True,
        gqa_layout=gqa_layout,
        global_window_size=global_window,
        local_window_size=local_window,
    )

    perf(  # noqa: F821
        lambda: swa_decode(
            query,
            k_cache,
            v_cache,
            total_seq_lens,
            block_tables,
            softmax_scale=softmax_scale,
        )
    )


def generate_sdpa_data(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_computed_len: int,
    dtype: torch.dtype,
):
    q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32)
    q_lens = torch.clamp(q_lens, min=1)
    cu_q_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)])

    if max_kv_computed_len <= 0:
        kv_cache_lens = None
        kv_lens = q_lens
    else:
        kv_cache_lens = torch.randint(max_kv_computed_len // 2, max_kv_computed_len, (batch_size,), dtype=torch.int32)
        kv_lens = q_lens + kv_cache_lens
    cu_total_seq_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0, dtype=torch.int32)])

    total_q_tokens = cu_q_lens[-1].item()
    total_kv_tokens = cu_total_seq_lens[-1].item()

    query = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)
    key = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    value = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)

    return query, key, value, cu_q_lens, cu_total_seq_lens


test_configs_swa_infer = [
    (2, 16, 4, 128, 1024, 0, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 1024, 0, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 16, 4, 128, 1024, 8192, torch.bfloat16, "M_BF16_WITH_CACHE"),
]


@pytest.mark.parametrize(
    "query, key, value, cu_q_lens, cu_total_seq_lens",
    [
        pytest.param(
            *generate_sdpa_data(
                batch_size=B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=Q_LEN,
                max_kv_computed_len=KV_COMPUTED_LEN,
                dtype=dtype,
            ),
            id=ID,
        )
        for B, Q_H, KV_H, D, Q_LEN, KV_COMPUTED_LEN, dtype, ID in test_configs_swa_infer
    ],
)
@pytest.mark.parametrize(
    "gqa_layout, global_window, local_window",
    [
        ("AABB", 4, 1023),
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_swa_infer(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_total_seq_lens: torch.Tensor,
    gqa_layout: str,
    global_window: int,
    local_window: int,
):
    swa_infer = MojoSWA(
        is_causal=True,
        gqa_layout=gqa_layout,
        global_window_size=global_window,
        local_window_size=local_window,
    )

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    perf(  # noqa: F821
        lambda: swa_infer(
            query,
            key,
            value,
            cu_q_lens,
            cu_total_seq_lens,
            softmax_scale=softmax_scale,
        )
    )
