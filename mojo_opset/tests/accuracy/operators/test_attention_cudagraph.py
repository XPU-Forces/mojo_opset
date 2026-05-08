import os
import math

import pytest
import torch

from mojo_opset import MojoPagedDecodeGQA
from mojo_opset import MojoPagedDecodeSWA
from mojo_opset import MojoPagedPrefillGQA
from mojo_opset import MojoPagedPrefillSWA
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.tests.utils import get_platform
from mojo_opset.utils.acc import check_tol_diff

pytestmark = pytest.mark.skipif(
    get_platform() != "ilu" or os.environ.get("MOJO_BACKEND", "").strip().lower() != "ttx",
    reason="CUDA Graph attention tests are only enabled on ILU platform with TTX backend.",
)

# ------------------------------------------------------------------------------------------------
# 1. Difference between generate_paged_prefill_data and generate_paged_prefill_data_with_graph:
# ------------------------------------------------------------------------------------------------
# - generate_paged_prefill_data samples q_lens and optional kv cache lengths,
#   builds compact caches for that sampled workload, and returns
#   cu_total_seq_lens=None when there is no extra cache.

# - generate_paged_prefill_data_with_graph fixes every batch at max_q_len and
#   max_kv_computed_len, always returns concrete cu_total_seq_lens, and sizes the
#   cache/block table for the maximum static shape. The CUDA Graph tests then
#   mutate these captured buffers in place with smaller random workloads before
#   replaying the graph.

# ------------------------------------------------------------------------------------------------
# 2. Difference between generate_paged_decode_data and generate_paged_decode_data_with_graph:
# ------------------------------------------------------------------------------------------------
# - generate_paged_decode_data creates random per-batch sequence lengths and
#   sizes the cache/block table to the actual sampled workload.
#
# - generate_paged_decode_data_with_graph creates max-shape static buffers:
#   every batch uses max_seq_len (or a non-zero 1..batch_size range for the
#   padding case), and the cache/block table is sized for batch_size *
#   max_num_blocks_per_seq. CUDA Graph capture needs these stable shapes so the
#   test can copy smaller random workloads into the same buffers and replay.

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

    if max_seq_len > 0:
        total_seq_lens = torch.randint(0, max_seq_len, (batch_size,), dtype=torch.int32)
        total_seq_lens = torch.clamp(total_seq_lens, min=1)
    else:
        total_seq_lens = torch.randperm(batch_size, dtype=torch.int32)

    max_total_seq_len = total_seq_lens.max().item()
    max_num_blocks_per_seq = (max_total_seq_len + block_size - 1) // block_size
    total_blocks_needed = int(torch.div(total_seq_lens + block_size - 1, block_size, rounding_mode="floor").sum().item())

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

    return query, k_cache, v_cache, total_seq_lens, block_tables, max_total_seq_len


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
    if max_q_len > 0:
        q_lens = torch.randint(max_q_len // 2, max_q_len, (batch_size,), dtype=torch.int32)
        q_lens = torch.clamp(q_lens, min=1)
    else:
        # max_q_len = 0 for testing padding logic, use randperm to generate a list with 0
        q_lens = torch.randperm(batch_size, dtype=torch.int32)
    cu_q_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)])

    if max_kv_computed_len <= 0:
        kv_cache_lens = None
        kv_lens = q_lens
    else:
        kv_cache_lens = torch.randint(max_kv_computed_len // 2, max_kv_computed_len, (batch_size,), dtype=torch.int32)
        kv_lens = q_lens + kv_cache_lens
        kv_lens = torch.where(q_lens > 0, kv_lens, torch.zeros_like(kv_lens))
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

    cu_total_seq_lens = None if kv_cache_lens is None else cu_total_seq_lens
    max_q_lens = int((cu_q_lens[1:] - cu_q_lens[:-1]).max().item()) if cu_q_lens.numel() > 1 else 0
    max_total_seq_lens = int(kv_lens.max().item()) if kv_lens.numel() > 0 else 0
    return query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens, max_q_lens, max_total_seq_lens


def generate_paged_decode_data_with_graph(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    block_size: int,
    dtype: torch.dtype,
):
    query = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype)
    if max_seq_len > 0:
        total_seq_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int32)
    else:
        total_seq_lens = torch.arange(1, batch_size + 1, dtype=torch.int32)
    max_total_seq_len = int(total_seq_lens.max().item())
    max_cache_seq_len = max_total_seq_len
    max_num_blocks_per_seq = (max_cache_seq_len + block_size - 1) // block_size
    total_blocks_needed = batch_size * max_num_blocks_per_seq
    num_total_blocks = total_blocks_needed + 10

    k_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)

    block_tables = torch.zeros(batch_size, max_num_blocks_per_seq, dtype=torch.int32)
    free_blocks = torch.randperm(num_total_blocks)

    current_block_offset = 0
    for i in range(batch_size):
        seq_len = total_seq_lens[i].item()
        num_blocks_for_seq = (seq_len + block_size - 1) // block_size

        if current_block_offset + num_blocks_for_seq > num_total_blocks:
            raise ValueError("Not enough blocks to generate test data.")

        assigned_blocks = free_blocks[current_block_offset : current_block_offset + num_blocks_for_seq]
        block_tables[i, :num_blocks_for_seq] = assigned_blocks
        current_block_offset += num_blocks_for_seq

    return query, k_cache, v_cache, total_seq_lens, block_tables, max_total_seq_len


def generate_paged_prefill_data_with_graph(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_computed_len: int,
    block_size: int,
    dtype: torch.dtype,
):
    q_lens = torch.full((batch_size,), max_q_len, dtype=torch.int32)
    cu_q_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)])

    kv_cache_lens = torch.full((batch_size,), max_kv_computed_len, dtype=torch.int32)
    kv_lens = q_lens + kv_cache_lens
    cu_total_seq_lens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(kv_lens, 0, dtype=torch.int32)])

    total_q_tokens = cu_q_lens[-1].item()
    query = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype)

    max_num_blocks_per_seq = (kv_lens.max().item() + block_size - 1) // block_size
    total_blocks_needed = batch_size * max_num_blocks_per_seq
    num_total_blocks = total_blocks_needed + 10

    k_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)
    v_cache = torch.randn(num_total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype)

    block_tables = torch.zeros(batch_size, max_num_blocks_per_seq, dtype=torch.int32)
    free_blocks = torch.randperm(num_total_blocks, dtype=torch.int32)

    current_block_offset = 0
    for i in range(batch_size):
        seq_len = kv_lens[i].item()
        num_blocks_for_seq = (seq_len + block_size - 1) // block_size

        if current_block_offset + num_blocks_for_seq > num_total_blocks:
            raise ValueError("Not enough blocks to generate test data.")

        assigned_blocks = free_blocks[current_block_offset : current_block_offset + num_blocks_for_seq]
        block_tables[i, :num_blocks_for_seq] = assigned_blocks
        current_block_offset += num_blocks_for_seq

    max_q_lens = max_q_len
    max_total_seq_lens = int(kv_lens.max().item())
    return query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens, max_q_lens, max_total_seq_lens


test_configs_decode_with_graph = [
    (16, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
    (8, 16, 4, 96, 1024, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (8, 8, 1, 128, 8192, 1024, torch.bfloat16, "M_BF16_LONG"),
    (8, 8, 1, 128, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
    (8, 8, 1, 128, 0, 1024, torch.bfloat16, "M_BF16_PADSEQ")
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, total_seq_lens, block_tables, max_total_seq_len",
    [
        pytest.param(
            *generate_paged_decode_data_with_graph(
                batch_size=MAX_B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_seq_len=MAX_S_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for MAX_B, Q_H, KV_H, D, MAX_S_LEN, BLK_S, dtype, ID in test_configs_decode_with_graph
    ],
)
@pytest.mark.parametrize("gqa_layout", ["AABB"])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_decode_gqa_with_graph(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    total_seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_total_seq_len: int,
    gqa_layout: str,
):
    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    with torch.no_grad():
        paged_decode_attn = MojoPagedDecodeGQA(
            is_causal=True,
            gqa_layout=gqa_layout,
        )
        paged_decode_attn(
            query,
            k_cache,
            v_cache,
            total_seq_lens,
            block_tables,
            softmax_scale=softmax_scale,
            max_total_seq_len=max_total_seq_len,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                output = paged_decode_attn(
                    query,
                    k_cache,
                    v_cache,
                    total_seq_lens,
                    block_tables,
                    softmax_scale=softmax_scale,
                    max_total_seq_len=max_total_seq_len,
                )
            torch.cuda.synchronize()
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"CUDA graph capture failed: {e}.")
            torch.cuda.empty_cache()
            return

    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    paged_decode_attn_ref = MojoPagedDecodeGQA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    ref_output = paged_decode_attn_ref(
        query,
        k_cache,
        v_cache,
        total_seq_lens,
        block_tables,
        softmax_scale=softmax_scale,
        max_total_seq_len=max_total_seq_len,
    )

    atol = 2e-2 if query.dtype != torch.float32 else 1e-5
    rtol = 2e-2 if query.dtype != torch.float32 else 1e-6

    check_tol_diff(output, ref_output, atol=atol, rtol=rtol)

    max_batch_size, num_q_heads, head_dim = query.shape
    _, num_kv_heads, block_size, _ = k_cache.shape
    for _ in range(5):
        current_batch_size = torch.randint(1, max_batch_size + 1, ()).item()

        cur_q, cur_k, cur_v, cur_total_seq_lens, cur_block_tables, cur_max_total_seq_len = (
            generate_paged_decode_data(
                batch_size=current_batch_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_seq_len=max_total_seq_len,
                block_size=block_size,
                dtype=query.dtype,
            )
        )

        current_num_blocks = cur_k.shape[0]
        k_cache[:current_num_blocks].copy_(cur_k)
        v_cache[:current_num_blocks].copy_(cur_v)
        query[:current_batch_size].copy_(cur_q)

        total_seq_lens[:current_batch_size].copy_(cur_total_seq_lens)
        total_seq_lens[current_batch_size:] = 0

        for i in range(current_batch_size):
            num_blocks_per_seq = (int(cur_total_seq_lens[i].item()) + block_size - 1) // block_size
            block_tables[i, :num_blocks_per_seq].copy_(cur_block_tables[i, :num_blocks_per_seq])
            if num_blocks_per_seq < block_tables.shape[1]:
                pad_id = (
                    int(block_tables[i, num_blocks_per_seq - 1].item()) if num_blocks_per_seq > 0 else 0
                )
                block_tables[i, num_blocks_per_seq:].fill_(pad_id)

        ref_output = paged_decode_attn_ref(
            cur_q,
            cur_k,
            cur_v,
            cur_total_seq_lens,
            cur_block_tables,
            softmax_scale=softmax_scale,
            max_total_seq_len=cur_max_total_seq_len,
        )

        reserved_unused_output = output[current_batch_size:].clone()

        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

        check_tol_diff(output[:current_batch_size], ref_output, atol=atol, rtol=rtol)
        check_tol_diff(output[current_batch_size:], reserved_unused_output, atol=atol, rtol=rtol)


test_configs_prefill_with_graph = [
    (2, 16, 4, 128, 1024, 1024, 32, torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 1024, 1024, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (2, 8, 1, 128, 4096, 8192, 128, torch.bfloat16, "M_BF16_WITH_CACHE"),
    (2, 8, 1, 128, 1024, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens, max_q_lens, max_total_seq_lens",
    [
        pytest.param(
            *generate_paged_prefill_data_with_graph(
                batch_size=MAX_B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=MAX_Q_LEN,
                max_kv_computed_len=MAX_KV_COMPUTED_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for MAX_B, Q_H, KV_H, D, MAX_Q_LEN, MAX_KV_COMPUTED_LEN, BLK_S, dtype, ID in test_configs_prefill_with_graph
    ],
)
@pytest.mark.parametrize("gqa_layout", ["AABB"])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_prefill_gqa_with_graph(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_q_lens: torch.Tensor,
    block_tables: torch.Tensor,
    cu_total_seq_lens: torch.Tensor,
    max_q_lens: int,
    max_total_seq_lens: int,
    gqa_layout: str,
):
    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    with torch.no_grad():
        paged_prefill_attn = MojoPagedPrefillGQA(
            is_causal=True,
            gqa_layout=gqa_layout,
        )
        paged_prefill_attn(
            query,
            k_cache,
            v_cache,
            cu_q_lens,
            block_tables=block_tables,
            softmax_scale=softmax_scale,
            cu_total_seq_lens=cu_total_seq_lens,
            max_q_lens=max_q_lens,
            max_total_seq_lens=max_total_seq_lens,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                output = paged_prefill_attn(
                    query,
                    k_cache,
                    v_cache,
                    cu_q_lens,
                    block_tables=block_tables,
                    softmax_scale=softmax_scale,
                    cu_total_seq_lens=cu_total_seq_lens,
                    max_q_lens=max_q_lens,
                    max_total_seq_lens=max_total_seq_lens,
                )
            torch.cuda.synchronize()
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"CUDA graph capture failed: {e}.")
            torch.cuda.empty_cache()
            return

    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    paged_prefill_attn_ref = MojoPagedPrefillGQA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
    )

    ref_output = paged_prefill_attn_ref(
        query,
        k_cache,
        v_cache,
        cu_q_lens,
        block_tables=block_tables,
        softmax_scale=softmax_scale,
        cu_total_seq_lens=cu_total_seq_lens,
        max_q_lens=max_q_lens,
        max_total_seq_lens=max_total_seq_lens,
    )

    atol = 2e-2 if query.dtype != torch.float32 else 1e-5
    rtol = 2e-2 if query.dtype != torch.float32 else 1e-6

    check_tol_diff(output, ref_output, atol=atol, rtol=rtol)

    max_batch_size = cu_q_lens.shape[0] - 1
    max_total_q_tokens, num_q_heads, head_dim = query.shape
    _, num_kv_heads, block_size, _ = k_cache.shape
    max_q_len_cfg = max_total_q_tokens // max_batch_size
    max_kv_computed_len_cfg = max_total_seq_lens - max_q_len_cfg

    for _ in range(5):
        current_batch_size = torch.randint(1, max_batch_size + 1, ()).item()

        (
            cur_q,
            cur_k,
            cur_v,
            cur_cu_q_lens,
            cur_block_tables,
            cur_cu_total_seq_lens,
            cur_max_q_lens,
            cur_max_total_seq_lens,
        ) = generate_paged_prefill_data(
            batch_size=current_batch_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_q_len=max_q_len_cfg,
            max_kv_computed_len=max_kv_computed_len_cfg,
            block_size=block_size,
            dtype=query.dtype,
        )

        if cur_cu_total_seq_lens is None:
            cur_cu_total_seq_lens = cur_cu_q_lens

        cur_total_q_tokens = int(cur_cu_q_lens[-1].item())
        current_num_blocks = cur_k.shape[0]

        k_cache[:current_num_blocks].copy_(cur_k)
        v_cache[:current_num_blocks].copy_(cur_v)
        query[:cur_total_q_tokens].copy_(cur_q)

        cu_q_lens[: current_batch_size + 1].copy_(cur_cu_q_lens)
        cu_q_lens[current_batch_size + 1 :] = cur_cu_q_lens[-1]
        cu_total_seq_lens[: current_batch_size + 1].copy_(cur_cu_total_seq_lens)
        cu_total_seq_lens[current_batch_size + 1 :] = cur_cu_total_seq_lens[-1]

        cur_total_seq_lens = cur_cu_total_seq_lens[1:] - cur_cu_total_seq_lens[:-1]
        for i in range(current_batch_size):
            num_blocks_per_seq = (int(cur_total_seq_lens[i].item()) + block_size - 1) // block_size
            block_tables[i, :num_blocks_per_seq].copy_(cur_block_tables[i, :num_blocks_per_seq])
            if num_blocks_per_seq < block_tables.shape[1]:
                pad_id = (
                    int(block_tables[i, num_blocks_per_seq - 1].item()) if num_blocks_per_seq > 0 else 0
                )
                block_tables[i, num_blocks_per_seq:].fill_(pad_id)
        if current_batch_size < max_batch_size:
            block_tables[current_batch_size:].copy_(
                block_tables[current_batch_size - 1 : current_batch_size].expand(
                    max_batch_size - current_batch_size, -1
                )
            )

        ref_output = paged_prefill_attn_ref(
            cur_q,
            cur_k,
            cur_v,
            cur_cu_q_lens,
            block_tables=cur_block_tables,
            softmax_scale=softmax_scale,
            cu_total_seq_lens=cur_cu_total_seq_lens,
            max_q_lens=cur_max_q_lens,
            max_total_seq_lens=cur_max_total_seq_lens,
        )

        reserved_unused_output = output[cur_total_q_tokens:].clone()

        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

        check_tol_diff(output[:cur_total_q_tokens], ref_output, atol=atol, rtol=rtol)
        pad_after_bytes = output[cur_total_q_tokens:].view(torch.uint8)
        pad_before_bytes = reserved_unused_output.view(torch.uint8)
        assert torch.equal(pad_after_bytes, pad_before_bytes), (
            "GQA paged prefill kernel modified the padded output region during"
            f" CUDA Graph replay (cur_total_q_tokens={cur_total_q_tokens})."
        )


def generate_paged_prefill_swa_data_with_graph(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_computed_len: int,
    block_size: int,
    dtype: torch.dtype,
):
    query, k_cache, v_cache, cu_q_lens, block_tables, cu_total_seq_lens, max_q_lens, max_total_seq_lens = (
        generate_paged_prefill_data_with_graph(
            batch_size=batch_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_q_len=max_q_len,
            max_kv_computed_len=max_kv_computed_len,
            block_size=block_size,
            dtype=dtype,
        )
    )
    seqlens_kv = cu_total_seq_lens[1:] - cu_total_seq_lens[:-1]
    return (
        query,
        k_cache,
        v_cache,
        cu_q_lens,
        cu_total_seq_lens,
        block_tables,
        seqlens_kv,
        max_q_lens,
        max_total_seq_lens,
    )


test_configs_swa_prefill_with_graph = [
    (2, 16, 4, 128, 1024, 1024, 32, 4, 255, "ABAB", torch.bfloat16, "M_BF16"),
    (2, 16, 4, 96, 1024, 1024, 128, 4, 255, "ABAB", torch.bfloat16, "M_BF16_PADDIM"),
    (2, 8, 1, 128, 256, 1024, 128, 4, 255, "AABB", torch.bfloat16, "M_BF16_WITH_CACHE"),
    (2, 8, 1, 128, 1024, 2048, 1024, 4, 1023, "AABB", torch.bfloat16, "M_BF16_BIGPAGE"),
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, cu_seqlens_q, cu_seqlens_kv, block_tables, seqlens_kv,"
    " max_seqlen_q, max_seqlen_k, global_window, local_window, gqa_layout",
    [
        pytest.param(
            *generate_paged_prefill_swa_data_with_graph(
                batch_size=MAX_B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_q_len=MAX_Q_LEN,
                max_kv_computed_len=MAX_KV_COMPUTED_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            GW,
            LW,
            LAYOUT,
            id=ID,
        )
        for (
            MAX_B,
            Q_H,
            KV_H,
            D,
            MAX_Q_LEN,
            MAX_KV_COMPUTED_LEN,
            BLK_S,
            GW,
            LW,
            LAYOUT,
            dtype,
            ID,
        ) in test_configs_swa_prefill_with_graph
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_paged_prefill_swa_with_graph(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    block_tables: torch.Tensor,
    seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    global_window: int,
    local_window: int,
    gqa_layout: str,
):
    del max_seqlen_q, max_seqlen_k  # SWA paged prefill API does not consume these

    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    with torch.no_grad():
        paged_prefill_swa = MojoPagedPrefillSWA(
            is_causal=True,
            gqa_layout=gqa_layout,
            global_window_size=global_window,
            local_window_size=local_window,
        )
        # Warm-up: trigger autotune / kernel compile before graph capture.
        paged_prefill_swa(
            query,
            k_cache,
            v_cache,
            cu_seqlens_q,
            block_tables,
            softmax_scale=softmax_scale,
            cu_total_seq_lens=cu_seqlens_kv,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                output = paged_prefill_swa(
                    query,
                    k_cache,
                    v_cache,
                    cu_seqlens_q,
                    block_tables,
                    softmax_scale=softmax_scale,
                    cu_total_seq_lens=cu_seqlens_kv,
                )
            torch.cuda.synchronize()
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"CUDA graph capture failed: {e}.")
            torch.cuda.empty_cache()
            return

    paged_prefill_swa_ref = MojoPagedPrefillSWA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
        global_window_size=global_window,
        local_window_size=local_window,
    )

    max_batch_size = cu_seqlens_q.shape[0] - 1
    max_total_q_tokens, num_q_heads, head_dim = query.shape
    _, num_kv_heads, block_size, _ = k_cache.shape
    max_q_len_cfg = max_total_q_tokens // max_batch_size
    max_kv_len_cfg = int(seqlens_kv.max().item())
    max_kv_computed_len_cfg = max_kv_len_cfg - max_q_len_cfg

    atol = 2e-2 if query.dtype != torch.float32 else 1e-5
    rtol = 2e-2 if query.dtype != torch.float32 else 1e-6

    for _ in range(5):
        current_batch_size = torch.randint(1, max_batch_size + 1, ()).item()

        (
            cur_q,
            cur_k,
            cur_v,
            cur_cu_q,
            cur_block_tables,
            cur_cu_kv,
            _cur_max_seqlen_q,
            _cur_max_seqlen_k,
        ) = generate_paged_prefill_data(
            batch_size=current_batch_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_q_len=max_q_len_cfg,
            max_kv_computed_len=max_kv_computed_len_cfg,
            block_size=block_size,
            dtype=query.dtype,
        )

        # generate_paged_prefill_data returns cu_total_seq_lens=None when there
        # is no extra cache (kv_len == q_len); SWA always needs concrete lengths.
        if cur_cu_kv is None:
            cur_seqlens_kv = cur_cu_q[1:] - cur_cu_q[:-1]
            cur_cu_kv = cur_cu_q
        else:
            cur_seqlens_kv = cur_cu_kv[1:] - cur_cu_kv[:-1]

        cur_T = int(cur_cu_q[-1].item())

        # In-place update static buffers captured by the graph.
        current_num_blocks = cur_k.shape[0]
        k_cache[:current_num_blocks].copy_(cur_k)
        v_cache[:current_num_blocks].copy_(cur_v)
        query[:cur_T].copy_(cur_q)

        # cu_seqlens_q: valid prefix mirrors cur, padded batches keep the final
        # cumulative value so q_len = 0 for them (per spec).
        cu_seqlens_q[: current_batch_size + 1].copy_(cur_cu_q)
        cu_seqlens_q[current_batch_size + 1 :] = cur_cu_q[-1]

        # seqlens_kv: valid batches copied, padded batches set to 0 (per spec).
        seqlens_kv[:current_batch_size].copy_(cur_seqlens_kv)
        seqlens_kv[current_batch_size:] = 0
        cu_seqlens_kv[: current_batch_size + 1].copy_(cur_cu_kv)
        cu_seqlens_kv[current_batch_size + 1 :] = cur_cu_kv[-1]

        # block_tables: valid entries mirror cur, padded batches use -1 (per
        # spec). The SWA kernel must not load these entries because the
        # per-batch q_len>0 && kv_len>0 guard short-circuits computation.
        block_tables[:current_batch_size, : cur_block_tables.shape[1]].copy_(cur_block_tables)
        block_tables[:current_batch_size, cur_block_tables.shape[1] :] = -1
        block_tables[current_batch_size:] = -1

        ref_output = paged_prefill_swa_ref(
            cur_q,
            cur_k,
            cur_v,
            cur_cu_q,
            cur_block_tables,
            softmax_scale=softmax_scale,
            cu_total_seq_lens=cur_cu_kv,
        )

        # Capture padded-tail outputs to verify CUDA Graph replay does not
        # touch them (output buffer is reused across replays, padding region
        # must remain bit-identical).
        reserved_unused_output = output[cur_T:].clone()

        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

        check_tol_diff(output[:cur_T], ref_output, atol=atol, rtol=rtol)
        # Padding region must remain bit-identical across replays. We compare
        # raw bytes (not float values) because torch.empty_like in the impl
        # may seed the buffer with NaN/Inf garbage, which the kernel
        # legitimately leaves untouched -- a per-element float comparison
        # would spuriously fail because NaN != NaN.
        pad_after_bytes = output[cur_T:].view(torch.uint8)
        pad_before_bytes = reserved_unused_output.view(torch.uint8)
        assert torch.equal(pad_after_bytes, pad_before_bytes), (
            "SWA paged prefill kernel modified the padded output region during"
            f" CUDA Graph replay (cur_T={cur_T})."
        )


test_configs_swa_decode_with_graph = [
    (16, 16, 4, 128, 1024, 32, torch.bfloat16, "M_BF16"),
    (8, 16, 4, 96, 1024, 128, torch.bfloat16, "M_BF16_PADDIM"),
    (8, 8, 1, 128, 8192, 1024, torch.bfloat16, "M_BF16_LONG"),
    (8, 8, 1, 128, 2048, 1024, torch.bfloat16, "M_BF16_BIGPAGE"),
    (8, 8, 1, 128, 0, 1024, torch.bfloat16, "M_BF16_PADSEQ")
]


@pytest.mark.parametrize(
    "query, k_cache, v_cache, total_seq_lens, block_tables, max_total_seq_len",
    [
        pytest.param(
            *generate_paged_decode_data_with_graph(
                batch_size=MAX_B,
                num_q_heads=Q_H,
                num_kv_heads=KV_H,
                head_dim=D,
                max_seq_len=MAX_S_LEN,
                block_size=BLK_S,
                dtype=dtype,
            ),
            id=ID,
        )
        for MAX_B, Q_H, KV_H, D, MAX_S_LEN, BLK_S, dtype, ID in test_configs_swa_decode_with_graph
    ],
)
@pytest.mark.parametrize("gqa_layout, global_window, local_window", [
    ("ABAB", 4, 255),
    ("AABB", 4, 1023),
])
@auto_switch_platform()
@bypass_not_implemented
def test_paged_decode_swa_with_graph(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    total_seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_total_seq_len: int,
    gqa_layout: str,
    global_window: int,
    local_window: int,
):
    head_dim = query.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)

    with torch.no_grad():
        paged_decode_swa = MojoPagedDecodeSWA(
            is_causal=True,
            gqa_layout=gqa_layout,
            global_window_size=global_window,
            local_window_size=local_window,
        )
        paged_decode_swa(
            query, k_cache, v_cache, total_seq_lens, block_tables, softmax_scale=softmax_scale
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                output = paged_decode_swa(
                    query, k_cache, v_cache, total_seq_lens, block_tables, softmax_scale=softmax_scale
                )

            torch.cuda.synchronize()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"CUDA graph capture failed: {e}.")
            torch.cuda.empty_cache()

    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    paged_decode_swa_ref = MojoPagedDecodeSWA._registry.get("torch")(
        is_causal=True,
        gqa_layout=gqa_layout,
        global_window_size=global_window,
        local_window_size=local_window,
    )

    ref_output = paged_decode_swa_ref(
        query, k_cache, v_cache, total_seq_lens, block_tables, softmax_scale=softmax_scale
    )

    atol = 2e-2 if query.dtype != torch.float32 else 1e-5
    rtol = 2e-2 if query.dtype != torch.float32 else 1e-6

    check_tol_diff(output, ref_output, atol=atol, rtol=rtol)

    max_batch_size, num_q_heads, head_dim = query.shape
    _, num_kv_heads, block_size, _ = k_cache.shape
    for _ in range(5):
        current_batch_size = torch.randint(1, max_batch_size + 1, ()).item()

        cur_q, cur_k, cur_v, cur_total_seq_lens, cur_block_tables, _cur_max_total_seq_len = (
            generate_paged_decode_data(
                batch_size=current_batch_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_seq_len=max_total_seq_len,
                block_size=block_size,
                dtype=query.dtype,
            )
        )

        current_num_blocks = cur_k.shape[0]
        k_cache[:current_num_blocks].copy_(cur_k)
        v_cache[:current_num_blocks].copy_(cur_v)
        query[:current_batch_size].copy_(cur_q)

        total_seq_lens[:current_batch_size].copy_(cur_total_seq_lens)
        total_seq_lens[current_batch_size:] = 0

        for i in range(current_batch_size):
            num_blocks_per_seq = (cur_total_seq_lens[i] + block_size - 1) // block_size
            block_tables[i, :num_blocks_per_seq].copy_(cur_block_tables[i, :num_blocks_per_seq])
            if num_blocks_per_seq < block_tables.shape[1]:
                pad_id = (
                    int(block_tables[i, num_blocks_per_seq - 1].item()) if num_blocks_per_seq > 0 else 0
                )
                block_tables[i, num_blocks_per_seq:].fill_(pad_id)

        ref_output = paged_decode_swa_ref(
            cur_q, cur_k, cur_v, cur_total_seq_lens, cur_block_tables, softmax_scale=softmax_scale
        )

        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()

        check_tol_diff(output[:current_batch_size], ref_output, atol=atol, rtol=rtol)
