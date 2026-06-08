# -*- coding: utf-8 -*-
"""Perf: MojoSALSIndexer.

pytest mojo_opset/tests/perf/test_sals_indexer.py -v
"""
from __future__ import annotations

import pytest
import torch

from mojo_opset import MojoSALSIndexer
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


HEAD_DIM = 128
SPARSE_BLOCK_SIZE = 64
MIN_SPARSE_LEN = 512

MODEL_SPECS = [
    ("new_model_1", 2, 128),
    ("new_model_2", 4, 128),
    ("new_model_3", 4, 128),
    ("new_model_4", 8, 128),
    ("new_model_5", 8, 128),
    ("new_model_6", 16, 128),
    ("M9-23B", 8, 128),
    ("M8-14B", 8, 128),
]

# (q_seqlen, share_len, kv_seqlen, sparsity, fixed_tail, cache_block_size, dtype)
SCENARIOS = [
    (8192,   128, 1024, 4, 8,  64,  torch.float16),
    (8192,   256, 2048, 2, 16, 256, torch.bfloat16),
    (10240,  256, 1024, 4, 8,  256, torch.float16),
    (16384,  128, 512,  4, 16, 64,  torch.bfloat16),
    (16384,  256, 256,  2, 8,  64,  torch.float16),
    (24576,  128, 2048, 2, 16, 256, torch.bfloat16),
    (32768,  256, 2048, 4, 8,  256, torch.float16),
]

# 64k/80k/128k only with lightweight model to avoid OOM
LARGE_SCENARIOS = [
    (65536,  256, 512,  4, 16, 64,  torch.float16),
    (81920,  256, 1024, 2, 8,  256, torch.bfloat16),
    (131072, 256, 256,  2, 16, 64,  torch.float16),
]

# Custom test cases:
# q_len = 4096, kv_len = 16384/32768/65536, q_head_num = 3/4, kv_head_num = 1,
# share_len = 256, cache_block_size = 64, fixed_tail_blocks = 4
CUSTOM_SCENARIOS = [
    (4096, 3, 1, 256, 16384, 64, 4, torch.bfloat16),
    (4096, 3, 1, 256, 32768, 64, 4, torch.bfloat16),
    (4096, 3, 1, 256, 65536, 64, 4, torch.bfloat16),
    (4096, 4, 1, 256, 16384, 64, 4, torch.bfloat16),
    (4096, 4, 1, 256, 32768, 64, 4, torch.bfloat16),
    (4096, 4, 1, 256, 65536, 64, 4, torch.bfloat16),
]

_SMALL_MODEL = MODEL_SPECS[0]  # new_model_1

_MODEL_SPEC_PARAMS = []
for _m in MODEL_SPECS:
    for _s in SCENARIOS:
        _MODEL_SPEC_PARAMS.append(pytest.param(
            *_m, *_s, id=f"{_m[0]}-q{_s[0]}-sl{_s[1]}",
        ))

_LARGE_PARAMS = []
for _s in LARGE_SCENARIOS:
    _LARGE_PARAMS.append(pytest.param(
        *_SMALL_MODEL, *_s, id=f"{_SMALL_MODEL[0]}-q{_s[0]}-sl{_s[1]}",
    ))

_CUSTOM_PARAMS = []
for _s in CUSTOM_SCENARIOS:
    q_seqlen, q_head_num, kv_head_num, share_len, kv_seqlen, cache_block_size, fixed_tail, dtype = _s
    _CUSTOM_PARAMS.append(pytest.param(
        q_seqlen, q_head_num, kv_head_num, share_len, kv_seqlen, cache_block_size, fixed_tail, dtype,
        id=f"q{q_seqlen}-qh{q_head_num}-kh{kv_head_num}-kv{kv_seqlen}-cb{cache_block_size}-ft{fixed_tail}",
    ))


def _device():
    device = get_torch_device()
    return "cpu" if device == "meta" else device


def _make_block_table(G, max_blocks, num_phys, device):
    table = torch.zeros(G, max_blocks, dtype=torch.int32, device=device)
    for g in range(G):
        n_blks = min(max_blocks, num_phys)
        perm = torch.randperm(num_phys, device=device)[:n_blks].to(torch.int32)
        table[g, :n_blks] = perm
        if n_blks < max_blocks:
            table[g, n_blks:] = perm[-1]
    return table


def _generate_sals_indexer_data(G, seq_lengths, *, dtype=torch.float16, sparsity=4,
                                fixed_tail_count=8, kv_heads=2, cache_block_size=64):
    sparse_ratio = 1.0 / sparsity
    device = _device()
    sbs = SPARSE_BLOCK_SIZE
    max_seqlen = max(seq_lengths) if seq_lengths else 0
    max_bpg = max((max_seqlen + cache_block_size - 1) // cache_block_size, 1)
    num_phys = max_bpg * G + 4

    max_count = max((max_seqlen + sbs - 1) // sbs, 0)
    fixed_tail_count = min(fixed_tail_count, max_count)
    sparse_count = max(1, min(
        int((max_count - fixed_tail_count) * sparse_ratio + 0.5) + fixed_tail_count,
        max_count,
    ))

    query = torch.randn(G, kv_heads, HEAD_DIM, dtype=dtype, device=device)
    key = torch.randn(num_phys, cache_block_size, kv_heads, HEAD_DIM, dtype=dtype, device=device)
    block_table = _make_block_table(G, max_bpg, num_phys, device)
    actual_seq_lengths_key = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
    act_n_counts = torch.tensor(
        [(max(s, 0) + sbs - 1) // sbs for s in seq_lengths],
        dtype=torch.int32, device=device,
    )

    return (
        query, key, block_table, actual_seq_lengths_key, act_n_counts,
        sbs, sparse_ratio, fixed_tail_count, sparse_count,
        "lse", max_seqlen,
    )


_PARAM_NAMES = (
    "model_name,kv_heads,head_dim,"
    "q_seqlen,share_len,kv_seqlen,sparsity,fixed_tail,cache_block_size,dtype"
)


@pytest.mark.parametrize(_PARAM_NAMES, _MODEL_SPEC_PARAMS)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_sals_indexer_perf(
    model_name, kv_heads, head_dim,
    q_seqlen, share_len, kv_seqlen, sparsity, fixed_tail, cache_block_size, dtype,
):
    G = (q_seqlen - MIN_SPARSE_LEN) // share_len
    (query, key, block_table, actual_seq_lengths_key, act_n_counts,
     sparse_block_size, sparse_ratio, fixed_tail_count, sparse_count,
     score_mode, max_seqlen_key) = _generate_sals_indexer_data(
        G=G, seq_lengths=[kv_seqlen] * G,
        dtype=dtype, sparsity=sparsity,
        fixed_tail_count=fixed_tail, kv_heads=kv_heads,
        cache_block_size=cache_block_size,
    )
    indexer = MojoSALSIndexer()
    perf(lambda: indexer(
        query, key, block_table, actual_seq_lengths_key, act_n_counts,
        sparse_block_size, sparse_ratio, fixed_tail_count, sparse_count,
        score_mode, max_seqlen_key,
    ))  # noqa: F821


@pytest.mark.parametrize(_PARAM_NAMES, _LARGE_PARAMS)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_sals_indexer_large_perf(
    model_name, kv_heads, head_dim,
    q_seqlen, share_len, kv_seqlen, sparsity, fixed_tail, cache_block_size, dtype,
):
    G = (q_seqlen - MIN_SPARSE_LEN) // share_len
    (query, key, block_table, actual_seq_lengths_key, act_n_counts,
     sparse_block_size, sparse_ratio, fixed_tail_count, sparse_count,
     score_mode, max_seqlen_key) = _generate_sals_indexer_data(
        G=G, seq_lengths=[kv_seqlen] * G,
        dtype=dtype, sparsity=sparsity,
        fixed_tail_count=fixed_tail, kv_heads=kv_heads,
        cache_block_size=cache_block_size,
    )
    indexer = MojoSALSIndexer()
    perf(lambda: indexer(
        query, key, block_table, actual_seq_lengths_key, act_n_counts,
        sparse_block_size, sparse_ratio, fixed_tail_count, sparse_count,
        score_mode, max_seqlen_key,
    ))  # noqa: F821


@pytest.mark.parametrize(
    "q_seqlen,q_head_num,kv_head_num,share_len,kv_seqlen,cache_block_size,fixed_tail,dtype",
    _CUSTOM_PARAMS,
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_sals_indexer_custom_perf(
    q_seqlen, q_head_num, kv_head_num, share_len, kv_seqlen, cache_block_size, fixed_tail, dtype,
):
    G = (q_seqlen - MIN_SPARSE_LEN) // share_len
    (query, key, block_table, actual_seq_lengths_key, act_n_counts,
     sparse_block_size, sparse_ratio, fixed_tail_count, sparse_count,
     score_mode, max_seqlen_key) = _generate_sals_indexer_data(
        G=G, seq_lengths=[kv_seqlen] * G,
        dtype=dtype, sparsity=4,
        fixed_tail_count=fixed_tail, kv_heads=kv_head_num,
        cache_block_size=cache_block_size,
    )
    indexer = MojoSALSIndexer()
    perf(lambda: indexer(
        query, key, block_table, actual_seq_lengths_key, act_n_counts,
        sparse_block_size, sparse_ratio, fixed_tail_count, sparse_count,
        score_mode, max_seqlen_key,
    ))  # noqa: F821


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
