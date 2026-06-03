# -*- coding: utf-8 -*-
"""Accuracy: MojoSALSIndexer (ttx vs torch reference).

Operator-level test comparing the TTX NPU kernel against the pure-PyTorch
reference for the SALS block-indexer.

pytest mojo_opset/tests/accuracy/operators/test_sals_indexer.py -v
"""
from __future__ import annotations

import pytest
import torch

from mojo_opset import MojoSALSIndexer
from mojo_opset.utils.platform import get_torch_device
from mojo_opset.tests.utils import auto_switch_platform, bypass_not_implemented


HEAD_DIM = 128
SPARSE_BLOCK_SIZE = 64

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

# Custom test cases as specified:
# q_len = 4096, kv_len = 16384/32768/65536, q_head_num = 3/4, kv_head_num = 1, sparse_block_size = 64, fixed_tail_blocks = 4
CUSTOM_SCENARIOS = [
    (4096, 3, 1, 16384, 64, 4, torch.bfloat16),
    (4096, 3, 1, 32768, 64, 4, torch.bfloat16),
    (4096, 3, 1, 65536, 64, 4, torch.bfloat16),
    (4096, 4, 1, 16384, 64, 4, torch.bfloat16),
    (4096, 4, 1, 32768, 64, 4, torch.bfloat16),
    (4096, 4, 1, 65536, 64, 4, torch.bfloat16),
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

# 64k/80k/128k only with lightweight model to avoid OOM - enhanced
LARGE_SCENARIOS = [
    (65536,  256, 512,  4, 16, 64,  torch.float16),
    (81920,  256, 1024, 2, 8,  256, torch.bfloat16),
    (131072, 256, 256,  2, 16, 64,  torch.float16),
]


_SMALL_MODEL = MODEL_SPECS[0]  # new_model_1

_ALL_PARAMS = []
for _m in MODEL_SPECS:
    for _s in SCENARIOS:
        _ALL_PARAMS.append(pytest.param(
            *_m, *_s, id=f"{_m[0]}-q{_s[0]}-sl{_s[1]}",
        ))
for _s in LARGE_SCENARIOS:
    _ALL_PARAMS.append(pytest.param(
        *_SMALL_MODEL, *_s, id=f"{_SMALL_MODEL[0]}-q{_s[0]}-sl{_s[1]}",
    ))

_CUSTOM_PARAMS = []
for _s in CUSTOM_SCENARIOS:
    q_seqlen, q_head_num, kv_head_num, kv_seqlen, cache_block_size, fixed_tail, dtype = _s
    _CUSTOM_PARAMS.append(pytest.param(
        q_seqlen, q_head_num, kv_head_num, kv_seqlen, cache_block_size, fixed_tail, dtype,
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


def _make_inputs(G, seq_lengths, *, dtype=torch.float16, sparsity=4,
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

    return dict(
        query=torch.randn(G, kv_heads, HEAD_DIM, dtype=dtype, device=device),
        key=torch.randn(num_phys, cache_block_size, kv_heads, HEAD_DIM, dtype=dtype, device=device),
        block_table=_make_block_table(G, max_bpg, num_phys, device),
        actual_seq_lengths_key=torch.tensor(seq_lengths, dtype=torch.int32, device=device),
        act_n_counts=torch.tensor(
            [(max(s, 0) + sbs - 1) // sbs for s in seq_lengths],
            dtype=torch.int32, device=device,
        ),
        sparse_block_size=sbs,
        sparse_ratio=sparse_ratio,
        fixed_tail_count=fixed_tail_count,
        sparse_count=sparse_count,
        score_mode="lse",
        max_seqlen_key=max_seqlen,
    )


def _assert_match(torch_out, ttx_out, G, N, fixed_tail_count):
    si_ref, ssl_ref = torch_out
    si_ttx, ssl_ttx = ttx_out

    assert ssl_ref.shape == ssl_ttx.shape
    torch.testing.assert_close(ssl_ref, ssl_ttx, atol=0, rtol=0)
    assert si_ref.shape == si_ttx.shape

    for g in range(G):
        keep = int(ssl_ref[g].item())
        for n in range(N):
            row_ref = si_ref[g, n, :keep].cpu()
            row_ttx = si_ttx[g, n, :keep].cpu()
            topk_n = max(0, keep - fixed_tail_count)

            if topk_n > 0:
                assert torch.equal(
                    torch.sort(row_ref[:topk_n])[0],
                    torch.sort(row_ttx[:topk_n])[0],
                ), (
                    f"[g={g} n={n}] topk: torch={row_ref[:topk_n].tolist()} "
                    f"ttx={row_ttx[:topk_n].tolist()}"
                )

            if keep - topk_n > 0:
                torch.testing.assert_close(row_ref[topk_n:], row_ttx[topk_n:], atol=0, rtol=0)


def test_sals_indexer_short_sequence_keeps_all_visible_blocks_torch():
    kw = _make_inputs(
        G=3,
        seq_lengths=[64, 128, 192],
        dtype=torch.float16,
        sparsity=4,
        fixed_tail_count=8,
        kv_heads=2,
    )
    out_indices, out_lens = MojoSALSIndexer()._registry.get("torch")().forward(**kw)

    torch.testing.assert_close(
        out_lens.cpu(),
        torch.tensor([1, 2, 3], dtype=torch.int32, device="cpu"),
        atol=0,
        rtol=0,
    )
    for group, keep in enumerate([1, 2, 3]):
        expected = torch.arange(keep, dtype=torch.int32, device="cpu")
        for head in range(out_indices.shape[1]):
            torch.testing.assert_close(
                out_indices[group, head, :keep].cpu(),
                expected,
                atol=0,
                rtol=0,
            )


@auto_switch_platform()
@bypass_not_implemented
def test_sals_indexer_short_sequence_keeps_all_visible_blocks_ttx():
    G = 3
    kv_heads = 2
    kw = _make_inputs(
        G=G,
        seq_lengths=[64, 128, 192],
        dtype=torch.float16,
        sparsity=4,
        fixed_tail_count=8,
        kv_heads=kv_heads,
    )
    indexer = MojoSALSIndexer()
    torch_out = indexer._registry.get("torch")().forward(**kw)
    ttx_out = indexer._registry.get("ttx")().forward(**kw)
    _assert_match(torch_out, ttx_out, G, kv_heads, kw["fixed_tail_count"])


@auto_switch_platform()
@bypass_not_implemented
def test_sals_indexer_sparse_block_size_can_differ_from_cache_page_size():
    device = _device()
    G, kv_heads, head_dim = 4, 2, HEAD_DIM
    sparse_block_size = 64
    cache_page_size = 32
    sparsity = 2
    sparse_ratio = 1.0 / sparsity
    max_seqlen = 1024
    max_sparse_blocks = (max_seqlen + sparse_block_size - 1) // sparse_block_size
    max_pages = (max_seqlen + cache_page_size - 1) // cache_page_size
    num_phys = G * max_pages + 8

    query = torch.randn(G, kv_heads, head_dim, dtype=torch.float16, device=device)
    key = torch.randn(num_phys, cache_page_size, kv_heads, head_dim, dtype=torch.float16, device=device)
    block_table = _make_block_table(G, max_pages, num_phys, device)
    seq_lens = torch.tensor([512, 640, 768, 1024], dtype=torch.int32, device=device)
    act_n_counts = torch.div(seq_lens + sparse_block_size - 1, sparse_block_size, rounding_mode="floor")
    fixed_tail_count = 8
    sparse_count = min(
        max_sparse_blocks,
        max(1, int((max_sparse_blocks - fixed_tail_count) * sparse_ratio + 0.5) + fixed_tail_count),
    )
    kw = dict(
        query=query,
        key=key,
        block_table=block_table,
        actual_seq_lengths_key=seq_lens,
        act_n_counts=act_n_counts,
        sparse_block_size=sparse_block_size,
        sparse_ratio=sparse_ratio,
        fixed_tail_count=fixed_tail_count,
        sparse_count=sparse_count,
        score_mode="lse",
        max_seqlen_key=max_seqlen,
    )

    indexer = MojoSALSIndexer()
    torch_out = indexer._registry.get("torch")().forward(**kw)
    ttx_out = indexer._registry.get("ttx")().forward(**kw)
    _assert_match(torch_out, ttx_out, G, kv_heads, fixed_tail_count)


@pytest.mark.parametrize(
    "model_name,kv_heads,head_dim,"
    "q_seqlen,share_len,kv_seqlen,sparsity,fixed_tail,cache_block_size,dtype",
    _ALL_PARAMS,
)
@auto_switch_platform()
@bypass_not_implemented
def test_sals_indexer_model_specs(
    model_name, kv_heads, head_dim,
    q_seqlen, share_len, kv_seqlen, sparsity, fixed_tail, cache_block_size, dtype,
):
    G = q_seqlen // share_len
    kw = _make_inputs(
        G=G, seq_lengths=[kv_seqlen] * G,
        dtype=dtype, sparsity=sparsity,
        fixed_tail_count=fixed_tail, kv_heads=kv_heads,
        cache_block_size=cache_block_size,
    )
    indexer = MojoSALSIndexer()
    torch_cls = indexer._registry.get("torch")
    ttx_cls = indexer._registry.get("ttx")
    torch_out = torch_cls().forward(**kw)
    ttx_out = ttx_cls().forward(**kw)
    _assert_match(torch_out, ttx_out, G, kv_heads, kw["fixed_tail_count"])


@pytest.mark.parametrize(
    "q_seqlen,q_head_num,kv_head_num,kv_seqlen,cache_block_size,fixed_tail,dtype",
    _CUSTOM_PARAMS,
)
@auto_switch_platform()
@bypass_not_implemented
def test_sals_indexer_custom(
    q_seqlen, q_head_num, kv_head_num, kv_seqlen, cache_block_size, fixed_tail, dtype,
):
    G = q_head_num
    sparse_ratio = 0.25
    sbs = SPARSE_BLOCK_SIZE
    device = _device()
    
    max_seqlen = kv_seqlen
    max_bpg = max((max_seqlen + cache_block_size - 1) // cache_block_size, 1)
    num_phys = max_bpg * G + 4
    
    max_count = max((max_seqlen + sbs - 1) // sbs, 0)
    fixed_tail_count = min(fixed_tail, max_count)
    sparse_count = max(1, min(
        int((max_count - fixed_tail_count) * sparse_ratio + 0.5) + fixed_tail_count,
        max_count,
    ))
    
    query = torch.randn(G, kv_head_num, HEAD_DIM, dtype=dtype, device=device)
    key = torch.randn(num_phys, cache_block_size, kv_head_num, HEAD_DIM, dtype=dtype, device=device)
    
    block_table = torch.zeros(G, max_bpg, dtype=torch.int32, device=device)
    for g in range(G):
        n_blks = min(max_bpg, num_phys)
        perm = torch.randperm(num_phys, device=device)[:n_blks].to(torch.int32)
        block_table[g, :n_blks] = perm
        if n_blks < max_bpg:
            block_table[g, n_blks:] = perm[-1]
    
    actual_seq_lengths_key = torch.tensor([kv_seqlen] * G, dtype=torch.int32, device=device)
    act_n_counts = torch.tensor(
        [(max(kv_seqlen, 0) + sbs - 1) // sbs] * G,
        dtype=torch.int32, device=device,
    )
    
    kw = dict(
        query=query,
        key=key,
        block_table=block_table,
        actual_seq_lengths_key=actual_seq_lengths_key,
        act_n_counts=act_n_counts,
        sparse_block_size=sbs,
        sparse_ratio=sparse_ratio,
        fixed_tail_count=fixed_tail_count,
        sparse_count=sparse_count,
        score_mode="lse",
        max_seqlen_key=max_seqlen,
    )
    
    indexer = MojoSALSIndexer()
    torch_out = indexer._registry.get("torch")().forward(**kw)
    ttx_out = indexer._registry.get("ttx")().forward(**kw)
    _assert_match(torch_out, ttx_out, G, kv_head_num, fixed_tail_count)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
