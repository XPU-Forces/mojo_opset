# -*- coding: utf-8 -*-
"""Accuracy: MojoSALSSFA (ttx vs torch reference).

Operator-level test comparing the TTX NPU kernel against the pure-PyTorch
reference for the SALS Sparse Flash Attention operator.

pytest mojo_opset/tests/accuracy/operators/test_sals_sfa.py -v
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mojo_opset import MojoSALSSFA
from mojo_opset.utils.platform import get_torch_device
from mojo_opset.tests.utils import auto_switch_platform, bypass_not_implemented


SPARSE_BLOCK_SIZE = 64
HEAD_DIM = 128
MIN_SPARSE_LEN = 512

MODEL_SPECS = [
    # (name, head_dim, num_query_heads, num_kv_heads)
    ("new_model_1", 128, 8, 2),
    ("new_model_2", 128, 8, 4),
    ("new_model_3", 128, 16, 4),
    ("new_model_4", 128, 32, 8),
    ("new_model_5", 128, 16, 8),
    ("new_model_6", 128, 32, 16),
    ("M9-23B", 128, 80, 8),
    ("M8-14B", 128, 64, 8),
]

# (q_seqlen, share_len, base_kv, sparsity, fixed_tail, cache_block_size, dtype)
# Covers all required parameter values:
#   share_len ∈ {128, 256}
#   sparsity ∈ {2, 4}  (sparse_ratio 0.5, 0.25)
#   fixed_tail ∈ {8, 16}
#   cache_block_size ∈ {64, 256}
#   dtype ∈ {fp16, bf16}
#   base_kv ∈ {0, 1024, 2048}
#   q_seqlen ∈ {8k, 10k, 16k, 24k, 32k}
SCENARIOS = [
    (8193,   128, 3,    4, 8,  64,  torch.float16),
    (8192,   128, 0,    4, 8,  64,  torch.float16),
    (8192,   256, 2048, 2, 16, 256, torch.bfloat16),
    (10240,  256, 0,    4, 8,  256, torch.float16),
    (16384,  128, 1024, 4, 16, 64,  torch.bfloat16),
    (16384,  256, 0,    2, 8,  64,  torch.float16),
    (24576,  128, 2048, 2, 16, 256, torch.bfloat16),
    (32771,  256, 1027, 4, 8,  256, torch.float16),
    (32768,  256, 0,    4, 8,  256, torch.float16),
]

# 64k/80k/128k only with lightweight model to avoid OOM - enhanced
LARGE_SCENARIOS = [
    (65536,  256, 0,    4, 16, 64,  torch.float16),
    (81920,  256, 1024, 2, 8,  256, torch.bfloat16),
    (131072, 256, 0,    2, 16, 64,  torch.float16),
]

_SMALL_MODEL = MODEL_SPECS[0]  # new_model_1 (qH=8, kvH=2)

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


def _compute_K(total_kv_len, sparse_block_size=SPARSE_BLOCK_SIZE,
               sparse_ratio=0.25, fixed_tail=8):
    num_blocks = (total_kv_len + sparse_block_size - 1) // sparse_block_size
    ft = min(fixed_tail, num_blocks)
    sort_blocks = max(num_blocks - ft, 0)
    topk = max(1, int(sort_blocks * sparse_ratio + 0.5)) if sort_blocks > 0 else 0
    return min(topk + ft, num_blocks)


def _device():
    device = get_torch_device()
    return "cpu" if device == "meta" else device


def _make_sfa_inputs(
    *,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int = HEAD_DIM,
    sparse_block_size: int = SPARSE_BLOCK_SIZE,
    q_seqlen: int,
    base_kv: int,
    G: int,
    sparse_ratio: float = 0.25,
    fixed_tail: int = 8,
    cache_block_size: int = 64,
    dtype: torch.dtype = torch.float16,
    head_reserve: int = 0,
    tail_reserve: int = 0,
) -> dict:
    device = _device()
    B_req = 1
    q_lens = [q_seqlen]
    base_kv_lens = [base_kv]

    total_kv = base_kv + q_seqlen
    K = _compute_K(total_kv, sparse_block_size, sparse_ratio, fixed_tail)

    T = q_seqlen
    softmax_scale = 1.0 / (head_dim ** 0.5)
    cumsum_q = [0, q_seqlen]

    max_blocks_needed = (total_kv + cache_block_size - 1) // cache_block_size
    table_len = max_blocks_needed
    num_blocks = table_len + 4
    num_sparse_blocks = (total_kv + sparse_block_size - 1) // sparse_block_size

    k_cache = torch.randn(num_blocks, cache_block_size, num_kv_heads, head_dim,
                           dtype=dtype, device=device)
    v_cache = torch.randn(num_blocks, cache_block_size, num_kv_heads, head_dim,
                           dtype=dtype, device=device)
    block_tables = torch.arange(0, table_len, dtype=torch.int32, device=device).unsqueeze(0)

    q = torch.randn(T, num_query_heads, head_dim, dtype=dtype, device=device)

    group_qid = torch.zeros(G, dtype=torch.int32, device=device)
    group_q_start = torch.zeros(G, dtype=torch.int32, device=device)
    group_q_len_t = torch.zeros(G, dtype=torch.int32, device=device)
    seq_len_flat = torch.zeros(G, dtype=torch.int32, device=device)
    indices_flat = torch.zeros(G, num_kv_heads, K, dtype=torch.int32, device=device)

    chunk_size = (q_seqlen - head_reserve - tail_reserve) // G if G > 0 else 0
    for i in range(G):
        group_qid[i] = 0
        group_q_start[i] = head_reserve + i * chunk_size
        group_q_len_t[i] = chunk_size if i < G - 1 else (q_seqlen - head_reserve - tail_reserve - i * chunk_size)

        ft, sort_n, topk, num_selected = 0, 0, 0, 0
        if num_sparse_blocks > 0:
            ft = min(fixed_tail, num_sparse_blocks)
            sort_n = max(num_sparse_blocks - ft, 0)
            topk = max(1, int(sort_n * sparse_ratio + 0.5)) if sort_n > 0 else 0
            num_selected = min(topk + ft, num_sparse_blocks, K)

        if num_sparse_blocks > 0 and num_selected > 0:
            if sort_n > 0 and num_selected > ft:
                n_topk = min(topk, num_selected - ft, sort_n)
            else:
                n_topk = 0
            n_recent = num_selected - n_topk

            # topk part: random selection from sortable region [0, sort_n)
            if n_topk > 0:
                topk_perm = torch.randperm(
                    sort_n, device=device, dtype=torch.int32
                )[:n_topk]
            else:
                topk_perm = torch.empty(0, dtype=torch.int32, device=device)

            # recent part: contiguous blocks at the end [sort_n, sort_n + n_recent)
            if n_recent > 0:
                recent_idx = torch.arange(
                    sort_n, sort_n + n_recent, device=device, dtype=torch.int32
                )
            else:
                recent_idx = torch.empty(0, dtype=torch.int32, device=device)

            idx = torch.cat([topk_perm, recent_idx])
            pad_val = int(idx[-1].item()) if idx.numel() > 0 else 0

            for h in range(num_kv_heads):
                indices_flat[i, h, :num_selected] = idx
                if num_selected < K:
                    indices_flat[i, h, num_selected:] = pad_val
        seq_len_flat[i] = num_selected

    # group_use_dense: [G], per-group dense flag (1=dense, 0=sparse)
    # In production, determined by layer-specific q-position ratio config.
    # For testing: mark first group as dense to exercise the dense code path.
    group_use_dense = torch.zeros(G, dtype=torch.int32, device=device)
    if G > 0:
        group_use_dense[0] = 1

    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k_scales=None,
        v_scales=None,
        block_tables=block_tables,
        indices_flat=indices_flat,
        seq_len_flat=seq_len_flat,
        group_qid=group_qid,
        group_q_start=group_q_start,
        group_q_len=group_q_len_t,
        cumsum_q_len=torch.tensor(cumsum_q, dtype=torch.int32, device=device),
        base_kv_len=torch.tensor(base_kv_lens, dtype=torch.int32, device=device),
        group_use_dense=group_use_dense,
        softmax_scale=softmax_scale,
        num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads,
        head_dim=head_dim,
        sparse_block_size=sparse_block_size,
    )


def _assert_match(torch_out: torch.Tensor, ttx_out: torch.Tensor, *, atol: float = 0.1):
    assert torch_out.shape == ttx_out.shape, (
        f"Shape mismatch: torch {torch_out.shape} vs ttx {ttx_out.shape}"
    )
    nonzero_mask = torch_out.abs() > 0
    if nonzero_mask.any():
        torch_flat = torch_out[nonzero_mask].float()
        ttx_flat = ttx_out[nonzero_mask].float()
        cos_sim = F.cosine_similarity(
            torch_flat.unsqueeze(0), ttx_flat.unsqueeze(0),
        ).item()
        assert cos_sim > 0.99, f"Cosine similarity {cos_sim:.6f} < 0.99"
    max_err = (torch_out.float() - ttx_out.float()).abs().max().item()
    assert max_err < atol, f"Max abs error {max_err:.6f} >= {atol}"


@auto_switch_platform()
@bypass_not_implemented
def test_sfa_ttx_uses_group_level_union_across_kv_heads():
    kw = _make_sfa_inputs(
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=HEAD_DIM,
        q_seqlen=1024,
        base_kv=0,
        G=2,
        sparse_ratio=0.25,
        fixed_tail=8,
        dtype=torch.float16,
    )
    device = kw["q"].device
    kw["indices_flat"] = torch.full((2, 2, 3), -1, dtype=torch.int32, device=device)
    kw["indices_flat"][0, 0, :2] = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kw["indices_flat"][0, 1, :2] = torch.tensor([2, 3], dtype=torch.int32, device=device)
    kw["indices_flat"][1, 0, :2] = torch.tensor([1, 2], dtype=torch.int32, device=device)
    kw["indices_flat"][1, 1, :2] = torch.tensor([0, 3], dtype=torch.int32, device=device)
    kw["seq_len_flat"] = torch.tensor([2, 2], dtype=torch.int32, device=device)

    op = MojoSALSSFA()
    torch_out = op._registry.get("torch")().forward(**kw)
    ttx_out = op._registry.get("ttx")().forward(**kw)
    _assert_match(torch_out, ttx_out)


@auto_switch_platform()
@bypass_not_implemented
def test_sfa_ttx_masks_partial_last_sparse_block():
    kw = _make_sfa_inputs(
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=HEAD_DIM,
        q_seqlen=1024,
        base_kv=0,
        G=1,
        sparse_ratio=0.25,
        fixed_tail=8,
        dtype=torch.float16,
    )
    device = kw["q"].device
    kw["indices_flat"] = torch.tensor([[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]], dtype=torch.int32, device=device)
    kw["seq_len_flat"] = torch.tensor([5], dtype=torch.int32, device=device)

    op = MojoSALSSFA()
    torch_out = op._registry.get("torch")().forward(**kw)
    ttx_out = op._registry.get("ttx")().forward(**kw)
    _assert_match(torch_out, ttx_out)


@pytest.mark.parametrize(
    "model_name,head_dim,num_query_heads,num_kv_heads,"
    "q_seqlen,share_len,base_kv,sparsity,fixed_tail,cache_block_size,dtype",
    _ALL_PARAMS,
)
@auto_switch_platform()
@bypass_not_implemented
def test_sfa_model_specs(
    model_name, head_dim, num_query_heads, num_kv_heads,
    q_seqlen, share_len, base_kv, sparsity, fixed_tail, cache_block_size, dtype,
):
    G = (q_seqlen - MIN_SPARSE_LEN) // share_len
    sr = 1.0 / sparsity
    kw = _make_sfa_inputs(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_seqlen=q_seqlen,
        base_kv=base_kv,
        G=G,
        sparse_ratio=sr,
        fixed_tail=fixed_tail,
        cache_block_size=cache_block_size,
        dtype=dtype,
        head_reserve=MIN_SPARSE_LEN // 2,
        tail_reserve=MIN_SPARSE_LEN // 2,
    )
    op = MojoSALSSFA()
    torch_cls = op._registry.get("torch")
    ttx_cls = op._registry.get("ttx")
    torch_out = torch_cls().forward(**kw)
    ttx_out = ttx_cls().forward(**kw)
    _assert_match(torch_out, ttx_out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
