# -*- coding: utf-8 -*-
"""Performance: MojoSALSSFA operator-level benchmark.

pytest mojo_opset/tests/perf/test_sals_sfa.py -v
"""
from __future__ import annotations

import pytest
import torch

from mojo_opset import MojoSALSSFA
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


SPARSE_BLOCK_SIZE = 64
HEAD_DIM = 128
DEFAULT_SPARSE_RATIO = 0.25
DEFAULT_FIXED_TAIL = 32

MODEL_SPECS = [
    ("new_model_1", 128, 8, 2),
    ("new_model_2", 128, 8, 4),
    ("new_model_3", 128, 16, 4),
    ("new_model_4", 128, 32, 8),
    ("new_model_5", 128, 16, 8),
    ("new_model_6", 128, 32, 16),
    ("M9-23B", 128, 80, 8),
    ("M8-14B", 128, 64, 8),
]

# (q_seqlen, share_len, base_kv, sparsity)
PREFILL_SCENARIOS = [
    (8192,  128, 1024, 4),
    (8192,  256, 0,    4),
    (16384, 256, 0,    4),
    (32768, 256, 0,    4),
]


def _compute_K(total_kv_len, sparse_ratio=DEFAULT_SPARSE_RATIO, fixed_tail=DEFAULT_FIXED_TAIL):
    num_blocks = (total_kv_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    ft = min(fixed_tail, num_blocks)
    sort_blocks = max(num_blocks - ft, 0)
    topk = max(1, int(sort_blocks * sparse_ratio + 0.5)) if sort_blocks > 0 else 0
    return min(topk + ft, num_blocks)


def _generate_sfa_data(num_query_heads, num_kv_heads, head_dim,
                       q_seqlen, base_kv, G, sparse_ratio):
    device = get_torch_device()
    dtype = torch.float16
    sbs = SPARSE_BLOCK_SIZE
    total_kv = base_kv + q_seqlen
    K = _compute_K(total_kv, sparse_ratio=sparse_ratio)
    softmax_scale = 1.0 / (head_dim ** 0.5)

    T = q_seqlen
    cumsum_q = [0, q_seqlen]

    max_blocks_needed = (total_kv + sbs - 1) // sbs
    table_len = max_blocks_needed
    num_blocks = table_len + 4

    k_cache = torch.randn(num_blocks, sbs, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_cache = torch.randn(num_blocks, sbs, num_kv_heads, head_dim, dtype=dtype, device=device)
    block_tables = torch.arange(0, table_len, dtype=torch.int32, device=device).unsqueeze(0)

    q = torch.randn(T, num_query_heads, head_dim, dtype=dtype, device=device)

    group_qid = torch.zeros(G, dtype=torch.int32, device=device)
    group_q_start = torch.zeros(G, dtype=torch.int32, device=device)
    group_q_len_t = torch.zeros(G, dtype=torch.int32, device=device)
    seq_len_flat = torch.zeros(G, dtype=torch.int32, device=device)
    indices_flat = torch.zeros(G, num_kv_heads, K, dtype=torch.int32, device=device)

    chunk_size = q_seqlen // G if G > 0 else 0
    fixed_tail = DEFAULT_FIXED_TAIL
    for i in range(G):
        group_qid[i] = 0
        group_q_start[i] = i * chunk_size
        group_q_len_t[i] = chunk_size if i < G - 1 else (q_seqlen - i * chunk_size)

        ft, sort_n, topk, num_selected = 0, 0, 0, 0
        if max_blocks_needed > 0:
            ft = min(fixed_tail, max_blocks_needed)
            sort_n = max(max_blocks_needed - ft, 0)
            topk = max(1, int(sort_n * sparse_ratio + 0.5)) if sort_n > 0 else 0
            num_selected = min(topk + ft, max_blocks_needed, K)
        if max_blocks_needed > 0 and num_selected > 0:
            if sort_n > 0 and num_selected > ft:
                n_topk = min(topk, num_selected - ft, sort_n)
            else:
                n_topk = 0
            n_recent = num_selected - n_topk

            # topk part: random selection from sortable region [0, sort_n)
            if n_topk > 0:
                topk_perm = torch.randperm(sort_n, device=device, dtype=torch.int32)[:n_topk]
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

    base_kv_len = torch.tensor([base_kv], dtype=torch.int32, device=device)
    cumsum_q_len = torch.tensor(cumsum_q, dtype=torch.int32, device=device)

    return (
        q, k_cache, v_cache,
        None, None,
        block_tables, indices_flat, seq_len_flat,
        group_qid, group_q_start, group_q_len_t,
        cumsum_q_len, base_kv_len, None,
        softmax_scale,
        num_kv_heads, num_query_heads, head_dim, sbs,
    )


@pytest.mark.parametrize("model_name,head_dim,num_query_heads,num_kv_heads", MODEL_SPECS)
@pytest.mark.parametrize("q_seqlen,share_len,base_kv,sparsity", PREFILL_SCENARIOS)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_sals_sfa_perf(model_name, head_dim, num_query_heads, num_kv_heads,
                       q_seqlen, share_len, base_kv, sparsity):
    G = q_seqlen // share_len
    sr = 1.0 / sparsity
    (q, k_cache, v_cache, k_scales, v_scales,
     block_tables, indices_flat, seq_len_flat,
     group_qid, group_q_start, group_q_len,
     cumsum_q_len, base_kv_len, group_use_dense,
     softmax_scale,
     num_kv_heads_out, num_query_heads_out, head_dim_out,
     sparse_block_size) = _generate_sfa_data(
        num_query_heads, num_kv_heads, head_dim, q_seqlen, base_kv, G, sr)
    op = MojoSALSSFA()
    perf(lambda: op(  # noqa: F821
        q, k_cache, v_cache, k_scales, v_scales,
        block_tables, indices_flat, seq_len_flat,
        group_qid, group_q_start, group_q_len,
        cumsum_q_len, base_kv_len, group_use_dense,
        softmax_scale,
        num_kv_heads_out, num_query_heads_out, head_dim_out, sparse_block_size,
    ))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
