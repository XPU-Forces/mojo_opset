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
SPARSE_BLOCK_SIZE = 16

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

# (q_seqlen, share_len, kv_seqlen, sparse_ratio, fixed_tail, dtype)
# Minimal set covering all required parameter values:
#   share_len ∈ {128, 256}
#   kv_seqlen ∈ {256, 512, 1024, 2048}
#   sparse_ratio ∈ {0.25, 0.5}  (sparsity 4, 2)
#   fixed_tail ∈ {32, 64}
#   dtype ∈ {fp16, bf16}
#   q_seqlen ∈ {8k, 16k, 32k, 64k, 128k}
SCENARIOS = [
    (8192,   128, 1024, 0.25, 32, torch.float16),
    (8192,   256, 2048, 0.5,  64, torch.bfloat16),
    (16384,  128, 512,  0.25, 64, torch.float16),
    (16384,  256, 256,  0.5,  32, torch.bfloat16),
    (32768,  256, 2048, 0.25, 32, torch.float16),
    (65536,  256, 512,  0.25, 32, torch.float16),
    (131072, 256, 256,  0.5,  64, torch.float16),
]


def _device():
    return get_torch_device()


def _make_block_table(G, max_blocks, num_phys, device):
    table = torch.zeros(G, max_blocks, dtype=torch.int32, device=device)
    for g in range(G):
        n_blks = min(max_blocks, num_phys)
        perm = torch.randperm(num_phys, device=device)[:n_blks].to(torch.int32)
        table[g, :n_blks] = perm
        if n_blks < max_blocks:
            table[g, n_blks:] = perm[-1]
    return table


def _make_inputs(G, seq_lengths, *, dtype=torch.float16, sparse_ratio=0.25,
                 fixed_tail_count=32, kv_heads=2):
    device = _device()
    sbs = SPARSE_BLOCK_SIZE
    max_seqlen = max(seq_lengths) if seq_lengths else 0
    max_bpg = max((max_seqlen + sbs - 1) // sbs, 1)
    num_phys = max_bpg * G + 4

    max_count = max((max_seqlen + sbs - 1) // sbs, 0)
    fixed_tail_count = min(fixed_tail_count, max_count)
    sparse_count = max(1, min(
        int((max_count - fixed_tail_count) * sparse_ratio + 0.5) + fixed_tail_count,
        max_count,
    ))

    return dict(
        query=torch.randn(G, kv_heads, HEAD_DIM, dtype=dtype, device=device),
        key=torch.randn(num_phys, sbs, kv_heads, HEAD_DIM, dtype=dtype, device=device),
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


@pytest.mark.parametrize("model_name,kv_heads,head_dim", MODEL_SPECS)
@pytest.mark.parametrize(
    "q_seqlen,share_len,kv_seqlen,sparse_ratio,fixed_tail,dtype", SCENARIOS,
)
@auto_switch_platform()
@bypass_not_implemented
def test_sals_indexer_model_specs(
    model_name, kv_heads, head_dim,
    q_seqlen, share_len, kv_seqlen, sparse_ratio, fixed_tail, dtype,
):
    G = q_seqlen // share_len
    kw = _make_inputs(
        G=G, seq_lengths=[kv_seqlen] * G,
        dtype=dtype, sparse_ratio=sparse_ratio,
        fixed_tail_count=fixed_tail, kv_heads=kv_heads,
    )
    indexer = MojoSALSIndexer()
    torch_cls = indexer._registry.get("torch")
    ttx_cls = indexer._registry.get("ttx")
    torch_out = torch_cls().forward(**kw)
    ttx_out = ttx_cls().forward(**kw)
    _assert_match(torch_out, ttx_out, G, kv_heads, kw["fixed_tail_count"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
