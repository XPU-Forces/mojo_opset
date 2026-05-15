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
KV_HEADS = 2
SPARSE_BLOCK_SIZE = 16
DEFAULT_SPARSE_RATIO = 0.25
DEFAULT_FIXED_TAIL = 8

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


def _make_inputs(
    G, seq_lengths, *,
    dtype=torch.float16,
    sparse_block_size=SPARSE_BLOCK_SIZE,
    sparse_ratio=DEFAULT_SPARSE_RATIO,
    fixed_tail_count=DEFAULT_FIXED_TAIL,
    kv_heads=KV_HEADS,
):
    """Build the kwargs dict that matches MojoSALSIndexer.forward() signature."""
    device = _device()
    max_seqlen = max(seq_lengths) if seq_lengths else 0
    sbs = sparse_block_size
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
    """Compare torch vs ttx outputs for the SALS indexer."""
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


@pytest.fixture
def assert_ttx_vs_torch():
    def _fn(kwargs):
        indexer = MojoSALSIndexer()
        torch_cls = indexer._registry.get("torch")
        ttx_cls = indexer._registry.get("ttx")

        torch_out = torch_cls().forward(**kwargs)
        ttx_out = ttx_cls().forward(**kwargs)
        G, N = kwargs["query"].shape[0], kwargs["query"].shape[1]
        _assert_match(torch_out, ttx_out, G, N, kwargs["fixed_tail_count"])
    return _fn


# ===== Basic shape tests =====
@pytest.mark.parametrize("G,seqs", [
    (8, [512] * 8),
    (4, [2048] * 4),
    (1, [512]),
    (3, [512, 0, 256]),
    (6, [512, 64, 1024, 128, 256, 32]),
    (4, [64, 96, 128, 160]),
])
@auto_switch_platform()
@bypass_not_implemented
def test_basic(G, seqs, assert_ttx_vs_torch):
    assert_ttx_vs_torch(_make_inputs(G=G, seq_lengths=seqs))


@pytest.mark.parametrize("seqlen", [512, 1024])
@auto_switch_platform()
@bypass_not_implemented
def test_long(seqlen, assert_ttx_vs_torch):
    assert_ttx_vs_torch(_make_inputs(G=4, seq_lengths=[seqlen] * 4))


# ===== Edge cases =====
@pytest.mark.parametrize("G,seqs", [
    (2, [192, 193]),
    (2, [208, 208]),
    (2, [8192, 4096]),
    (8, [16, 32, 48, 64, 80, 96, 112, 128]),
])
@auto_switch_platform()
@bypass_not_implemented
def test_edge(G, seqs, assert_ttx_vs_torch):
    assert_ttx_vs_torch(_make_inputs(G=G, seq_lengths=seqs))


@auto_switch_platform()
@bypass_not_implemented
def test_large_G(assert_ttx_vs_torch):
    seqs = [256 + (i % 3) * 128 for i in range(32)]
    assert_ttx_vs_torch(_make_inputs(G=32, seq_lengths=seqs))


# ===== Dtype tests =====
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@auto_switch_platform()
@bypass_not_implemented
def test_dtype(dtype, assert_ttx_vs_torch):
    assert_ttx_vs_torch(_make_inputs(G=4, seq_lengths=[512] * 4, dtype=dtype))


# ===== Block table variants =====
@auto_switch_platform()
@bypass_not_implemented
def test_block_table_identity(assert_ttx_vs_torch):
    kw = _make_inputs(G=4, seq_lengths=[512] * 4)
    mb = (512 + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    kw["block_table"] = (
        torch.arange(kw["key"].shape[0], device=_device(), dtype=torch.int32)
        [:mb].unsqueeze(0).expand(4, -1).contiguous()
    )
    assert_ttx_vs_torch(kw)


@auto_switch_platform()
@bypass_not_implemented
def test_block_table_reverse(assert_ttx_vs_torch):
    kw = _make_inputs(G=4, seq_lengths=[512] * 4)
    mb = (512 + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    np_ = kw["key"].shape[0]
    rev = torch.arange(np_ - 1, max(np_ - mb - 1, -1), -1,
                        device=_device(), dtype=torch.int32)[:mb]
    kw["block_table"] = rev.unsqueeze(0).expand(4, -1).contiguous()
    assert_ttx_vs_torch(kw)


# ===== Determinism =====
@auto_switch_platform()
@bypass_not_implemented
def test_determinism():
    kw = _make_inputs(G=4, seq_lengths=[512] * 4)
    indexer = MojoSALSIndexer()
    torch_cls = indexer._registry.get("torch")
    ttx_cls = indexer._registry.get("ttx")

    for cls in [torch_cls, ttx_cls]:
        op = cls()
        a = op.forward(**kw)
        b = op.forward(**kw)
        torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
        torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)


# ===== Critical model spec tests (MUST INCLUDE) =====
@pytest.mark.parametrize("model_name,kv_heads,head_dim", MODEL_SPECS)
@auto_switch_platform()
@bypass_not_implemented
def test_sals_indexer_model_specs(model_name, kv_heads, head_dim, assert_ttx_vs_torch):
    assert_ttx_vs_torch(_make_inputs(G=4, seq_lengths=[512] * 4, kv_heads=kv_heads))
    assert_ttx_vs_torch(_make_inputs(G=2, seq_lengths=[1024, 2048], kv_heads=kv_heads))


if __name__ == "__main__":
    pytest.main([__file__ + "::test_sals_indexer_model_specs"])
