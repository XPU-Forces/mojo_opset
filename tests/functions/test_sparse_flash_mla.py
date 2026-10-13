from __future__ import annotations

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_close


def generate_smla_data(
    s1: int,
    s2: int,
    n1: int,
    d: int,
    dtype: torch.dtype,
    page: int = 128,
    seed: int = 0,
    device=None,
):
    """Random paged-KV SMLA inputs with a shuffled block table (mirrors real page allocation)."""
    torch.manual_seed(seed)
    pages = (s2 + page - 1) // page
    q = torch.rand(s1, n1, d, dtype=dtype, device=device) * 4 - 2
    kv = torch.randn(pages, page, 1, d, dtype=dtype, device=device) * 0.5
    block_table = torch.randperm(pages, device=device).to(torch.int32).reshape(1, -1)
    cu_seqlens_q = torch.tensor([0, s1], dtype=torch.int32, device=device)
    seqused_ori_kv = torch.tensor([s2], dtype=torch.int32, device=device)
    return q, kv, block_table, cu_seqlens_q, seqused_ori_kv


_SMLA_CASES = [
    # --- basic coverage ---
    (1, 1024, 64, 512, 127, 0),  # single-token decode (wheel demo case)
    (4, 2048, 64, 512, 127, 0),  # multi-token queries, KV spans 16 pages
    (1, 1024, 128, 512, 127, 0),  # 128 query heads
    (8, 16384, 64, 512, 127, 0),  # large paged KV (128 pages)
    (2, 4096, 64, 512, 255, 0),  # wider SWA window (256, crosses page)
    # --- window variants (bidirectional / narrow / near-full) ---
    (1, 1024, 64, 512, 63, 63),  # bidirectional window (64 left + 64 right)
    (4, 2048, 64, 512, 127, 63),  # bidirectional window, multi-token
    (8, 16384, 64, 512, 255, 63),  # wide bidirectional window
    (2, 4096, 64, 512, 63, 0),  # narrow window (64)
    (1, 512, 64, 512, 511, 0),  # near-full window
    # --- KV length variants (short / non-page-aligned / huge) ---
    (1, 512, 64, 512, 127, 0),  # short KV (4 pages)
    (4, 512, 64, 512, 127, 0),  # short KV, multi-token
    (1, 1000, 64, 512, 127, 0),  # non-page-aligned KV (1000, partial last page)
    (2, 1000, 64, 512, 127, 0),  # non-page-aligned KV, multi-token
    (16, 16384, 64, 512, 127, 0),  # long queries over large KV
    (32, 8192, 64, 512, 127, 0),  # very long query (prefill-like)
    # --- 128-head variants ---
    (4, 2048, 128, 512, 127, 0),  # 128 heads, multi-token
    (8, 16384, 128, 512, 127, 0),  # 128 heads, large paged KV
    (16, 8192, 128, 512, 127, 0),  # 128 heads, long query
    (2, 4096, 128, 512, 255, 0),  # 128 heads, wider window
    # --- boundary combos ---
    (1, 1000, 128, 512, 63, 63),  # 128 heads + partial page + bidirectional
    (32, 4096, 64, 512, 63, 63),  # long query + bidirectional window
    (1, 128, 64, 512, 127, 0),  # single page KV, window wider than KV (clamped)
    (2, 256, 64, 512, 127, 0),  # two-page KV
    (4, 1000, 128, 512, 255, 63),  # mixed: partial page + wide bidirectional
]


def generate_smla_cmp_data(
    b,
    s1,
    s2,
    n1,
    d,
    dtype,
    layer,
    ratio,
    t3,
    k=0,
    layout="PA",
    ori_page=128,
    cmp_page=64,
    seed=0,
    device=None,
):
    """SMLA inputs with a compressed-KV branch (CSA when k > 0, HCA otherwise).

    PA layout uses shuffled block tables (mirrors real page allocation); TND
    uses dense per-batch contiguous storage with cumulative prefixes.
    """
    torch.manual_seed(seed)
    q = torch.rand(b * s1, n1, d, dtype=dtype, device=device) * 4 - 2
    cu_seqlens_q = torch.tensor([i * s1 for i in range(b + 1)], dtype=torch.int32, device=device)
    seqused_ori_kv = torch.full((b,), s2, dtype=torch.int32, device=device)
    if layout == "PA":
        pages = (s2 + ori_page - 1) // ori_page
        kv = torch.randn(b * pages, ori_page, 1, d, dtype=dtype, device=device) * 0.5
        block_table = torch.stack([torch.randperm(b * pages, device=device)[:pages] for _ in range(b)]).to(torch.int32)
        cu_seqlens_ori_kv = None
    else:
        kv = torch.randn(b * s2, 1, d, dtype=dtype, device=device) * 0.5
        block_table = None
        cu_seqlens_ori_kv = torch.tensor([i * s2 for i in range(b + 1)], dtype=torch.int32, device=device)
    inputs = dict(
        q=q,
        ori_kv=kv,
        ori_block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_ori_kv=seqused_ori_kv,
        layout_kv="PA_BBND" if layout == "PA" else "TND",
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
    )
    if layer == "SWA":
        return inputs, dict(cmp_ratio=1)

    seqused_cmp_kv = torch.full((b,), t3, dtype=torch.int32, device=device)
    cmp_residual_kv = torch.zeros(b, dtype=torch.int32, device=device)
    if layout == "PA":
        cmp_pages = (t3 + cmp_page - 1) // cmp_page
        cmp_kv = torch.randn(b * cmp_pages, cmp_page, 1, d, dtype=dtype, device=device) * 0.5
        cmp_block_table = torch.stack(
            [torch.randperm(b * cmp_pages, device=device)[:cmp_pages] for _ in range(b)]
        ).to(torch.int32)
        cu_seqlens_cmp_kv = None
    else:
        cmp_kv = torch.randn(b * t3, 1, d, dtype=dtype, device=device) * 0.5
        cmp_block_table = None
        cu_seqlens_cmp_kv = torch.tensor([i * t3 for i in range(b + 1)], dtype=torch.int32, device=device)
    cmp_sparse_indices = None
    if layer == "CSA":
        cmp_sparse_indices = torch.stack(
            [torch.randperm(t3, device=device)[:k] for _ in range(b * s1)]
        ).to(torch.int32)
    inputs.update(
        cmp_kv=cmp_kv,
        cmp_block_table=cmp_block_table,
        seqused_cmp_kv=seqused_cmp_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        cmp_sparse_indices=cmp_sparse_indices,
    )
    return inputs, dict(cmp_ratio=ratio)


# (b, s1, s2, n1, win_left, win_right, ratio, t3, k, layout)
_SMLA_CSA_CASES = [
    (1, 64, 1024, 64, 127, 0, 4, 256, 128, "PA"),  # basic C4
    (1, 64, 1024, 64, 127, 0, 4, 256, 128, "TND"),  # basic C4, dense KV
    (2, 32, 2048, 64, 127, 0, 4, 512, 256, "PA"),  # multi-batch
    (2, 32, 2048, 128, 127, 0, 4, 512, 256, "TND"),  # multi-batch + 128 heads
    (1, 64, 512, 64, 127, 0, 4, 128, 128, "PA"),  # k == t3 (every token selected)
    (1, 32, 1000, 64, 63, 63, 4, 250, 64, "PA"),  # bidirectional window + non-divisible lengths
    (1, 64, 2048, 64, 255, 0, 8, 256, 256, "TND"),  # ratio 8
    (1, 64, 1024, 128, 127, 0, 4, 256, 512, "PA"),  # 128 heads, wide topk
]

# (b, s1, s2, n1, win_left, win_right, ratio, t3, layout)
_SMLA_HCA_CASES = [
    (1, 64, 1024, 64, 127, 0, 128, 8, "PA"),  # basic C128
    (1, 64, 1024, 64, 127, 0, 128, 8, "TND"),  # basic C128, dense KV
    (2, 32, 1024, 64, 127, 0, 128, 16, "PA"),  # multi-batch
    (2, 32, 1024, 128, 127, 0, 128, 16, "TND"),  # multi-batch + 128 heads
    (1, 32, 1000, 64, 63, 63, 8, 125, "PA"),  # ratio 8 (revert == s2) + bidirectional
    (1, 64, 2048, 64, 255, 0, 64, 32, "TND"),  # ratio 64
    (1, 64, 1024, 128, 127, 0, 128, 8, "PA"),  # 128 heads
]


def _assert_smla_close(actual, expected, dtype):
    # Original comparison ran in fp32 with these operator-specific limits.
    assert_close(actual, expected, dtype, rtol=1e-2, atol=5e-2)


@pytest.mark.api("functions.sparse_flash_mla_infer")
@pytest.mark.parametrize(
    "s1, s2, n1, d, win_left, win_right",
    _SMLA_CASES,
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.accuracy
def test_sparse_flash_mla(sparse_flash_mla_backend, s1, s2, n1, d, win_left, win_right, dtype):
    implementation, device = sparse_flash_mla_backend
    q, kv, block_table, cu_seqlens_q, seqused_ori_kv = generate_smla_data(s1, s2, n1, d, dtype, device=device)

    kwargs = dict(win_left=win_left, win_right=win_right)
    actual = F.sparse_flash_mla_infer(
        q, kv, block_table, cu_seqlens_q, seqused_ori_kv, implementation=implementation, **kwargs
    )
    expected = F.sparse_flash_mla_infer(
        q, kv, block_table, cu_seqlens_q, seqused_ori_kv, implementation="torch_reference", **kwargs
    )
    _assert_smla_close(actual, expected, dtype)


def _run_cmp_case(sparse_flash_mla_backend, dtype, layer, case):
    implementation, device = sparse_flash_mla_backend
    if layer == "CSA":
        b, s1, s2, n1, win_left, win_right, ratio, t3, k, layout = case
    else:
        b, s1, s2, n1, win_left, win_right, ratio, t3, layout = case
        k = 0
    cmp_page = 64 if ratio <= 8 else 2
    inputs, extra = generate_smla_cmp_data(
        b, s1, s2, n1, 512, dtype, layer, ratio, t3, k=k, layout=layout, cmp_page=cmp_page, device=device
    )

    kwargs = dict(win_left=win_left, win_right=win_right, cmp_mask_mode=3, **extra)
    actual = F.sparse_flash_mla_infer(implementation=implementation, **inputs, **kwargs)
    expected = F.sparse_flash_mla_infer(implementation="torch_reference", **inputs, **kwargs)
    _assert_smla_close(actual, expected, dtype)


@pytest.mark.api("functions.sparse_flash_mla_infer")
@pytest.mark.parametrize(
    "case",
    _SMLA_CSA_CASES,
    ids=[f"b{c[0]}-s1_{c[1]}-s2_{c[2]}-n{c[3]}-w{c[4]}_{c[5]}-r{c[6]}-t3_{c[7]}-k{c[8]}-{c[9]}" for c in _SMLA_CSA_CASES],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.accuracy
def test_sparse_flash_mla_csa(sparse_flash_mla_backend, case, dtype):
    _run_cmp_case(sparse_flash_mla_backend, dtype, "CSA", case)


@pytest.mark.api("functions.sparse_flash_mla_infer")
@pytest.mark.parametrize(
    "case",
    _SMLA_HCA_CASES,
    ids=[f"b{c[0]}-s1_{c[1]}-s2_{c[2]}-n{c[3]}-w{c[4]}_{c[5]}-r{c[6]}-t3_{c[7]}-{c[8]}" for c in _SMLA_HCA_CASES],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.accuracy
def test_sparse_flash_mla_hca(sparse_flash_mla_backend, case, dtype):
    _run_cmp_case(sparse_flash_mla_backend, dtype, "HCA", case)
