from __future__ import annotations

import pytest
import torch

from mojo_opset import functions


# Standard SMLA benchmark cases (cases_pa.csv): PA_BBND paged KV with the csv
# block_size for both ori and cmp storage, shuffled block tables, bf16,
# layout_q=TND, layout_kv=PA_BBND, win=(127, 0). SWA has no compressed KV;
# C4 is CSA (sparse topk over compressed KV, cmp_ratio=4); C128 is HCA (dense
# compressed prefix, cmp_ratio=128). Compressed length t3 = s2 / cmp_ratio.
# (case_id, batch, s1, s2, num_heads_q, layer, cmp_ratio, t3, topk, page)
_SMLA_CASES = [
    ("001_SWA_b1_s18192_s28192_n64", 1, 8192, 8192, 64, "SWA", 1, 0, 0, 256),
    ("002_C4_b1_s18192_s28192_n64", 1, 8192, 8192, 64, "CSA", 4, 2048, 512, 64),
    ("003_C128_b1_s18192_s28192_n64", 1, 8192, 8192, 64, "HCA", 128, 64, 0, 2),
    ("004_SWA_b5_s18192_s265536_n64", 5, 8192, 65536, 64, "SWA", 1, 0, 0, 256),
    ("005_C4_b5_s18192_s265536_n64", 5, 8192, 65536, 64, "CSA", 4, 16384, 512, 64),
    ("006_C128_b5_s18192_s265536_n64", 5, 8192, 65536, 64, "HCA", 128, 512, 0, 2),
    ("007_SWA_b1_s165536_s2524288_n64", 1, 65536, 524288, 64, "SWA", 1, 0, 0, 256),
    ("008_C4_b1_s165536_s2524288_n64", 1, 65536, 524288, 64, "CSA", 4, 131072, 512, 64),
    ("009_C128_b1_s165536_s2524288_n64", 1, 65536, 524288, 64, "HCA", 128, 4096, 0, 2),
    ("010_SWA_b1_s18192_s28192_n128", 1, 8192, 8192, 128, "SWA", 1, 0, 0, 256),
    ("011_C4_b1_s18192_s28192_n128", 1, 8192, 8192, 128, "CSA", 4, 2048, 1024, 64),
    ("012_C128_b1_s18192_s28192_n128", 1, 8192, 8192, 128, "HCA", 128, 64, 0, 2),
    ("013_SWA_b1_s18192_s265536_n128", 1, 8192, 65536, 128, "SWA", 1, 0, 0, 256),
    ("014_C4_b1_s18192_s265536_n128", 1, 8192, 65536, 128, "CSA", 4, 16384, 1024, 64),
    ("015_C128_b1_s18192_s265536_n128", 1, 8192, 65536, 128, "HCA", 128, 512, 0, 2),
    ("016_SWA_b1_s18192_s2524288_n128", 1, 8192, 524288, 128, "SWA", 1, 0, 0, 256),
    ("017_C4_b1_s18192_s2524288_n128", 1, 8192, 524288, 128, "CSA", 4, 131072, 1024, 64),
    ("018_C128_b1_s18192_s2524288_n128", 1, 8192, 524288, 128, "HCA", 128, 4096, 0, 2),
]


def generate_smla_pa_data(s1, s2, n1, b=1, layer="SWA", ratio=1, t3=0, k=0, page=256, d=512, seed=43, device=None):
    """Paged-KV SMLA inputs (PA_BBND) with shuffled block tables and csv block sizes."""
    torch.manual_seed(seed)
    pages = (s2 + page - 1) // page
    q = torch.rand(b * s1, n1, d, dtype=torch.bfloat16, device=device) * 4 - 2
    kv = (torch.randn(b * s2, 1, d, dtype=torch.bfloat16, device=device) * 0.5).reshape(b * pages, page, 1, d)
    block_table = torch.stack(
        [torch.randperm(b * pages, device=device)[:pages] for _ in range(b)]
    ).to(torch.int32).to(device)
    cu_seqlens_q = torch.tensor([i * s1 for i in range(b + 1)], dtype=torch.int32, device=device)
    seqused_ori_kv = torch.full((b,), s2, dtype=torch.int32, device=device)
    cmp_kv = cmp_block_table = seqused_cmp_kv = cmp_residual_kv = cmp_sparse_indices = None
    if layer != "SWA":
        cmp_pages = (t3 + page - 1) // page
        cmp_kv = (torch.randn(b * t3, 1, d, dtype=torch.bfloat16, device=device) * 0.5).reshape(
            b * cmp_pages, page, 1, d
        )
        cmp_block_table = torch.stack(
            [torch.randperm(b * cmp_pages, device=device)[:cmp_pages] for _ in range(b)]
        ).to(torch.int32).to(device)
        seqused_cmp_kv = torch.full((b,), t3, dtype=torch.int32, device=device)
        cmp_residual_kv = torch.zeros(b, dtype=torch.int32, device=device)
        if layer == "CSA":
            cmp_sparse_indices = torch.randint(0, t3, (b * s1, k), dtype=torch.int32, device=device)
    return (q, kv, block_table, cu_seqlens_q, seqused_ori_kv, cmp_kv, cmp_block_table,
            seqused_cmp_kv, cmp_residual_kv, cmp_sparse_indices)


def generate_smla_tnd_data(s1, s2, n1, b=1, layer="SWA", ratio=1, t3=0, k=0, d=512, seed=43, device=None):
    """Dense contiguous SMLA inputs (layout_kv=TND) with per-batch storage prefixes."""
    torch.manual_seed(seed)
    q = torch.rand(b * s1, n1, d, dtype=torch.bfloat16, device=device) * 4 - 2
    kv = torch.randn(b * s2, 1, d, dtype=torch.bfloat16, device=device) * 0.5
    cu_seqlens_q = torch.tensor([i * s1 for i in range(b + 1)], dtype=torch.int32, device=device)
    cu_seqlens_ori_kv = torch.tensor([i * s2 for i in range(b + 1)], dtype=torch.int32, device=device)
    seqused_ori_kv = torch.full((b,), s2, dtype=torch.int32, device=device)
    cmp_kv = cu_seqlens_cmp_kv = seqused_cmp_kv = cmp_residual_kv = cmp_sparse_indices = None
    if layer != "SWA":
        cmp_kv = torch.randn(b * t3, 1, d, dtype=torch.bfloat16, device=device) * 0.5
        cu_seqlens_cmp_kv = torch.tensor([i * t3 for i in range(b + 1)], dtype=torch.int32, device=device)
        seqused_cmp_kv = torch.full((b,), t3, dtype=torch.int32, device=device)
        cmp_residual_kv = torch.zeros(b, dtype=torch.int32, device=device)
        if layer == "CSA":
            cmp_sparse_indices = torch.randint(0, t3, (b * s1, k), dtype=torch.int32, device=device)
    return (q, kv, cu_seqlens_q, cu_seqlens_ori_kv, seqused_ori_kv, cmp_kv,
            cu_seqlens_cmp_kv, seqused_cmp_kv, cmp_residual_kv, cmp_sparse_indices)


@pytest.fixture
def sparse_flash_mla_perf_backend(perf_environment):
    _, device, target, implementation = perf_environment
    if implementation == "torch_reference":
        return implementation, device
    if not target.startswith("npu.a5") or implementation not in (None, "cannbotdsl"):
        pytest.skip("Sparse flash MLA performance currently has an A5 cannbotdsl provider only")
    return implementation, device


@pytest.mark.api("functions.sparse_flash_mla_infer")
@pytest.mark.parametrize(
    "case_id, b, s1, s2, n1, layer, ratio, t3, k, page",
    _SMLA_CASES,
    ids=[case_id + "_pa" for (case_id, *_) in _SMLA_CASES],
)
def test_sparse_flash_mla_pa(benchmark, sparse_flash_mla_perf_backend, case_id, b, s1, s2, n1, layer, ratio, t3, k, page):
    implementation, device = sparse_flash_mla_perf_backend

    def factory():
        (q, kv, block_table, cu_seqlens_q, seqused_ori_kv, cmp_kv, cmp_block_table,
         seqused_cmp_kv, cmp_residual_kv, cmp_sparse_indices) = generate_smla_pa_data(
            s1, s2, n1, b=b, layer=layer, ratio=ratio, t3=t3, k=k, page=page, device=device
        )

        def run():
            return functions.sparse_flash_mla_infer(
                q, kv, block_table, cu_seqlens_q, seqused_ori_kv,
                win_left=127, win_right=0,
                cmp_kv=cmp_kv, cmp_block_table=cmp_block_table,
                seqused_cmp_kv=seqused_cmp_kv, cmp_residual_kv=cmp_residual_kv,
                cmp_sparse_indices=cmp_sparse_indices, cmp_ratio=ratio,
                implementation=implementation,
            )

        return run

    benchmark(
        factory=factory,
        op="sparse_flash_mla_infer",
        case_id=case_id,
        layer=layer,
        batch=b,
        s1=s1,
        s2=s2,
        heads=n1,
        cmp_ratio=ratio,
        t3=t3,
        topk=k,
        page=page,
        layout="PA_BBND",
    )


@pytest.mark.api("functions.sparse_flash_mla_infer")
@pytest.mark.parametrize(
    "case_id, b, s1, s2, n1, layer, ratio, t3, k, page",
    _SMLA_CASES,
    ids=[case_id + "_tnd" for (case_id, *_) in _SMLA_CASES],
)
def test_sparse_flash_mla_tnd(benchmark, sparse_flash_mla_perf_backend, case_id, b, s1, s2, n1, layer, ratio, t3, k, page):
    implementation, device = sparse_flash_mla_perf_backend

    def factory():
        (q, kv, cu_seqlens_q, cu_seqlens_ori_kv, seqused_ori_kv, cmp_kv,
         cu_seqlens_cmp_kv, seqused_cmp_kv, cmp_residual_kv, cmp_sparse_indices) = generate_smla_tnd_data(
            s1, s2, n1, b=b, layer=layer, ratio=ratio, t3=t3, k=k, device=device
        )

        def run():
            return functions.sparse_flash_mla_infer(
                q, kv, None, cu_seqlens_q, seqused_ori_kv,
                win_left=127, win_right=0, layout_kv="TND",
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cmp_kv=cmp_kv, cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_cmp_kv=seqused_cmp_kv, cmp_residual_kv=cmp_residual_kv,
                cmp_sparse_indices=cmp_sparse_indices, cmp_ratio=ratio,
                implementation=implementation,
            )

        return run

    benchmark(
        factory=factory,
        op="sparse_flash_mla_infer",
        case_id=case_id,
        layer=layer,
        batch=b,
        s1=s1,
        s2=s2,
        heads=n1,
        cmp_ratio=ratio,
        t3=t3,
        topk=k,
        page=None,
        layout="TND",
    )
