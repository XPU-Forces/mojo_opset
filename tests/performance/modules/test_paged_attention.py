import pytest
import torch

from mojo_opset import modules

# Shape union from Mojo perf/test_attention.py and perf_new/operators/attention.py.
DECODE = [
    (8, 16, 4, 128, 1024, 32),
    (8, 16, 4, 128, 1024, 128),
    (8, 8, 1, 128, 8192, 1024),
    (8, 8, 1, 128, 2048, 1024),
    (8, 8, 1, 128, 0, 1024),
    (8, 8, 1, 128, 16384, 128),
    (8, 8, 1, 128, 32768, 128),
    (8, 16, 4, 96, 1024, 128),
    (8, 8, 1, 128, 8192, 128),
]
SWA_DECODE = [
    (8, 16, 4, 128, 1024, 32),
    (8, 16, 4, 128, 1024, 128),
    (8, 16, 4, 128, 8192, 128),
    (4, 16, 4, 128, 1024, 512),
    (8, 16, 4, 128, 2048, 128),
    (8, 8, 1, 128, 4096, 128),
    (2, 8, 1, 128, 2048, 1024),
    (2, 8, 1, 128, 0, 1024),
    (2, 8, 2, 128, 2048, 1024),
    (2, 24, 8, 128, 2048, 1024),
    (8, 16, 4, 128, 16384, 128),
    (8, 16, 4, 128, 32768, 128),
    (8, 16, 4, 96, 1024, 128),
]
PREFILL = [
    (2, 16, 4, 128, 1024, 0, 32),
    (2, 16, 4, 128, 1024, 0, 128),
    (2, 8, 1, 128, 4096, 8192, 128),
    (2, 8, 1, 128, 1024, 2048, 1024),
    (2, 8, 1, 128, 0, 0, 1024),
    (2, 8, 1, 128, 16384, 8192, 128),
    (2, 8, 1, 128, 32768, 10240, 128),
    (1, 12, 4, 128, 131072, 0, 128),
    (2, 16, 4, 96, 1024, 0, 128),
]
SWA_PREFILL = [
    (2, 16, 4, 128, 1024, 0, 32),
    (2, 16, 4, 128, 2048, 0, 128),
    (2, 8, 1, 128, 256, 1024, 128),
    (2, 8, 1, 128, 1024, 2048, 1024),
    (2, 8, 1, 128, 0, 0, 1024),
    (2, 8, 2, 128, 2048, 0, 1024),
    (2, 24, 8, 128, 1024, 1024, 1024),
    (2, 8, 1, 128, 16348, 1024, 128),
    (2, 8, 1, 128, 32768, 1024, 128),
    (2, 16, 4, 96, 1024, 0, 128),
    (2, 16, 4, 128, 1024, 8192, 128),
]


def make_cache(lengths, heads, dim, block_size, device):
    blocks = (lengths + block_size - 1) // block_size
    count = int(blocks.sum()) + 10
    table = torch.full((len(lengths), int(blocks.max())), -1, dtype=torch.int32)
    free = torch.randperm(count, dtype=torch.int32)
    offset = 0
    for i, num in enumerate(blocks.tolist()):
        table[i, :num] = free[offset : offset + num]
        offset += num
    cache_shape = (count, heads, block_size, dim)
    k, v = [torch.randn(cache_shape, dtype=torch.bfloat16, device=device) for _ in range(2)]
    return k, v, table.to(device)


@pytest.mark.parametrize(
    "sliding,case,layout",
    [
        pytest.param(
            sliding, case, layout,
            marks=pytest.mark.api(
                "modules.PagedDecodeSWAInfer" if sliding else "modules.PagedDecodeGQAInfer",
                ops=["paged_decode_swa_infer" if sliding else "paged_decode_gqa_infer"],
            ),
        )
        for sliding, case, layout in [
            *[(False, case, layout) for case in DECODE for layout in ("AABB", "ABAB")],
            *[(True, case, "AABB") for case in SWA_DECODE],
        ]
    ],
)
def test_decode(benchmark, perf_environment, sliding, case, layout):
    _, device, _, implementation = perf_environment
    batch, qh, kh, dim, max_len, block = case
    op = "paged_decode_swa_infer" if sliding else "paged_decode_gqa_infer"
    rng = torch.Generator().manual_seed(43)
    lengths = (
        torch.randint(0, max_len, (batch,), generator=rng, dtype=torch.int32).clamp(min=1)
        if max_len
        else torch.randperm(batch, generator=rng, dtype=torch.int32)
    )

    def factory():
        cls = modules.PagedDecodeSWAInfer if sliding else modules.PagedDecodeGQAInfer
        kwargs = dict(local_window_size=1023, global_window_size=4) if sliding else {}
        module = cls(gqa_layout=layout, implementation=implementation, **kwargs)
        q = torch.randn(batch, qh, dim, dtype=torch.bfloat16, device=device)
        k, v, table = make_cache(lengths, kh, dim, block, device)
        lens = lengths.to(device)
        return lambda: module(q, k, v, lens, table, softmax_scale=dim**-0.5)

    benchmark(factory=factory, op=op, shape=list(case), layout=layout, dtype="bfloat16", phase="forward")


@pytest.mark.parametrize(
    "sliding,case,layout",
    [
        pytest.param(
            sliding, case, layout,
            marks=pytest.mark.api(
                "modules.PagedPrefillSWAInfer" if sliding else "modules.PagedPrefillGQAInfer",
                ops=["paged_prefill_swa_infer" if sliding else "paged_prefill_gqa_infer"],
            ),
        )
        for sliding, case, layout in [
            *[(False, case, layout) for case in PREFILL for layout in ("AABB", "ABAB")],
            *[(True, case, "AABB") for case in SWA_PREFILL],
        ]
    ],
)
def test_prefill(benchmark, perf_environment, sliding, case, layout):
    _, device, _, implementation = perf_environment
    batch, qh, kh, dim, max_q, max_cache, block = case
    op = "paged_prefill_swa_infer" if sliding else "paged_prefill_gqa_infer"
    if implementation == "torch_npu" and (max_cache or dim % 128 or block % 128 or block > 512):
        pytest.skip("Inherited torch_npu prefill restrictions: no prefix cache, aligned dim/pages, page <= 512")
    rng = torch.Generator().manual_seed(43)
    qlens = (
        torch.randint(max_q // 2, max_q, (batch,), generator=rng, dtype=torch.int32).clamp(min=1)
        if max_q
        else torch.randperm(batch, generator=rng, dtype=torch.int32)
    )
    klens = qlens.clone()
    if max_cache:
        klens += torch.randint(max_cache // 2, max_cache, (batch,), generator=rng, dtype=torch.int32)
        klens = torch.where(qlens > 0, klens, 0)

    def factory():
        cls = modules.PagedPrefillSWAInfer if sliding else modules.PagedPrefillGQAInfer
        kwargs = dict(local_window_size=1023, global_window_size=4) if sliding else {}
        module = cls(gqa_layout=layout, implementation=implementation, **kwargs)
        q = torch.randn(int(qlens.sum()), qh, dim, dtype=torch.bfloat16, device=device)
        k, v, table = make_cache(klens, kh, dim, block, device)
        cq = torch.cat((torch.zeros(1, dtype=torch.int32), qlens)).cumsum(0, dtype=torch.int32).to(device)
        ck = (
            torch.cat((torch.zeros(1, dtype=torch.int32), klens)).cumsum(0, dtype=torch.int32).to(device)
            if max_cache
            else None
        )
        if not sliding:
            module.prepare_metadata(cq, ck, qh, kh, block)
        max_q_len, max_total_seq_len = int(qlens.max()), int(klens.max())
        return lambda: module(
            q,
            k,
            v,
            cq,
            table,
            softmax_scale=dim**-0.5,
            cu_total_seq_lens=ck,
            max_q_len=max_q_len,
            max_total_seq_len=max_total_seq_len,
        )

    benchmark(factory=factory, op=op, shape=list(case), layout=layout, dtype="bfloat16", phase="forward")
