import pytest
import torch

from mojo_opset import modules
from mojo_opset.utils.kv_cache_metadata import build_paged_kv_chunk_metadata


@pytest.mark.api("modules.StorePagedKVCache", ops=["store_paged_kv_cache"])
@pytest.mark.parametrize(
    "heads,dim,block_size,context,queries",
    [
        (2, 128, 128, [0, 0], [130, 33]),
        (2, 128, 128, [32, 35], [1, 1]),
        (2, 128, 128, [15, 40], [788, 126]),
        (2, 128, 256, [15, 40], [788, 126]),
        (1, 128, 128, [0], [5]),
        (1, 128, 128, [5], [1]),
        (2, 128, 128, [224, 542, 34, 41, 54, 57, 65, 0], [432, 84, 977, 93, 23, 89, 31, 555]),
        (2, 128, 128, [772, 974, 3232, 43, 77, 7633, 888, 1], [1] * 8),
    ],
)
def test_store_paged_kv(benchmark, perf_environment, heads, dim, block_size, context, queries):
    _, device, _, implementation = perf_environment
    blocks = [(c + q + block_size - 1) // block_size for c, q in zip(context, queries)]

    def factory():
        module = modules.StorePagedKVCache(implementation=implementation)
        key, value = (torch.randn(sum(queries), heads, dim, dtype=torch.bfloat16, device=device) for _ in range(2))
        kc, vc = (
            torch.zeros(sum(blocks) + 10, heads, block_size, dim, dtype=key.dtype, device=device) for _ in range(2)
        )
        table = torch.full((len(context), max(blocks) + 2), -1, dtype=torch.int32)
        offset = 0
        for index, count in enumerate(blocks):
            table[index, :count] = torch.arange(offset, offset + count)
            offset += count
        cu = (
            None
            if all(q == 1 for q in queries)
            else torch.tensor([0, *queries], dtype=torch.int32).cumsum(0, dtype=torch.int32)
        )
        metadata = build_paged_kv_chunk_metadata(table, cu, torch.tensor(context, dtype=torch.int32), block_size).to(
            device
        )
        # Fixed metadata overwrites the same slots; it does not append across calls.
        return lambda: module(key, value, kc, vc, chunk_metadata=metadata)

    benchmark(
        factory=factory,
        op="store_paged_kv_cache",
        heads=heads,
        dim=dim,
        block_size=block_size,
        context=context,
        queries=queries,
        dtype="bfloat16",
        phase="forward",
    )
