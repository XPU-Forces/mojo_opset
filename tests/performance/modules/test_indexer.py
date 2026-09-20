import pytest
import torch

from mojo_opset import modules


@pytest.mark.api("modules.LightningIndexer", ops=["lightning_indexer"])
@pytest.mark.parametrize(
    "batch,m,n,heads,k", [(128, 256, 256, 64, 128), (24, 1024, 1024, 128, 128), (24, 1, 16384, 128, 128)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_lightning_indexer(benchmark, perf_environment, batch, m, n, heads, k, dtype):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.LightningIndexer(implementation=implementation)
        query = torch.randn(batch, m, heads, k, dtype=dtype, device=device)
        key = torch.randn(batch, n, k, dtype=dtype, device=device)
        qs = torch.randn(batch, m, heads, device=device)
        ks = torch.randn(batch, n, device=device)
        return lambda: module(query, qs, key, ks)

    benchmark(
        factory=factory,
        op="lightning_indexer",
        batch=batch,
        m=m,
        n=n,
        heads=heads,
        k=k,
        dtype=str(dtype),
        phase="forward",
    )
