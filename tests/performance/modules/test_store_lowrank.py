import pytest
import torch

from mojo_opset import modules


@pytest.mark.api("modules.StoreLowrank", ops=["store_lowrank"])
@pytest.mark.parametrize("heads", [1, 8])
@pytest.mark.parametrize("tokens", [1024, 2048, 4096, 8192, 13312])
@pytest.mark.parametrize("mapping", ["permutation", "random"])
def test_store_lowrank(benchmark, perf_environment, heads, tokens, mapping):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.StoreLowrank(implementation=implementation)
        cache = torch.zeros(256, heads, 512, 128, dtype=torch.bfloat16, device=device)
        key = torch.randn(tokens, heads, 128, dtype=torch.bfloat16, device=device)
        if mapping == "permutation":
            slots = torch.randperm(tokens, device=device)
            blocks, offsets = (slots // 512).int(), (slots % 512).int()
        else:
            # perf_new used independent random block/token indices (duplicates possible).
            blocks = torch.randint(256, (tokens,), dtype=torch.int32, device=device)
            offsets = torch.randint(512, (tokens,), dtype=torch.int32, device=device)
        return lambda: module(cache, key, blocks, offsets, tokens)

    benchmark(
        factory=factory,
        op="store_lowrank",
        heads=heads,
        tokens=tokens,
        mapping=mapping,
        dtype="bfloat16",
        phase="forward",
    )
