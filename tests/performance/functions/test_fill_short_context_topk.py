"""fill_short_context_topk performance cases (production shapes, cold L2 via the harness)."""

import pytest
import torch

from mojo_opset import functions


@pytest.mark.api("functions.fill_short_context_topk")
@pytest.mark.parametrize(
    "num_tokens,top_k,compress_ratio",
    [(256, 512, 2), (1024, 2048, 4), (2048, 2048, 4), (16384, 2048, 4)],
)
def test_fill_short_context_topk(benchmark, perf_environment, num_tokens, top_k, compress_ratio):
    _, device, _, implementation = perf_environment

    def factory():
        buffer = torch.empty(num_tokens, top_k, dtype=torch.int32, device=device)
        positions = torch.randint(
            0, top_k * compress_ratio, (num_tokens,), dtype=torch.int64, device=device
        )
        return lambda: functions.fill_short_context_topk(
            buffer, positions, top_k, compress_ratio, implementation=implementation
        )

    benchmark(
        factory=factory,
        op="fill_short_context_topk",
        num_tokens=num_tokens,
        top_k=top_k,
        compress_ratio=compress_ratio,
        dtype="int32",
        phase="forward",
    )
