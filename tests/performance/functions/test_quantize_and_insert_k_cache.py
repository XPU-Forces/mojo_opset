import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training

# (num_tokens, block_size, bytes_per_token)
CASES = [
    (4096, 64, 584),
    (8192, 64, 584),
    (16384, 64, 584),
    (4096, 64, 528),
    (8192, 64, 528),
    (16384, 64, 528),
    (8192, 128, 584),
    (8192, 128, 528),
    (16384, 128, 584),
    (16384, 128, 528),
]


def _case_id(case):
    num_tokens, block_size, bpt = case
    return f"n{num_tokens}-b{block_size}-r{bpt}"


@pytest.mark.api("functions.quantize_and_insert_k_cache", ops=["quantize_and_insert_k_cache"])
@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_quantize_and_insert_k_cache(benchmark, perf_environment, case):
    num_tokens, block_size, bytes_per_token = case
    _, device, _, implementation = perf_environment

    def factory():
        k = torch.randn(num_tokens, 512, dtype=torch.bfloat16, device=device)
        num_blocks = (num_tokens + block_size - 1) // block_size + 2
        k_cache = torch.zeros(
            num_blocks, block_size * bytes_per_token, dtype=torch.uint8, device=device
        )
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
        # Fixed slots overwrite the same pages; the cache does not grow.
        return training(
            lambda: functions.quantize_and_insert_k_cache(
                k,
                k_cache,
                slot_mapping,
                block_size,
                bytes_per_token=bytes_per_token,
                implementation=implementation,
            ),
            (k, k_cache, slot_mapping),
            "forward",
        )

    benchmark(
        factory=factory,
        op="quantize_and_insert_k_cache",
        num_tokens=num_tokens,
        block_size=block_size,
        bytes_per_token=bytes_per_token,
        dtype="bfloat16",
        phase="forward",
    )
