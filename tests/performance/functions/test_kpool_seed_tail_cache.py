import pytest
import torch

from mojo_opset import functions as F
from tests.functions.test_kpool_seed_tail_cache import GPU_CASE_ARGS
from tests.functions.test_kpool_seed_tail_cache import GPU_CASES
from tests.functions.test_kpool_seed_tail_cache import make_case


@pytest.mark.api("functions.kpool_seed_tail_cache")
@pytest.mark.parametrize(GPU_CASE_ARGS, GPU_CASES)
def test_kpool_seed_tail_cache(benchmark, perf_environment, requests, ring, slot_dtype, num_blocks, padded):
    _, device, _, implementation = perf_environment

    def factory():
        backing, shape, strides, key, gate, slots = make_case(requests, ring, slot_dtype, num_blocks, padded)
        cache = backing.to(device).as_strided(shape, strides)
        key, gate, slots = (tensor.to(device) for tensor in (key, gate, slots))
        # Fixed slots overwrite the same locations on each call.
        return lambda: F.kpool_seed_tail_cache(cache, key, gate, slots, implementation=implementation)

    benchmark(
        factory=factory,
        op="kpool_seed_tail_cache",
        request_lengths=[length for length, _, _ in requests],
        tokens=sum(length for length, _, _ in requests) + (2 * len(requests) if padded else 0),
        ring=ring,
        padded=padded,
        dtype=str(torch.bfloat16),
        slot_dtype=str(slot_dtype),
        phase="forward",
    )
