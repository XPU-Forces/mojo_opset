import pytest
import torch

from mojo_opset import functions as F
from tests.functions.test_kpool_seed_tail_cache import make_case


@pytest.mark.api("functions.kpool_seed_tail_cache")
@pytest.mark.parametrize("dtype,slot_dtype", [(torch.bfloat16, torch.int64), (torch.float16, torch.int32)])
@pytest.mark.parametrize("n,ring", [(1, 8), (9, 8), (33, 16), (16384, 16)])
def test_kpool_seed_tail_cache(benchmark, perf_environment, dtype, slot_dtype, n, ring):
    _, device, _, implementation = perf_environment

    def factory():
        backing, key, gate, slots = make_case(n, ring, dtype, slot_dtype, "cpu")
        backing, key, gate, slots = (tensor.to(device) for tensor in (backing, key, gate, slots))
        cache = backing[:, :, :ring]
        # Fixed slots overwrite the same locations on each call.
        return lambda: F.kpool_seed_tail_cache(cache, key, gate, slots, implementation=implementation)

    benchmark(
        factory=factory,
        op="kpool_seed_tail_cache",
        tokens=n,
        ring=ring,
        dtype=str(dtype),
        slot_dtype=str(slot_dtype),
        phase="forward",
    )
