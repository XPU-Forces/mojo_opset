"""Measure the complete public API on the delivered 22-case matrix."""

import pytest

from mojo_opset import functions as F
from tests._convert_req_index import CASES
from tests._convert_req_index import make_inputs


@pytest.mark.api("functions.triton_convert_req_index_to_global_index")
@pytest.mark.parametrize("case", CASES, ids=[f"case{i:02d}" for i in range(1, len(CASES) + 1)])
def test_conversion(benchmark, perf_environment, case):
    _, device, _, implementation = perf_environment
    instance = 0

    def factory():
        nonlocal instance
        inputs = make_inputs(case, device, seed=42 + instance)
        instance += 1
        return lambda: F.triton_convert_req_index_to_global_index(*inputs, implementation=implementation)

    benchmark(
        factory=factory, op="triton_convert_req_index_to_global_index", case=list(case), dtype="int32", phase="forward"
    )
