"""Measure the complete public API on the seven GLM-5 prefill workloads."""

import pytest

from mojo_opset import functions as F
from tests.functions.test_triton_convert_req_index_to_global_index import PERFORMANCE_CASES
from tests.functions.test_triton_convert_req_index_to_global_index import build_balanced_inputs


@pytest.mark.api("functions.triton_convert_req_index_to_global_index")
@pytest.mark.parametrize("case", PERFORMANCE_CASES, ids=[c["name"] for c in PERFORMANCE_CASES])
def test_conversion(benchmark, perf_environment, case):
    _, device, _, implementation = perf_environment

    def factory():
        inputs = build_balanced_inputs(case, device)
        return lambda: F.triton_convert_req_index_to_global_index(*inputs, implementation=implementation)

    benchmark(factory=factory, op="triton_convert_req_index_to_global_index", case=case, dtype="int32", phase="forward")
