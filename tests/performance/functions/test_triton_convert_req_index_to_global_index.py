"""Measure the complete public API on the balanced GLM-5 14-case matrix."""

import pytest

from mojo_opset import functions as F
from tests._convert_req_index_balanced import CASES
from tests._convert_req_index_balanced import build_inputs


@pytest.mark.api("functions.triton_convert_req_index_to_global_index")
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_conversion(benchmark, perf_environment, case):
    _, device, _, implementation = perf_environment

    def factory():
        inputs = build_inputs(case, device)
        return lambda: F.triton_convert_req_index_to_global_index(*inputs, implementation=implementation)

    benchmark(factory=factory, op="triton_convert_req_index_to_global_index", case=case, dtype="int32", phase="forward")
