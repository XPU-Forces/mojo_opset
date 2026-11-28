# SPDX-License-Identifier: Apache-2.0

import pytest

from mojo_opset import functions as F
from tests.functions.test_indexed_gate import OP
from tests.functions.test_indexed_gate import PERF_CASES
from tests.functions.test_indexed_gate import make_model_case
from tests.functions.test_indexed_gate import shape_id


@pytest.mark.api("functions." + OP)
@pytest.mark.parametrize("case", PERF_CASES, ids=shape_id)
def test_indexed_gate(benchmark, perf_environment, case):
    _, device, _, implementation = perf_environment

    def factory():
        args = make_model_case(case, device)
        return lambda: F.indexed_gate_infer(*args, implementation=implementation)

    benchmark(
        factory=factory,
        op=OP,
        shape=shape_id(case),
        rows=case[0],
        hidden=5376,
        conditions=case[1],
        dtype="torch.bfloat16",
        phase="forward",
    )
