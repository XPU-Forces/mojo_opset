import pytest
import torch

from mojo_opset import functions as F
from tests.functions.test_indexed_gate_rms_norm_scale_shift import MODEL_CASE_PARAMS
from tests.functions.test_indexed_gate_rms_norm_scale_shift import OP
from tests.functions.test_indexed_gate_rms_norm_scale_shift import make_case


@pytest.mark.api("functions." + OP)
@pytest.mark.parametrize("case", MODEL_CASE_PARAMS)
def test_indexed_gate_rms_norm_scale_shift(benchmark, perf_environment, case):
    _, device, _, implementation = perf_environment

    def factory():
        args = make_case(case, device)
        return lambda: F.indexed_gate_rms_norm_scale_shift_infer(*args, implementation=implementation)

    benchmark(
        factory=factory,
        op=OP,
        case_id=case["id"],
        rows=case["rows"],
        hidden=case["hidden_size"],
        conditions=case["conditions"],
        dtype=str(torch.bfloat16),
        phase="forward",
    )
