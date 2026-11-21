import pytest
import torch

from mojo_opset import functions
from tests.functions.test_indexed_rms_norm_scale_shift import MODEL_CASES
from tests.functions.test_indexed_rms_norm_scale_shift import OP
from tests.functions.test_indexed_rms_norm_scale_shift import make_case


@pytest.mark.api("functions." + OP)
@pytest.mark.parametrize("case", MODEL_CASES)
def test_indexed_rms_norm_scale_shift(benchmark, perf_environment, case):
    _, device, _, implementation = perf_environment

    def factory():
        args = make_case(case, device)
        return lambda: functions.indexed_rms_norm_scale_shift_infer(*args, implementation=implementation)

    benchmark(
        factory=factory,
        op=OP,
        case_id=case["id"],
        rows=case["rows"],
        hidden=case["hidden_size"],
        conditions=case["conditions"],
        active_conditions=case["active_conditions"],
        table_row_stride=6 * case["hidden_size"],
        indices_sha256=case["indices_sha256"],
        eps=case["eps"],
        dtype=str(torch.bfloat16),
        phase="forward",
    )
