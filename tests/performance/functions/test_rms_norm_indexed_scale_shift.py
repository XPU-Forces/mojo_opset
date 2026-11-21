import pytest
import torch

from mojo_opset import functions
from tests.functions.test_rms_norm_indexed_scale_shift import MODEL_CASES
from tests.functions.test_rms_norm_indexed_scale_shift import OP
from tests.functions.test_rms_norm_indexed_scale_shift import case_shape
from tests.functions.test_rms_norm_indexed_scale_shift import make_case
from tests.functions.test_rms_norm_indexed_scale_shift import shape_label

PERFORMANCE_SHAPES = {(15056, 5376, 9, 4), (10688, 5376, 12, 5)}
PERFORMANCE_CASES = [case for case in MODEL_CASES if case_shape(case) in PERFORMANCE_SHAPES]


@pytest.mark.api("functions." + OP)
@pytest.mark.parametrize("case", PERFORMANCE_CASES, ids=shape_label)
def test_rms_norm_indexed_scale_shift(benchmark, perf_environment, case):
    _, device, _, implementation = perf_environment

    def factory():
        args = make_case(case, device)
        return lambda: functions.rms_norm_indexed_scale_shift(*args, implementation=implementation)

    benchmark(
        factory=factory,
        op=OP,
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
