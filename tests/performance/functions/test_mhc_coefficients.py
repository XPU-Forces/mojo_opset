import pytest

from mojo_opset import functions as F
from tests.functions.test_mhc_coefficients import CASE_ARGS
from tests.functions.test_mhc_coefficients import PERF_CASES
from tests.functions.test_mhc_coefficients import make_inputs


@pytest.mark.api("functions.mhc_coefficients")
@pytest.mark.parametrize(CASE_ARGS, PERF_CASES)
def test_mhc_coefficients(benchmark, perf_environment, batch, seq, hidden, dtype):
    _, device, _, implementation = perf_environment

    def factory():
        inputs = tuple(tensor.to(device) for tensor in make_inputs(batch, seq, hidden, dtype))
        return lambda: F.mhc_coefficients(*inputs, implementation=implementation)

    benchmark(
        factory=factory,
        op="mhc_coefficients",
        shape=[batch, seq, 4, hidden],
        tokens=batch * seq,
        dtype=str(dtype),
        phase="forward",
    )
