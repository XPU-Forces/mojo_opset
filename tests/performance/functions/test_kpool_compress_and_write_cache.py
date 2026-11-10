import pytest

from tests.functions.test_kpool_compress_and_write_cache import PERF_CASES
from tests.functions.test_kpool_compress_and_write_cache import make_inputs
from tests.functions.test_kpool_compress_and_write_cache import run_case


@pytest.mark.api("functions.kpool_compress_and_write_cache")
@pytest.mark.parametrize("case", PERF_CASES, ids=lambda case: case.case_id)
def test_kpool_compress_and_write_cache(benchmark, perf_environment, case):
    _, device, _, implementation = perf_environment

    def factory():
        values = tuple(value.to(device) if value is not None else None for value in make_inputs(case))
        # Each iteration overwrites the same cache slots.
        return lambda: run_case(case, values, implementation)

    benchmark(
        factory=factory,
        op="kpool_compress_and_write_cache",
        n=case.n,
        pool_size=case.pool,
        page_size=case.page,
        dtype="bfloat16",
        phase="forward",
    )
