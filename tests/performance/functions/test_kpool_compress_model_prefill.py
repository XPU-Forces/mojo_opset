"""Operator-only timing on model-shaped inputs; the caller's gather is excluded."""

import pytest

from tests.functions.test_kpool_compress_and_write_cache import run_case
from tests.functions.test_kpool_compress_model_prefill import MODEL_PREFILL_CASES
from tests.functions.test_kpool_compress_model_prefill import make_model_prefill_inputs


@pytest.mark.api("functions.kpool_compress_and_write_cache")
@pytest.mark.parametrize("case", MODEL_PREFILL_CASES, ids=lambda case: case.case_id)
def test_kpool_compress_model_prefill(benchmark, perf_environment, case):
    _, device, _, implementation = perf_environment

    def factory():
        values = tuple(value.to(device) for value in make_model_prefill_inputs(case))
        return lambda: run_case(case.operator_case, values, implementation)

    benchmark(
        factory=factory,
        op="kpool_compress_and_write_cache",
        n=case.n,
        pool_size=case.pool,
        page_size=64,
        dtype="bfloat16",
        phase="forward",
        fixture_origin="model_caller_and_configuration_derived_not_trace",
        request_lengths=case.request_lengths,
        start_position=case.start_position,
        context_limit=case.context_limit,
        valid_pools=case.writes,
        caller_gather_in_timing=False,
    )
