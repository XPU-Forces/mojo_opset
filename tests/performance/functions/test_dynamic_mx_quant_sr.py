from __future__ import annotations

import pytest
import torch

from mojo_opset import functions

# Representative benchmark cases per kernel layout branch: tail (dense decode
# scale), stride-2 fused/non-fused, wide-axis lanes, and the transpose
# fallback.  (case_id, shape, axis)
_MXQ_CASES = [
    ("001_tail_16k", (128, 128), -1),
    ("002_tail_1m", (1024, 1024), -1),
    ("003_stride2_fused", (2, 1024, 2), 1),
    ("004_stride2", (2, 1152, 2), 1),
    ("005_width8", (4096, 8), 0),
    ("006_width16", (2048, 16), 0),
    ("007_width32", (512, 128, 32), -2),
    ("008_width64", (256, 128, 64), -2),
    ("009_width128", (128, 128, 128), -2),
    ("010_transpose", (64, 48, 3), 1),
]


@pytest.fixture
def dynamic_mx_quant_sr_perf_backend(perf_environment):
    _, device, target, implementation = perf_environment
    if implementation == "torch_reference":
        return implementation, device
    if not target.startswith("npu.a5") or implementation not in (None, "cannbotdsl"):
        pytest.skip("Dynamic MX quant SR performance currently has an A5 cannbotdsl provider only")
    return implementation, device


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize(
    "case_id, shape, axis",
    _MXQ_CASES,
    ids=[case_id for case_id, *_ in _MXQ_CASES],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_dynamic_mx_quant_sr(benchmark, dynamic_mx_quant_sr_perf_backend, case_id, shape, axis, dtype):
    implementation, device = dynamic_mx_quant_sr_perf_backend

    def factory():
        generator = torch.Generator().manual_seed(43)
        x = (torch.randn(shape, generator=generator) * 7).to(dtype).to(device)

        def run():
            return functions.dynamic_mx_quant_sr(x, axis=axis, implementation=implementation)

        return run

    benchmark(
        factory=factory,
        op="dynamic_mx_quant_sr",
        case_id=case_id,
        shape=str(shape),
        axis=axis,
        dtype=str(dtype).removeprefix("torch."),
    )
