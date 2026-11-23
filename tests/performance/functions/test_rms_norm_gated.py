import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training

# Production shapes from tests/modules/test_rms_norm_gated.py; all cases are
# kept above the 20us measurement floor.
CASES = [
    ((2, 32, 2048, 128), "silu", torch.float16),
    ((2, 32, 1024, 256), "sigmoid", torch.bfloat16),
    ((2, 16, 1024, 512), "sigmoid", torch.float16),
    ((1, 16, 1024, 1024), "silu", torch.float16),
    ((2, 8, 1024, 1024), "sigmoid", torch.bfloat16),
    ((2, 4, 2048, 768), "sigmoid", torch.float16),
    ((4, 8, 1024, 600), "sigmoid", torch.bfloat16),
    ((2, 4, 2048, 1200), "silu", torch.bfloat16),
    ((131072, 128), "silu", torch.bfloat16),
    ((16384, 1024), "sigmoid", torch.float16),
]

_DTYPE_NAME = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}


def _case_id(case):
    shape, activation, dtype = case
    shape_id = "_".join(str(dim) for dim in shape)
    return f"{shape_id}_{activation}_{_DTYPE_NAME[dtype]}"


@pytest.mark.api("functions.rms_norm_gated", ops=["rms_norm_gated"])
@pytest.mark.parametrize("shape,activation,dtype", CASES, ids=[_case_id(case) for case in CASES])
@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_rms_norm_gated(benchmark, perf_environment, shape, activation, dtype, phase):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
        g = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
        weight = torch.randn(shape[-1], dtype=dtype, device=device, requires_grad=True)
        return training(
            lambda: functions.rms_norm_gated(
                x,
                g,
                weight,
                activation=activation,
                implementation=implementation,
            ),
            (x, g, weight),
            phase,
        )

    benchmark(
        factory=factory,
        op="rms_norm_gated",
        shape=list(shape),
        dtype=str(dtype),
        activation=activation,
        phase=phase,
    )
