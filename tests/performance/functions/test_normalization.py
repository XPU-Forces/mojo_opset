import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training


# Union of original perf and perf_new function cases.
@pytest.mark.api("functions.rms_norm")
@pytest.mark.parametrize(
    "shape,dtype,gradient",
    [
        (shape, dtype, "uniform")
        for shape in [(32, 1024), (64, 8192), (57, 7338), (77, 489), (763, 8777), (7762, 18778)]
        for dtype in [torch.float32, torch.bfloat16]
    ]
    + [(shape, torch.bfloat16, "normal") for shape in [(32, 1024), (64, 8192)]],
)
@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_rms_norm(benchmark, perf_environment, shape, dtype, gradient, phase):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
        weight = torch.randn(shape[-1], dtype=torch.float32, device=device, requires_grad=True)
        return training(
            lambda: functions.rms_norm(x, weight, 1e-6, implementation=implementation),
            (x, weight),
            phase,
            grad_factory=torch.rand_like if gradient == "uniform" else torch.randn_like,
        )

    benchmark(
        factory=factory,
        op="rms_norm",
        shape=list(shape),
        dtype=str(dtype),
        gradient=gradient,
        weight_dtype="float32",
        phase=phase,
    )
