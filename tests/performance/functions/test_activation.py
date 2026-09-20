import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(
            op,
            marks=pytest.mark.api(
                "functions." + op,
                ops=[op],
            ),
        )
        for op in ["silu", "gelu"]
    ],
)
@pytest.mark.parametrize("shape", [(128, 1024), (1024, 4096)], ids=["small", "large"])
@pytest.mark.parametrize("phase", ["forward", "backward", "forward_backward"])
def test_activation(benchmark, perf_environment, op, shape, phase):
    _, device, _, implementation = perf_environment
    function = getattr(functions, op)

    def factory():
        x = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=phase != "forward")
        grad = torch.randn_like(x) if phase != "forward" else None
        saved_output = function(x, implementation=implementation) if phase == "backward" else None

        def run():
            output = saved_output if phase == "backward" else function(x, implementation=implementation)
            if phase != "forward":
                return torch.autograd.grad(output, x, grad_outputs=grad, retain_graph=phase == "backward")
            return output

        return run

    benchmark(factory=factory, op=op, shape=list(shape), dtype="bfloat16", phase=phase)


# Original Mojo perf + perf_new function matrices; keep dtype/distribution distinct.
@pytest.mark.api("functions.silu")
@pytest.mark.parametrize(
    "shape,dtype,distribution",
    [
        ((128, 128), torch.float32, "rand"),
        ((999, 9999), torch.float32, "rand"),
        ((1024, 10240), torch.float32, "rand"),
        ((1024, 1024), torch.float16, "randn"),
        ((4096, 4096), torch.float16, "randn"),
    ],
)
@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_silu(benchmark, perf_environment, shape, dtype, distribution, phase):
    _, device, _, implementation = perf_environment

    def factory():
        x = getattr(torch, distribution)(shape, dtype=dtype, device=device, requires_grad=True)
        call = lambda: functions.silu(x, implementation=implementation)
        return training(call, (x,), phase, grad_factory=torch.rand_like if distribution == "rand" else torch.randn_like)

    benchmark(factory=factory, op="silu", shape=list(shape), dtype=str(dtype), distribution=distribution, phase=phase)
