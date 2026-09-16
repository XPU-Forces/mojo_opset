import pytest
import torch

from mojo_opset import functions


@pytest.mark.parametrize("op", ["silu", "gelu"])
@pytest.mark.parametrize("shape", [(128, 1024), (1024, 4096)], ids=["small", "large"])
@pytest.mark.parametrize("phase", ["forward", "backward", "forward_backward"])
def test_activation(benchmark, perf_environment, op, shape, phase):
    _, device, _, implementation, _ = perf_environment
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
