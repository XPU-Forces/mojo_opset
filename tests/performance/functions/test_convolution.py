import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training


@pytest.mark.api("functions.causal_conv1d")
@pytest.mark.parametrize("batch,tokens,hidden,width,extras", [(2, 64, 1024, 3, True), (2, 128, 4096, 4, False)])
@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_causal_conv1d(benchmark, perf_environment, batch, tokens, hidden, width, extras, phase):
    _, device, _, implementation = perf_environment
    shape = (batch, tokens, hidden)

    def factory():
        def tensor(shape):
            return torch.rand(shape, dtype=torch.float16, device=device, requires_grad=True)

        x, weight = tensor(shape), tensor((hidden, width))
        bias = tensor((hidden,)) if extras else None
        residual = tensor(shape) if extras else None
        call = lambda: functions.causal_conv1d(
            x, weight, bias, residual=residual, activation="swish", implementation=implementation
        )
        return training(call, (x, weight, bias, residual), phase, grad_factory=torch.rand_like)

    benchmark(
        factory=factory,
        op="causal_conv1d",
        shape=list(shape),
        width=width,
        bias=extras,
        residual=extras,
        activation="swish",
        dtype="float16",
        phase=phase,
    )
