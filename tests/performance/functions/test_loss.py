import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training


@pytest.mark.api("functions.linear_cross_entropy")
@pytest.mark.parametrize("batch,hidden,vocab", [(128, 256, 1024), (2048, 1024, 4096)])
@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_linear_cross_entropy(benchmark, perf_environment, batch, hidden, vocab, phase):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randn(batch, hidden, dtype=torch.bfloat16, device=device, requires_grad=True)
        weight = torch.randn(vocab, hidden, dtype=torch.bfloat16, device=device, requires_grad=True)
        target = torch.randint(vocab, (batch,), device=device)
        # Original Mojo CE contract, not the distinct Ext fused CE interface.
        call = lambda: functions.linear_cross_entropy(x, weight, target, implementation=implementation)
        return training(call, (x, weight), phase, grad_factory=torch.rand_like)

    benchmark(
        factory=factory,
        op="linear_cross_entropy",
        batch=batch,
        hidden=hidden,
        vocab=vocab,
        dtype="bfloat16",
        phase=phase,
    )
