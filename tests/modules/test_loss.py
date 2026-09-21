import pytest
import torch

from mojo_opset import modules
from tests._checks import assert_accuracy
from tests._checks import assert_repeatable


@pytest.mark.api("modules.LinearCrossEntropyLoss", ops=["linear_cross_entropy"])
@pytest.mark.accuracy
def test_linear_ce(accuracy_backend):
    _, _, device = accuracy_backend
    labels = torch.randint(128, (17,), device=device)

    def run(x, w, **selection):
        return modules.LinearCrossEntropyLoss(**selection)(w, x, labels)

    assert_accuracy(
        run,
        (torch.randn(17, 64, device=device), torch.randn(128, 64, device=device)),
        implementation=accuracy_backend[0],
    )


@pytest.mark.api("modules.LinearCrossEntropyLoss", ops=["linear_cross_entropy"])
@pytest.mark.bitwise
def test_linear_ce_bitwise(accuracy_backend):
    implementation, _, device = accuracy_backend
    x = torch.randn(17, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(128, 64, device=device, dtype=x.dtype, requires_grad=True)
    labels = torch.randint(128, (17,), device=device)
    module = modules.LinearCrossEntropyLoss(implementation=implementation)

    def run(x, w):
        return module(w, x, labels)

    assert_repeatable(run, (x, w))
