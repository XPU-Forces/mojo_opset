import pytest
import torch

from tests.utils import auto_switch_platform, bypass_not_implemented
from mojo_opset import MojoRMSNormFunction


@pytest.mark.parametrize(
    "x, w",
    [(torch.rand(128, 128, requires_grad=True), torch.rand(128, requires_grad=True))],
)
@auto_switch_platform()
@bypass_not_implemented
def test_rmsnorm_forward_backward_diff(monkeypatch, x, w):
    monkeypatch.setenv("MOJORMSNORMFUNCTION_FWD_MODE", "DIFF")
    monkeypatch.setenv("MOJORMSNORMFUNCTION_BACKWARD_MODE", "DIFF")

    y = MojoRMSNormFunction.apply(x, w)

    grad_output = torch.rand_like(y)
    y.backward(grad_output)
