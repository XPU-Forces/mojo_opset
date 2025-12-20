import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoSiluFunction


@pytest.mark.parametrize("x", [torch.rand(128, 128, requires_grad=True)])
@auto_switch_platform()
@bypass_not_implemented
def test_silu_forward_backward_diff(monkeypatch, x):
    monkeypatch.setenv("MOJOSILUFUNCTION_FWD_MODE", "DIFF")
    monkeypatch.setenv("MOJOSILUFUNCTION_BWD_MODE", "DIFF")

    y = MojoSiluFunction.apply(x)

    grad_output = torch.rand_like(y)
    y.backward(grad_output)
