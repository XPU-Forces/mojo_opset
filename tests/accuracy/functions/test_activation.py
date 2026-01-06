import pytest
import torch

from tests.utils import MockFunctionCtx
from tests.utils import assert_close
from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoSiluFunction


@pytest.mark.parametrize("x", [torch.rand(128, 128, requires_grad=True)])
@auto_switch_platform()
@bypass_not_implemented
def test_silu_forward_backward_diff(monkeypatch, x):
    ctx = MockFunctionCtx()
    y = MojoSiluFunction.forward(ctx, x)

    ctx_ref = MockFunctionCtx()
    y_ref = MojoSiluFunction._registry.get("ref").forward(ctx_ref, x)
    assert_close(y, y_ref)

    dy = torch.rand_like(y)
    dx = MojoSiluFunction.backward(ctx, dy)
    dx_ref = MojoSiluFunction._registry.get("ref").backward(ctx_ref, dy)
    assert_close(dx, dx_ref)
