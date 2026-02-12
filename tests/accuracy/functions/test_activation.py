import pytest
import torch

from tests.utils import MockFunctionCtx
from tests.utils import assert_close
from tests.utils import bypass_not_implemented

from mojo_opset import MojoSiluFunction


@pytest.mark.parametrize("shape", [([128, 128]), ([999, 9999]), ([1024, 10240]),])
@bypass_not_implemented
def test_silu_forward_backward_diff(shape):
    x = torch.rand(*shape, requires_grad=True)
    ctx = MockFunctionCtx()
    y = MojoSiluFunction.forward(ctx, x)

    ctx_ref = MockFunctionCtx()
    y_ref = MojoSiluFunction._registry.get("torch").forward(ctx_ref, x)
    assert_close(y, y_ref)

    dy = torch.rand_like(y)
    dx = MojoSiluFunction.backward(ctx, dy)
    dx_ref = MojoSiluFunction._registry.get("torch").backward(ctx_ref, dy)
    assert_close(dx, dx_ref)
