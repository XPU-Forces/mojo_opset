import pytest
import torch

from tests.utils import MockFunctionCtx
from tests.utils import assert_close
from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset.training.operators.activation import MojoSiluFunction


@pytest.mark.parametrize("x", [torch.rand(128, 128, requires_grad=True)])
@auto_switch_platform()
@bypass_not_implemented
def test_silu_forward_backward_diff(monkeypatch, x):
    # silu = MojoSiluModule()
    # y = silu(x)

    # silu_ref = MojoSiluFunction._registry.get("torch")
    # y_ref = silu_ref(x)
    # assert_close(y, y_ref)

    # dx = silu.backward(dy)
    # dx_ref = silu_ref.backward(dy)
    # assert_close(dx, dx_ref)

    import os

    os.environ["MOJO_BACKEND"] = "npu"
    ctx = MockFunctionCtx()
    y = MojoSiluFunction.forward(ctx, x)
    dy = torch.rand_like(y)

    os.environ["MOJO_BACKEND"] = "torch"
    ctx_ref = MockFunctionCtx()
    y_ref = MojoSiluFunction.forward(ctx_ref, x)
    assert_close(y, y_ref)

    os.environ["MOJO_BACKEND"] = "npu"
    dx = MojoSiluFunction.backward(ctx, dy)
    os.environ["MOJO_BACKEND"] = "torch"
    dx_ref = MojoSiluFunction.backward(ctx_ref, dy)
    assert_close(dx, dx_ref)

    # assert_close(dx, dx_func)
