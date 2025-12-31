import pytest
import torch

from tests.utils import MockFunctionCtx
from tests.utils import assert_close
from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoRMSNormFunction


@pytest.mark.parametrize(
    "x, w",
    [(torch.rand(128, 128, requires_grad=True), torch.rand(128, requires_grad=True))],
)
@auto_switch_platform()
@bypass_not_implemented
def test_rmsnorm_forward_backward_diff(x, w):
    ctx = MockFunctionCtx()
    y = MojoRMSNormFunction.forward(ctx, x, w, 1e-6)

    ctx_ref = MockFunctionCtx()
    y_ref = MojoRMSNormFunction._registry.get("ref").forward(ctx_ref, x, w, 1e-6)
    assert_close(y, y_ref)

    dy = torch.rand_like(y)
    grads = MojoRMSNormFunction.backward(ctx, dy)
    grads_ref = MojoRMSNormFunction._registry.get("ref").backward(ctx_ref, dy)
    
    assert_close(grads, grads_ref)
