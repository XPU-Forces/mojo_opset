import pytest
import torch

from tests.utils import MockFunctionCtx
from tests.utils import assert_close
from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoRMSNormFunction

shapes = [
    (32, 1024),
    (64, 8192),
    (57, 7338),
    (763, 8777),
    (7762, 18778),
]
dtypes = [torch.float32, torch.bfloat16]


@pytest.mark.parametrize(
    "x, weight",
    [
        (
            torch.randn(size=shape, dtype=dtype),
            torch.randn(size=(shape[-1],), dtype=torch.float32),
        )
        for dtype in dtypes
        for shape in shapes
    ],
)
@pytest.mark.parametrize("eps", [1e-5])
@auto_switch_platform()
@bypass_not_implemented
def test_rmsnorm_forward_backward_diff(x, weight, eps):
    ctx = MockFunctionCtx()
    y = MojoRMSNormFunction.forward(ctx, x, weight, 1e-6)

    ctx_ref = MockFunctionCtx()
    y_ref = MojoRMSNormFunction._registry.get("torch").forward(ctx_ref, x, weight, 1e-6)
    assert_close(y, y_ref)

    dy = torch.rand_like(y)
    grads = MojoRMSNormFunction.backward(ctx, dy)
    grads_ref = MojoRMSNormFunction._registry.get("torch").backward(ctx_ref, dy)

    assert_close(grads, grads_ref)
