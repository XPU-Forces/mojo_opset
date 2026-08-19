import pytest
import torch
from mojo_opset import MojoSiluFunction
from mojo_opset import MojoGelu
from mojo_opset import MojoSilu
from mojo_opset import MojoSwiGLU
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.tests.utils import MockFunctionCtx
from mojo_opset.utils.platform import get_torch_device

device = get_torch_device()

@pytest.mark.parametrize(
    "x",
    [
        (torch.rand(128, 128)),
        (torch.rand(1024, 10240)),
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_gelu(x):
    gelu = MojoGelu()
    perf(lambda: gelu(x))  # noqa: F821


@pytest.mark.parametrize(
    "x",
    [
        (torch.rand(128, 128)),
        (torch.rand(256, 128)),
        (torch.rand(1024, 10240)),
        (torch.rand(999, 9999)),
    ]
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_silu(x):
    silu = MojoSilu()
    perf(lambda: silu(x))  # noqa: F821

@pytest.mark.parametrize("shape", [([128, 128]), ([999, 9999]), ([1024, 10240]),])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_silu_forward_backward_diff(shape):
    x = torch.rand(*shape, requires_grad=True, device=device)
    ctx = MockFunctionCtx()
    perf(lambda: MojoSiluFunction.forward(ctx, x))
    y = MojoSiluFunction.forward(ctx, x)
    dy = torch.rand_like(y)
    perf(lambda: MojoSiluFunction.backward(ctx, dy))



@pytest.mark.parametrize(
    "shape",
    [
        ([256, 128]),
        ([1024, 10240]),
        ([999, 9999]),
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_swiglu(shape):
    gate_out = torch.rand(*shape, dtype=torch.bfloat16, device=device,)
    up_out = torch.rand(*shape, dtype=torch.bfloat16, device=device,)
    swiglu = MojoSwiGLU()
    perf(lambda: swiglu(gate_out, up_out))  # noqa: F821
