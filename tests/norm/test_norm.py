import pytest
import torch

from tests.utils import auto_switch_platform, bypass_not_implemented

from mojo_opset import MojoNorm


@pytest.mark.parametrize(
    "x, weight",
    [
        (
            torch.randn(size=(256, 128), dtype=dtype),
            torch.randn(size=(128,), dtype=torch.float32),
        )
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ],
)
@pytest.mark.parametrize("eps", [1e-5])
@auto_switch_platform()
@bypass_not_implemented
def test_rmsnorm(x, weight, eps):
    rmsnorm = MojoNorm(
        hidden_size=x.shape[-1],
        eps=eps,
        norm_type="rmsnorm",
    ).to(x.device)

    with torch.no_grad():
        rmsnorm.weight.copy_(weight.to(torch.float32))

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 3e-2, 6e-3
    rmsnorm.forward_diff(x, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "x, weight, bias",
    [
        (
            torch.randn(size=(256, 128), dtype=dtype),
            torch.randn(size=(128,), dtype=torch.float32),
            torch.randn(size=(128,), dtype=torch.float32),
        )
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ],
)
@pytest.mark.parametrize("eps", [1e-5])
@auto_switch_platform()
@bypass_not_implemented
def test_layernorm(x, weight, bias, eps):
    layernorm = MojoNorm(
        hidden_size=x.shape[-1],
        eps=eps,
        norm_type="layernorm",
    ).to(x.device)

    with torch.no_grad():
        layernorm.weight.copy_(weight)
        layernorm.bias.copy_(bias)

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 3e-2, 6e-3
    layernorm.forward_diff(x, atol=atol, rtol=rtol)
