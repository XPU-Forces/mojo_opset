import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoNorm


@pytest.mark.parametrize(
    "x, gamma",
    [
        (
            torch.randn(size=(1, 32, 2048), dtype=dtype),
            torch.randn(size=(2048,), dtype=torch.float32),
        )
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ],
)
@pytest.mark.parametrize("epsilon", [1e-5])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_rmsnorm(x, gamma, epsilon):
    rmsnorm = MojoNorm(
        epsilon=epsilon,
        norm_type="rmsnorm",
        gamma=gamma,
    ).to(x.device)

    with torch.no_grad():
        rmsnorm.gamma.copy_(gamma.to(torch.float32))

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 3e-2, 6e-3
    perf(rmsnorm(x, atol=atol, rtol=rtol))  # noqa: F821


@pytest.mark.parametrize(
    "x, gamma, beta",
    [
        (
            torch.randn(size=(256, 128), dtype=dtype),
            torch.randn(size=(128,), dtype=torch.float32),
            torch.randn(size=(128,), dtype=torch.float32),
        )
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ],
)
@pytest.mark.parametrize("epsilon", [1e-5])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_layernorm(x, gamma, beta, epsilon):
    layernorm = MojoNorm(
        epsilon=epsilon,
        norm_type="layernorm",
        gamma=gamma,
        beta=beta,
    ).to(x.device)

    with torch.no_grad():
        layernorm.gamma.copy_(gamma.to(torch.float32))
        layernorm.beta.copy_(beta.to(torch.float32))

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 3e-2, 6e-3
    perf(layernorm(x, atol=atol, rtol=rtol))  # noqa: F821
