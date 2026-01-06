import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoNorm
from mojo_opset import MojoResidualAddNorm

torch.manual_seed(42)

shapes = [
    (32, 1024),
    (64, 8192),
    (57, 7338),
    (2, 256),
    (7762, 18778),
]
dtypes = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize(
    "x, gamma",
    [
        (
            torch.randn(size=shape, dtype=dtype),
            torch.randn(size=(shape[-1],), dtype=torch.float32),
        )
        for dtype in dtypes
        for shape in shapes
    ],
)
@pytest.mark.parametrize("epsilon", [1e-5])
@auto_switch_platform()
@bypass_not_implemented
def test_rmsnorm(x, gamma, epsilon):
    rmsnorm = MojoNorm(
        epsilon=epsilon,
        norm_type="rmsnorm",
        gamma=gamma,
    ).to(x.device)

    rmsnorm_ref = MojoNorm._registry.get("ref")(
        epsilon=epsilon,
        norm_type="rmsnorm",
        gamma=gamma,
    ).to(x.device)

    with torch.no_grad():
        rmsnorm.gamma.copy_(gamma.to(torch.float32))
        rmsnorm_ref.gamma.copy_(gamma.to(torch.float32))

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 3e-2, 6e-3
    rmsnorm.forward_diff_with(rmsnorm_ref, x, atol=atol, rtol=rtol)


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
@auto_switch_platform()
@bypass_not_implemented
def test_layernorm(x, gamma, beta, epsilon):
    layernorm = MojoNorm(
        epsilon=epsilon,
        norm_type="layernorm",
        gamma=gamma,
        beta=beta,
    ).to(x.device)

    layernorm_ref = MojoNorm._registry.get("ref")(
        epsilon=epsilon,
        norm_type="layernorm",
        gamma=gamma,
        beta=beta,
    ).to(x.device)

    with torch.no_grad():
        layernorm.gamma.copy_(gamma.to(torch.float32))
        layernorm.beta.copy_(beta.to(torch.float32))
        layernorm_ref.gamma.copy_(gamma.to(torch.float32))
        layernorm_ref.beta.copy_(beta.to(torch.float32))

    if x.dtype == torch.float32:
        atol, rtol = 1e-4, 1e-5
    else:
        atol, rtol = 5e-2, 1e-2
    layernorm.forward_diff_with(layernorm_ref, x, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "x, residual, gamma, beta",
    [
        (
            torch.randn(size=(128, 128), dtype=dtype),
            torch.randn(size=(128, 128), dtype=dtype),
            torch.randn(size=(128,), dtype=dtype),
            torch.randn(size=(128,), dtype=dtype),
        )
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ],
)
@pytest.mark.parametrize("epsilon", [1e-5])
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
@pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
@auto_switch_platform()
@bypass_not_implemented
def test_residual_add_norm(x, residual, gamma, beta, norm_type, norm_pos, epsilon):
    beta = beta if norm_type == "layernorm" else None

    add_norm = MojoResidualAddNorm(
        gamma=gamma,
        beta=beta,
        epsilon=epsilon,
        norm_pos=norm_pos,
        norm_type=norm_type,
    )
    add_norm_ref = MojoResidualAddNorm._registry.get("ref")(
        gamma=gamma,
        beta=beta,
        epsilon=epsilon,
        norm_pos=norm_pos,
        norm_type=norm_type,
    )

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 3e-2, 6e-3

    add_norm.forward_diff_with(
        add_norm_ref,
        x,
        residual,
        atol=atol,
        rtol=rtol,
    )
