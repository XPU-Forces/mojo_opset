import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoNorm
from mojo_opset import MojoResidualAddNorm
from mojo_opset.backends.reference.operators.normalization import RefNorm
from mojo_opset.backends.reference.operators.normalization import RefResidualAddNorm


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
@auto_switch_platform(set_perf=True)
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
    add_norm_ref = RefResidualAddNorm(
        gamma=gamma,
        beta=beta,
        epsilon=epsilon,
        norm_pos=norm_pos,
        norm_type=norm_type,
    )

    perf(lambda: add_norm_ref(x, residual))  # noqa: F821
    perf(lambda: add_norm(x, residual))  # noqa: F821


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
    rmsnorm_ref = RefNorm(
        epsilon=epsilon,
        norm_type="rmsnorm",
        gamma=gamma,
    ).to(x.device)

    with torch.no_grad():
        rmsnorm.gamma.copy_(gamma.to(torch.float32))

    perf(lambda: rmsnorm_ref(x))  # noqa: F821
    perf(lambda: rmsnorm(x))  # noqa: F821


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
    layernorm_ref = RefNorm(
        epsilon=epsilon,
        norm_type="layernorm",
        gamma=gamma,
        beta=beta,
    ).to(x.device)

    with torch.no_grad():
        layernorm.gamma.copy_(gamma.to(torch.float32))
        layernorm.beta.copy_(beta.to(torch.float32))

    perf(lambda: layernorm_ref(x))  # noqa: F821
    perf(lambda: layernorm(x))  # noqa: F821
