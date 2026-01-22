import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoLayerNorm
from mojo_opset import MojoResidualAddLayerNorm
from mojo_opset import MojoResidualAddRMSNorm
from mojo_opset import MojoRMSNorm


@pytest.mark.parametrize(
    "x, residual, weight",
    [
        (
            torch.randn(size=(128, 128), dtype=dtype),
            torch.randn(size=(128, 128), dtype=dtype),
            torch.randn(size=(128,), dtype=dtype),
        )
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ],
)
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_residual_add_rmsnorm(x, residual, weight, norm_pos, eps):
    add_norm = MojoResidualAddRMSNorm(
        weight=weight,
        eps=eps,
        norm_pos=norm_pos,
    )
    add_norm_ref = MojoResidualAddRMSNorm._registry.get("torch")(
        weight=weight,
        eps=eps,
        norm_pos=norm_pos,
    )

    perf(lambda: add_norm_ref(x, residual))  # noqa: F821
    perf(lambda: add_norm(x, residual))  # noqa: F821


@pytest.mark.parametrize(
    "x, residual, weight, bias",
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
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_residual_add_layernorm(x, residual, weight, bias, norm_pos, eps):
    add_norm = MojoResidualAddLayerNorm(
        weight=weight,
        bias=bias,
        eps=eps,
        norm_pos=norm_pos,
    )
    add_norm_ref = MojoResidualAddLayerNorm._registry.get("torch")(
        weight=weight,
        bias=bias,
        eps=eps,
        norm_pos=norm_pos,
    )

    perf(lambda: add_norm_ref(x, residual))  # noqa: F821
    perf(lambda: add_norm(x, residual))  # noqa: F821


@pytest.mark.parametrize(
    "x, weight",
    [
        (
            torch.randn(size=(1, 32, 2048), dtype=dtype),
            torch.randn(size=(2048,), dtype=torch.float32),
        )
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ],
)
@pytest.mark.parametrize("eps", [1e-5])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_rmsnorm(x, weight, eps):
    rmsnorm = MojoRMSNorm(
        weight,
        eps,
    ).to(x.device)

    rmsnorm_ref = MojoRMSNorm._registry.get("torch")(
        weight,
        eps,
    ).to(x.device)

    with torch.no_grad():
        rmsnorm.weight.copy_(weight.to(torch.float32))

    perf(lambda: rmsnorm_ref(x))  # noqa: F821
    perf(lambda: rmsnorm(x))  # noqa: F821


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
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_layernorm(x, weight, bias, eps):
    layernorm = MojoLayerNorm(
        weight=weight,
        bias=bias,
        eps=eps,
    ).to(x.device)
    layernorm_ref = MojoLayerNorm._registry.get("torch")(
        weight=weight,
        bias=bias,
        eps=eps,
    ).to(x.device)

    with torch.no_grad():
        layernorm.weight.copy_(weight.to(torch.float32))
        layernorm.bias.copy_(bias.to(torch.float32))

    perf(lambda: layernorm_ref(x))  # noqa: F821
    perf(lambda: layernorm(x))  # noqa: F821
