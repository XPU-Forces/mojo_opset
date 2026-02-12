import pytest
import torch

from mojo_opset import MojoLayerNorm
from mojo_opset import MojoResidualAddLayerNorm
from mojo_opset import MojoResidualAddRMSNorm
from mojo_opset import MojoRMSNorm
from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented


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
        hidden_size=weight.size(0),
        eps=eps,
        norm_pos=norm_pos,
    ).to(x.device)
    add_norm.weight = torch.nn.Parameter(weight)

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
        hidden_size=weight.size(0),
        eps=eps,
        norm_pos=norm_pos,
    )
    add_norm.weight = torch.nn.Parameter(weight)
    add_norm.bias = torch.nn.Parameter(bias)

    perf(lambda: add_norm(x, residual))  # noqa: F821


@pytest.mark.parametrize(
    "x, weight",
    [
        (
            torch.randn(size=(1, 32, 2048), dtype=dtype),
            torch.nn.Parameter(torch.randn(size=(2048,), dtype=torch.float32)),
        )
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ],
)
@pytest.mark.parametrize("eps", [1e-5])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_rmsnorm(x, weight, eps):
    rmsnorm = MojoRMSNorm(
        weight.size(0),
        eps,
    ).to(x.device)

    with torch.no_grad():
        rmsnorm.weight.copy_(weight.to(torch.float32))

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
        hidden_size=weight.size(0),
        eps=eps,
    ).to(x.device)

    with torch.no_grad():
        layernorm.weight.copy_(weight.to(torch.float32))
        layernorm.bias.copy_(bias.to(torch.float32))

    perf(lambda: layernorm(x))  # noqa: F821
