import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoLayerNorm
from mojo_opset import MojoResidualAddLayerNorm
from mojo_opset import MojoResidualAddRMSNorm
from mojo_opset import MojoRMSNorm

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
def test_rmsnorm(x, weight, eps):
    rmsnorm = MojoRMSNorm(eps=eps, norm_size=weight.size(0), device=x.device, dtype=weight.dtype)

    rmsnorm_ref = (
        MojoRMSNorm._registry.get("torch")(
            eps=eps,
            norm_size=weight.size(0),
        )
        .to(x.device)
        .to(weight.dtype)
    )

    with torch.no_grad():
        rmsnorm.weight.copy_(weight.to(torch.float32))
        rmsnorm_ref.weight.copy_(weight.to(torch.float32))

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 3e-2, 6e-3
    rmsnorm.forward_diff_with(rmsnorm_ref, x, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "x, weight, bias",
    [
        (
            torch.randn(size=(256, 128), dtype=dtype),
            torch.randn(size=(128,), dtype=dtype),
            torch.randn(size=(128,), dtype=dtype),
        )
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ],
)
@pytest.mark.parametrize("eps", [1e-5])
@auto_switch_platform()
@bypass_not_implemented
def test_layernorm(x, weight, bias, eps):
    layernorm = MojoLayerNorm(eps=eps, norm_size=weight.size(0), dtype=weight.dtype, device=x.device)

    layernorm_ref = (
        MojoLayerNorm._registry.get("torch")(
            eps=eps,
            norm_size=weight.size(0),
        )
        .to(x.device)
        .to(weight.dtype)
    )

    with torch.no_grad():
        layernorm.weight.copy_(weight.to(torch.float32))
        layernorm.bias.copy_(bias.to(torch.float32))
        layernorm_ref.weight.copy_(weight.to(torch.float32))
        layernorm_ref.bias.copy_(bias.to(torch.float32))

    if x.dtype == torch.float32:
        atol, rtol = 1e-4, 1e-5
    else:
        atol, rtol = 5e-2, 1e-2
    layernorm.forward_diff_with(layernorm_ref, x, atol=atol, rtol=rtol)


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
@auto_switch_platform()
@bypass_not_implemented
def test_residual_add_rms_norm(x, residual, weight, norm_pos, eps):
    add_norm = MojoResidualAddRMSNorm(
        norm_size=weight.size(0), eps=eps, norm_pos=norm_pos, device=x.device, dtype=weight.dtype
    )
    add_norm_ref = MojoResidualAddRMSNorm._registry.get("torch")(
        norm_size=weight.size(0),
        eps=eps,
        norm_pos=norm_pos,
    )

    add_norm_ref.weight = add_norm.weight = torch.nn.Parameter(weight)

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
@auto_switch_platform()
@bypass_not_implemented
def test_residual_add_layernorm(x, residual, weight, bias, norm_pos, eps):
    add_norm = MojoResidualAddLayerNorm(
        norm_size=weight.size(0),
        eps=eps,
        norm_pos=norm_pos,
    )
    add_norm_ref = MojoResidualAddLayerNorm._registry.get("torch")(
        norm_size=weight.size(0),
        eps=eps,
        norm_pos=norm_pos,
    )

    add_norm.weight = torch.nn.Parameter(weight)
    add_norm.bias = torch.nn.Parameter(bias)
    add_norm_ref.weight = torch.nn.Parameter(weight)
    add_norm_ref.bias = torch.nn.Parameter(bias)

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
