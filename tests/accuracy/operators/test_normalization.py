import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoLayerNorm
from mojo_opset import MojoResidualAddLayerNorm
from mojo_opset import MojoResidualAddRMSNorm
from mojo_opset import MojoRMSNorm
from mojo_opset import MojoResidualAddRMSNormCast
from mojo_opset import MojoResidualAddLayerNormCast

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
    rmsnorm = MojoRMSNorm(
        eps=eps,
        weight=weight,
    ).to(x.device)

    rmsnorm_ref = MojoRMSNorm._registry.get("torch")(
        eps=eps,
        weight=weight,
    ).to(x.device)

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
    layernorm = MojoLayerNorm(
        eps=eps,
        weight=weight,
        bias=bias,
    ).to(x.device)

    layernorm_ref = MojoLayerNorm._registry.get("torch")(
        eps=eps,
        weight=weight,
        bias=bias,
    ).to(x.device)

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
        weight=weight,
        eps=eps,
        norm_pos=norm_pos,
    )
    add_norm_ref = MojoResidualAddRMSNorm._registry.get("torch")(
        weight=weight,
        eps=eps,
        norm_pos=norm_pos,
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
@pytest.mark.parametrize("cast_type", ["int8", "float32"])
@auto_switch_platform()
@bypass_not_implemented
def test_residual_add_rms_norm_cast(x, residual, weight, norm_pos, eps, cast_type):
    device = x.device

    add_norm_cast = MojoResidualAddRMSNormCast(
        norm_size=weight.size(0), 
        eps=eps, 
        norm_pos=norm_pos,
        cast_type=cast_type,
        device=device, 
        dtype=weight.dtype
    )

    add_norm_cast_ref = MojoResidualAddRMSNormCast._registry.get("torch")(
        norm_size=weight.size(0),
        eps=eps,
        norm_pos=norm_pos,
        cast_type=cast_type,
    ).to(device).to(weight.dtype)
    
    with torch.no_grad():
        add_norm_cast.weight.copy_(weight)
        add_norm_cast_ref.weight.copy_(weight)

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    elif x.dtype == torch.float16:
        atol, rtol = 3e-2, 6e-3
    else:  # bfloat16
        atol, rtol = 5e-2, 1e-2

    if cast_type == "int8":
        atol *= 10
        rtol *= 10

    add_norm_cast.forward_diff_with(
        add_norm_cast_ref,
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
@pytest.mark.parametrize("cast_type", ["int8", "float32"])
@auto_switch_platform()
@bypass_not_implemented
def test_residual_add_layer_norm_cast(x, residual, weight, bias, norm_pos, eps, cast_type):
    device = x.device

    add_norm_cast = MojoResidualAddLayerNormCast(
        norm_size=weight.size(0),
        eps=eps,
        norm_pos=norm_pos,
        cast_type=cast_type,
        device=device,
        dtype=weight.dtype
    )

    add_norm_cast_ref = MojoResidualAddLayerNormCast._registry.get("torch")(
        norm_size=weight.size(0),
        eps=eps,
        norm_pos=norm_pos,
        cast_type=cast_type,
    ).to(device).to(weight.dtype)

    with torch.no_grad():
        add_norm_cast.weight.copy_(weight)
        add_norm_cast.bias.copy_(bias)
        add_norm_cast_ref.weight.copy_(weight)
        add_norm_cast_ref.bias.copy_(bias)

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    elif x.dtype == torch.float16:
        atol, rtol = 3e-2, 6e-3
    else:  # bfloat16
        atol, rtol = 5e-2, 1e-2

    if cast_type == "int8":
        atol *= 10
        rtol *= 10

    add_norm_cast.forward_diff_with(
        add_norm_cast_ref,
        x,
        residual,
        atol=atol,
        rtol=rtol,
    )