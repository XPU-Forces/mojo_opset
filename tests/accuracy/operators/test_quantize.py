import pytest
import torch

from mojo_opset.utils.platform import get_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoDequantSwiGLUQuant
from mojo_opset import MojoDequant
from mojo_opset import MojoDynamicQuant
from mojo_opset import MojoQuant

torch.manual_seed(42)

dtypes = [torch.float16, torch.bfloat16]


# ---------------------------------------------------------------------------
# Helpers: pre-compute scale / zero_point outside the operator
# ---------------------------------------------------------------------------

def make_per_token_scale_sym(x: torch.Tensor, q_max: int = 127) -> torch.Tensor:
    """Per-token symmetric scale: shape (..., 1)."""
    return (x.float().abs().amax(dim=-1, keepdim=True) / q_max).clamp(min=1e-10)


def make_per_tensor_scale_sym(x: torch.Tensor, q_max: int = 127) -> torch.Tensor:
    """Per-tensor symmetric scale: scalar tensor."""
    return (x.float().abs().amax() / q_max).clamp(min=1e-10)


def make_per_token_scale_asym(x: torch.Tensor, q_max: int = 127, q_min: int = 0):
    """Per-token asymmetric scale + zero_point: shapes (..., 1)."""
    x_fp = x.float()
    val_min = x_fp.amin(dim=-1, keepdim=True)
    val_max = x_fp.amax(dim=-1, keepdim=True)
    scale = ((val_max - val_min) / (q_max - q_min)).clamp(min=1e-10)
    zero_point = torch.round(-val_min / scale).clamp(q_min, q_max)
    return scale, zero_point


def make_per_group_scale_sym(x: torch.Tensor, group_size: int, q_max: int = 127) -> torch.Tensor:
    """Per-group symmetric scale: shape (..., K // group_size, 1)."""
    orig = x.shape
    xg = x.float().reshape(*orig[:-1], -1, group_size)
    return (xg.abs().amax(dim=-1, keepdim=True) / q_max).clamp(min=1e-10)


# ---------------------------------------------------------------------------
# MojoQuant: symmetric, per-token (user provides scale)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        (32, 128),
        (64, 1024),
        (1, 4096),
        (128, 8192),
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_quant_symmetric_per_token(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype)
    scale = make_per_token_scale_sym(x)

    quant = MojoQuant(quant_dtype=torch.int8, symmetric=True)
    quant_ref = MojoQuant._registry.get("torch")(quant_dtype=torch.int8, symmetric=True)
    quant.forward_diff_with(quant_ref, x, scale, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# MojoQuant: symmetric, per-tensor (user provides scale)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        (32, 128),
        (64, 1024),
        (128, 8192),
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_quant_symmetric_per_tensor(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype)
    scale = make_per_tensor_scale_sym(x)

    quant = MojoQuant(quant_dtype=torch.int8, symmetric=True)
    quant_ref = MojoQuant._registry.get("torch")(quant_dtype=torch.int8, symmetric=True)
    quant.forward_diff_with(quant_ref, x, scale, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# MojoQuant: asymmetric, per-token (user provides scale + zero_point)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        (32, 128),
        (64, 1024),
        (128, 8192),
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_quant_asymmetric_per_token(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype)
    scale, zero_point = make_per_token_scale_asym(x)

    quant = MojoQuant(quant_dtype=torch.int8, symmetric=False)
    quant_ref = MojoQuant._registry.get("torch")(quant_dtype=torch.int8, symmetric=False)
    quant.forward_diff_with(quant_ref, x, scale, zero_point, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# MojoQuant: symmetric, per-group (user provides scale)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape, group_size",
    [
        ((32, 128), 64),
        ((64, 1024), 128),
        ((16, 4096), 256),
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_quant_symmetric_per_group(shape, group_size, dtype):
    x = torch.randn(size=shape, dtype=dtype)
    scale = make_per_group_scale_sym(x, group_size)

    quant = MojoQuant(quant_dtype=torch.int8, symmetric=True, group_size=group_size)
    quant_ref = MojoQuant._registry.get("torch")(
        quant_dtype=torch.int8, symmetric=True, group_size=group_size
    )
    quant.forward_diff_with(quant_ref, x, scale, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# MojoQuant: float8_e4m3fn symmetric
# ---------------------------------------------------------------------------
_requires_cpu = pytest.mark.skipif(
    get_platform() != "cpu",
    reason="float8_e4m3fn not supported on NPU; reference-only test requires CPU",
)


@_requires_cpu
@pytest.mark.parametrize("shape", [(32, 128), (64, 1024)])
@pytest.mark.parametrize("dtype", dtypes)
def test_quant_float8_symmetric(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = (x.float().abs().amax(dim=-1, keepdim=True) / fp8_max).clamp(min=1e-10)

    quant = MojoQuant._registry.get("torch")(quant_dtype=torch.float8_e4m3fn, symmetric=True)
    out = quant(x, scale)
    expected = torch.clamp(torch.round(x.float() / scale.float()), -fp8_max, fp8_max).to(torch.float8_e4m3fn)
    torch.testing.assert_close(out.float(), expected.float(), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# MojoQuant: unsupported dtype raises NotImplementedError
# ---------------------------------------------------------------------------
def test_quant_unsupported_dtype_raises():
    with pytest.raises(NotImplementedError, match="Unsupported quant_dtype"):
        MojoQuant._registry.get("torch")(quant_dtype=torch.float32)


# ---------------------------------------------------------------------------
# MojoQuant: asymmetric requires zero_point (error path)
# ---------------------------------------------------------------------------
def test_quant_asymmetric_missing_zero_point_raises():
    x = torch.randn(8, 64)
    scale = make_per_token_scale_sym(x)
    quant = MojoQuant._registry.get("torch")(quant_dtype=torch.int8, symmetric=False)
    with pytest.raises(ValueError, match="zero_point"):
        quant(x, scale)


# ---------------------------------------------------------------------------
# MojoDequant: symmetric
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        (32, 128),
        (64, 1024),
        (128, 8192),
    ],
)
@pytest.mark.parametrize("output_dtype", dtypes)
@bypass_not_implemented
def test_dequant_symmetric(shape, output_dtype):
    x = torch.randn(size=shape, dtype=output_dtype)
    scale = make_per_token_scale_sym(x)
    quant_op = MojoQuant._registry.get("torch")(quant_dtype=torch.int8, symmetric=True)
    quantized = quant_op(x, scale)

    dequant = MojoDequant(output_dtype=output_dtype, symmetric=True)
    dequant_ref = MojoDequant._registry.get("torch")(output_dtype=output_dtype, symmetric=True)
    dequant.forward_diff_with(dequant_ref, quantized, scale, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# MojoDequant: asymmetric
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        (32, 128),
        (64, 1024),
    ],
)
@pytest.mark.parametrize("output_dtype", dtypes)
@bypass_not_implemented
def test_dequant_asymmetric(shape, output_dtype):
    x = torch.randn(size=shape, dtype=output_dtype)
    scale, zero_point = make_per_token_scale_asym(x)
    quant_op = MojoQuant._registry.get("torch")(quant_dtype=torch.int8, symmetric=False)
    quantized = quant_op(x, scale, zero_point)

    dequant = MojoDequant(output_dtype=output_dtype, symmetric=False)
    dequant_ref = MojoDequant._registry.get("torch")(output_dtype=output_dtype, symmetric=False)
    dequant.forward_diff_with(dequant_ref, quantized, scale, zero_point, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# MojoDequant: per-group symmetric
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape, group_size",
    [
        ((32, 128), 64),
        ((64, 1024), 128),
    ],
)
@pytest.mark.parametrize("output_dtype", dtypes)
@bypass_not_implemented
def test_dequant_symmetric_per_group(shape, group_size, output_dtype):
    x = torch.randn(size=shape, dtype=output_dtype)
    scale = make_per_group_scale_sym(x, group_size)
    quant_op = MojoQuant._registry.get("torch")(
        quant_dtype=torch.int8, symmetric=True, group_size=group_size
    )
    quantized = quant_op(x, scale)

    dequant = MojoDequant(output_dtype=output_dtype, symmetric=True, group_size=group_size)
    dequant_ref = MojoDequant._registry.get("torch")(
        output_dtype=output_dtype, symmetric=True, group_size=group_size
    )
    dequant.forward_diff_with(dequant_ref, quantized, scale, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Round-trip: quant → dequant should approximate original
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        (64, 256),
        (128, 1024),
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_quant_dequant_roundtrip(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype)
    scale = make_per_token_scale_sym(x)

    quant_op = MojoQuant._registry.get("torch")(quant_dtype=torch.int8, symmetric=True)
    dequant_op = MojoDequant._registry.get("torch")(output_dtype=dtype, symmetric=True)

    quantized = quant_op(x, scale)
    recovered = dequant_op(quantized, scale)

    torch.testing.assert_close(recovered.to(torch.float32), x.to(torch.float32), atol=5e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# MojoDynamicQuant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_dynamic_quant_reference(dtype):
    x = torch.randn(32, 128, dtype=dtype)
    smooth_scale = torch.randn(128, dtype=torch.float32)

    op = MojoDynamicQuant._registry.get("torch")(quant_dtype=torch.int8, smooth_input=True)
    out, scale = op(x, smooth_scale)

    expected_input = x.float() * smooth_scale.float().unsqueeze(0)
    expected_scale = expected_input.abs().amax(dim=-1).clamp(min=1e-12) / 127
    expected_out = torch.clamp(
        torch.round(expected_input / expected_scale.unsqueeze(-1)),
        -128,
        127,
    ).to(torch.int8)

    torch.testing.assert_close(out, expected_out, atol=0, rtol=0)
    torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_dynamic_quant_backend(dtype):
    x = torch.randn(24, 128, dtype=dtype)
    smooth_scale = torch.randn(128, dtype=torch.float32)

    op = MojoDynamicQuant(quant_dtype=torch.int8, smooth_input=True)
    op_ref = MojoDynamicQuant._registry.get("torch")(quant_dtype=torch.int8, smooth_input=True)
    op.forward_diff_with(op_ref, x, smooth_scale, atol=(1, 2e-4), rtol=(0, 2e-4))


@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_dynamic_quant_backend_moe(dtype):
    x = torch.randn(12, 128, dtype=dtype)
    smooth_scale = torch.randn(3, 128, dtype=torch.float32)
    token_count = torch.tensor([4, 3, 5], dtype=torch.int64)

    op = MojoDynamicQuant(quant_dtype=torch.int8, smooth_input=True, moe_mode=True)
    op_ref = MojoDynamicQuant._registry.get("torch")(quant_dtype=torch.int8, smooth_input=True, moe_mode=True)
    op.forward_diff_with(
        op_ref,
        x,
        smooth_scale,
        token_count,
        atol=(1, 2e-4),
        rtol=(0, 2e-4),
    )


# ---------------------------------------------------------------------------
# MojoDequantSwiGLUQuant
# ---------------------------------------------------------------------------
@bypass_not_implemented
def test_dequant_swiglu_quant_backend():
    tokens = 12
    hidden = 64
    token_count = torch.tensor([5, 7], dtype=torch.int64)

    x = torch.randint(-1024, 1024, (tokens, hidden * 2), dtype=torch.int32)
    weight_scale = torch.rand(2, hidden * 2, dtype=torch.float32)
    activation_scale = torch.rand(tokens, dtype=torch.float32)
    quant_scale = torch.rand(2, hidden, dtype=torch.float32)

    op = MojoDequantSwiGLUQuant(activate_left=False, quant_mode=1)
    op_ref = MojoDequantSwiGLUQuant._registry.get("torch")(activate_left=False, quant_mode=1)
    op.forward_diff_with(
        op_ref,
        x,
        weight_scale,
        activation_scale,
        None,
        quant_scale,
        None,
        token_count,
        atol=(0, 1e-4),
        rtol=(0, 1e-4),
    )
