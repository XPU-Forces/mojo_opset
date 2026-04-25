import pytest
import torch

from mojo_opset.utils.platform import get_platform
from mojo_opset.tests.utils import bypass_not_implemented

from mojo_opset import MojoDequantSwiGLUQuant
from mojo_opset import MojoDequant
from mojo_opset import MojoDynamicQuant
from mojo_opset import MojoMoEDynamicQuant
from mojo_opset import MojoStaticQuant

torch.manual_seed(42)

dtypes = [torch.float16, torch.bfloat16]


# ---------------------------------------------------------------------------
# Helpers: pre-compute scale outside the operator
# ---------------------------------------------------------------------------

def make_per_token_scale_sym(x: torch.Tensor, q_max: int = 127) -> torch.Tensor:
    """Per-token symmetric scale: shape (..., 1)."""
    return (x.float().abs().amax(dim=-1, keepdim=True) / q_max).clamp(min=1e-10)


def make_per_tensor_scale_sym(x: torch.Tensor, q_max: int = 127) -> torch.Tensor:
    """Per-tensor symmetric scale: scalar tensor."""
    return (x.float().abs().amax() / q_max).clamp(min=1e-10)


def make_per_channel_scale_sym(x: torch.Tensor, q_max: int = 127) -> torch.Tensor:
    """Per-channel symmetric scale: shape (K,)."""
    return (x.float().abs().amax(dim=0) / q_max).clamp(min=1e-10)


def load_params(module: torch.nn.Module, **params):
    module.load_state_dict(params, strict=False)
    return module


# ---------------------------------------------------------------------------
# MojoStaticQuant: per-channel scale parameter
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
    scale = make_per_channel_scale_sym(x)

    quant = load_params(MojoStaticQuant(input_size=shape[-1], quant_dtype=torch.int8), scale=scale)
    quant_ref = load_params(
        MojoStaticQuant._registry.get("torch")(input_size=shape[-1], quant_dtype=torch.int8),
        scale=scale.clone(),
    )
    quant.forward_diff_with(quant_ref, x, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# MojoStaticQuant: scalar-like scale parameter
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
    scale = make_per_tensor_scale_sym(x).expand(shape[-1]).clone()

    quant = load_params(MojoStaticQuant(input_size=shape[-1], quant_dtype=torch.int8), scale=scale)
    quant_ref = load_params(
        MojoStaticQuant._registry.get("torch")(input_size=shape[-1], quant_dtype=torch.int8),
        scale=scale.clone(),
    )
    quant.forward_diff_with(quant_ref, x, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# MojoStaticQuant: float8_e4m3fn
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
    scale = (x.float().abs().amax(dim=0) / fp8_max).clamp(min=1e-10)

    quant = load_params(
        MojoStaticQuant._registry.get("torch")(input_size=shape[-1], quant_dtype=torch.float8_e4m3fn),
        scale=scale,
    )
    out = quant(x)
    expected = torch.clamp(torch.round(x.float() / scale.float()), -fp8_max, fp8_max).to(torch.float8_e4m3fn)
    torch.testing.assert_close(out.float(), expected.float(), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# MojoStaticQuant: unsupported dtype raises NotImplementedError
# ---------------------------------------------------------------------------
def test_quant_unsupported_dtype_raises():
    with pytest.raises(NotImplementedError, match="Unsupported quant_dtype"):
        MojoStaticQuant._registry.get("torch")(input_size=1, quant_dtype=torch.float32)


# ---------------------------------------------------------------------------
# MojoDequant
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
    scale = make_per_channel_scale_sym(x)
    quant_op = load_params(
        MojoStaticQuant._registry.get("torch")(input_size=shape[-1], quant_dtype=torch.int8),
        scale=scale,
    )
    quantized = quant_op(x)

    dequant = load_params(MojoDequant(input_size=shape[-1], output_dtype=output_dtype), scale=scale)
    dequant_ref = load_params(
        MojoDequant._registry.get("torch")(input_size=shape[-1], output_dtype=output_dtype),
        scale=scale.clone(),
    )
    dequant.forward_diff_with(dequant_ref, quantized, atol=0, rtol=0)


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
    scale = make_per_channel_scale_sym(x)

    quant_op = load_params(
        MojoStaticQuant._registry.get("torch")(input_size=shape[-1], quant_dtype=torch.int8),
        scale=scale,
    )
    dequant_op = load_params(
        MojoDequant._registry.get("torch")(input_size=shape[-1], output_dtype=dtype),
        scale=scale.clone(),
    )

    quantized = quant_op(x)
    recovered = dequant_op(quantized)

    torch.testing.assert_close(recovered.to(torch.float32), x.to(torch.float32), atol=5e-2, rtol=5e-2)


def test_dequant_invalid_output_dtype_raises():
    with pytest.raises(NotImplementedError, match="Unsupported output_dtype"):
        MojoDequant._registry.get("torch")(input_size=1, output_dtype=torch.int8)


def test_quant_scales_are_parameters():
    static_quant = MojoStaticQuant._registry.get("torch")(input_size=16, quant_dtype=torch.int8)
    dequant = MojoDequant._registry.get("torch")(input_size=16, output_dtype=torch.bfloat16)
    dynamic_quant = MojoDynamicQuant._registry.get("torch")(input_size=16, quant_dtype=torch.int8)
    moe_dynamic_quant = MojoMoEDynamicQuant._registry.get("torch")(
        expert_num=2,
        input_size=16,
        quant_dtype=torch.int8,
    )
    swiglu_quant = MojoDequantSwiGLUQuant._registry.get("torch")(
        expert_num=2,
        hidden_size=16,
        quant_dtype=torch.int8,
    )

    assert isinstance(static_quant.scale, torch.nn.Parameter)
    assert isinstance(dequant.scale, torch.nn.Parameter)
    assert isinstance(dynamic_quant.smooth_scale, torch.nn.Parameter)
    assert isinstance(moe_dynamic_quant.smooth_scale, torch.nn.Parameter)
    assert isinstance(swiglu_quant.weight_scale, torch.nn.Parameter)
    assert isinstance(swiglu_quant.quant_scale, torch.nn.Parameter)
    assert moe_dynamic_quant.smooth_scale.shape == (2, 16)
    assert swiglu_quant.weight_scale.shape == (2, 32)
    assert swiglu_quant.quant_scale.shape == (2, 16)

    assert set(static_quant.state_dict()) == {"scale"}
    assert set(dequant.state_dict()) == {"scale"}
    assert set(dynamic_quant.state_dict()) == {"smooth_scale"}
    assert set(moe_dynamic_quant.state_dict()) == {"smooth_scale"}
    assert set(swiglu_quant.state_dict()) == {"weight_scale", "quant_scale"}


# ---------------------------------------------------------------------------
# MojoDynamicQuant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_dynamic_quant_reference(dtype):
    x = torch.randn(32, 128, dtype=dtype)
    smooth_scale = torch.randn(128, dtype=torch.float32)

    op = load_params(
        MojoDynamicQuant._registry.get("torch")(input_size=smooth_scale.numel(), quant_dtype=torch.int8),
        smooth_scale=smooth_scale,
    )
    out, scale = op(x)

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

    op = load_params(MojoDynamicQuant(input_size=smooth_scale.numel(), quant_dtype=torch.int8), smooth_scale=smooth_scale)
    op_ref = load_params(
        MojoDynamicQuant._registry.get("torch")(input_size=smooth_scale.numel(), quant_dtype=torch.int8),
        smooth_scale=smooth_scale.clone(),
    )
    op.forward_diff_with(op_ref, x, atol=(1, 2e-4), rtol=(0, 2e-4))


@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_dynamic_quant_backend_moe(dtype):
    x = torch.randn(12, 128, dtype=dtype)
    smooth_scale = torch.randn(3, 128, dtype=torch.float32)
    token_count = torch.tensor([4, 3, 5], dtype=torch.int32)

    op = load_params(
        MojoMoEDynamicQuant(expert_num=3, input_size=128, quant_dtype=torch.int8),
        smooth_scale=smooth_scale,
    )
    op_ref = load_params(
        MojoMoEDynamicQuant._registry.get("torch")(expert_num=3, input_size=128, quant_dtype=torch.int8),
        smooth_scale=smooth_scale.clone(),
    )
    op.forward_diff_with(
        op_ref,
        x,
        token_count,
        atol=(1, 2e-4),
        rtol=(0, 2e-4),
    )


def test_dynamic_quant_rejects_token_count():
    x = torch.randn(4, 16)
    token_count = torch.tensor([4], dtype=torch.int32)
    op = MojoDynamicQuant._registry.get("torch")(quant_dtype=torch.int8)
    with pytest.raises(TypeError):
        op(x, token_count)


def test_moe_dynamic_quant_token_count_shape_check():
    x = torch.randn(4, 16)
    token_count = torch.tensor([2, 1], dtype=torch.int32)
    op = MojoMoEDynamicQuant._registry.get("torch")(expert_num=2, input_size=16, quant_dtype=torch.int8)
    with pytest.raises(ValueError, match="token_count sum"):
        op(x, token_count)


def test_moe_dynamic_quant_requires_smooth_scale():
    x = torch.randn(4, 16)
    token_count = torch.tensor([4], dtype=torch.int32)
    with pytest.raises(TypeError, match="expert_num"):
        MojoMoEDynamicQuant._registry.get("torch")(quant_dtype=torch.int8)(x, token_count)


# ---------------------------------------------------------------------------
# MojoDequantSwiGLUQuant
# ---------------------------------------------------------------------------
@bypass_not_implemented
def test_dequant_swiglu_quant_backend():
    tokens = 12
    hidden = 64
    # torch_npu reqires int64
    token_count = torch.tensor([5, 7], dtype=torch.int64)

    x = torch.randint(-1024, 1024, (tokens, hidden * 2), dtype=torch.int32)
    weight_scale = torch.rand(2, hidden * 2, dtype=torch.float32)
    activation_scale = torch.rand(tokens, dtype=torch.float32)
    quant_scale = torch.rand(2, hidden, dtype=torch.float32)

    op = load_params(
        MojoDequantSwiGLUQuant(
            expert_num=2,
            hidden_size=hidden,
            activate_left=False,
            quant_mode=1,
        ),
        weight_scale=weight_scale,
        quant_scale=quant_scale,
    )
    op_ref = load_params(
        MojoDequantSwiGLUQuant._registry.get("torch")(
            expert_num=2,
            hidden_size=hidden,
            activate_left=False,
            quant_mode=1,
        ),
        weight_scale=weight_scale.clone(),
        quant_scale=quant_scale.clone(),
    )
    op.forward_diff_with(
        op_ref,
        x,
        activation_scale,
        None,
        None,
        token_count,
        atol=(0, 1e-4),
        rtol=(0, 1e-4),
    )
