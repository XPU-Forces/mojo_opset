import pytest
import torch

from mojo_opset import MojoDequant
from mojo_opset import MojoDequantSwiGLUQuant
from mojo_opset import MojoDynamicQuant
from mojo_opset import MojoMoEDynamicQuant
from mojo_opset import MojoStaticQuant
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_platform

torch.manual_seed(42)

dtypes = [torch.float16, torch.bfloat16]

static_quant_shapes = [
    (1, 128),
    (2, 256),
    (8, 512),
    (32, 1024),
    (64, 4096),
    (57, 7338),
    (128, 8192),
]

dequant_shapes = [
    (1, 128),
    (4, 128),
    (16, 512),
    (32, 1024),
    (96, 4096),
    (128, 8192),
]

dynamic_quant_shapes = [
    (1, 128),
    (8, 128),
    (17, 320),
    (24, 512),
    (48, 1536),
    (64, 2048),
]

moe_dynamic_quant_cases = [
    (8, 128, [8]),
    (12, 256, [4, 3, 5]),
    (18, 512, [6, 6, 4, 2]),
    (21, 1024, [2, 5, 1, 7, 6]),
    (32, 2048, [8, 7, 5, 6, 4, 2]),
]

dequant_swiglu_quant_cases = [
    (8, 32, [3, 5]),
    (12, 64, [4, 3, 5]),
    (20, 128, [6, 4, 7, 3]),
    (24, 256, [5, 8, 4, 7]),
    (30, 512, [6, 3, 8, 5, 8]),
]


def load_params(module: torch.nn.Module, **params):
    module.load_state_dict(params, strict=False)
    return module


def make_scale(x: torch.Tensor, q_max: float) -> torch.Tensor:
    return (x.float().abs().amax(dim=0) / q_max).clamp(min=1e-10)


def has_ixformer_quant_kernel(name: str) -> bool:
    if get_platform() != "ilu":
        return True
    try:
        from ixformer import functions as ixf_f
    except ImportError:
        return False
    return hasattr(ixf_f, name)


@pytest.mark.parametrize("shape", static_quant_shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("quant_dtype", [torch.int8])
@bypass_not_implemented
def test_static_quant(shape, dtype, quant_dtype):
    if quant_dtype == torch.int8 and not has_ixformer_quant_kernel("static_quant"):
        pytest.skip("static_quant kernel is not available on the current ixformer build")

    x = torch.randn(shape, dtype=dtype)
    q_max = 127
    scale = make_scale(x, q_max)

    quant = load_params(MojoStaticQuant(input_size=shape[-1], quant_dtype=quant_dtype), scale=scale)
    quant_ref = load_params(
        MojoStaticQuant._registry.get("torch")(input_size=shape[-1], quant_dtype=quant_dtype),
        scale=scale.clone(),
    )

    atol = 1 if quant_dtype == torch.int8 else 0
    quant.forward_diff_with(quant_ref, x, atol=atol, rtol=0)


@pytest.mark.parametrize("shape", dequant_shapes)
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_dequant(shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    scale = make_scale(x, 127)

    quant_ref = load_params(
        MojoStaticQuant._registry.get("torch")(input_size=shape[-1], quant_dtype=torch.int8),
        scale=scale,
    )
    quantized, quant_scale = quant_ref(x)

    dequant = MojoDequant(output_dtype=dtype)
    dequant_ref = MojoDequant._registry.get("torch")(output_dtype=dtype)
    dequant.forward_diff_with(dequant_ref, quantized, quant_scale, atol=0, rtol=0)


@pytest.mark.parametrize("shape", dynamic_quant_shapes)
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_dynamic_quant(shape, dtype):
    if not has_ixformer_quant_kernel("dynamic_quant"):
        pytest.skip("dynamic_quant kernel is not available on the current ixformer build")

    x = torch.randn(shape, dtype=dtype)
    smooth_scale = torch.rand(shape[-1], dtype=torch.float32) + 0.1
    inv_smooth_scale = 1.0 / smooth_scale

    quant = load_params(
        MojoDynamicQuant(input_size=shape[-1], quant_dtype=torch.int8),
        inv_smooth_scale=inv_smooth_scale,
    )
    quant_ref = load_params(
        MojoDynamicQuant._registry.get("torch")(input_size=shape[-1], quant_dtype=torch.int8),
        inv_smooth_scale=inv_smooth_scale,
    )
    quant.forward_diff_with(quant_ref, x, atol=(1, 2e-3), rtol=(0, 2e-3))


@pytest.mark.parametrize("tokens, hidden_size, token_count", moe_dynamic_quant_cases)
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_moe_dynamic_quant(tokens, hidden_size, token_count, dtype):
    x = torch.randn(tokens, hidden_size, dtype=dtype)
    expert_num = len(token_count)
    token_count = torch.tensor(token_count, dtype=torch.int32)
    smooth_scale = torch.rand(expert_num, hidden_size, dtype=torch.float32) + 0.1
    inv_smooth_scale = 1.0 / smooth_scale

    quant = load_params(
        MojoMoEDynamicQuant(expert_num=expert_num, input_size=hidden_size, quant_dtype=torch.int8),
        inv_smooth_scale=inv_smooth_scale,
    )
    quant_ref = load_params(
        MojoMoEDynamicQuant._registry.get("torch")(
            expert_num=expert_num,
            input_size=hidden_size,
            quant_dtype=torch.int8,
        ),
        inv_smooth_scale=inv_smooth_scale,
    )
    quant.forward_diff_with(quant_ref, x, token_count, atol=(1, 2e-3), rtol=(0, 2e-3))


@pytest.mark.parametrize("tokens, hidden_size, token_count", dequant_swiglu_quant_cases)
@bypass_not_implemented
def test_dequant_swiglu_quant(tokens, hidden_size, token_count):
    expert_num = len(token_count)
    token_count = torch.tensor(token_count, dtype=torch.int64)

    x = torch.randint(-1024, 1024, (tokens, hidden_size * 2), dtype=torch.int32)
    activation_scale = torch.rand(tokens, dtype=torch.float32)
    weight_scale = torch.rand(expert_num, hidden_size * 2, dtype=torch.float32)
    quant_scale = torch.rand(expert_num, hidden_size, dtype=torch.float32)

    quant = load_params(
        MojoDequantSwiGLUQuant(
            expert_num=expert_num,
            hidden_size=hidden_size,
            activate_left=False,
            quant_mode=1,
        ),
        weight_scale=weight_scale,
        quant_scale=quant_scale,
    )
    quant_ref = load_params(
        MojoDequantSwiGLUQuant._registry.get("torch")(
            expert_num=expert_num,
            hidden_size=hidden_size,
            activate_left=False,
            quant_mode=1,
        ),
        weight_scale=weight_scale.clone(),
        quant_scale=quant_scale.clone(),
    )
    quant.forward_diff_with(
        quant_ref,
        x,
        activation_scale,
        None,
        None,
        token_count,
        atol=(0, 1e-4),
        rtol=(0, 1e-4),
    )
