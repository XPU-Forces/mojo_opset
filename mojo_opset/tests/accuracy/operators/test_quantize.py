import pytest
import torch

from mojo_opset import MojoDequant
from mojo_opset import MojoDequantSwiGLUClampQuant
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

static_quant_grouped_cases = [
    ((2, 4, 128), (4, 128)),
    ((3, 8, 64), (8, 64)),
    ((5, 2, 16, 32), (16, 32)),
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
    (3, 129),
    (7, 257),
    (15, 511),
    (31, 1023),
    (63, 2047),
    (96, 4097),
    (128, 6144),
    (257, 8192),
]

moe_dynamic_quant_cases = [
    (8, 128, [8]),
    (12, 256, [4, 3, 5]),
    (18, 512, [6, 6, 4, 2]),
    (21, 1024, [2, 5, 1, 7, 6]),
    (32, 2048, [8, 7, 5, 6, 4, 2]),
]

dequant_swiglu_quant_cases = [
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


def make_grouped_scale(x: torch.Tensor, q_max: float, scale_shape: tuple[int, ...]) -> torch.Tensor:
    reduce_dims = tuple(range(x.dim() - len(scale_shape)))
    return (x.float().abs().amax(dim=reduce_dims) / q_max).clamp(min=1e-10)


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


@pytest.mark.parametrize("shape,scale_shape", static_quant_grouped_cases)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("quant_dtype", [torch.int8])
@bypass_not_implemented
def test_static_quant_grouped(shape, scale_shape, dtype, quant_dtype):
    if quant_dtype == torch.int8 and not has_ixformer_quant_kernel("static_quant"):
        pytest.skip("static_quant kernel is not available on the current ixformer build")

    x = torch.randn(shape, dtype=dtype)
    q_max = 127
    scale = make_grouped_scale(x, q_max, scale_shape)

    quant = load_params(MojoStaticQuant(input_size=scale_shape, quant_dtype=quant_dtype), scale=scale)
    quant_ref = load_params(
        MojoStaticQuant._registry.get("torch")(input_size=scale_shape, quant_dtype=quant_dtype),
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


@pytest.mark.parametrize("shape,scale_shape", static_quant_grouped_cases)
@pytest.mark.parametrize("dtype", dtypes)
@bypass_not_implemented
def test_dequant_grouped(shape, scale_shape, dtype):
    x = torch.randn(shape, dtype=dtype)
    scale = make_grouped_scale(x, 127, scale_shape)

    quant_ref = load_params(
        MojoStaticQuant._registry.get("torch")(input_size=scale_shape, quant_dtype=torch.int8),
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


def test_dequant_swiglu_clamp_quant_torch_reference():
    input = torch.tensor(
        [[1, 2, 3, 4], [2, -3, 4, -5], [-1, 4, -2, 3]],
        dtype=torch.int32,
        device="cpu",
    )
    group_list = torch.tensor([1, 2], dtype=torch.int32, device="cpu")
    weight_scale = torch.tensor([[0.5, 1.0, 0.25, 0.5], [1.0, 0.5, 0.5, 0.25]], device="cpu")
    quant_scale = torch.tensor([[1.0, 0.5], [0.75, 1.25]], device="cpu")
    activation_scale = torch.tensor([[0.5], [1.0], [2.0]], device="cpu")

    op = MojoDequantSwiGLUClampQuant._registry.get("torch")(
        expert_num=2, hidden_size=2, clamp_limit=2.0, device="cpu"
    )
    op.load_state_dict({"weight_scale": weight_scale, "quant_scale": quant_scale}, strict=True)
    output, scale = op(input, activation_scale, group_list)

    expanded_weight_scale = weight_scale.repeat_interleave(group_list, dim=0)
    hidden = input.float() * expanded_weight_scale * activation_scale
    gate, up = hidden.chunk(2, dim=-1)
    hidden = torch.nn.functional.silu(gate.clamp(max=2.0)) * up.clamp(min=-2.0, max=2.0)
    hidden *= quant_scale.repeat_interleave(group_list, dim=0)
    expected_scale = hidden.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127
    expected = torch.clamp(torch.round(hidden / expected_scale), -128, 127).to(torch.int8)

    assert torch.equal(output, expected)
    assert torch.equal(scale, expected_scale)


@pytest.mark.parametrize("backend", ["torch", "torch_npu"])
@pytest.mark.parametrize("hidden_size, counts, limit", [(64, [1, 2], 2), (128, [0, 1, 0, 2, 0], 4)])
def test_dequant_swiglu_clamp_quant_boundaries(backend, hidden_size, counts, limit):
    if backend == "torch_npu":
        if get_platform() != "npu":
            pytest.skip("Requires an NPU")
        import torch_npu

        if not hasattr(torch_npu, "npu_dequant_swiglu_clamp_quant"):
            pytest.skip("torch_npu.npu_dequant_swiglu_clamp_quant is unavailable; fused path not validated")

    # Include an unclamped negative gate, the boundary, and values beyond it.
    gate = torch.tensor([-limit - 1, limit, limit + 1, limit + 2], dtype=torch.int32, device="cpu")
    up = torch.tensor([limit + 1, -limit - 1, limit, -limit], dtype=torch.int32, device="cpu")
    input = torch.cat((gate.repeat(hidden_size // 4), up.repeat(hidden_size // 4))).repeat(3, 1)
    input[1].zero_()
    # A literal post-clamp oracle avoids duplicating the implementation's clamps.
    gate_ref = torch.tensor([-limit - 1, limit, limit, limit], dtype=torch.float32, device="cpu")
    up_ref = torch.tensor([limit, -limit, limit, -limit], dtype=torch.float32, device="cpu")
    hidden = (torch.nn.functional.silu(gate_ref) * up_ref).repeat(hidden_size // 4).repeat(3, 1)
    hidden[1].zero_()
    expected_scale = hidden.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127
    expected = torch.round(hidden / expected_scale).to(torch.int8)

    device = "npu" if backend == "torch_npu" else "cpu"
    op = MojoDequantSwiGLUClampQuant._registry.get(backend)(
        expert_num=len(counts), hidden_size=hidden_size, clamp_limit=limit, device=device
    )
    assert op._backend == backend
    op.load_state_dict(
        {
            "weight_scale": torch.ones(len(counts), hidden_size * 2, device="cpu"),
            "quant_scale": torch.ones(len(counts), hidden_size, device="cpu"),
        },
        strict=True,
    )
    activation_scale = torch.ones(3, 1, dtype=torch.float32, device=device)
    token_count = torch.tensor(counts, dtype=torch.int64, device=device)
    output, scale = op(input.to(device), activation_scale, token_count)
    output, scale = output.cpu(), scale.cpu()

    assert output.shape == (3, hidden_size)
    assert output.dtype == torch.int8
    assert scale.shape == (3, 1)
    assert scale.dtype == torch.float32
    assert torch.isfinite(scale).all() and (scale > 0).all()
    assert torch.equal(output, expected)
    if backend == "torch":
        assert torch.equal(scale, expected_scale)
    else:
        torch.testing.assert_close(scale, expected_scale, atol=0, rtol=1e-4)
