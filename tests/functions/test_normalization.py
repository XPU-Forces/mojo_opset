from functools import partial

import pytest
import torch

from mojo_opset import Target
from mojo_opset import functions
from mojo_opset import functions as F
from mojo_opset import preload
from mojo_opset.functions._dispatch import resolve_implementation
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph

from .._checks import assert_close
from .._checks import assert_mojo_close
from .._checks import clone_with_grad

GROUP_LEADING_SHAPES = [((7, 128), (3, 128)), ((7, 2, 128), (3, 4, 128))]


GROUP_SHAPES = [
    (1024, (16, 4), 96),
    (798, (16, 4, 8, 2), 128),
    (8000, (48, 8, 16, 4), 128),
    (17, (3, 5), 128),
    (33, (2, 7, 1), 128),
    (65, (4, 4, 4, 4), 128),
    (129, (1, 3, 5, 7), 128),
    (257, (6, 2), 192),
    (513, (8, 8, 8), 256),
    (1025, (12, 6, 3, 1), 128),
    (2049, (5, 9, 7, 3), 64),
]


NORM_SHAPES = [(32, 1024), (64, 8192), (57, 7338), (2, 256), (7762, 18778)]


RESIDUAL_SHAPES = NORM_SHAPES[:-1]


def assert_norm_close(actual, expected, *, rms=False):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype and actual.device == expected.device
    rtol, atol = (6e-3, 3e-2) if rms else (1e-2, 5e-2)
    assert_close(actual.float(), expected.float(), torch.float32, rtol=rtol, atol=atol)


M15_RMS_SHAPES = [(13, 192), (257, 6144)]


MOJO_RMS_SHAPES = [(32, 1024), (64, 8192), (57, 7338), (77, 489), (763, 8777), (7762, 18778)]


RMS_NORM_CASES = [(shape, torch.float32) for shape in MOJO_RMS_SHAPES + M15_RMS_SHAPES] + [((7, 513), None)]


@pytest.mark.api("functions.rms_norm")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("shape,weight_dtype", RMS_NORM_CASES)
@pytest.mark.accuracy
def test_rms_norm(accuracy_backend, dtype, shape, weight_dtype):
    implementation, target, device = accuracy_backend
    reference_inputs = (
        torch.randn(shape, device=device, dtype=dtype, requires_grad=True),
        torch.randn(shape[-1], device=device, dtype=weight_dtype or dtype, requires_grad=True),
    )
    actual_inputs = clone_with_grad(reference_inputs)
    grad_output = (
        torch.randn_like(reference_inputs[0])
        if shape == (7, 513) or shape in M15_RMS_SHAPES
        else torch.rand_like(reference_inputs[0])
    )

    expected = F.rms_norm(*reference_inputs, implementation="torch_reference")
    actual = F.rms_norm(*actual_inputs, implementation=implementation)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output)
    actual_grads = torch.autograd.grad(actual, actual_inputs, grad_output)

    m15 = shape in M15_RMS_SHAPES and dtype == torch.bfloat16
    if shape in MOJO_RMS_SHAPES:
        for index, (a, b) in enumerate(zip((actual, *actual_grads), (expected, *expected_grads))):
            assert_mojo_close(a, b, name=f"output/grad[{index}]")
        return
    assert_close(actual, expected, dtype, **(dict(rtol=2e-2, atol=2e-2) if m15 else {}))
    for index, (actual_grad, expected_grad) in enumerate(zip(actual_grads, expected_grads)):
        tolerance = dict(rtol=2e-2, atol=5e-2 if index == 1 else 2e-2) if m15 else {}
        assert_close(actual_grad, expected_grad, dtype, **tolerance)


@pytest.mark.parametrize(
    "op", [pytest.param(op, marks=pytest.mark.api("functions." + op)) for op in ["layer_norm_infer", "rms_norm_infer"]]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim", [128, 513])
@pytest.mark.accuracy
def test_norm_infer(accuracy_backend, op, dtype, dim):
    implementation, _, device = accuracy_backend
    x = torch.randn(2, 13, dim, dtype=dtype, device=device).transpose(0, 1)
    weight = torch.randn(dim, dtype=dtype, device=device)
    inputs = (x, weight)
    if op == "layer_norm_infer":
        inputs += (torch.randn_like(weight),)
    fn = getattr(F, op)
    actual = fn(*inputs, implementation=implementation)
    expected = fn(*inputs, implementation="torch_reference")
    assert_close(actual, expected, dtype)
    assert actual.dtype == x.dtype and actual.is_contiguous()


@pytest.mark.api("functions.layer_norm_infer")
@pytest.mark.parametrize("has_weight,has_bias", [(False, False), (True, False), (False, True)])
@pytest.mark.accuracy
def test_layer_norm_affine(accuracy_backend, has_weight, has_bias):
    implementation, target, device = accuracy_backend
    selected = implementation or resolve_implementation("layer_norm_infer", Target.parse(target))
    if selected == "ixformer":
        pytest.skip("ixformer LayerNorm requires affine parameters")
    x = torch.randn(17, 128, device=device)
    weight = torch.randn(128, device=device) if has_weight else None
    bias = torch.randn(128, device=device) if has_bias else None
    actual = F.layer_norm_infer(x, weight, bias, implementation=implementation)
    expected = F.layer_norm_infer(x, weight, bias, implementation="torch_reference")
    assert_close(actual, expected, x.dtype)


@pytest.mark.api("functions.group_rms_norm_infer")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.accuracy
def test_group_rms_norm(accuracy_backend, dtype, affine):
    implementation, _, device = accuracy_backend
    packed = torch.randn(17, 6, 128, device=device, dtype=dtype)
    groups = [packed[:, :4], packed[:, 4:]]
    weight = torch.randn(2, 128, device=device, dtype=dtype) if affine else None
    actual = F.group_rms_norm_infer(groups, weight, implementation=implementation)
    expected = F.group_rms_norm_infer(groups, weight, implementation="torch_reference")
    for output, ref, original in zip(actual, expected, groups):
        assert_close(output, ref, dtype)
        assert output.shape == original.shape and output.dtype == dtype and output.is_contiguous()


@pytest.mark.api("functions.group_rms_norm_infer")
@pytest.mark.parametrize("shapes", GROUP_LEADING_SHAPES, ids=["2d", "different_tokens"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.accuracy
def test_group_layout(accuracy_backend, shapes, dtype):
    implementation, target, device = accuracy_backend
    if implementation != "torch_reference" and not target.startswith("npu."):
        pytest.skip("NPU grouped normalization layout regression")
    selected = implementation or resolve_implementation("group_rms_norm_infer", Target.parse(target))
    groups = [torch.randn(shape, device=device, dtype=dtype) for shape in shapes]
    weight = torch.randn(len(groups), shapes[0][-1], device=device, dtype=dtype)
    if selected == "triton":
        with pytest.raises(NotImplementedError, match="Triton group RMSNorm requires"):
            F.group_rms_norm_infer(groups, weight, implementation=implementation)
        return
    actual = F.group_rms_norm_infer(groups, weight, implementation=implementation)
    expected = F.group_rms_norm_infer(groups, weight, implementation="torch_reference")
    for output, ref in zip(actual, expected):
        assert_norm_close(output, ref, rms=True)


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(op, marks=pytest.mark.api("functions." + op))
        for op in ["residual_add_layer_norm_infer", "residual_add_rms_norm_infer"]
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
@pytest.mark.accuracy
def test_residual_norm(accuracy_backend, op, dtype, norm_pos):
    implementation, _, device = accuracy_backend
    x = torch.randn(2, 13, 128, device=device, dtype=dtype).transpose(0, 1)
    residual = torch.randn_like(x)
    weight = torch.randn(128, device=device, dtype=dtype)
    inputs = (x, residual, weight)
    if op == "residual_add_layer_norm_infer":
        inputs += (torch.randn_like(weight),)
    original_x, original_residual = x.clone(), residual.clone()
    fn = getattr(F, op)
    actual = fn(*inputs, norm_pos=norm_pos, implementation=implementation)
    expected = fn(*inputs, norm_pos=norm_pos, implementation="torch_reference")
    for output, ref in zip(actual, expected):
        assert_close(output, ref, dtype)
        assert output.dtype == dtype and output.is_contiguous()
    assert_close(x, original_x, rtol=0, atol=0)
    assert_close(residual, original_residual, rtol=0, atol=0)
    if norm_pos == "post":
        assert actual[0] is actual[1]


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(op, marks=pytest.mark.api("functions." + op))
        for op in [
            "layer_norm_infer",
            "rms_norm_infer",
            "group_rms_norm_infer",
            "residual_add_layer_norm_infer",
            "residual_add_rms_norm_infer",
        ]
    ],
)
@pytest.mark.accuracy
@pytest.mark.reference
def test_infer_autograd(op):
    x = torch.randn(2, 4, 128, requires_grad=True)
    weight = torch.ones(128)
    if op == "group_rms_norm_infer":
        args = ([x], weight[None])
    elif op.startswith("residual"):
        args = (x, torch.randn_like(x), weight)
        if op == "residual_add_layer_norm_infer":
            args += (torch.zeros_like(weight),)
    else:
        args = (x, weight)
    with pytest.raises(RuntimeError, match="does not support autograd"):
        getattr(F, op)(*args, implementation="torch_reference")


# Keep the original master operator matrix in addition to the
# small layout/contract cases above. In particular, these exercise multi-pass
# column reductions (>2048), not only the newly added inference wrappers.
@pytest.mark.parametrize(
    "op", [pytest.param(op, marks=pytest.mark.api("functions." + op)) for op in ["layer_norm_infer", "rms_norm_infer"]]
)
@pytest.mark.parametrize("shape", NORM_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.accuracy
def test_norm_shapes(accuracy_backend, op, shape, dtype):
    implementation, _, device = accuracy_backend
    torch.manual_seed(43)
    inputs = (torch.randn(shape, device=device, dtype=dtype), torch.randn(shape[-1], device=device, dtype=dtype))
    if op == "layer_norm_infer":
        inputs += (torch.randn_like(inputs[1]),)
    actual = getattr(F, op)(*inputs, eps=1e-5, implementation=implementation)
    expected = getattr(F, op)(*inputs, eps=1e-5, implementation="torch_reference")
    assert_norm_close(actual, expected, rms=op == "rms_norm_infer")


@pytest.mark.api("functions.group_rms_norm_infer")
@pytest.mark.parametrize("tokens,heads,dim", GROUP_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.accuracy
def test_group_shapes(accuracy_backend, tokens, heads, dim, dtype):
    implementation, _, device = accuracy_backend
    torch.manual_seed(43)
    packed = torch.randn(tokens, sum(heads), dim, device=device, dtype=dtype)
    groups = list(torch.split(packed, heads, dim=1))
    weight = torch.randn(len(heads), dim, device=device, dtype=dtype)
    actual = F.group_rms_norm_infer(groups, weight, eps=1e-5, implementation=implementation)
    expected = F.group_rms_norm_infer(groups, weight, eps=1e-5, implementation="torch_reference")
    for output, ref in zip(actual, expected):
        assert_norm_close(output, ref, rms=True)


@pytest.mark.api("functions.group_rms_norm_infer")
@pytest.mark.parametrize(
    "group_count,dim",
    [
        (1, 128),
        (5, 128),
        (4, 2048),
        (2, 4096),
        (4, 4096),
        (2, 8192),
        (4, 10240),
        (2, 16384),
    ],
)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.accuracy
def test_group_dispatch(accuracy_backend, group_count, dim, affine, dtype):
    implementation, _, device = accuracy_backend
    groups = [torch.randn(3, heads + 1, dim, device=device, dtype=dtype) for heads in range(group_count)]
    weight = torch.randn(group_count, dim, device=device, dtype=dtype) if affine else None
    actual = F.group_rms_norm_infer(groups, weight, implementation=implementation)
    expected = F.group_rms_norm_infer(groups, weight, implementation="torch_reference")
    for output, ref in zip(actual, expected):
        assert_close(output, ref, dtype)


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(op, marks=pytest.mark.api("functions." + op))
        for op in ["residual_add_layer_norm_infer", "residual_add_rms_norm_infer"]
    ],
)
@pytest.mark.parametrize("shape", RESIDUAL_SHAPES)
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.accuracy
def test_residual_shapes(accuracy_backend, op, shape, norm_pos, dtype):
    implementation, _, device = accuracy_backend
    torch.manual_seed(43)
    inputs = (
        torch.randn(shape, device=device, dtype=dtype),
        torch.randn(shape, device=device, dtype=dtype),
        torch.randn(shape[-1], device=device, dtype=dtype),
    )
    if op == "residual_add_layer_norm_infer":
        inputs += (torch.randn_like(inputs[-1]),)
    actual = getattr(F, op)(*inputs, norm_pos=norm_pos, eps=1e-5, implementation=implementation)
    expected = getattr(F, op)(*inputs, norm_pos=norm_pos, eps=1e-5, implementation="torch_reference")
    for output, ref in zip(actual, expected):
        assert_norm_close(output, ref)


@pytest.mark.api("functions.rms_norm")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape,weight_dtype", RMS_NORM_CASES)
@pytest.mark.bitwise
def test_rms_norm_bitwise(accuracy_backend, dtype, shape, weight_dtype):
    implementation, _, device = accuracy_backend
    inputs = (
        torch.randn(shape, device=device, dtype=dtype, requires_grad=True),
        torch.randn(shape[-1], device=device, dtype=weight_dtype or dtype, requires_grad=True),
    )
    assert_repeatable(partial(F.rms_norm, implementation=implementation), inputs)


@pytest.mark.accuracy
@pytest.mark.api("functions.rms_norm")
@pytest.mark.parametrize("op", ["rms_norm"])
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_compiled_layout(accuracy_backend, op, backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu."):
        pytest.skip("NPU Triton contiguous-output contract")
    implementation = "triton"
    scaled = op == "swiglu_scaled"
    op = "swiglu" if scaled else op
    preload(op, implementation=implementation)
    x = torch.randn(17, 7, device=device, dtype=torch.bfloat16).T.requires_grad_()
    inputs = (x,)
    kwargs = {}
    if op == "rms_norm":
        # A strided vector also exercises backward weight metadata.
        weight = torch.randn(34, device=device)[::2].requires_grad_()
        inputs = (x, weight)

    def run(*args):
        output = getattr(F, op)(*args, **kwargs, implementation=implementation)
        outputs = output if isinstance(output, tuple) else (output,)
        return outputs, tuple((y.stride(), y.is_contiguous()) for y in outputs)

    try:
        eager, metadata = run(*inputs)
        actual, captured_metadata = compile_fullgraph(run, backend=backend)(*inputs)
        assert captured_metadata == metadata
        assert all(y.is_contiguous() for y in actual)
        assert captured_metadata == tuple((y.stride(), y.is_contiguous()) for y in actual)
        reference = getattr(F, op)(*inputs, **kwargs, implementation="torch_reference")
        reference = reference if isinstance(reference, tuple) else (reference,)
        for y, expected in zip(actual, reference):
            torch.testing.assert_close(y, expected, rtol=2e-2, atol=2e-2)
        upstream = tuple(torch.randn_like(y) for y in eager)
        grads = torch.autograd.grad(actual, inputs, upstream)
        expected_grads = torch.autograd.grad(reference, inputs, upstream)
        for grad, expected in zip(grads, expected_grads):
            torch.testing.assert_close(grad, expected, rtol=2e-2, atol=2e-2)
    finally:
        torch._dynamo.reset()


@pytest.mark.accuracy
@pytest.mark.api("functions.rms_norm")
@pytest.mark.parametrize("shape", [(1, 4), (4, 1), ()])
def test_weight_shape(shape):
    x = torch.randn(2, 4, requires_grad=True)
    weight = torch.ones(shape, requires_grad=True)
    with pytest.raises(ValueError, match="one-dimensional"):
        functions.rms_norm(x, weight, implementation="torch_reference")
