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


RMS_NORM_CASES = [(shape, torch.float32) for shape in MOJO_RMS_SHAPES + M15_RMS_SHAPES] + [
    ((7, 513), None),
    ((2, 7, 513), None),
]


@pytest.mark.api("functions.rms_norm")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("shape,weight_dtype", RMS_NORM_CASES)
@pytest.mark.accuracy
def test_rms_norm_accuracy(accuracy_backend, dtype, shape, weight_dtype):
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

    values = reference_inputs[0].float()
    expected = (reference_inputs[1] * (values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + 1e-6))).to(dtype)
    actual = F.rms_norm(*actual_inputs, implementation=implementation)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output)
    actual_grads = torch.autograd.grad(actual, actual_inputs, grad_output)

    m15 = shape in M15_RMS_SHAPES and dtype == torch.bfloat16
    if shape in MOJO_RMS_SHAPES:
        for index, (a, b) in enumerate(zip((actual, *actual_grads), (expected, *expected_grads))):
            # Parameter gradients can be FP32 even when activations are BF16.
            rtol, atol = (6e-3, 6e-3) if a.dtype == torch.float32 else (5e-2, 1e-1)
            assert_close(a, b, rtol=rtol, atol=atol, name=f"output/grad[{index}]")
        return
    assert_close(actual, expected, dtype, **(dict(rtol=2e-2, atol=2e-2) if m15 else {}))
    for index, (actual_grad, expected_grad) in enumerate(zip(actual_grads, expected_grads)):
        tolerance = dict(rtol=2e-2, atol=5e-2 if index == 1 else 2e-2) if m15 else {}
        assert_close(actual_grad, expected_grad, dtype, **tolerance)


@pytest.mark.accuracy
@pytest.mark.parametrize(
    "op,shape,dtype,strided,has_weight,has_bias,shape_budget",
    [
        pytest.param(op, shape, dtype, strided, weight, bias, budget, marks=pytest.mark.api("functions." + op))
        for op in ("layer_norm_infer", "rms_norm_infer")
        for shape, dtype, strided, weight, bias, budget in [
            (shape, dtype, False, True, True, True)
            for shape in NORM_SHAPES
            for dtype in (torch.float16, torch.bfloat16)
        ]
        + [
            ((2, 13, dim), dtype, True, True, True, False)
            for dim in (128, 513)
            for dtype in (torch.float32, torch.bfloat16)
        ]
        + (
            [
                ((17, 128), torch.float32, False, weight, bias, False)
                for weight, bias in ((False, False), (True, False), (False, True))
            ]
            if op == "layer_norm_infer"
            else []
        )
    ],
)
def test_normalization_infer_accuracy(accuracy_backend, op, shape, dtype, strided, has_weight, has_bias, shape_budget):
    implementation, target, device = accuracy_backend
    if not has_weight or not has_bias:
        selected = implementation or resolve_implementation(op, Target.parse(target))
        if selected == "ixformer":
            pytest.skip("ixformer LayerNorm requires affine parameters")
    if shape_budget:
        torch.manual_seed(43)
    x = torch.randn(shape, dtype=dtype, device=device)
    if strided:
        x = x.transpose(0, 1)
    weight = torch.randn(shape[-1], dtype=dtype, device=device) if has_weight else None
    inputs = (x, weight)
    if op == "layer_norm_infer":
        inputs += (torch.randn(shape[-1], dtype=dtype, device=device) if has_bias else None,)
    fn = getattr(F, op)
    actual = fn(*inputs, eps=1e-5, implementation=implementation)
    expected = fn(*inputs, eps=1e-5, implementation="torch")
    if shape_budget:
        assert_norm_close(actual, expected, rms=op == "rms_norm_infer")
    else:
        assert_close(actual, expected, dtype)
    if strided:
        assert actual.is_contiguous()


@pytest.mark.api("functions.group_rms_norm_infer")
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "tokens,heads,dim,dtype,affine,packed,shape_budget",
    [
        (17, (4, 2), 128, dtype, affine, True, False)
        for dtype in (torch.float32, torch.bfloat16)
        for affine in (True, False)
    ]
    + [
        (tokens, heads, dim, dtype, True, True, True)
        for tokens, heads, dim in GROUP_SHAPES
        for dtype in (torch.float16, torch.bfloat16)
    ]
    + [
        (3, tuple(range(1, count + 1)), dim, dtype, affine, False, False)
        for count, dim in ((1, 128), (5, 128), (4, 2048), (2, 4096), (4, 4096), (2, 8192), (4, 10240), (2, 16384))
        for dtype in (torch.float32, torch.bfloat16)
        for affine in (True, False)
    ],
)
def test_group_rms_norm_infer_accuracy(accuracy_backend, tokens, heads, dim, dtype, affine, packed, shape_budget):
    implementation, _, device = accuracy_backend
    if shape_budget:
        torch.manual_seed(43)
    if packed:
        groups = list(torch.randn(tokens, sum(heads), dim, device=device, dtype=dtype).split(heads, dim=1))
    else:
        groups = [torch.randn(tokens, head, dim, device=device, dtype=dtype) for head in heads]
    weight = torch.randn(len(heads), dim, device=device, dtype=dtype) if affine else None
    actual = F.group_rms_norm_infer(groups, weight, eps=1e-5, implementation=implementation)
    expected = F.group_rms_norm_infer(groups, weight, eps=1e-5, implementation="torch")
    for output, ref in zip(actual, expected):
        if shape_budget:
            assert_norm_close(output, ref, rms=True)
        else:
            assert_close(output, ref, dtype)
        if packed and not shape_budget:
            assert output.is_contiguous()


@pytest.mark.api("functions.group_rms_norm_infer")
@pytest.mark.parametrize("shapes", GROUP_LEADING_SHAPES, ids=["2d", "different_tokens"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.accuracy
def test_group_layout(accuracy_backend, shapes, dtype):
    implementation, target, device = accuracy_backend
    selected = implementation or resolve_implementation("group_rms_norm_infer", Target.parse(target))
    if selected != "torch" and not target.startswith("npu."):
        pytest.skip("NPU grouped normalization layout regression")
    groups = [torch.randn(shape, device=device, dtype=dtype) for shape in shapes]
    weight = torch.randn(len(groups), shapes[0][-1], device=device, dtype=dtype)
    if selected == "triton":
        with pytest.raises(NotImplementedError, match="Triton group RMSNorm requires"):
            F.group_rms_norm_infer(groups, weight, implementation=implementation)
        return
    actual = F.group_rms_norm_infer(groups, weight, implementation=implementation)
    expected = F.group_rms_norm_infer(groups, weight, implementation="torch")
    for output, ref in zip(actual, expected):
        assert_norm_close(output, ref, rms=True)


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(op, marks=pytest.mark.api("functions." + op))
        for op in ("residual_add_layer_norm_infer", "residual_add_rms_norm_infer")
    ],
)
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
@pytest.mark.parametrize(
    "shape,dtype,strided",
    [((2, 13, 128), dtype, True) for dtype in (torch.float32, torch.bfloat16)]
    + [(shape, dtype, False) for shape in RESIDUAL_SHAPES for dtype in (torch.float16, torch.bfloat16)],
)
@pytest.mark.accuracy
def test_residual_add_normalization_infer_accuracy(accuracy_backend, op, norm_pos, shape, dtype, strided):
    implementation, _, device = accuracy_backend
    if not strided:
        torch.manual_seed(43)
    x = torch.randn(shape, dtype=dtype, device=device)
    if strided:
        x = x.transpose(0, 1)
    residual = torch.randn_like(x)
    weight = torch.randn(shape[-1], dtype=dtype, device=device)
    inputs = (x, residual, weight)
    if op == "residual_add_layer_norm_infer":
        inputs += (torch.randn_like(weight),)
    originals = (x.clone(), residual.clone()) if strided else None
    fn = getattr(F, op)
    actual = fn(*inputs, norm_pos=norm_pos, eps=1e-5, implementation=implementation)
    expected = fn(*inputs, norm_pos=norm_pos, eps=1e-5, implementation="torch")
    for output, ref in zip(actual, expected):
        if strided:
            assert_close(output, ref, dtype)
            assert output.is_contiguous()
        else:
            assert_norm_close(output, ref)
    if strided:
        for tensor, original in zip((x, residual), originals):
            assert_close(tensor, original, rtol=0, atol=0)
    if strided and norm_pos == "post":
        assert actual[0] is actual[1]


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


@pytest.mark.api("functions.rms_norm")
def test_rms_norm_fake_metadata(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("NPU Triton fake metadata")
    preload("rms_norm", implementation="triton", target=target)
    ops = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    x = torch.randn(17, 7, device=device, dtype=torch.bfloat16).T
    weight = torch.randn(34, device=device)[::2]
    torch.library.opcheck(ops.rms_norm_fwd.default, (x, weight, 1e-6), test_utils=("test_faketensor",))
    # The backward wrapper normalizes inputs before invoking its two leaves.
    x, weight = x.contiguous(), weight.contiguous()
    rstd = torch.ones(x.shape[0], device=device)
    torch.library.opcheck(
        ops.rms_norm_bwd_partial.default,
        (torch.randn_like(x), x, weight, rstd, 2),
        test_utils=("test_faketensor",),
    )
    torch.library.opcheck(
        ops.rms_norm_dw_reduce.default,
        (torch.randn(2, x.shape[1], device=device), weight.dtype),
        test_utils=("test_faketensor",),
    )


@pytest.mark.reference
@pytest.mark.api("functions.rms_norm")
@pytest.mark.parametrize("shape", [(1, 4), (4, 1), ()])
def test_rms_norm_rejects_invalid_weight_shape(shape):
    x = torch.randn(2, 4, requires_grad=True)
    weight = torch.ones(shape, requires_grad=True)
    with pytest.raises(ValueError, match="one-dimensional"):
        functions.rms_norm(x, weight, implementation="torch")


@pytest.mark.api("functions.rms_norm")
@pytest.mark.parametrize("op", ["rms_norm"])
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_compiled_layout(accuracy_backend, op, backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton", "torch") or not target.startswith("npu."):
        pytest.skip("Torch/NPU Triton contiguous-output contract")
    implementation = implementation or "triton"
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
        reference = getattr(F, op)(*inputs, **kwargs, implementation="torch")
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
