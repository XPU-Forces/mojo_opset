import pytest
import torch

from mojo_opset import Target
from mojo_opset import modules
from mojo_opset.functions._dispatch import resolve_implementation
from tests._checks import assert_repeatable

from .._checks import assert_close

NORM_OPS = {
    "LayerNormInfer": "layer_norm_infer",
    "RMSNormInfer": "rms_norm_infer",
    "GroupRMSNormInfer": "group_rms_norm_infer",
    "ResidualAddLayerNormInfer": "residual_add_layer_norm_infer",
    "ResidualAddRMSNormInfer": "residual_add_rms_norm_infer",
}


GROUP_LEADING_SHAPES = [((7, 128), (3, 128)), ((7, 2, 128), (3, 4, 128))]


NORM_SHAPES = [(32, 1024), (64, 8192), (57, 7338), (2, 256), (7762, 18778)]


NORM_MODULE_CASES = [(name, shape, "pre") for name in ("LayerNormInfer", "RMSNormInfer") for shape in NORM_SHAPES]


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


NORM_MODULE_CASES += [("GroupRMSNormInfer", shape, "pre") for shape in GROUP_SHAPES]


RESIDUAL_SHAPES = NORM_SHAPES[:-1]


NORM_MODULE_CASES += [
    (name, shape, pos)
    for name in ("ResidualAddLayerNormInfer", "ResidualAddRMSNormInfer")
    for shape in RESIDUAL_SHAPES
    for pos in ("pre", "post")
]


def assert_norm_close(actual, expected, *, rms=False):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype and actual.device == expected.device
    rtol, atol = (6e-3, 3e-2) if rms else (1e-2, 5e-2)
    assert_close(actual.float(), expected.float(), torch.float32, rtol=rtol, atol=atol)


def make_inference_norm_modules(name, shape, norm_pos, dtype, backend, *, strided=False, affine=True):
    implementation, target, device = backend
    if name == "LayerNormInfer" and not affine:
        selected = implementation or resolve_implementation("layer_norm_infer", Target.parse(target))
        if selected == "ixformer":
            pytest.skip("ixformer LayerNorm requires affine parameters")
    if name == "GroupRMSNormInfer":
        tokens, heads, dim = shape
        packed = torch.randn(tokens, sum(heads), dim, device=device, dtype=dtype)
        inputs = tuple(torch.split(packed, heads, dim=1))
        options = dict(num_groups=len(heads), normalized_shape=dim, elementwise_affine=affine)
    else:
        x = torch.randn(shape, device=device, dtype=dtype)
        inputs = (x.transpose(0, 1) if strided else x,)
        options = dict(normalized_shape=shape[-1])
        if name == "LayerNormInfer":
            options["elementwise_affine"] = affine
        if name.startswith("Residual"):
            inputs += (torch.randn_like(inputs[0]),)
            options["norm_pos"] = norm_pos
    actual = getattr(modules, name)(**options, eps=1e-5, device=device, dtype=dtype, implementation=implementation)
    reference = getattr(modules, name)(**options, eps=1e-5, device=device, dtype=dtype, implementation="torch")
    with torch.no_grad():
        for parameter in actual.parameters():
            parameter.normal_()
        reference.load_state_dict(actual.state_dict())
    return actual, reference, inputs


M15_RMS_SHAPES = [(13, 192), (257, 6144)]


MOJO_RMS_SHAPES = [(32, 1024), (64, 8192), (57, 7338), (77, 489), (763, 8777), (7762, 18778)]


RMS_NORM_CASES = [(shape, torch.float32) for shape in MOJO_RMS_SHAPES + M15_RMS_SHAPES] + [((7, 513), None)]


@pytest.mark.api("modules.RMSNorm", ops=["rms_norm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape,weight_dtype", RMS_NORM_CASES)
@pytest.mark.accuracy
def test_rms_norm_accuracy(accuracy_backend, dtype, shape, weight_dtype):
    implementation, target, device = accuracy_backend
    actual = modules.RMSNorm(shape[-1], device=device, dtype=weight_dtype, implementation=implementation)
    expected = modules.RMSNorm(shape[-1], device=device, dtype=weight_dtype, implementation="torch")
    with torch.no_grad():
        actual.weight.normal_()
        expected.weight.copy_(actual.weight)
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    y, reference_y = actual(x), expected(reference_x)
    upstream = torch.randn_like(y) if shape == (7, 513) or shape in M15_RMS_SHAPES else torch.rand_like(y)
    actual_grads = torch.autograd.grad(y, (x, actual.weight), upstream)
    reference_grads = torch.autograd.grad(reference_y, (reference_x, expected.weight), upstream)
    m15 = shape in M15_RMS_SHAPES and dtype == torch.bfloat16
    for index, (a, b) in enumerate(zip((y, *actual_grads), (reference_y, *reference_grads))):
        if shape in MOJO_RMS_SHAPES:
            # Parameter gradients can be FP32 even when activations are BF16.
            rtol, atol = (6e-3, 6e-3) if a.dtype == torch.float32 else (5e-2, 1e-1)
            assert_close(a, b, rtol=rtol, atol=atol, name=f"output/grad[{index}]")
            continue
        tolerance = dict(rtol=2e-2, atol=5e-2 if index == 2 else 2e-2) if m15 else {}
        assert_close(a, b, dtype, **tolerance)


@pytest.mark.accuracy
@pytest.mark.parametrize(
    "name,shape,norm_pos,dtype,affine,layout_case",
    [
        pytest.param(
            name, shape, pos, dtype, affine, layout, marks=pytest.mark.api("modules." + name, ops=[NORM_OPS[name]])
        )
        for name, shape, pos, dtype, affine, layout in [
            (name, shape, pos, dtype, True, False)
            for name, shape, pos in NORM_MODULE_CASES
            for dtype in (torch.float16, torch.bfloat16)
        ]
        + [
            (name, (17, (4, 2), 128) if name == "GroupRMSNormInfer" else (2, 13, 128), "pre", dtype, True, True)
            for name in NORM_OPS
            for dtype in (torch.float32, torch.bfloat16)
        ]
        + [
            (
                name,
                (17, (4, 2), 128) if name == "GroupRMSNormInfer" else (2, 13, 128),
                "pre",
                torch.float32,
                False,
                True,
            )
            for name in ("LayerNormInfer", "GroupRMSNormInfer")
        ]
        + [
            (name, (2, 13, 128), "post", torch.bfloat16, True, True)
            for name in ("ResidualAddLayerNormInfer", "ResidualAddRMSNormInfer")
        ]
    ],
)
def test_normalization_infer_accuracy(accuracy_backend, name, shape, norm_pos, dtype, affine, layout_case):
    actual, reference, inputs = make_inference_norm_modules(
        name, shape, norm_pos, dtype, accuracy_backend, strided=layout_case, affine=affine
    )
    args = (list(inputs),) if name == "GroupRMSNormInfer" else inputs
    with torch.no_grad():
        outputs, expected = actual(*args), reference(*args)
    if not affine:
        assert list(actual.parameters()) == []
    if norm_pos == "post" and layout_case:
        assert outputs[0] is outputs[1]
    if isinstance(outputs, torch.Tensor):
        outputs, expected = (outputs,), (expected,)
    for output, ref in zip(outputs, expected):
        if layout_case:
            assert_close(output, ref, dtype)
            assert output.is_contiguous()
        else:
            assert_norm_close(output, ref, rms=name in ("RMSNormInfer", "GroupRMSNormInfer"))


@pytest.mark.api("modules.GroupRMSNormInfer", ops=["group_rms_norm_infer"])
@pytest.mark.parametrize("shapes", GROUP_LEADING_SHAPES, ids=["2d", "different_tokens"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.accuracy
def test_group_layout(accuracy_backend, shapes, dtype):
    implementation, target, device = accuracy_backend
    selected = implementation or resolve_implementation("group_rms_norm_infer", Target.parse(target))
    if selected != "torch" and not target.startswith("npu."):
        pytest.skip("NPU grouped normalization layout regression")
    groups = [torch.randn(shape, device=device, dtype=dtype) for shape in shapes]
    actual = modules.GroupRMSNormInfer(
        len(groups), shapes[0][-1], device=device, dtype=dtype, implementation=implementation
    )
    reference = modules.GroupRMSNormInfer(
        len(groups), shapes[0][-1], device=device, dtype=dtype, implementation="torch"
    )
    with torch.no_grad():
        actual.weight.normal_()
        reference.load_state_dict(actual.state_dict())
        if selected == "triton":
            with pytest.raises(NotImplementedError, match="Triton group RMSNorm requires"):
                actual(groups)
            return
        for output, ref in zip(actual(groups), reference(groups)):
            assert_norm_close(output, ref, rms=True)


@pytest.mark.api("modules.RMSNorm", ops=["rms_norm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape,weight_dtype", RMS_NORM_CASES)
@pytest.mark.bitwise
def test_rms_norm_bitwise(accuracy_backend, dtype, shape, weight_dtype):
    implementation, _, device = accuracy_backend
    module = modules.RMSNorm(shape[-1], device=device, dtype=weight_dtype, implementation=implementation)
    with torch.no_grad():
        module.weight.normal_()
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    assert_repeatable(module, (x,), parameters=tuple(module.parameters()))
