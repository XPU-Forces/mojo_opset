import pytest
import torch

from mojo_opset import Target
from mojo_opset import functions
from mojo_opset import modules
from mojo_opset.functions._dispatch import resolve_implementation
from tests._checks import assert_accuracy
from tests._checks import assert_close
from tests._checks import assert_mojo_close
from tests._checks import assert_repeatable
from tests._checks import clone_with_grad

ACTIVATION_SHAPES = [(7, 513), (128, 128), (256, 128), (999, 9999), (1024, 10240)]


SWIGLU_SHAPES = ACTIVATION_SHAPES + [(3072, 3072), (4096, 2048), (257, 480)]


@pytest.mark.parametrize(
    "name,limit,shape",
    [
        pytest.param(
            name,
            limit,
            shape,
            marks=pytest.mark.api("modules." + name, ops=[{"GELU": "gelu", "SiLU": "silu", "SwiGLU": "swiglu"}[name]]),
        )
        for name, limit in [("GELU", 0), ("SiLU", 0), ("SwiGLU", 0), ("SwiGLU", 0.5)]
        for shape in (SWIGLU_SHAPES if name == "SwiGLU" else ACTIVATION_SHAPES)
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.accuracy
def test_activation(accuracy_backend, name, dtype, limit, shape):
    options = {"swiglu_limit": limit} if name == "SwiGLU" else {}

    def run(*inputs, **selection):
        return getattr(modules, name)(**options, **selection)(*inputs)

    generate = torch.rand if shape in ACTIVATION_SHAPES[1:] else torch.randn
    inputs = tuple(
        generate(shape, device=accuracy_backend[2], dtype=dtype) for _ in range(2 if name == "SwiGLU" else 1)
    )
    if name == "SiLU" and dtype == torch.float32 and shape in ((128, 128), (999, 9999), (1024, 10240)):
        inputs, reference = clone_with_grad(inputs), clone_with_grad(inputs)
        actual = run(*inputs, implementation=accuracy_backend[0])
        expected = run(*reference, implementation="torch_reference")
        upstream = torch.rand_like(actual)
        (actual_grad,) = torch.autograd.grad(actual, inputs, upstream)
        (expected_grad,) = torch.autograd.grad(expected, reference, upstream)
        assert_mojo_close(actual, expected)
        assert_mojo_close(actual_grad, expected_grad, name="grad")
    else:
        assert_accuracy(run, inputs, implementation=accuracy_backend[0])


@pytest.mark.parametrize(
    "name,limit,shape",
    [
        pytest.param(
            name,
            limit,
            shape,
            marks=pytest.mark.api("modules." + name, ops=[{"GELU": "gelu", "SiLU": "silu", "SwiGLU": "swiglu"}[name]]),
        )
        for name, limit in [("GELU", 0), ("SiLU", 0), ("SwiGLU", 0), ("SwiGLU", 0.5)]
        for shape in (SWIGLU_SHAPES if name == "SwiGLU" else ACTIVATION_SHAPES)
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.bitwise
def test_activation_bitwise(accuracy_backend, name, limit, dtype, shape):
    implementation, _, device = accuracy_backend
    options = {"swiglu_limit": limit} if name == "SwiGLU" else {}
    module = getattr(modules, name)(implementation=implementation, **options)
    generate = torch.rand if shape in ACTIVATION_SHAPES[1:] else torch.randn
    inputs = tuple(
        generate(shape, device=device, dtype=dtype, requires_grad=True) for _ in range(2 if name == "SwiGLU" else 1)
    )
    assert_repeatable(module, inputs, parameters=tuple(module.parameters()))


@pytest.mark.api("modules.GELU", ops=["gelu"])
@pytest.mark.parametrize("shape", ACTIVATION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.accuracy
def test_gelu_exact(accuracy_backend, shape, dtype):
    implementation, target, device = accuracy_backend
    selected = implementation or resolve_implementation("gelu", Target.parse(target))
    if selected == "triton":
        pytest.skip("Triton GELU supports approximate='tanh' only")
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = modules.GELU(approximate="none", implementation=implementation)(x)
    expected = functions.gelu(reference_x, approximate="none", implementation="torch_reference")
    upstream = torch.rand_like(x)
    (actual_grad,) = torch.autograd.grad(actual, x, upstream)
    (expected_grad,) = torch.autograd.grad(expected, reference_x, upstream)
    assert_close(
        actual, expected, rtol=1e-2 if dtype == torch.bfloat16 else 1e-4, atol=1e-2 if dtype == torch.bfloat16 else 1e-4
    )
    assert_close(
        actual_grad,
        expected_grad,
        rtol=1e-2 if dtype == torch.bfloat16 else 1e-4,
        atol=1e-2 if dtype == torch.bfloat16 else 1e-4,
    )


@pytest.mark.api("modules.GELU", ops=["gelu"])
@pytest.mark.parametrize("shape", ACTIVATION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.bitwise
def test_gelu_exact_bitwise(accuracy_backend, shape, dtype):
    implementation, target, device = accuracy_backend
    selected = implementation or resolve_implementation("gelu", Target.parse(target))
    if selected == "triton":
        pytest.skip("Triton GELU supports approximate='tanh' only")
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    function = modules.GELU(approximate="none", implementation=implementation)
    assert_repeatable(function, (x,))
