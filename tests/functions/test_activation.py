from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_repeatable

from .._checks import assert_close
from .._checks import assert_mojo_close
from .._checks import clone_with_grad

ACTIVATION_SHAPES = [(7, 513), (128, 128), (256, 128), (999, 9999), (1024, 10240)]


SWIGLU_SHAPES = ACTIVATION_SHAPES + [(3072, 3072), (4096, 2048), (257, 480)]


@pytest.mark.api("functions.silu", "functions.gelu")
@pytest.mark.parametrize("function_name", ("silu", "gelu"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("shape", ACTIVATION_SHAPES)
@pytest.mark.accuracy
def test_unary(accuracy_backend, function_name, dtype, shape):
    implementation, target, device = accuracy_backend
    generate = torch.randn if shape == (7, 513) else torch.rand
    reference_x = generate(shape, device=device, dtype=dtype, requires_grad=True)
    actual_x = reference_x.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(reference_x) if shape == (7, 513) else torch.rand_like(reference_x)
    function = getattr(F, function_name)

    expected = function(reference_x, implementation="torch_reference")
    actual = function(actual_x, implementation=implementation)
    (expected_grad,) = torch.autograd.grad(expected, reference_x, grad_output)
    (actual_grad,) = torch.autograd.grad(actual, actual_x, grad_output)

    original_silu = (
        function_name == "silu" and dtype == torch.float32 and shape in ((128, 128), (999, 9999), (1024, 10240))
    )
    check = assert_mojo_close if original_silu else partial(assert_close, dtype=dtype)
    check(actual, expected)
    check(actual_grad, expected_grad, name="grad")


@pytest.mark.api("functions.swiglu")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("limit", [0.0, 0.5])
@pytest.mark.parametrize("shape", SWIGLU_SHAPES)
@pytest.mark.accuracy
def test_swiglu(accuracy_backend, dtype, limit, shape):
    implementation, target, device = accuracy_backend
    generate = torch.rand if shape in ACTIVATION_SHAPES[1:] else torch.randn
    reference_inputs = tuple(generate(shape, device=device, dtype=dtype, requires_grad=True) for _ in range(2))
    actual_inputs = clone_with_grad(reference_inputs)
    grad_output = torch.randn_like(reference_inputs[0])

    gate, up = (x.float() for x in reference_inputs)
    expected = (
        torch.nn.functional.silu(gate.clamp(max=limit) if limit > 0 else gate)
        * (up.clamp(-limit, limit) if limit > 0 else up)
    ).to(dtype)
    actual = F.swiglu(*actual_inputs, swiglu_limit=limit, implementation=implementation)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output)
    actual_grads = torch.autograd.grad(actual, actual_inputs, grad_output)

    assert_close(actual, expected, dtype)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert_close(actual_grad, expected_grad, dtype)


@pytest.mark.api("functions.silu", "functions.gelu", "functions.swiglu")
@pytest.mark.parametrize(
    "op,limit,shape",
    [
        (op, limit, shape)
        for op, limit in [("silu", 0), ("gelu", 0), ("swiglu", 0), ("swiglu", 0.5)]
        for shape in (SWIGLU_SHAPES if op == "swiglu" else ACTIVATION_SHAPES)
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.bitwise
def test_activation_bitwise(accuracy_backend, op, limit, dtype, shape):
    implementation, _, device = accuracy_backend
    generate = torch.rand if shape in ACTIVATION_SHAPES[1:] else torch.randn
    x = generate(shape, device=device, dtype=dtype, requires_grad=True)
    inputs = (x,)
    if op == "swiglu":
        inputs += (generate(shape, device=device, dtype=dtype, requires_grad=True),)
    options = {"swiglu_limit": limit} if op == "swiglu" else {}
    assert_repeatable(partial(getattr(F, op), implementation=implementation, **options), inputs)


@pytest.mark.api("functions.gelu")
@pytest.mark.parametrize("shape", ACTIVATION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.accuracy
def test_gelu_exact(accuracy_backend, shape, dtype):
    implementation, _, device = accuracy_backend
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    try:
        actual = F.gelu(x, approximate="none", implementation=implementation)
    except NotImplementedError as error:
        assert "Triton GELU supports approximate='tanh' only" in str(error)
        return
    expected = torch.nn.functional.gelu(reference_x.float(), approximate="none").to(dtype)
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


@pytest.mark.api("functions.gelu")
@pytest.mark.parametrize("shape", ACTIVATION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.bitwise
def test_gelu_exact_bitwise(accuracy_backend, shape, dtype):
    implementation, _, device = accuracy_backend
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    function = partial(F.gelu, approximate="none", implementation=implementation)
    try:
        assert_repeatable(function, (x,))
    except NotImplementedError as error:
        assert "Triton GELU supports approximate='tanh' only" in str(error)
