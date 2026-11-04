"""Tests for functions.gelu."""

from functools import partial

import pytest
import torch

from mojo_opset import Target
from mojo_opset import functions as F
from mojo_opset import preload
from mojo_opset.functions._dispatch import resolve_implementation
from tests._checks import assert_close
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph

ACTIVATION_SHAPES = [(7, 513), (128, 128), (256, 128), (999, 9999), (1024, 10240)]


@pytest.mark.api("functions.gelu")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("shape", ACTIVATION_SHAPES)
@pytest.mark.parametrize("approximate", ["tanh", "none"])
@pytest.mark.accuracy
def test_gelu_accuracy(accuracy_backend, dtype, shape, approximate):
    implementation, target, device = accuracy_backend
    selected = implementation or resolve_implementation("gelu", Target.parse(target))
    if approximate == "none" and selected == "triton":
        pytest.skip("Triton GELU supports approximate='tanh' only")
    generate = torch.randn if approximate == "none" or shape == (7, 513) else torch.rand
    reference_x = generate(shape, device=device, dtype=dtype, requires_grad=True)
    actual_x = reference_x.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(reference_x) if shape == (7, 513) else torch.rand_like(reference_x)
    function = partial(F.gelu, approximate=approximate)

    expected = function(reference_x, implementation="torch")
    actual = function(actual_x, implementation=implementation)
    (expected_grad,) = torch.autograd.grad(expected, reference_x, grad_output)
    (actual_grad,) = torch.autograd.grad(actual, actual_x, grad_output)

    check = partial(assert_close, dtype=dtype)
    check(actual, expected)
    check(actual_grad, expected_grad, name="grad")


@pytest.mark.api("functions.gelu")
@pytest.mark.parametrize("shape", ACTIVATION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("approximate", ["tanh", "none"])
@pytest.mark.bitwise
def test_gelu_bitwise(accuracy_backend, dtype, shape, approximate):
    implementation, target, device = accuracy_backend
    selected = implementation or resolve_implementation("gelu", Target.parse(target))
    if approximate == "none" and selected == "triton":
        pytest.skip("Triton GELU supports approximate='tanh' only")
    generate = torch.randn if approximate == "none" or shape == (7, 513) else torch.rand
    x = generate(shape, device=device, dtype=dtype, requires_grad=True)
    inputs = (x,)
    assert_repeatable(partial(F.gelu, approximate=approximate, implementation=implementation), inputs)


@pytest.mark.reference
@pytest.mark.accuracy
@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_gelu_reference_matches_cpu_fp64(accuracy_backend, approximate, dtype):
    # A CPU FP64 oracle catches vendor-specific dispatch bugs in the reference itself.
    x = torch.linspace(-5, 5, 1001, device=accuracy_backend[2], dtype=dtype).requires_grad_()
    reference_x = x.detach().cpu().double().requires_grad_()
    actual = F.gelu(x, approximate=approximate, implementation="torch")
    expected = torch.nn.functional.gelu(reference_x, approximate=approximate)
    actual_grad = torch.autograd.grad(actual.sum(), x)[0]
    expected_grad = torch.autograd.grad(expected.sum(), reference_x)[0]
    assert_close(actual, expected.to(x), dtype)
    assert_close(actual_grad, expected_grad.to(x), dtype)


@pytest.mark.api("functions.gelu")
def test_gelu_fake_metadata(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("NPU Triton fake metadata")
    preload("gelu", implementation="triton", target=target)
    ops = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    x = torch.randn(17, 7, device=device, dtype=torch.bfloat16).T
    torch.library.opcheck(ops.gelu_fwd.default, (x,), test_utils=("test_faketensor",))
    torch.library.opcheck(ops.gelu_bwd.default, (torch.randn_like(x), x), test_utils=("test_faketensor",))


@pytest.mark.api("functions.gelu")
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_gelu_compiled_layout(accuracy_backend, backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation == "torch":
        pytest.skip("NPU contiguous-output contract")
    implementation = implementation or "triton"
    preload("gelu", implementation=implementation)
    x = torch.randn(17, 7, device=device, dtype=torch.bfloat16).T.requires_grad_()
    inputs = (x,)

    def run(*args):
        output = F.gelu(*args, implementation=implementation)
        outputs = output if isinstance(output, tuple) else (output,)
        return outputs, tuple((y.stride(), y.is_contiguous()) for y in outputs)

    try:
        eager, metadata = run(*inputs)
        actual, captured_metadata = compile_fullgraph(run, backend=backend)(*inputs)
        assert captured_metadata == metadata
        assert all(y.is_contiguous() for y in actual)
        assert captured_metadata == tuple((y.stride(), y.is_contiguous()) for y in actual)
        reference = F.gelu(*inputs, implementation="torch")
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
