"""Tests for functions.silu."""

from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_close
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph

ACTIVATION_SHAPES = [(7, 513), (128, 128), (256, 128), (999, 9999), (1024, 10240)]


@pytest.mark.api("functions.silu")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("shape", ACTIVATION_SHAPES)
@pytest.mark.accuracy
def test_silu_accuracy(accuracy_backend, dtype, shape):
    implementation, target, device = accuracy_backend
    generate = torch.randn if shape == (7, 513) else torch.rand
    reference_x = generate(shape, device=device, dtype=dtype, requires_grad=True)
    actual_x = reference_x.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(reference_x) if shape == (7, 513) else torch.rand_like(reference_x)
    function = F.silu

    expected = function(reference_x, implementation="torch")
    actual = function(actual_x, implementation=implementation)
    (expected_grad,) = torch.autograd.grad(expected, reference_x, grad_output)
    (actual_grad,) = torch.autograd.grad(actual, actual_x, grad_output)

    original_silu = dtype == torch.float32 and shape in ((128, 128), (999, 9999), (1024, 10240))
    # Preserve the pointwise budget of the original FP32 training cases.
    tolerance = dict(rtol=6e-3, atol=6e-3) if original_silu else {}
    check = partial(assert_close, dtype=dtype, **tolerance)
    check(actual, expected)
    check(actual_grad, expected_grad, name="grad")


@pytest.mark.api("functions.silu")
@pytest.mark.parametrize("shape", ACTIVATION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.bitwise
def test_silu_bitwise(accuracy_backend, dtype, shape):
    implementation, _, device = accuracy_backend
    generate = torch.rand if shape in ACTIVATION_SHAPES[1:] else torch.randn
    x = generate(shape, device=device, dtype=dtype, requires_grad=True)
    inputs = (x,)
    assert_repeatable(partial(F.silu, implementation=implementation), inputs)


@pytest.mark.api("functions.silu")
def test_silu_fake_metadata(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("NPU Triton fake metadata")
    preload("silu", implementation="triton", target=target)
    ops = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    x = torch.randn(17, 7, device=device, dtype=torch.bfloat16).T
    torch.library.opcheck(ops.silu_fwd.default, (x,), test_utils=("test_faketensor",))
    torch.library.opcheck(ops.silu_bwd.default, (torch.randn_like(x), x), test_utils=("test_faketensor",))


@pytest.mark.api("functions.silu")
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_silu_compiled_layout(accuracy_backend, backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation == "torch":
        pytest.skip("NPU contiguous-output contract")
    implementation = implementation or "triton"
    preload("silu", implementation=implementation)
    x = torch.randn(17, 7, device=device, dtype=torch.bfloat16).T.requires_grad_()
    inputs = (x,)

    def run(*args):
        output = F.silu(*args, implementation=implementation)
        outputs = output if isinstance(output, tuple) else (output,)
        return outputs, tuple((y.stride(), y.is_contiguous()) for y in outputs)

    try:
        eager, metadata = run(*inputs)
        actual, captured_metadata = compile_fullgraph(run, backend=backend)(*inputs)
        assert captured_metadata == metadata
        assert all(y.is_contiguous() for y in actual)
        assert captured_metadata == tuple((y.stride(), y.is_contiguous()) for y in actual)
        reference = F.silu(*inputs, implementation="torch")
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
