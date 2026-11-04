"""Tests for functions.swiglu."""

from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_close
from tests._checks import assert_repeatable
from tests._checks import clone_with_grad
from tests._checks import comparison_dtype
from tests._compile import compile_fullgraph

ACTIVATION_SHAPES = [(7, 513), (128, 128), (256, 128), (999, 9999), (1024, 10240)]
SWIGLU_SHAPES = ACTIVATION_SHAPES + [(3072, 3072), (4096, 2048), (257, 480), (8192, 1536), (8192, 2048)]


@pytest.mark.api("functions.swiglu")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
@pytest.mark.parametrize("limit", [0.0, 0.5])
@pytest.mark.parametrize("shape", SWIGLU_SHAPES)
@pytest.mark.parametrize("scale_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.accuracy
def test_swiglu_accuracy(accuracy_backend, dtype, limit, shape, scale_dtype):
    implementation, target, device = accuracy_backend
    generate = torch.rand if shape in ACTIVATION_SHAPES[1:] else torch.randn
    reference_inputs = tuple(generate(shape, device=device, dtype=dtype, requires_grad=True) for _ in range(2))
    if scale_dtype is not None:
        reference_inputs += (torch.randn(shape[:-1], device=device, dtype=scale_dtype, requires_grad=True),)
    actual_inputs = clone_with_grad(reference_inputs)
    grad_output = torch.randn_like(reference_inputs[0])

    gate, up = (x.float() for x in reference_inputs[:2])
    expected = torch.nn.functional.silu(gate.clamp(max=limit) if limit > 0 else gate) * (
        up.clamp(-limit, limit) if limit > 0 else up
    )
    if scale_dtype is not None:
        expected = expected * reference_inputs[2].float().unsqueeze(-1)
    expected = expected.to(dtype)
    actual = F.swiglu(*actual_inputs, swiglu_limit=limit, implementation=implementation)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output)
    actual_grads = torch.autograd.grad(actual, actual_inputs, grad_output)

    assert_close(actual, expected, dtype)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert_close(actual_grad, expected_grad, comparison_dtype((actual_inputs[0], actual_grad)))


@pytest.mark.api("functions.swiglu")
@pytest.mark.parametrize("limit", [0.0, 0.5])
@pytest.mark.parametrize("shape", SWIGLU_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scale_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.bitwise
def test_swiglu_bitwise(accuracy_backend, limit, dtype, shape, scale_dtype):
    implementation, _, device = accuracy_backend
    generate = torch.rand if shape in ACTIVATION_SHAPES[1:] else torch.randn
    x = generate(shape, device=device, dtype=dtype, requires_grad=True)
    inputs = (x, generate(shape, device=device, dtype=dtype, requires_grad=True))
    if scale_dtype is not None:
        inputs += (torch.randn(shape[:-1], device=device, dtype=scale_dtype, requires_grad=True),)
    options = {"swiglu_limit": limit}
    assert_repeatable(partial(F.swiglu, implementation=implementation, **options), inputs)


@pytest.mark.api("functions.swiglu")
@pytest.mark.parametrize("scale_dtype", [None, torch.bfloat16, torch.float32])
def test_swiglu_fake_metadata(accuracy_backend, scale_dtype):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("NPU Triton fake metadata")
    preload("swiglu", implementation="triton", target=target)
    ops = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    x = torch.randn(17, 7, device=device, dtype=torch.bfloat16).T
    scales = torch.randn(14, device=device, dtype=scale_dtype)[::2] if scale_dtype is not None else None
    inputs = (x, torch.randn_like(x), scales)
    torch.library.opcheck(ops.swiglu_fwd.default, inputs, test_utils=("test_faketensor",))
    torch.library.opcheck(ops.swiglu_bwd.default, (torch.randn_like(x), *inputs), test_utils=("test_faketensor",))


@pytest.mark.api("functions.swiglu")
@pytest.mark.parametrize("scale_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_swiglu_compiled_layout(accuracy_backend, backend, scale_dtype):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation == "torch":
        pytest.skip("NPU contiguous-output contract")
    implementation = implementation or "triton"
    preload("swiglu", implementation=implementation)
    x = torch.randn(17, 7, device=device, dtype=torch.bfloat16).T.requires_grad_()
    inputs = (x, torch.randn_like(x, requires_grad=True))
    if scale_dtype is not None:
        scales = torch.randn(14, device=device, dtype=scale_dtype)[::2].requires_grad_()
        inputs += (scales,)

    def run(*args):
        output = F.swiglu(*args, implementation=implementation)
        outputs = output if isinstance(output, tuple) else (output,)
        return outputs, tuple((y.stride(), y.is_contiguous()) for y in outputs)

    try:
        eager, metadata = run(*inputs)
        actual, captured_metadata = compile_fullgraph(run, backend=backend)(*inputs)
        assert captured_metadata == metadata
        assert all(y.is_contiguous() for y in actual)
        assert captured_metadata == tuple((y.stride(), y.is_contiguous()) for y in actual)
        reference = F.swiglu(*inputs, implementation="torch")
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
