"""Tests for functions.situ_and_mul."""

from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_close
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph

# Kimi-K3 SituAndMul consumes the concatenated [gate | up] projections, so the
# last dimension is 2 * intermediate. Routed experts use moe_intermediate_size=3072
# with rows = tokens * num_experts_per_token(16) once grouped; shared experts use
# 2 * 3072; dense KimiMLP layers use intermediate_size=33792.
SITU_AND_MUL_SHAPES = [
    (7, 1026),
    (128, 1280),
    (257, 960),
    (16, 6144),  # kimi moe decode: 1 token * top-16 experts
    (8192, 6144),  # kimi moe prefill: 512 tokens * top-16 experts
    (1024, 12288),  # kimi shared experts: 2 * 3072
    (64, 67584),  # kimi dense mlp: intermediate_size=33792
]

# Kimi-K3 activation_situ_beta=4.0, activation_situ_linear_beta=25.0.
SITU_AND_MUL_PARAMS = [(1.0, None), (4.0, 25.0)]


@pytest.mark.api("functions.situ_and_mul")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
@pytest.mark.parametrize("beta,linear_beta", SITU_AND_MUL_PARAMS)
@pytest.mark.parametrize("shape", SITU_AND_MUL_SHAPES)
@pytest.mark.accuracy
def test_situ_and_mul_accuracy(accuracy_backend, dtype, beta, linear_beta, shape):
    implementation, target, device = accuracy_backend
    reference_x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    actual_x = reference_x.detach().clone().requires_grad_(True)
    grad_output = torch.randn(shape[:-1] + (shape[-1] // 2,), device=device, dtype=dtype)

    gate, up = (half.float() for half in reference_x.chunk(2, dim=-1))
    expected = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    expected = (expected * up).to(dtype)
    actual = F.situ_and_mul(actual_x, beta=beta, linear_beta=linear_beta, implementation=implementation)
    (expected_grad,) = torch.autograd.grad(expected, reference_x, grad_output)
    (actual_grad,) = torch.autograd.grad(actual, actual_x, grad_output)

    check = partial(assert_close, dtype=dtype)
    check(actual, expected)
    check(actual_grad, expected_grad, name="grad")


@pytest.mark.api("functions.situ_and_mul")
@pytest.mark.parametrize("shape", [(128, 1280), (16, 6144)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.bitwise
def test_situ_and_mul_bitwise(accuracy_backend, dtype, shape):
    implementation, _, device = accuracy_backend
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    inputs = (x,)
    assert_repeatable(partial(F.situ_and_mul, beta=4.0, linear_beta=25.0, implementation=implementation), inputs)


@pytest.mark.api("functions.situ_and_mul")
def test_situ_and_mul_fake_metadata(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("NPU Triton fake metadata")
    preload("situ_and_mul", implementation="triton", target=target)
    ops = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    # Even last dimension for the [gate | up] halves, non-contiguous like silu's .T case.
    x = torch.randn(14, 18, device=device, dtype=torch.bfloat16).T
    grad = torch.randn(x.shape[:-1] + (x.shape[-1] // 2,), device=device, dtype=x.dtype)
    torch.library.opcheck(ops.situ_and_mul_fwd.default, (x, 4.0, 25.0), test_utils=("test_faketensor",))
    torch.library.opcheck(ops.situ_and_mul_bwd.default, (grad, x, 4.0, 25.0), test_utils=("test_faketensor",))


@pytest.mark.api("functions.situ_and_mul")
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_situ_and_mul_compiled_layout(accuracy_backend, backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation == "torch":
        pytest.skip("NPU contiguous-output contract")
    implementation = implementation or "triton"
    preload("situ_and_mul", implementation=implementation)
    # Keep the non-contiguous .T layout but an even last dimension for [gate | up].
    x = torch.randn(14, 18, device=device, dtype=torch.bfloat16).T.requires_grad_()
    inputs = (x,)

    def run(*args):
        output = F.situ_and_mul(*args, beta=4.0, linear_beta=25.0, implementation=implementation)
        outputs = output if isinstance(output, tuple) else (output,)
        return outputs, tuple((y.stride(), y.is_contiguous()) for y in outputs)

    try:
        eager, metadata = run(*inputs)
        actual, captured_metadata = compile_fullgraph(run, backend=backend)(*inputs)
        assert captured_metadata == metadata
        assert all(y.is_contiguous() for y in actual)
        assert captured_metadata == tuple((y.stride(), y.is_contiguous()) for y in actual)
        reference = F.situ_and_mul(*inputs, beta=4.0, linear_beta=25.0, implementation="torch")
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


@pytest.mark.reference
@pytest.mark.accuracy
@pytest.mark.parametrize("beta,linear_beta", SITU_AND_MUL_PARAMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_situ_and_mul_reference_matches_cpu_fp64(accuracy_backend, beta, linear_beta, dtype):
    # A CPU FP64 oracle catches vendor-specific dispatch bugs in the reference itself.
    device = accuracy_backend[2]
    gate = torch.linspace(-8, 8, 501, device=device, dtype=torch.float32)
    up = torch.linspace(-64, 64, 501, device=device, dtype=torch.float32)
    x = torch.cat((gate, up), dim=-1).to(dtype).requires_grad_()
    reference_x = x.detach().cpu().double().requires_grad_()

    actual = F.situ_and_mul(x, beta=beta, linear_beta=linear_beta, implementation="torch")
    reference_gate, reference_up = reference_x.chunk(2, dim=-1)
    expected = beta * torch.tanh(reference_gate / beta) * torch.sigmoid(reference_gate)
    if linear_beta is not None:
        reference_up = linear_beta * torch.tanh(reference_up / linear_beta)
    expected = (expected * reference_up).to(x.dtype)
    actual_grad = torch.autograd.grad(actual.sum(), x)[0]
    expected_grad = torch.autograd.grad(expected.sum(), reference_x)[0]
    assert_close(actual, expected.to(x), dtype)
    assert_close(actual_grad, expected_grad.to(x), dtype)
