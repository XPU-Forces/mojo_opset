from functools import partial

import pytest
import torch

from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

from mojo_opset import functions
from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_accuracy
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph

from .._checks import assert_close
from .._checks import assert_mojo_close
from .._checks import clone_with_grad

CONVOLUTION_CASES = [
    # batch, tokens, channels, width, activation, bias, residual, dtype, sequences
    (2, 64, 1024, 3, "swish", True, True, torch.float16, None),
    (2, 128, 8192, 4, "swish", False, True, torch.float16, None),
    (2, 64, 8192, 3, "swish", True, False, torch.float16, None),
    (2, 128, 4096, 4, "swish", False, False, torch.float16, None),
    (2, 64, 8192, 3, None, True, False, torch.float16, None),
    (3, 1446, 8192, 4, None, False, False, torch.float16, None),
    (1, 500, 1024, 3, "silu", True, False, torch.float16, 4),
    (1, 1024, 256, 4, "silu", False, False, torch.float16, 3),
    (1, 500, 1024, 3, None, True, False, torch.float16, 4),
    (1, 1024, 1024, 4, None, False, False, torch.float16, 4),
    (1, 8192, 8192, 4, None, False, False, torch.float32, 5),
    (1, 7666, 8192, 4, None, False, False, torch.float32, 3),
    (1, 12291, 8192, 4, None, False, False, torch.bfloat16, 5),
    (1, 12291, 8192, 4, "silu", False, False, torch.bfloat16, 5),
    (1, 11357, 8192, 4, None, False, False, torch.bfloat16, 6),
    (1, 10287, 8192, 4, "silu", False, False, torch.bfloat16, 9),
    (2, 31, 64, 4, "silu", True, False, torch.bfloat16, None),
]


def make_convolution_case(case, device):
    batch, tokens, channels, width, activation, bias, residual, dtype, sequences = case
    generator = torch.Generator().manual_seed(41 if sequences else 42)
    cu = None
    if sequences:
        middle = torch.arange(16, tokens)[torch.randperm(tokens - 16, generator=generator)[: sequences - 1]]
        cu = torch.cat((torch.tensor([0]), middle, torch.tensor([tokens]))).sort().values.to(device)
    generate = torch.randn if (tokens, channels) == (31, 64) else torch.rand
    x = generate(batch, tokens, channels, generator=generator).to(device=device, dtype=dtype).requires_grad_()
    weight = generate(channels, width, generator=generator).to(device=device, dtype=dtype).requires_grad_()
    inputs = [x, weight]
    if bias:
        inputs.append(generate(channels, generator=generator).to(device=device, dtype=dtype).requires_grad_())
    if residual:
        inputs.append(x.detach().clone().requires_grad_())
    upstream = generate(batch, tokens, channels, generator=generator).to(device=device, dtype=dtype)
    return tuple(inputs), dict(activation=activation, cu_seqlens=cu), upstream


UPDATE_LENGTHS = [(1, 4, 4), (2, 3, 4), (17, 8, 3), (257, 4, 4), (3, 0, 1)]
UPDATE_SHAPES = [
    (1, 12291, 8192, 4, "swish"),
    (1, 5000, 2048, 4, "swish"),
    (2, 64, 128, 3, "swish"),
    (2, 128, 128, 4, "swish"),
    (2, 64, 128, 3, None),
    (3, 1446, 256, 4, None),
    (1, 32, 32, 4, None),
]


@pytest.mark.api("functions.causal_conv1d_update_state_infer")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("length,state_length,width", UPDATE_LENGTHS)
@pytest.mark.parametrize("activation", [None, "swish"])
@pytest.mark.accuracy
def test_update(accuracy_backend, dtype, length, state_length, width, activation):
    implementation, _, device = accuracy_backend
    x = torch.randn(2, 17, length, device=device, dtype=dtype)
    state = torch.randn(2, 17, state_length, device=device, dtype=dtype)
    weight = torch.randn(17, width, device=device, dtype=dtype)
    bias = torch.randn(17, device=device, dtype=dtype)
    reference_state = state.clone()
    ptr = state.data_ptr()
    for current in (x, x[..., :1]):
        actual = F.causal_conv1d_update_state_infer(
            current, state, weight, bias, activation, implementation=implementation
        )
        expected = F.causal_conv1d_update_state_infer(
            current, reference_state, weight, bias, activation, implementation="torch_reference"
        )
        assert_close(actual, expected, dtype)
        assert_close(state, reference_state, rtol=0, atol=0)
        assert state.data_ptr() == ptr


@pytest.mark.api("functions.causal_conv1d_update_state_infer")
@pytest.mark.parametrize("b,t,d,w,activation", UPDATE_SHAPES)
@pytest.mark.accuracy
def test_update_master(accuracy_backend, b, t, d, w, activation):
    implementation, _, device = accuracy_backend
    x = torch.randn(b, d, t, device=device, dtype=torch.float16)
    state = torch.randn(b, d, w, device=device, dtype=x.dtype)
    reference_state = state.clone()
    weight = torch.randn(d, w, device=device, dtype=x.dtype)
    actual = F.causal_conv1d_update_state_infer(x, state, weight, activation=activation, implementation=implementation)
    expected = F.causal_conv1d_update_state_infer(
        x, reference_state, weight, activation=activation, implementation="torch_reference"
    )
    assert_mojo_close(actual, expected)
    assert_close(state, reference_state, rtol=0, atol=0)


@pytest.mark.api("functions.causal_conv1d")
@pytest.mark.parametrize("state", [False, True])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.accuracy
def test_options(accuracy_backend, state, varlen, activation):
    _, _, device = accuracy_backend
    x = torch.randn(1 if varlen else 2, 17, 64, device=device)
    w = torch.randn(64, 4, device=device)
    bias = torch.randn(64, device=device)
    residual = torch.randn_like(x)
    h = torch.randn(2, 64, 3, device=device)
    cu = torch.tensor([0, 2, 17], device=device, dtype=torch.int32) if varlen else None

    def run(x, w, bias, residual, h, **selection):
        return F.causal_conv1d(
            x,
            w,
            bias,
            residual=residual,
            initial_state=h,
            output_final_state=state,
            activation=activation,
            cu_seqlens=cu,
            **selection,
        )

    assert_accuracy(run, (x, w, bias, residual, h), implementation=accuracy_backend[0])


@pytest.mark.api("functions.causal_conv1d")
@pytest.mark.accuracy
def test_streaming(accuracy_backend):
    implementation, target, device = accuracy_backend
    x = torch.randn(1, 8, 64, device=device, requires_grad=True)
    w = torch.randn(64, 4, device=device, requires_grad=True)
    options = dict(implementation=implementation)
    full, _ = F.causal_conv1d(x, w, **options)
    first, state = F.causal_conv1d(x[:, :2], w, output_final_state=True, **options)
    last, _ = F.causal_conv1d(x[:, 2:], w, initial_state=state, **options)
    assert_close(torch.cat((first, last), 1), full, rtol=1.3e-6, atol=1e-5)
    _, final = F.causal_conv1d(x, w, output_final_state=True, **options)
    grad = torch.autograd.grad(final.sum(), x)[0]
    expected = torch.zeros_like(x)
    expected[:, -3:] = 1
    assert_close(grad, expected, rtol=1.3e-6, atol=1e-5)


@pytest.mark.api("functions.causal_conv1d")
@pytest.mark.accuracy
def test_strides(accuracy_backend):
    device = accuracy_backend[2]
    x = torch.randn(2, 64, 17, device=device).transpose(1, 2)
    weight = torch.randn(4, 64, device=device).transpose(0, 1)
    assert_accuracy(
        F.causal_conv1d, (x, weight), implementation=accuracy_backend[0], activation="silu", output_final_state=True
    )


@pytest.mark.api("functions.causal_conv1d")
@pytest.mark.parametrize("case", CONVOLUTION_CASES)
@pytest.mark.accuracy
def test_causal_conv1d(accuracy_backend, case):
    implementation, target, device = accuracy_backend
    dtype = case[7]
    reference_inputs, options, grad_output = make_convolution_case(case, device)
    actual_inputs = clone_with_grad(reference_inputs)

    def run(inputs, selected):
        x, weight, *optional = inputs
        bias = optional.pop(0) if case[5] else None
        residual = optional.pop(0) if case[6] else None
        return F.causal_conv1d(x, weight, bias, residual=residual, implementation=selected, **options)[0]

    expected = run(reference_inputs, "torch_reference")
    actual = run(actual_inputs, implementation)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output)
    actual_grads = torch.autograd.grad(actual, actual_inputs, grad_output)

    check = partial(assert_close, dtype=dtype) if case[1:3] == (31, 64) else assert_mojo_close
    check(actual, expected)
    for index, (actual_grad, expected_grad) in enumerate(zip(actual_grads, expected_grads)):
        check(actual_grad, expected_grad, name=f"grad[{index}]")


@pytest.mark.api("functions.causal_conv1d")
@pytest.mark.parametrize("case", CONVOLUTION_CASES)
@pytest.mark.bitwise
def test_convolution_bitwise(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    inputs, options, upstream = make_convolution_case(case, device)
    del upstream

    def run(x, weight, *optional):
        bias = optional[0] if case[5] else None
        residual = optional[int(case[5])] if case[6] else None
        return F.causal_conv1d(x, weight, bias, residual=residual, implementation=implementation, **options)

    assert_repeatable(run, inputs)


@pytest.mark.accuracy
@pytest.mark.api("functions.causal_conv1d")
def test_compiled_leaves(accuracy_backend):
    op = "causal_conv1d"
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu."):
        pytest.skip("NPU Triton leaf decomposition contract")
    preload(op, implementation=implementation, target=target)
    nodes = {"forward": [], "backward": []}

    def compiler(phase):
        def capture(graph, inputs):
            nodes[phase].extend(str(node.target) for node in graph.graph.nodes if node.op == "call_function")
            return make_boxed_func(graph.forward)

        return capture

    backend = aot_autograd(fw_compiler=compiler("forward"), bw_compiler=compiler("backward"))
    inputs = tuple(
        torch.randn(shape, device=device, requires_grad=True) for shape in ((2, 17, 64), (64, 4), (2, 64, 3))
    )

    def run(x, weight, state):
        return F.causal_conv1d(
            x,
            weight,
            initial_state=state,
            output_final_state=True,
            activation="silu",
            implementation=implementation,
        )

    expected_forward = {"causal_conv1d_forward", "causal_conv1d_state"}
    expected_backward = {
        "causal_conv1d_forward",
        "silu_bwd",
        "causal_conv1d_backward",
        "causal_conv1d_state_backward",
    }
    eager_inputs = tuple(t.detach().clone().requires_grad_() for t in inputs)
    try:
        compiled = compile_fullgraph(run, backend=backend)
        output, expected = compiled(*inputs), run(*eager_inputs)
        output = (output,) if isinstance(output, torch.Tensor) else output
        expected = (expected,) if isinstance(expected, torch.Tensor) else expected
        upstream = tuple(torch.randn_like(t) for t in output)
        for a, b in zip(output, expected):
            torch.testing.assert_close(a, b)
        actual_grads = torch.autograd.grad(output, inputs, upstream)
        expected_grads = torch.autograd.grad(expected, eager_inputs, upstream)
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b)
        for phase, leaves in (("forward", expected_forward), ("backward", expected_backward)):
            names = {name.split(".")[-2] for name in nodes[phase] if name.startswith("mojo_")}
            assert leaves <= names, nodes[phase]
    finally:
        torch._dynamo.reset()


@pytest.mark.accuracy
@pytest.mark.api("functions.causal_conv1d_update_state_infer")
def test_update_compile(accuracy_backend):
    implementation, _, device = accuracy_backend
    preload("causal_conv1d_update_state_infer", implementation=implementation)
    x = torch.randn(2, 64, 3, device=device)
    w = torch.randn(64, 4, device=device)
    state = torch.randn(2, 64, 5, device=device)
    eager_state = state.clone()

    def run(x, w, state):
        return functions.causal_conv1d_update_state_infer(x, state, w, implementation=implementation)

    try:
        compiled = compile_fullgraph(run)
        for _ in range(2):
            torch.testing.assert_close(compiled(x, w, state), run(x, w, eager_state))
            torch.testing.assert_close(state, eager_state, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()
