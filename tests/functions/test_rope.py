from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_close
from tests._checks import assert_repeatable
from tests._checks import clone_with_grad
from tests._compile import compile_fullgraph

ROPE_CASES = [
    pytest.param(
        batch,
        tokens,
        q_heads,
        k_heads,
        head_first,
        dim,
        rotary_dim,
        id=f"b{batch}-t{tokens}-h{q_heads}-{k_heads}-{'BHTD' if head_first else 'BTHD'}-d{dim}-r{rotary_dim}",
    )
    for batch, tokens in [(1, 124), (6, 555), (2, 2048)]
    for q_heads, k_heads, head_first in [
        (32, 8, True),
        (32, 4, False),
        (8, 2, True),
        (16, 1, False),
        (16, 8, True),
        (64, 8, False),
        (64, 4, True),
        (2, 1, False),
    ]
    for dim, rotary_dim in [(128, 128), (88, 88), (128, 48)]
] + [
    # Exercise vector repeat-count and FP32 mask-width boundaries.
    pytest.param(1, 555, 4, 2, False, 128, 16, id="repeat-boundary-r16"),
    pytest.param(1, 7, 4, 2, False, 256, 256, id="mask-boundary-r256"),
    # Real head counts and rotary width; T=1024 is a single-card CI truncation.
    pytest.param(1, 1024, 48, 8, False, 128, 48, id="h48-kv8-t1024-r48"),
    pytest.param(1, 1024, 56, 8, False, 128, 48, id="h56-kv8-t1024-r48"),
]


def make_rope_case(device, dtype, batch, tokens, q_heads, k_heads, head_first, dim, rotary_dim):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    angles = torch.outer(torch.arange(tokens, device=device).float(), inv_freq)
    angles = torch.cat((angles, angles), dim=-1)
    inputs = tuple(
        torch.randn(
            (batch, heads, tokens, dim) if head_first else (batch, tokens, heads, dim),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        for heads in (q_heads, k_heads)
    )
    return inputs, angles.cos(), angles.sin()


def rope_reference(q, k, cos, sin, unsqueeze_dim):
    def rotate(x):
        prefix = x.shape[-1] - cos.shape[-1]
        values = x[..., prefix:].float()
        first, second = values.chunk(2, dim=-1)
        output = values * cos.float().unsqueeze(unsqueeze_dim)
        output = output + torch.cat((-second, first), dim=-1) * sin.float().unsqueeze(unsqueeze_dim)
        return torch.cat((x[..., :prefix], output.to(x.dtype)), dim=-1)

    return rotate(q), rotate(k)


@pytest.mark.api("functions.rope")
@pytest.mark.bitwise
def test_rope_bitwise(accuracy_backend):
    implementation, _, device = accuracy_backend
    q, k = (torch.randn(1, h, 7, 16, device=device, dtype=torch.bfloat16, requires_grad=True) for h in (4, 2))
    angles = torch.randn(1, 7, 16, device=device)
    assert_repeatable(
        partial(F.rope, unsqueeze_dim=1, implementation=implementation),
        (q, k, angles.cos().bfloat16(), angles.sin().bfloat16()),
    )


@pytest.mark.api("functions.rope")
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "case,dtype,cache,q_transposed,k_transposed,table_dtype",
    [
        pytest.param(case.values, dtype, "sinusoidal", False, False, dtype, id=f"{case.id}-{dtype}")
        for case in ROPE_CASES
        for dtype in (torch.float16, torch.bfloat16)
    ]
    + [
        ((1, 7, 4, 2, True, 16, 16), torch.bfloat16, "batched", q, k, torch.bfloat16)
        for q, k in ((False, True), (True, False), (True, True))
    ]
    + [
        ((2, 9, 4, 2, head_first, 88, rotary_dim), torch.bfloat16, "repeated", False, False, table_dtype)
        for head_first in (False, True)
        for rotary_dim in (48, 88)
        for table_dtype in (torch.bfloat16, torch.float32)
    ]
    + [
        ((1, 124, 2, 1, False, 88, 88), dtype, "independent", False, False, dtype)
        for dtype in (torch.float16, torch.bfloat16, torch.float32)
    ],
)
def test_rope_accuracy(accuracy_backend, case, dtype, cache, q_transposed, k_transposed, table_dtype):
    implementation, _, device = accuracy_backend
    batch, tokens, q_heads, k_heads, head_first, dim, rotary_dim = case
    inputs, cos, sin = make_rope_case(device, dtype, *case)
    if cache != "sinusoidal":
        shape = (
            (1, tokens, rotary_dim)
            if cache == "batched"
            else (tokens, rotary_dim // 2 if cache == "repeated" else rotary_dim)
        )
        angles = torch.randn(shape, device=device)
        if cache == "repeated":
            angles = torch.cat((angles, angles), dim=-1)
        cos, sin = angles.cos().to(table_dtype), angles.sin().to(table_dtype)
    if q_transposed or k_transposed:
        inputs = tuple(
            torch.randn(batch, tokens, heads, dim, device=device, dtype=dtype).transpose(1, 2).requires_grad_()
            if transposed
            else original
            for heads, transposed, original in zip((q_heads, k_heads), (q_transposed, k_transposed), inputs)
        )
    options = dict(unsqueeze_dim=1 if cache == "batched" or not head_first else 0)
    reference_inputs = clone_with_grad(inputs)
    actual = F.rope(*inputs, cos, sin, implementation=implementation, **options)
    expected = rope_reference(*reference_inputs, cos, sin, options["unsqueeze_dim"])
    assert tuple(t.dtype for t in actual) == tuple(t.dtype for t in inputs)
    gradient = torch.rand_like if cache == "sinusoidal" else torch.randn_like
    upstream = tuple(gradient(output) for output in actual)
    actual_grads = torch.autograd.grad(actual, inputs, upstream)
    expected_grads = torch.autograd.grad(expected, reference_inputs, upstream)
    # Shape cases retain their historical budget; the small cache/layout cases stay stricter.
    tolerance = {}
    if cache == "sinusoidal":
        rtol, atol = (2e-2, 2e-2) if dtype == torch.float16 else (5e-2, 1e-1)
        tolerance = dict(rtol=rtol, atol=atol)
    for index, (a, b) in enumerate(zip((*actual, *actual_grads), (*expected, *expected_grads))):
        assert_close(a, b, dtype, **tolerance, name=f"output/grad[{index}]")


@pytest.mark.api("functions.rope")
@pytest.mark.parametrize("batch,tokens,q_heads,k_heads,head_first,dim,rotary_dim", ROPE_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.bitwise
def test_rope_shapes_bitwise(accuracy_backend, batch, tokens, q_heads, k_heads, head_first, dim, rotary_dim, dtype):
    implementation, _, device = accuracy_backend
    inputs, cos, sin = make_rope_case(device, dtype, batch, tokens, q_heads, k_heads, head_first, dim, rotary_dim)
    assert_repeatable(
        partial(F.rope, cos=cos, sin=sin, unsqueeze_dim=0 if head_first else 1, implementation=implementation),
        inputs,
    )


@pytest.mark.api("functions.rope")
def test_rope_fake_metadata(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("NPU Triton fake metadata")
    preload("rope", implementation="triton", target=target)
    ops = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    q = torch.randn(1, 4, 7, 16, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 7, 2, 16, device=device, dtype=q.dtype).transpose(1, 2)
    angles = torch.randn(1, 7, 16, device=device)
    inputs = (q, k, angles.cos().bfloat16(), angles.sin().bfloat16(), 1)
    for leaf in (ops.rope_fwd.default, ops.rope_bwd.default):
        torch.library.opcheck(leaf, inputs, test_utils=("test_faketensor",))


@pytest.mark.api("functions.rope")
@pytest.mark.parametrize("op", ["rope"])
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_compiled_layout(accuracy_backend, op, backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton", "native") or not target.startswith("npu."):
        pytest.skip("NPU contiguous-output contract")
    implementation = implementation or "triton"
    scaled = op == "swiglu_scaled"
    op = "swiglu" if scaled else op
    preload(op, implementation=implementation)
    x = torch.randn(17, 7, device=device, dtype=torch.bfloat16).T.requires_grad_()
    inputs = (x,)
    kwargs = {}
    if op == "rope":
        q = torch.randn(1, 4, 7, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, 7, 2, 16, device=device, dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
        angles = torch.randn(1, 7, 16, device=device)
        inputs = (q, k)
        kwargs = {"cos": angles.cos().bfloat16(), "sin": angles.sin().bfloat16(), "unsqueeze_dim": 1}

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
