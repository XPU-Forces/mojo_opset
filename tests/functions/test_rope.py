from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_accuracy
from tests._checks import assert_mojo_close
from tests._checks import assert_repeatable
from tests._checks import clone_with_grad

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


@pytest.mark.parametrize("q_transposed,k_transposed", [(False, True), (True, False), (True, True)])
@pytest.mark.accuracy
def test_rope_strides(accuracy_backend, q_transposed, k_transposed):
    device = accuracy_backend[2]
    angles = torch.randn(1, 7, 16, device=device)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()

    def input_tensor(heads, transposed):
        x = torch.randn(1, 7, heads, 16, device=device, dtype=torch.bfloat16).transpose(1, 2)
        return x if transposed else x.contiguous()

    def run(q, k, **selection):
        return F.apply_rope(q, k, cos, sin, unsqueeze_dim=1, **selection)

    assert_accuracy(
        run, (input_tensor(4, q_transposed), input_tensor(2, k_transposed)), implementation=accuracy_backend[0]
    )


@pytest.mark.api("functions.apply_rope")
@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("rotary_dim", [48, 88])
@pytest.mark.accuracy
def test_rope_layout(accuracy_backend, head_first, rotary_dim):
    device = accuracy_backend[2]
    angles = torch.randn(9, rotary_dim // 2, device=device)
    angles = torch.cat((angles, angles), -1)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    shapes = [(2, h, 9, 88) if head_first else (2, 9, h, 88) for h in (4, 2)]

    def run(q, k, **selection):
        return F.apply_rope(q, k, cos, sin, unsqueeze_dim=0 if head_first else 1, **selection)

    assert_accuracy(
        run,
        tuple(torch.randn(shape, device=device, dtype=torch.bfloat16) for shape in shapes),
        implementation=accuracy_backend[0],
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.accuracy
def test_rope_cache(accuracy_backend, dtype):
    device = accuracy_backend[2]
    # Unlike the historical inference cache, the two halves need not repeat.
    angles = torch.randn(124, 88, device=device)
    cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)

    def run(q, k, **selection):
        return F.apply_rope(q, k, cos, sin, **selection)

    assert_accuracy(
        run,
        tuple(torch.randn(1, 124, heads, 88, device=device, dtype=dtype) for heads in (2, 1)),
        implementation=accuracy_backend[0],
    )


@pytest.mark.api("functions.apply_rope")
@pytest.mark.bitwise
def test_rope_bitwise(accuracy_backend):
    implementation, _, device = accuracy_backend
    q, k = (torch.randn(1, h, 7, 16, device=device, dtype=torch.bfloat16, requires_grad=True) for h in (4, 2))
    angles = torch.randn(1, 7, 16, device=device)
    assert_repeatable(
        partial(F.apply_rope, unsqueeze_dim=1, implementation=implementation),
        (q, k, angles.cos().bfloat16(), angles.sin().bfloat16()),
    )


@pytest.mark.api("functions.apply_rope")
@pytest.mark.parametrize("batch,tokens,q_heads,k_heads,head_first,dim,rotary_dim", ROPE_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.accuracy
def test_rope(accuracy_backend, batch, tokens, q_heads, k_heads, head_first, dim, rotary_dim, dtype):
    implementation, _, device = accuracy_backend
    inputs, cos, sin = make_rope_case(device, dtype, batch, tokens, q_heads, k_heads, head_first, dim, rotary_dim)
    reference_inputs = clone_with_grad(inputs)
    actual = F.apply_rope(*inputs, cos, sin, unsqueeze_dim=0 if head_first else 1, implementation=implementation)
    expected = F.apply_rope(
        *reference_inputs, cos, sin, unsqueeze_dim=0 if head_first else 1, implementation="torch_reference"
    )
    upstream = tuple(torch.rand_like(output) for output in actual)
    actual_grads = torch.autograd.grad(actual, inputs, upstream)
    expected_grads = torch.autograd.grad(expected, reference_inputs, upstream)
    for index, (a, b) in enumerate(zip((*actual, *actual_grads), (*expected, *expected_grads))):
        assert_mojo_close(a, b, name=f"output/grad[{index}]")


@pytest.mark.api("functions.apply_rope")
@pytest.mark.parametrize("batch,tokens,q_heads,k_heads,head_first,dim,rotary_dim", ROPE_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.bitwise
def test_rope_shapes_bitwise(accuracy_backend, batch, tokens, q_heads, k_heads, head_first, dim, rotary_dim, dtype):
    implementation, _, device = accuracy_backend
    inputs, cos, sin = make_rope_case(device, dtype, batch, tokens, q_heads, k_heads, head_first, dim, rotary_dim)
    assert_repeatable(
        partial(F.apply_rope, cos=cos, sin=sin, unsqueeze_dim=0 if head_first else 1, implementation=implementation),
        inputs,
    )
