import pytest
import torch

from mojo_opset import Target
from mojo_opset import modules
from mojo_opset.functions._dispatch import resolve_implementation
from tests._checks import assert_repeatable
from tests._checks import check_forward_backward

ACTIVATION_CASES = [
    pytest.param(name, op, options, marks=pytest.mark.api("modules." + name, ops=[op]))
    for name, op, options in [
        ("SiLU", "silu", {}),
        ("GELU", "gelu", {"approximate": "tanh"}),
        ("GELU", "gelu", {"approximate": "none"}),
        ("SwiGLU", "swiglu", {"swiglu_limit": 0.0}),
        ("SwiGLU", "swiglu", {"swiglu_limit": 0.5}),
    ]
]


@pytest.mark.parametrize("name,op,options", ACTIVATION_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.accuracy
def test_activation_accuracy(accuracy_backend, name, op, options, dtype):
    implementation, target, device = accuracy_backend
    selected = implementation or resolve_implementation(op, Target.parse(target))
    if options.get("approximate") == "none" and selected == "triton":
        pytest.skip("Triton GELU supports approximate='tanh' only")

    def run(*inputs, **selection):
        return getattr(modules, name)(**options, **selection)(*inputs)

    inputs = tuple(torch.randn(7, 513, device=device, dtype=dtype) for _ in range(2 if op == "swiglu" else 1))
    check_forward_backward(run, inputs, implementation=implementation)


@pytest.mark.parametrize("name,op,options", ACTIVATION_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.bitwise
def test_activation_bitwise(accuracy_backend, name, op, options, dtype):
    implementation, target, device = accuracy_backend
    selected = implementation or resolve_implementation(op, Target.parse(target))
    if options.get("approximate") == "none" and selected == "triton":
        pytest.skip("Triton GELU supports approximate='tanh' only")
    module = getattr(modules, name)(implementation=implementation, **options)
    inputs = tuple(
        torch.randn(7, 513, device=device, dtype=dtype, requires_grad=True) for _ in range(2 if op == "swiglu" else 1)
    )
    assert_repeatable(module, inputs)
