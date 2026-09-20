import pytest
import torch

from mojo_opset import modules
from tests._checks import assert_accuracy
from tests._checks import assert_repeatable


@pytest.mark.api("modules.ApplyRoPE", ops=["apply_rope"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.accuracy
def test_rope(accuracy_backend, dtype, head_first):
    device = accuracy_backend[2]
    angles = torch.randn(1, 7, 16, device=device)
    cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
    shape = lambda h: (1, h, 7, 16) if head_first else (1, 7, h, 16)

    def run(q, k, **selection):
        module = modules.ApplyRoPE(unsqueeze_dim=1 if head_first else 2, **selection)
        return module(q, k, cos, sin)

    assert_accuracy(
        run,
        tuple(torch.randn(shape(h), device=device, dtype=dtype) for h in (4, 2)),
        implementation=accuracy_backend[0],
    )


@pytest.mark.api("modules.ApplyRoPE", ops=["apply_rope"])
@pytest.mark.bitwise
def test_rope_bitwise(accuracy_backend):
    implementation, _, device = accuracy_backend
    module = modules.ApplyRoPE(unsqueeze_dim=1, implementation=implementation)
    q, k = (torch.randn(1, h, 7, 16, device=device, dtype=torch.bfloat16, requires_grad=True) for h in (4, 2))
    angles = torch.randn(1, 7, 16, device=device)
    assert_repeatable(module, (q, k, angles.cos().bfloat16(), angles.sin().bfloat16()))
