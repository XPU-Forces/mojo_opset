import pytest
import torch

from mojo_opset import modules
from tests._checks import assert_close

ROTATE_SHAPES = [
    (batch, tokens, heads, dim)
    for batch in (2, 8, 32)
    for tokens in (1, 2048)
    for heads in (1, 32)
    for dim in (128, 1024)
]


def _rotation(implementation):
    if implementation not in (None, "torch_reference"):
        pytest.skip("RotateActivation has only the explicit Torch reference provider")
    return modules.RotateActivation(implementation=implementation)


def _reference(x):
    dim = x.shape[-1]
    matrix = x.new_ones((1, 1))
    for _ in range(dim.bit_length() - 1):
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    return torch.nn.functional.linear(x, matrix) * dim**-0.5


@pytest.mark.api("modules.RotateActivation")
@pytest.mark.parametrize("shape", ROTATE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.accuracy
def test_rotate(accuracy_backend, shape, dtype):
    implementation, _, device = accuracy_backend
    function = _rotation(implementation)
    x = torch.randn(shape, device=device, dtype=dtype)
    actual = function(x)
    expected = _reference(x)
    atol = {torch.bfloat16: 1.6e-2, torch.float16: 1e-3, torch.float32: 1.3e-6}[dtype]
    assert_close(actual, expected, rtol=1e-5, atol=atol)
