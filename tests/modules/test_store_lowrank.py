import pytest
import torch

from mojo_opset import modules
from tests._checks import assert_close

STORE_LENGTHS = [1024, 2048, 4096, 8192, 13312]


STORE_SHAPES = [(256, 1, 512, 128), (256, 8, 512, 128)]


def make_store_case(shape, tokens, device):
    _, heads, block_size, dim = shape
    slots = torch.randperm(tokens, device=device)
    return (
        torch.zeros(shape, device=device, dtype=torch.bfloat16),
        torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16),
        (slots // block_size).to(torch.int32),
        (slots % block_size).to(torch.int32),
    )


@pytest.mark.api("modules.StoreLowrank")
@pytest.mark.parametrize("shape", STORE_SHAPES)
@pytest.mark.parametrize("tokens", STORE_LENGTHS)
@pytest.mark.accuracy
def test_store(accuracy_backend, shape, tokens):
    implementation, _, device = accuracy_backend
    cache, *inputs = make_store_case(shape, tokens, device)
    actual = modules.StoreLowrank(implementation=implementation)(cache, *inputs, tokens)
    expected_cache = torch.zeros_like(cache)
    expected = modules.StoreLowrank(implementation="torch_reference")(expected_cache, *inputs, tokens)
    assert actual is cache
    assert_close(actual, expected, rtol=0, atol=0)
