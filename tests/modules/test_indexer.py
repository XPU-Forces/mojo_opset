import pytest
import torch

from mojo_opset import modules
from tests._checks import assert_close

INDEXER_SHAPES = [
    (8, 1024, 1024, 64, 64),
    (128, 256, 256, 64, 128),
    (24, 1024, 1024, 128, 128),
    (24, 1, 16384, 128, 128),
]


def assert_lightning_scores(actual, expected):
    # Original LightningIndexer uses forward_diff_with defaults for every dtype:
    # FP32 comparison, rtol=atol=1e-2, and all elements must pass (ptol=1).
    assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2, name="index_score")


def make_indexer_case(shape, dtype, device):
    batch, q_len, k_len, heads, dim = shape
    return (
        torch.randn(batch, q_len, heads, dim, device=device, dtype=dtype),
        torch.randn(batch, q_len, heads, device=device, dtype=torch.float32),
        torch.randn(batch, k_len, dim, device=device, dtype=dtype),
        torch.randn(batch, k_len, device=device, dtype=torch.float32),
    )


@pytest.mark.api("modules.LightningIndexer", ops=["lightning_indexer"])
@pytest.mark.parametrize("shape", INDEXER_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.accuracy
def test_indexer(accuracy_backend, shape, dtype):
    implementation, _, device = accuracy_backend
    inputs = make_indexer_case(shape, dtype, device)
    actual = modules.LightningIndexer(implementation=implementation)(*inputs)
    expected = modules.LightningIndexer(implementation="torch_reference")(*inputs)
    assert_lightning_scores(actual, expected)
