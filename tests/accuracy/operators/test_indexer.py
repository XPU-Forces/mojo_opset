import random

import pytest
import torch

from mojo_opset import MojoLightningIndex
from mojo_opset.utils.platform import get_platform
from tests.utils import auto_switch_platform, bypass_not_implemented

TEST_SHAPES = [
    (128, 256, 256, 64, 128),
    (24, 1024, 1024, 128, 128),
    (24, 1, 16384, 128, 128),
]
TEST_DTYPES = [torch.bfloat16, torch.float16, torch.float32]


@pytest.mark.parametrize(
    "query, query_scale, key, key_scale",
    [
        (
            torch.randn(B, M, H, K, dtype=dtype),
            torch.randn(B, M, H, dtype=torch.float32),
            torch.randn(B, N, K, dtype=dtype),
            torch.randn(B, N, K, dtype=torch.float32),
        )
        for (B, M, N, H, K) in TEST_SHAPES
        for dtype in TEST_DTYPES
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_lightning_index(query, query_scale, key, key_scale):
    indexer = MojoLightningIndex()
    indexer_ref = indexer._registry.get("torch")()

    indexer.forward_diff_with(indexer_ref, query, query_scale, key, key_scale)
