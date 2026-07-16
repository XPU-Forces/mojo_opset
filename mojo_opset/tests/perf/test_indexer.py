import os

import pytest
import torch

from mojo_opset.experimental import MojoLightningIndexer
from mojo_opset.tests.utils import auto_switch_platform, bypass_not_implemented

TEST_SHAPES = [
    (128, 256, 256, 64, 128),
    (24, 1024, 1024, 128, 128),
    (24, 1, 16384, 128, 128),
]
TEST_DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def _set_lightning_indexer_tle_branch(lightning_indexer_tle):
    os.environ["MOJO_TTX_LIGHTNING_INDEXER_TLE"] = lightning_indexer_tle


@pytest.mark.parametrize("lightning_indexer_tle", ["1", "0"], ids=["tle", "fallback"])
@pytest.mark.parametrize(
    "query, query_scale, key, key_scale",
    [
        (
            torch.randn(B, M, H, K, dtype=dtype),
            torch.randn(B, M, H, dtype=torch.float32),
            torch.randn(B, N, K, dtype=dtype),
            torch.randn(B, N, dtype=torch.float32),
        )
        for (B, M, N, H, K) in TEST_SHAPES
        for dtype in TEST_DTYPES
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_lightning_index(query, query_scale, key, key_scale, lightning_indexer_tle):
    _set_lightning_indexer_tle_branch(lightning_indexer_tle)
    indexer = MojoLightningIndexer()
    perf(lambda: indexer(query, query_scale, key, key_scale))
