import random

import pytest
import torch

from mojo_opset import MojoLightningIndex
from mojo_opset.utils.platform import get_platform
from tests.utils import auto_switch_platform, bypass_not_implemented


@pytest.mark.parametrize(
    "query, query_scale, key, key_scale",
    [
        (
            torch.randn(B, M, H, K, dtype=dtype),
            torch.randn(B, N, K, dtype=dtype),
            torch.randn(B, M, H, dtype=torch.float32),
            torch.randn(B, N, K, dtype=torch.float32),
        ) for (B, M, N, H, K, dtype) in [
            (
                24, 1024, 1024, 128, 128, dtype
            ) for dtype in [torch.bfloat16, torch.float16, torch.float32]
        ]
    ]
)
@auto_switch_platform()
@bypass_not_implemented
def test_lightning_index(query, query_scale, key, key_scale):
    indexer_func = MojoLightningIndex()
    indexer_func_ref = indexer_func._registry.get("torch")()

    indexer_func.forward_diff_with(
        indexer_func_ref, query, query_scale, key, key_scale
    )
