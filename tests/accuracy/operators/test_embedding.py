import pytest
import torch

from tests.utils import bypass_not_implemented

from mojo_opset import MojoRelativeEmbedding


@pytest.mark.parametrize("num_buckets", [32, 64])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize(
    "lq, lk",
    [
        (64, 64),
        (128, 512),
        (33, 97),
    ],
)
@bypass_not_implemented
def test_relative_embedding(num_buckets, num_heads, bidirectional, lq, lk):
    emb = MojoRelativeEmbedding(num_buckets=num_buckets, num_heads=num_heads, bidirectional=bidirectional)
    emb_ref = MojoRelativeEmbedding._registry.get("torch")(
        num_buckets=num_buckets, num_heads=num_heads, bidirectional=bidirectional
    )

    with torch.no_grad():
        weight = torch.randn(num_buckets, num_heads, dtype=torch.float32)
        emb.embedding.weight.copy_(weight)
        emb_ref.embedding.weight.copy_(weight)

    emb.forward_diff_with(emb_ref, lq, lk, atol=1e-5, rtol=1e-6)
