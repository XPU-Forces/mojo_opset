import pytest
import torch
import torch.nn.functional as F

from mojo_opset.tests.utils import bypass_not_implemented

from mojo_opset import MojoEmbedding, MojoParallelEmbedding, MojoRelativeEmbedding

# ---------------------------------------------------------------------------
# MojoEmbedding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_embeddings", [128, 1024])
@pytest.mark.parametrize("embedding_dim", [64, 256])
@pytest.mark.parametrize("padding_idx", [None, 0])
@pytest.mark.parametrize(
    "input_shape",
    [(32,), (8, 16), (2, 4, 8)],
)
@bypass_not_implemented
def test_embedding(num_embeddings, embedding_dim, padding_idx, input_shape):
    ids = torch.randint(0, num_embeddings, input_shape)
    op = MojoEmbedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    op_ref = MojoEmbedding._registry.get("torch")(
        num_embeddings, embedding_dim, padding_idx=padding_idx
    )
    with torch.no_grad():
        op_ref.weight.copy_(op.weight)

    op.forward_diff_with(op_ref, ids, atol=0, rtol=0)

    ref = F.embedding(ids, op.weight, padding_idx=padding_idx)
    torch.testing.assert_close(op(ids), ref, atol=0, rtol=0)


@bypass_not_implemented
def test_embedding_padding_idx_zeroed():
    """Ensure the padding_idx row stays all-zero after reset_parameters."""
    pad = 5
    op = MojoEmbedding._registry.get("torch")(32, 16, padding_idx=pad)
    torch.testing.assert_close(
        op.weight[pad], torch.zeros(16), atol=0, rtol=0
    )


# ---------------------------------------------------------------------------
# MojoParallelEmbedding — single-rank (dist NOT initialised → identity)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_embeddings", [128, 1024])
@pytest.mark.parametrize("embedding_dim", [64, 256])
@pytest.mark.parametrize("padding_idx", [None, 0])
@bypass_not_implemented
def test_parallel_embedding_single_rank(num_embeddings, embedding_dim, padding_idx):
    """Without dist, MojoParallelEmbedding should equal MojoEmbedding."""
    ids = torch.randint(0, num_embeddings, (8, 16))

    op = MojoParallelEmbedding._registry.get("torch")(
        num_embeddings, embedding_dim, padding_idx=padding_idx
    )
    ref = F.embedding(ids, op.weight, padding_idx=padding_idx)
    torch.testing.assert_close(op(ids), ref, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# MojoRelativeEmbedding
# ---------------------------------------------------------------------------

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
