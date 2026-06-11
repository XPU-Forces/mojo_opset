import pytest
import torch

from mojo_opset import MojoRotaryEmbedding
from mojo_opset import MojoApplyRoPE
from mojo_opset.experimental import MojoRelativeEmbedding
from mojo_opset.utils.platform import get_torch_device
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented


@pytest.mark.parametrize("bs", [32])
@pytest.mark.parametrize("seqlen", [8192])
@pytest.mark.parametrize(
    "q_heads, k_heads",
    [
        # (32, 32),
        (32, 8),
        # (16, 1),
    ],
)
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_pos_emb(bs, seqlen, q_heads, k_heads, head_dim, dtype):
    device = get_torch_device()
    x = torch.randn(bs, seqlen, q_heads * head_dim, device=device, dtype=dtype)
    rot_pos_emb = MojoRotaryEmbedding(
        rope_theta=10000.0, rope_dim=head_dim, init_max_length=seqlen,
    ).to(device)
    cos, sin = rot_pos_emb(x)

    # [B, S, N, D] -> [B, N, S, D]
    q = torch.randn(bs, seqlen, q_heads, head_dim, device=device, dtype=dtype).transpose(1, 2)
    k = torch.randn(bs, seqlen, k_heads, head_dim, device=device, dtype=dtype).transpose(1, 2)

    rope = MojoApplyRoPE()

    perf(lambda: rope(q, k, cos, sin, head_first=True))  # noqa: F821


# UC fixed-shape contract: num_buckets=32, num_heads=16, lq*lk=256.
# Off-grid combinations raise NotImplementedError on UC and are skipped
# by @bypass_not_implemented (TTX/torch_native still measured normally).
@pytest.mark.parametrize("num_buckets", [32])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("bidirectional", [True])
@pytest.mark.parametrize(
    "lq, lk",
    [
        (16, 16),
        (8, 32),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_relative_embedding(num_buckets, num_heads, bidirectional, lq, lk, dtype):
    device = get_torch_device()
    emb = MojoRelativeEmbedding(
        num_buckets=num_buckets, num_heads=num_heads, bidirectional=bidirectional,
    ).to(device, dtype=dtype)
    emb.embedding = emb.embedding.to(dtype=dtype)
    # Steady-state benchmark — caller pattern is "call once per layer per
    # fwd pass with stable weight & shape", which matches the UC wrapper's
    # output-cache hot path (best-practices §D.2).
    perf(lambda: emb(lq, lk))  # noqa: F821

