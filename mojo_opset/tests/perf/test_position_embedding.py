import pytest
import torch

from mojo_opset import MojoRotaryEmbedding
from mojo_opset import MojoApplyRoPE
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
