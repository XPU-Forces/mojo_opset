import pytest
import torch

from mojo_opset import MojoRoPE
from mojo_opset.utils.platform import get_platform
from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented


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
    device = get_platform()
    # [B, S, N, D]
    q = torch.randn(bs, seqlen, q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(bs, seqlen, k_heads, head_dim, device=device, dtype=dtype)

    rope = MojoRoPE()

    # Mock real inference memory layout: [B, N, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim))
    t = torch.arange(seqlen, device=q.device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    # [1, 1, S, D]
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]

    perf(lambda: rope(q, k, cos, sin))  # noqa: F821
