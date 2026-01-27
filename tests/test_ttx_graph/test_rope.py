import pytest
import torch

from tests.utils import auto_switch_platform


@pytest.mark.parametrize(
    "q, k",
    [
        (
            torch.randn(1, 32, 4096, 128),
            torch.randn(1, 8, 4096, 128),
        )
    ],
)
@auto_switch_platform()
def test_rope(q, k):
    # Transpose q and k to mock the memory layout transformation used in the real inference framework.
    _, _, seqlen, head_dim = q.shape

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim))
    t = torch.arange(seqlen, device=q.device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]

    torch.library.opcheck(torch.ops.ttx.rope, (q, k, sin, cos))
    torch.library.opcheck(torch.ops.ttx.rope_bwd, (q, k, sin, cos))
