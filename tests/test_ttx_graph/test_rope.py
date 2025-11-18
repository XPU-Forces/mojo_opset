import pytest
import torch

from tests.utils import auto_switch_platform


@pytest.mark.parametrize(
    "q, k",
    [
        (
            torch.randn(1, 32, 4096, 32),
            torch.randn(1, 8, 4096, 32),
        )
    ],
)
@auto_switch_platform()
def test_rope(q, k):
    # Transpose q and k to mock the memory layout transformation used in the real inference framework.
    _, head_num, _, head_size = q.shape

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_size, 2).float().to(q.device) / head_size))
    t = torch.arange(head_num, device=q.device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]

    torch.library.opcheck(torch.ops.ttx.rope, (q, k, sin, cos))
    torch.library.opcheck(torch.ops.ttx.rope_bwd, (q, k, sin, cos))
