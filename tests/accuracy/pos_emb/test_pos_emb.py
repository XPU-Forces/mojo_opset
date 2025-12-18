import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoRoPE
from mojo_opset.backends.reference.pos_emb import RefRoPE


@pytest.mark.parametrize(
    "q, k",
    [
        (
            torch.randn(1, 1024, 32, 32),  # [BSND]
            torch.randn(1, 1024, 8, 32),  # [BSND]
        )
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_pos_emb(q, k):
    rope = MojoRoPE(is_varlen=False)
    rope_ref = RefRoPE(is_varlen=False)

    # Transpose q and k to mock the memory layout transformation used in the real inference framework.

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, q.size(-1), 2).float().to(q.device) / q.size(-1)))
    t = torch.arange(q.size(-2), device=q.device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]

    rope_ref.forward_diff_with(rope, q, k, cos, sin)
