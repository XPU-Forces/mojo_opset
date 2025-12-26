import pytest
import torch

from tests.utils import auto_switch_platform, bypass_not_implemented
from mojo_opset import MojoRoPEFunction


@pytest.mark.parametrize(
    "q, k",
    [
        (
            torch.randn(1, 4096, 32, 32, requires_grad=True),
            torch.randn(1, 4096, 8, 32, requires_grad=True),
        )
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_rope_forward_backward_diff(monkeypatch, q, k):
    monkeypatch.setenv("MOJOROPEFUNCTION_FWD_MODE", "DIFF")
    monkeypatch.setenv("MOJOROPEFUNCTION_BACKWARD_MODE", "DIFF")

    q, k = q.transpose(1, 2), k.transpose(1, 2)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, q.size(-1), 2).float().to(q.device) / q.size(-1)))
    t = torch.arange(q.size(-2), device=q.device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]

    q_rot, k_rot = MojoRoPEFunction.apply(q, k, cos, sin)
    grad_q_out = torch.rand_like(q_rot)
    grad_k_out = torch.rand_like(k_rot)

    torch.autograd.backward([q_rot, k_rot], [grad_q_out, grad_k_out])
