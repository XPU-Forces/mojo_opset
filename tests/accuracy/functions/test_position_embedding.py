import pytest
import torch

from tests.utils import auto_switch_platform, bypass_not_implemented, MockFunctionCtx, assert_close
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
def test_rope_forward_backward_diff(q, k):

    q, k = q.transpose(1, 2), k.transpose(1, 2)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, q.size(-1), 2).float().to(q.device) / q.size(-1)))
    t = torch.arange(q.size(-2), device=q.device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]

    ctx = MockFunctionCtx()
    q_rot, k_rot = MojoRoPEFunction.forward(ctx, q, k, cos, sin)

    ctx_ref = MockFunctionCtx()
    q_rot_ref, k_rot_ref = MojoRoPEFunction._registry.get("ref").forward(ctx_ref, q, k, cos, sin)

    assert_close(q_rot, q_rot_ref)
    assert_close(k_rot, k_rot_ref)

    grad_q_out = torch.rand_like(q_rot)
    grad_k_out = torch.rand_like(k_rot)

    grads = MojoRoPEFunction.backward(ctx, grad_q_out, grad_k_out)
    grads_ref = MojoRoPEFunction._registry.get("ref").backward(ctx_ref, grad_q_out, grad_k_out)

    assert_close(grads, grads_ref)

