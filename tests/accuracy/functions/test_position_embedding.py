import pytest
import torch

from tests.utils import MockFunctionCtx
from tests.utils import assert_close
from tests.utils import bypass_not_implemented

from mojo_opset import MojoRoPEFunction
from mojo_opset.utils.platform import get_platform


@pytest.mark.parametrize("bs", [1, 6])
@pytest.mark.parametrize("seqlen", [124, 555, 2048])
@pytest.mark.parametrize(
    "q_heads, k_heads",
    [
        (32, 8),
        (32, 4),
        (8, 2),
        (16, 1),
        (16, 8),
        (64, 8),
        (64, 4),
        (2, 1),
    ],
)
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@bypass_not_implemented
def test_pos_emb(bs, seqlen, q_heads, k_heads, head_dim, dtype):
    device = get_platform()
    # [B, S, N, D]
    q = torch.randn(bs, seqlen, q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(bs, seqlen, k_heads, head_dim, device=device, dtype=dtype)

    # Mock real inference memory layout: [B, N, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim))
    t = torch.arange(seqlen, device=q.device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    # [1, 1, S, D]
    cos = emb.cos()[None, :, :]
    sin = emb.sin()[None, :, :]

    ctx = MockFunctionCtx()
    q_rot, k_rot = MojoRoPEFunction.forward(ctx, q.clone(), k.clone(), cos, sin)

    ctx_ref = MockFunctionCtx()
    q_rot_ref, k_rot_ref = MojoRoPEFunction._registry.get("torch").forward(ctx_ref, q.clone(), k.clone(), cos, sin)

    assert_close(q_rot, q_rot_ref)
    assert_close(k_rot, k_rot_ref)

    grad_q_out = torch.rand_like(q_rot)
    grad_k_out = torch.rand_like(k_rot)

    grads = MojoRoPEFunction.backward(ctx, grad_q_out.clone(), grad_k_out.clone())
    grads_ref = MojoRoPEFunction._registry.get("torch").backward(ctx_ref, grad_q_out.clone(), grad_k_out.clone())

    assert_close(grads, grads_ref)
