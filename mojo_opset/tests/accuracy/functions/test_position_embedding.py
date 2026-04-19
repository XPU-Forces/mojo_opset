import pytest
import torch

from mojo_opset.tests.utils import MockFunctionCtx
from mojo_opset.tests.utils import assert_close
from mojo_opset.tests.utils import bypass_not_implemented

from mojo_opset import MojoApplyRoPEFunction
from mojo_opset import MojoRotaryEmbedding
from mojo_opset.utils.platform import get_torch_device


@pytest.mark.parametrize("bs, seqlen", [
    (1, 124), 
    (6, 555),
    (2, 2048)
])
@pytest.mark.parametrize(
    "q_heads, k_heads, head_first",
    [
        (32, 8, True),
        (32, 4, False),
        (8, 2, True),
        (16, 1, False),
        (16, 8, True),
        (64, 8, False),
        (64, 4, True),
        (2, 1, False),
    ],
)
@pytest.mark.parametrize(
    "head_dim, rope_percentage",
    [
        (128, 1.0),
        (88, 1.0),
        (128, 0.375),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@bypass_not_implemented
def test_rope_function(bs, seqlen, q_heads, k_heads, head_dim, rope_percentage, head_first, dtype):
    """Test MojoApplyRoPEFunction with pre-extracted cos/sin, head_first, and nope_dim support."""
    device = get_torch_device()
    rope_dim = int(head_dim * rope_percentage)
    max_seq_len = 32768

    rot_pos_emb = MojoRotaryEmbedding(rope_theta=10000.0, rope_dim=rope_dim, init_max_length=max_seq_len).to(device)
    position_ids = torch.arange(seqlen, dtype=torch.int32, device=device)
    hidden_size = q_heads * head_dim
    x = torch.randn(seqlen, hidden_size, device=device, dtype=dtype)
    cos, sin = rot_pos_emb(x, position_ids=position_ids)

    if head_first:
        q = torch.randn(bs, q_heads, seqlen, head_dim, device=device, dtype=dtype)
        k = torch.randn(bs, k_heads, seqlen, head_dim, device=device, dtype=dtype)
    else:
        q = torch.randn(bs, seqlen, q_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(bs, seqlen, k_heads, head_dim, device=device, dtype=dtype)

    ctx = MockFunctionCtx()
    q_rot, k_rot = MojoApplyRoPEFunction.forward(ctx, q.clone(), k.clone(), cos, sin, head_first)

    ctx_ref = MockFunctionCtx()
    q_rot_ref, k_rot_ref = MojoApplyRoPEFunction._registry.get("torch").forward(
        ctx_ref, q.clone(), k.clone(), cos, sin, head_first,
    )

    assert_close(q_rot, q_rot_ref)
    assert_close(k_rot, k_rot_ref)

    grad_q_out = torch.rand_like(q_rot)
    grad_k_out = torch.rand_like(k_rot)

    grads = MojoApplyRoPEFunction.backward(ctx, grad_q_out.clone(), grad_k_out.clone())
    grads_ref = MojoApplyRoPEFunction._registry.get("torch").backward(ctx_ref, grad_q_out.clone(), grad_k_out.clone())

    assert_close(grads, grads_ref)
