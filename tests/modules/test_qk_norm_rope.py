"""Tests for modules.QkNormRope."""

import pytest
import torch

from mojo_opset.modules.qk_norm_rope import QkNormRope

HEAD_DIM = 128
ROTARY_DIM = 96
EPS = 1e-6


def make_inputs(tokens, heads, device="cpu", seed=0):
    gen = torch.Generator().manual_seed(seed)
    half = ROTARY_DIM // 2
    theta = torch.rand(tokens, half, generator=gen, dtype=torch.float32) * 6.283185307179586
    q = (torch.randn(tokens, heads, HEAD_DIM, generator=gen) * 0.5).to(torch.bfloat16)
    k = (torch.randn(tokens, heads, HEAD_DIM, generator=gen) * 0.5).to(torch.bfloat16)
    return q.to(device), k.to(device), theta.to(device)


@pytest.mark.reference
def test_qk_norm_rope_module_rope_table_pack():
    """rope_table must be [cos | sin] with theta of width rotary_dim/2."""
    mod = QkNormRope(HEAD_DIM, ROTARY_DIM, EPS)
    theta = torch.rand(4, ROTARY_DIM // 2)
    table = mod.rope_table(theta)
    assert table.shape == (4, ROTARY_DIM)
    assert torch.allclose(table[:, : ROTARY_DIM // 2], theta.cos())
    assert torch.allclose(table[:, ROTARY_DIM // 2 :], theta.sin())


@pytest.mark.reference
def test_qk_norm_rope_module_rejects_bad_theta():
    mod = QkNormRope(HEAD_DIM, ROTARY_DIM, EPS)
    with pytest.raises(ValueError):
        mod.rope_table(torch.rand(4, ROTARY_DIM))


@pytest.mark.reference
def test_qk_norm_rope_module_weight_shape_and_defaults():
    mod = QkNormRope(HEAD_DIM, ROTARY_DIM, EPS)
    assert mod.q_weight.shape == (HEAD_DIM,) and mod.k_weight.shape == (HEAD_DIM,)
    assert mod.q_weight.dtype == torch.bfloat16
    # Ones is the neutral choice: it makes the module a pure norm+rotate until loaded.
    assert torch.equal(mod.q_weight, torch.ones(HEAD_DIM, dtype=torch.bfloat16))
    assert "head_dim=128" in repr(mod)


@pytest.mark.reference
def test_qk_norm_rope_module_rejects_bad_geometry():
    """rotary_dim must be even and must not exceed head_dim, at construction time.

    An odd rotary_dim would make ``rope_table`` pack one angle too few, i.e. the module would
    advertise a width it cannot build.
    """
    with pytest.raises(ValueError):
        QkNormRope(HEAD_DIM, ROTARY_DIM - 1, EPS)
    with pytest.raises(ValueError):
        QkNormRope(HEAD_DIM, HEAD_DIM + 2, EPS)
    with pytest.raises(ValueError):
        QkNormRope(HEAD_DIM, 0, EPS)
    with pytest.raises(ValueError):
        QkNormRope(0, ROTARY_DIM, EPS)


@pytest.mark.reference
def test_qk_norm_rope_module_matches_function():
    """The module must be a thin wrapper: same bits as the function with the same weights."""
    from mojo_opset import functions as F

    mod = QkNormRope(HEAD_DIM, ROTARY_DIM, EPS)
    with torch.no_grad():
        mod.q_weight.copy_((torch.rand(HEAD_DIM) + 0.5).to(torch.bfloat16))
        mod.k_weight.copy_((torch.rand(HEAD_DIM) + 0.5).to(torch.bfloat16))
    q, k, theta = make_inputs(8, 4)
    table = mod.rope_table(theta)

    with torch.no_grad():
        got = mod(q, k, table, implementation="torch")
        ref = F.qk_norm_rope_infer(q, k, mod.q_weight, mod.k_weight, table, EPS, implementation="torch")
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


@pytest.mark.reference
def test_qk_norm_rope_module_requires_no_grad():
    """The Parameters make the shared inference guard reject a plain forward call.

    Same convention as ``RMSNorm``: the module is forward-only, so a caller outside
    ``torch.no_grad()`` gets a RuntimeError instead of a silently detached result.
    """
    mod = QkNormRope(HEAD_DIM, ROTARY_DIM, EPS)
    q, k, theta = make_inputs(4, 4)
    table = mod.rope_table(theta)

    with pytest.raises(RuntimeError):
        mod(q, k, table, implementation="torch")
    with torch.no_grad():
        out = mod(q, k, table, implementation="torch")
    assert out[0].shape == q.shape

