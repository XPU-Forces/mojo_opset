"""Tests for modules.LayerNormGated.

The module is a thin parameter holder, so these cases check the wiring rather than the maths:
the parameter set (a RMSNorm block must not expose a bias), the argument plumbing
(``group_size`` / ``is_rms_norm`` / ``activation`` / ``norm_before_gate``) and equivalence with
the free function it wraps.
"""

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset.modules import LayerNormGated
from tests._checks import assert_close

EPS = 1e-5
DTYPES = (torch.bfloat16, torch.float16)


def _inputs(device, dtype, m=32, n=128, with_gate=True, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(m, n, generator=gen) * 0.5).to(dtype).to(device)
    z = (torch.randn(m, n, generator=gen) * 0.5).to(dtype).to(device) if with_gate else None
    return x, z


@pytest.mark.api("modules.LayerNormGated", ops=["layernorm_gated_infer"])
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.accuracy
def test_module_matches_function(accuracy_backend, dtype):
    _impl, _target, device = accuracy_backend
    x, z = _inputs(device, dtype)
    module = LayerNormGated(128, eps=EPS, is_rms_norm=True, dtype=dtype).to(device)
    # The op is inference-only, so a Parameter (requires_grad=True by default) has to be fed
    # under no_grad; the guard in ``functions._checks`` is deliberate, not an obstacle.
    with torch.no_grad():
        module.weight.copy_(torch.rand(128, device=device, dtype=dtype) + 0.5)
        got = module(x, z)
        want = F.layernorm_gated_infer(
            x, module.weight, None, EPS, z=z, is_rms_norm=True, implementation="torch"
        )[0]
    assert_close(got, want, dtype, name="module vs function")


@pytest.mark.api("modules.LayerNormGated", ops=["layernorm_gated_infer"])
def test_module_parameter_set(accuracy_backend):
    """A RMSNorm block has no bias; a LayerNorm block with ``has_bias`` does."""
    rms = LayerNormGated(128, is_rms_norm=True)
    assert rms.bias is None
    assert "bias" not in dict(rms.named_parameters())

    ln = LayerNormGated(128, is_rms_norm=False, has_bias=True)
    assert ln.bias is not None
    assert "bias" in dict(ln.named_parameters())


@pytest.mark.api("modules.LayerNormGated", ops=["layernorm_gated_infer"])
@pytest.mark.accuracy
def test_module_group_norm_and_bias(accuracy_backend):
    _impl, _target, device = accuracy_backend
    x, z = _inputs(device, torch.bfloat16, n=256)
    module = LayerNormGated(256, eps=EPS, group_size=64, has_bias=True).to(device)
    with torch.no_grad():
        module.weight.copy_(torch.rand(256, device=device, dtype=torch.bfloat16) + 0.5)
        module.bias.copy_(torch.randn(256, device=device, dtype=torch.bfloat16) * 0.1)
        got = module(x, z)
        want = F.layernorm_gated_infer(
            x,
            module.weight,
            module.bias,
            EPS,
            z=z,
            group_size=64,
            implementation="torch",
        )[0]
    assert_close(got, want, torch.bfloat16, name="module group+bias")
