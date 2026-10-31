"""Coverage for the alias spellings of the gated-normalization entry points.

``layer_norm_gated`` is the canonical name; the alias spellings exist so that
callers written against the earlier native API keep working.  The module skips
itself when the aliases are absent, which is the state of the tree if the
canonical front end is taken from the community acceptance suite
(``XPU-Forces/mojo_opset#525``) instead of from this branch.
"""

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_close

ALIASES = ("layernorm_gated", "rms_norm_gated", "_layer_norm_fwd")


@pytest.fixture(autouse=True)
def require_aliases():
    missing = [name for name in ALIASES if not hasattr(F, name)]
    if missing:
        pytest.skip(f"alias entry points are not exported: {missing}")


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_alias_entry_points(accuracy_backend, dtype):
    implementation, _, device = accuracy_backend
    torch.manual_seed(20261031)
    x = torch.randn((128, 128), device=device, dtype=dtype)
    weight = torch.randn((128,), device=device, dtype=dtype)
    bias = torch.randn((128,), device=device, dtype=dtype)
    gate = torch.randn((128, 128), device=device, dtype=dtype)

    aliased = F.layernorm_gated(
        x,
        weight,
        bias,
        1e-6,
        z=gate,
        is_rms_norm=True,
        implementation=implementation,
    )
    canonical = F.layer_norm_gated(
        x,
        weight,
        bias,
        z=gate,
        eps=1e-6,
        is_rms_norm=True,
        implementation=implementation,
    )
    assert_close(aliased[0], canonical[0], dtype, name="layernorm_gated")

    actual = F.rms_norm_gated(
        x=x,
        weight=weight,
        bias=bias,
        z=gate,
        is_rms_norm=True,
        implementation=implementation,
    )
    expected = F.rms_norm_gated(
        x=x,
        weight=weight,
        bias=bias,
        z=gate,
        is_rms_norm=True,
        implementation="torch",
    )
    assert_close(actual, expected, dtype, name="rms_norm_gated")

    forward, mean, rstd = F._layer_norm_fwd(
        x,
        weight,
        bias,
        1e-6,
        z=gate,
        is_rms_norm=True,
    )
    assert_close(forward, expected, dtype, name="_layer_norm_fwd")
    assert mean is None
    assert rstd.shape == (x.shape[0],)
