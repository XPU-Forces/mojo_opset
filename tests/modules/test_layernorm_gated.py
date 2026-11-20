"""Accuracy suite for the gated normalization modules.

The modules are thin parameter holders around ``functions.rms_norm_gated``; each
case compares the selected implementation against the ``torch`` reference with
the shared per-dtype tolerance.
"""

import pytest
import torch

from mojo_opset import modules

from .._checks import assert_close
from tests._fixtures import check_implementation

GATE_HEADS = 4
GATE_PAD = 32

MODULE_CASES = [
    ("LayerNormGated", "layer_norm_gated"),
    ("RMSNormGated", "rms_norm_gated"),
]


def _make_gate(device, dtype, M, N, z_mode, seed):
    torch.manual_seed(seed)
    if z_mode == "none":
        return None
    if z_mode == "2d":
        return torch.randn(M, N, device=device, dtype=dtype)
    assert M % GATE_HEADS == 0
    base = torch.randn(M // GATE_HEADS, GATE_HEADS, N + GATE_PAD, device=device, dtype=dtype)
    return base[:, :, :N]


@pytest.mark.accuracy
@pytest.mark.parametrize(
    "name,op",
    [pytest.param(name, op, marks=pytest.mark.api("modules." + name, ops=[op])) for name, op in MODULE_CASES],
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("z_mode", ("none", "2d", "3d"))
def test_gated_norm_module(accuracy_backend, name, op, dtype, z_mode):
    implementation, target, device = accuracy_backend
    implementation = check_implementation([op], implementation, target)[op]
    hidden, rows = 128, 128
    x = torch.randn(rows, hidden, device=device, dtype=dtype)
    z = _make_gate(device, dtype, rows, hidden, z_mode, seed=37)

    actual = getattr(modules, name)(hidden, device=device, dtype=dtype, implementation=implementation)
    expected = getattr(modules, name)(hidden, device=device, dtype=dtype, implementation="torch")
    with torch.no_grad():
        actual.weight.normal_()
        expected.weight.copy_(actual.weight)
        if actual.bias is not None:
            actual.bias.normal_()
            expected.bias.copy_(actual.bias)
        got = actual(x, z)
        want = expected(x, z)

    assert tuple(got.shape) == tuple(x.shape) and got.dtype == dtype
    assert_close(got, want, dtype, name=f"{name} output")
