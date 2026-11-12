import pytest
import torch

from mojo_opset import modules
from tests._checks import assert_close

# (name, hidden_size, is_rms_norm, group_size, norm_before_gate, has_bias, z_kind)
MODULE_CASES = [
    ("LayerNormGated", 128, False, None, True, True, "3d"),
    ("RMSNormGated", 128, True, None, True, False, "3d"),
    ("RMSNormGated", 256, True, 64, False, False, "2d"),
    ("LayerNormGated", 256, False, 64, False, True, "2d"),
]
MODULE_IDS = [
    f"{name}-{hidden}{'-g' + str(group) if group else ''}-{'before' if before else 'after'}-z{z_kind}"
    for name, hidden, _, group, before, _, z_kind in MODULE_CASES
]


def _inputs(hidden_size, z_kind, dtype, device):
    torch.manual_seed(20261031)
    x = torch.randn((132, hidden_size), device=device, dtype=dtype)
    z = None
    if z_kind == "2d":
        z = torch.randn_like(x)
    elif z_kind == "3d":
        heads = 4
        padded = torch.randn((132 // heads, heads, hidden_size + 32), device=device, dtype=dtype)
        z = padded[..., :hidden_size]
    return x, z


@pytest.mark.api("modules.LayerNormGated", ops=["layer_norm_gated"])
@pytest.mark.api("modules.RMSNormGated", ops=["layer_norm_gated"])
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", MODULE_CASES, ids=MODULE_IDS)
def test_gated_norm_module(accuracy_backend, dtype, case):
    implementation, _, device = accuracy_backend
    name, hidden_size, _, group_size, norm_before_gate, _, z_kind = case
    options = {
        "group_size": group_size,
        "norm_before_gate": norm_before_gate,
        "eps": 1e-6,
        "device": device,
        "dtype": dtype,
    }
    actual = getattr(modules, name)(hidden_size, implementation=implementation, **options)
    expected = getattr(modules, name)(hidden_size, implementation="torch", **options)
    with torch.no_grad():
        actual.weight.normal_()
        expected.load_state_dict(actual.state_dict())
    x, z = _inputs(hidden_size, z_kind, dtype, device)

    with torch.no_grad():
        output = actual(x, z)
        reference = expected(x, z)

    assert output.shape == reference.shape == x.shape
    assert_close(output, reference, dtype, name=name)
