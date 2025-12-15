import pytest
import torch

from tests.utils import auto_switch_platform, bypass_not_implemented

from mojo_opset import MojoGelu, MojoSilu, MojoSiluMul, MojoSwiglu


@pytest.mark.parametrize(
    "x",
    [(torch.rand(128, 128))],
)
@auto_switch_platform()
@bypass_not_implemented
def test_gelu(x):
    gelu = MojoGelu()
    gelu.forward_diff(x)


@pytest.mark.parametrize(
    "x",
    [(torch.rand(128, 128))],
)
@auto_switch_platform()
@bypass_not_implemented
def test_silu(x):
    silu = MojoSilu()
    silu.forward_diff(x)


@pytest.mark.parametrize(
    "gate_out, up_out",
    [
        (
            torch.rand(size=(256, 128)),
            torch.rand(size=(256, 128)),
        )
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_silu_mul(gate_out, up_out):
    silu = MojoSiluMul()
    silu.forward_diff(gate_out, up_out)


@pytest.mark.parametrize(
    "x",
    [torch.rand(size=(256, 256))],
)
@auto_switch_platform()
@bypass_not_implemented
def test_swiglu(x):
    swiglu = MojoSwiglu()
    swiglu.forward_diff(x)
