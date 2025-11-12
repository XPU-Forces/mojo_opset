import pytest
import torch

from tests.utils import auto_switch_platform, bypass_not_implemented

from mojo_opset import MojoGelu, MojoSilu, MojoSwiGLU


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
def test_swiglu(gate_out, up_out):
    swiglu = MojoSwiGLU()
    swiglu.forward_diff(gate_out, up_out)
