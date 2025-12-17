import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoGelu
from mojo_opset import MojoSilu
from mojo_opset import MojoSwiGLU
from mojo_opset.backends.reference.activation import RefGelu
from mojo_opset.backends.reference.activation import RefSilu
from mojo_opset.backends.reference.activation import RefSwiGLU


@pytest.mark.parametrize(
    "x",
    [(torch.rand(128, 128))],
)
@auto_switch_platform()
@bypass_not_implemented
def test_gelu(x):
    gelu_ref = RefGelu()
    gelu = MojoGelu()
    gelu_ref.forward_diff_with(gelu, x)


@pytest.mark.parametrize(
    "x",
    [(torch.rand(128, 128))],
)
@auto_switch_platform()
@bypass_not_implemented
def test_silu(x):
    silu_ref = RefSilu()
    silu = MojoSilu()
    silu_ref.forward_diff_with(silu, x)


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
    swiglu_ref = RefSwiGLU()
    swiglu = MojoSwiGLU()
    swiglu_ref.forward_diff_with(swiglu, gate_out, up_out)
