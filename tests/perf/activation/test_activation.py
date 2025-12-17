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
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_gelu(x):
    gelu = MojoGelu()
    gelu_ref = RefGelu()
    perf(lambda: gelu_ref(x))  # noqa: F821
    perf(lambda: gelu(x))  # noqa: F821


@pytest.mark.parametrize(
    "x",
    [(torch.rand(128, 128))],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_silu(x):
    silu = MojoSilu()
    silu_ref = RefSilu()
    perf(lambda: silu_ref(x))  # noqa: F821
    perf(lambda: silu(x))  # noqa: F821


@pytest.mark.parametrize(
    "gate_out, up_out",
    [
        (
            torch.rand(size=(256, 128)),
            torch.rand(size=(256, 128)),
        )
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_swiglu(gate_out, up_out):
    swiglu = MojoSwiGLU()
    swiglu_ref = RefSwiGLU()
    perf(lambda: swiglu_ref(gate_out, up_out))  # noqa: F821
    perf(lambda: swiglu(gate_out, up_out))  # noqa: F821
