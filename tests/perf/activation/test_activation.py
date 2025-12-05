import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoGelu
from mojo_opset import MojoSilu
from mojo_opset import MojoSwiGLU


@pytest.mark.parametrize(
    "x",
    [(torch.rand(128, 128))],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_gelu(x):
    gelu = MojoGelu()
    perf(lambda: gelu.forward_ref(x))  # noqa: F821
    perf(lambda: gelu(x))  # noqa: F821


@pytest.mark.parametrize(
    "x",
    [(torch.rand(128, 128))],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_silu(x):
    silu = MojoSilu()
    perf(lambda: silu.forward_ref(x))  # noqa: F821
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
    perf(lambda: swiglu.forward_ref(gate_out, up_out))  # noqa: F821
    perf(lambda: swiglu(gate_out, up_out))  # noqa: F821
