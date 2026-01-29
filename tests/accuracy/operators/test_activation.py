import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoGelu
from mojo_opset import MojoSilu
from mojo_opset import MojoSwiGLU


@pytest.mark.parametrize(
    "x",
    [
        (torch.rand(128, 128, dtype=torch.bfloat16)),
        (torch.rand(999, 9999, dtype=torch.bfloat16)),
        (torch.rand(1024, 10240, dtype=torch.bfloat16)),
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_gelu(x):
    gelu = MojoGelu()
    gelu_ref = MojoGelu._registry.get("torch")()
    gelu.forward_diff_with(gelu_ref, x)


@pytest.mark.parametrize(
    "x",
    [
        (torch.rand(128, 128, dtype=torch.bfloat16)),
        (torch.rand(999, 9999, dtype=torch.bfloat16)),
        (torch.rand(1024, 10240, dtype=torch.bfloat16)),
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_silu(x):
    silu = MojoSilu()
    silu_ref = MojoSilu._registry.get("torch")()
    silu.forward_diff_with(silu_ref, x)


@pytest.mark.parametrize(
    "gate_out, up_out",
    [
        (
            torch.rand(size=(256, 128), dtype=torch.bfloat16),
            torch.rand(size=(256, 128), dtype=torch.bfloat16),
        ),
        (
            torch.rand(size=(1024, 10240), dtype=torch.bfloat16),
            torch.rand(size=(1024, 10240), dtype=torch.bfloat16),
        ),
        (
            torch.rand(size=(999, 9999), dtype=torch.bfloat16),
            torch.rand(size=(999, 9999), dtype=torch.bfloat16),
        ),
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_swiglu(gate_out, up_out):
    swiglu = MojoSwiGLU()
    swiglu_ref = MojoSwiGLU._registry.get("torch")()
    swiglu.forward_diff_with(swiglu_ref, gate_out, up_out)
