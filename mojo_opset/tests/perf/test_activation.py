import pytest
import torch

from mojo_opset import MojoGelu
from mojo_opset import MojoSilu
from mojo_opset import MojoSwiGLU
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented


@pytest.mark.parametrize(
    "x",
    [
        (torch.rand(128, 128)),
        (torch.rand(1024, 10240)),
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_gelu(x):
    gelu = MojoGelu._registry.get("ttx")()
    gelu_ref = MojoGelu._registry.get("torch_npu")()
    perf(lambda: gelu(x))
    perf(lambda: gelu_ref(x))


@pytest.mark.parametrize(
    "x",
    [(torch.rand(128, 128))],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_silu(x):
    silu = MojoSilu._registry.get("ttx")()
    silu_ref = MojoSilu._registry.get("torch")()
    perf(lambda: silu(x))
    # perf(lambda: silu_ref(x))


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
    swiglu = MojoSwiGLU._registry.get("ttx")()
    swiglu_ref = MojoSwiGLU._registry.get("torch")()
    perf(lambda: swiglu(gate_out, up_out))
    perf(lambda: swiglu_ref(gate_out, up_out))
