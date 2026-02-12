import pytest
import torch

from tests.utils import bypass_not_implemented

from mojo_opset import MojoGelu
from mojo_opset import MojoSilu
from mojo_opset import MojoSwiGLU


@pytest.mark.parametrize(
    "shape",
    [
        ([128, 128]),
        ([999, 9999]),
        ([1024, 10240]),
    ],
)
@bypass_not_implemented
def test_gelu(shape):
    x = torch.rand(*shape, dtype=torch.bfloat16)
    gelu = MojoGelu()
    gelu_ref = MojoGelu._registry.get("torch")()
    gelu.forward_diff_with(gelu_ref, x)


@pytest.mark.parametrize(
    "shape",
    [
        ([128, 128]),
        ([999, 9999]),
        ([1024, 10240]),
    ],
)
@bypass_not_implemented
def test_silu(shape):
    x = torch.rand(*shape, dtype=torch.bfloat16)
    silu = MojoSilu()
    silu_ref = MojoSilu._registry.get("torch")()
    silu.forward_diff_with(silu_ref, x)


@pytest.mark.parametrize(
    "shape",
    [
        ([256, 128]),
        ([1024, 10240]),
        ([999, 9999]),
    ],
)
@bypass_not_implemented
def test_swiglu(shape):
    gate_out = torch.rand(*shape, dtype=torch.bfloat16)
    up_out = torch.rand(*shape, dtype=torch.bfloat16)
    swiglu = MojoSwiGLU()
    swiglu_ref = MojoSwiGLU._registry.get("torch")()
    swiglu.forward_diff_with(swiglu_ref, gate_out, up_out)

