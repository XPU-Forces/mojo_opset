import pytest
import torch

from tests.utils import auto_switch_platform


@pytest.mark.parametrize(
    "x",
    [(torch.rand(128, 128))],
)
@auto_switch_platform()
def test_gelu(x):
    torch.library.opcheck(torch.ops.ttx.gelu, (x,))


@pytest.mark.parametrize(
    "dy, x",
    [
        (
            torch.rand(128, 128),
            torch.rand(128, 128),
        )
    ],
)
@auto_switch_platform()
def test_gelu_bwd(dy, x):
    torch.library.opcheck(torch.ops.ttx.gelu_bwd, (dy, x))


@pytest.mark.parametrize(
    "x",
    [(torch.rand(128, 128))],
)
@auto_switch_platform()
def test_silu(x):
    torch.library.opcheck(torch.ops.ttx.silu, (x,))


@pytest.mark.parametrize(
    "dy, x",
    [
        (
            torch.rand(128, 128),
            torch.rand(128, 128),
        )
    ],
)
@auto_switch_platform()
def test_silu_bwd(dy, x):
    torch.library.opcheck(torch.ops.ttx.silu_bwd, (dy, x))


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
def test_swiglu(gate_out, up_out):
    torch.library.opcheck(torch.ops.ttx.swiglu, (gate_out, up_out))


@pytest.mark.parametrize(
    "dc, a, b",
    [
        (
            torch.rand(size=(256, 128)),
            torch.rand(size=(256, 128)),
            torch.rand(size=(256, 128)),
        )
    ],
)
@auto_switch_platform()
def test_swiglu_bwd(dc, a, b):
    torch.library.opcheck(torch.ops.ttx.swiglu_bwd, (dc, a, b))


if __name__ == "__main__":
    print(torch.library.opcheck(torch.ops.ttx.gelu, (torch.rand(128, 128).npu(),)))
    print(torch.library.opcheck(torch.ops.ttx.gelu_bwd, (torch.rand(128, 128).npu(), torch.rand(128, 128).npu())))
