import pytest
import torch

from mojo_opset.tests.utils import auto_switch_platform


@pytest.mark.parametrize(
    "x",
    [(torch.rand(128, 128))],
)
@auto_switch_platform(set_perf=True)
def test_gelu(x):
    torch.library.opcheck(torch.ops.ttx.gelu, (x,))
    op = torch.ops.ttx.gelu
    perf(lambda: op(x))

@pytest.mark.parametrize(
    "dy, x",
    [
        (
            torch.rand(128, 128),
            torch.rand(128, 128),
        )
    ],
)
@auto_switch_platform(set_perf=True)
def test_gelu_bwd(dy, x):
    torch.library.opcheck(torch.ops.ttx.gelu_bwd, (dy, x))
    op = torch.ops.ttx.gelu_bwd
    perf(lambda: op(dy, x))

@pytest.mark.parametrize(
    "x",
    [(torch.rand(128, 128))],
)
@auto_switch_platform(set_perf=True)
def test_silu(x):
    torch.library.opcheck(torch.ops.ttx.silu, (x,))
    op = torch.ops.ttx.silu
    perf(lambda: op(x))


@pytest.mark.parametrize(
    "dy, x",
    [
        (
            torch.rand(128, 128),
            torch.rand(128, 128),
        )
    ],
)
@auto_switch_platform(set_perf=True)
def test_silu_bwd(dy, x):
    torch.library.opcheck(torch.ops.ttx.silu_bwd, (dy, x))
    op = torch.ops.ttx.silu_bwd
    perf(lambda: op(dy, x))

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
def test_swiglu(gate_out, up_out):
    torch.library.opcheck(torch.ops.ttx.swiglu, (gate_out, up_out))
    op = torch.ops.ttx.swiglu
    perf(lambda: op(gate_out, up_out))


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
@auto_switch_platform(set_perf=True)
def test_swiglu_bwd(dc, a, b):
    torch.library.opcheck(torch.ops.ttx.swiglu_bwd, (dc, a, b))
    op = torch.ops.ttx.swiglu_bwd
    perf(lambda: op(dc, a, b))