import pytest
import torch

from tests.utils import auto_switch_platform


@pytest.mark.parametrize(
    "x, gamma, dy, rstd",
    [
        (
            torch.randn(size=(256, 128), dtype=torch.bfloat16),
            torch.randn(size=(128,), dtype=torch.float32),
            torch.randn(size=(256, 128), dtype=torch.bfloat16),
            torch.randn(size=(256,), dtype=torch.float32),
        ),
    ],
)
@pytest.mark.parametrize("eps, offset", [(1e-5, 0.0)])
@pytest.mark.parametrize("casting_mode_int", [0])
@auto_switch_platform()
def test_rmsnorm(x, gamma, dy, eps, rstd, offset, casting_mode_int):
    torch.library.opcheck(torch.ops.ttx.rmsnorm_infer, (x, gamma, eps))
    torch.library.opcheck(torch.ops.ttx.rmsnorm_fwd, (x, gamma, eps, offset, casting_mode_int))
    torch.library.opcheck(torch.ops.ttx.rmsnorm_bwd, (dy, x, gamma, rstd, offset, casting_mode_int, x.dtype))
