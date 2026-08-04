import pytest
import torch

from mojo_opset import MojoHcPost
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


@pytest.mark.parametrize(
    "shape",
    [
        (16, 4096),
        (16, 7168),
        (32, 16, 4096),
        (32, 16, 7168),
        (64, 16, 4096),
        (64, 16, 7168),
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_hc_post(shape):
    hc = 4
    dtype = torch.bfloat16
    device = get_torch_device()
    torch.manual_seed(42)

    if len(shape) == 2:
        bs, d = shape
        x = torch.randn(bs, d, dtype=dtype, device=device)
        residual = torch.randn(bs, hc, d, dtype=dtype, device=device)
        post = torch.randn(bs, hc, dtype=dtype, device=device)
        comb = torch.randn(bs, hc, hc, dtype=dtype, device=device)
    else:
        b, s, d = shape
        x = torch.randn(b, s, d, dtype=dtype, device=device)
        residual = torch.randn(b, s, hc, d, dtype=dtype, device=device)
        post = torch.randn(b, s, hc, dtype=dtype, device=device)
        comb = torch.randn(b, s, hc, hc, dtype=dtype, device=device)

    op = MojoHcPost()

    perf(lambda: op.forward(x, residual, post, comb))  # noqa: F821
