import pytest
import torch

from mojo_opset import MojoHcPost
from mojo_opset.tests.utils import bypass_not_implemented


@pytest.mark.parametrize(
    "shape",
    [
        (16, 4096),
        (16, 7168),
        (4, 4, 4096),
        (4, 4, 7168),
        (8, 16, 4096),
        (8, 16, 7168),
    ],
)
@bypass_not_implemented
def test_hc_post(shape):
    hc = 4
    torch.manual_seed(0)

    if len(shape) == 2:
        bs, d = shape
        x = torch.randn(bs, d, dtype=torch.bfloat16)
        residual = torch.randn(bs, hc, d, dtype=torch.bfloat16)
        post = torch.randn(bs, hc, dtype=torch.bfloat16)
        comb = torch.randn(bs, hc, hc, dtype=torch.bfloat16)
    else:
        b, s, d = shape
        x = torch.randn(b, s, d, dtype=torch.bfloat16)
        residual = torch.randn(b, s, hc, d, dtype=torch.bfloat16)
        post = torch.randn(b, s, hc, dtype=torch.bfloat16)
        comb = torch.randn(b, s, hc, hc, dtype=torch.bfloat16)

    mojo_op = MojoHcPost()
    ref_op = MojoHcPost._registry.get("torch")()

    out = mojo_op.forward(x, residual, post, comb)
    ref_out = ref_op.forward(x, residual, post, comb)

    torch.testing.assert_close(out.float(), ref_out.float(), atol=5e-3, rtol=5e-3)
