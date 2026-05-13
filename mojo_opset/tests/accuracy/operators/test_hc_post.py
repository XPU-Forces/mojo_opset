import pytest
import torch

from mojo_opset import MojoHcPost
from mojo_opset.tests.utils import bypass_not_implemented


def _hc_post_ref(x, residual, post, comb):
    data_type = x.dtype
    x_fp = x.float()
    residual_fp = residual.float()
    post_fp = post.float()
    comb_fp = comb.float()
    out_shape = list(residual.shape)
    out = post_fp.unsqueeze(-1) * x_fp.unsqueeze(-2) + torch.sum(
        comb_fp.unsqueeze(-1) * residual_fp.unsqueeze(-2), dim=x.dim() - 1
    )
    out = out.to(data_type)
    out = out.reshape(out_shape)
    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16])
class TestHcPost:
    @bypass_not_implemented
    def test_hc_post_2d(self, dtype):
        bs = 16
        hc = 4
        d = 4096

        torch.manual_seed(0)
        x = torch.randn(bs, d, dtype=dtype)
        residual = torch.randn(bs, hc, d, dtype=dtype)
        post = torch.randn(bs, hc, dtype=dtype)
        comb = torch.randn(bs, hc, hc, dtype=dtype)

        mojo_op = MojoHcPost()
        ref_op = MojoHcPost._registry.get("torch")()

        out = mojo_op.forward(x, residual, post, comb)
        ref_out = ref_op.forward(x, residual, post, comb)

        torch.testing.assert_close(out.float(), ref_out.float(), atol=1e-2, rtol=1e-2)

    @bypass_not_implemented
    def test_hc_post_3d(self, dtype):
        b = 4
        s = 4
        hc = 4
        d = 4096

        torch.manual_seed(0)
        x = torch.randn(b, s, d, dtype=dtype)
        residual = torch.randn(b, s, hc, d, dtype=dtype)
        post = torch.randn(b, s, hc, dtype=dtype)
        comb = torch.randn(b, s, hc, hc, dtype=dtype)

        mojo_op = MojoHcPost()
        ref_op = MojoHcPost._registry.get("torch")()

        out = mojo_op.forward(x, residual, post, comb)
        ref_out = ref_op.forward(x, residual, post, comb)

        torch.testing.assert_close(out.float(), ref_out.float(), atol=1e-2, rtol=1e-2)

    @bypass_not_implemented
    def test_hc_post_2d_different_shapes(self, dtype):
        hc = 4
        for bs, d in [(16, 4096), (16, 7168), (2, 4096)]:
            torch.manual_seed(0)
            x = torch.randn(bs, d, dtype=dtype)
            residual = torch.randn(bs, hc, d, dtype=dtype)
            post = torch.randn(bs, hc, dtype=dtype)
            comb = torch.randn(bs, hc, hc, dtype=dtype)

            mojo_op = MojoHcPost()
            ref_op = MojoHcPost._registry.get("torch")()

            out = mojo_op.forward(x, residual, post, comb)
            ref_out = ref_op.forward(x, residual, post, comb)

            torch.testing.assert_close(out.float(), ref_out.float(), atol=1e-2, rtol=1e-2)

    @bypass_not_implemented
    def test_hc_post_3d_different_shapes(self, dtype):
        hc = 4
        for b, s, d in [(4, 4, 4096), (4, 4, 7168), (2, 8, 4096)]:
            torch.manual_seed(0)
            x = torch.randn(b, s, d, dtype=dtype)
            residual = torch.randn(b, s, hc, d, dtype=dtype)
            post = torch.randn(b, s, hc, dtype=dtype)
            comb = torch.randn(b, s, hc, hc, dtype=dtype)

            mojo_op = MojoHcPost()
            ref_op = MojoHcPost._registry.get("torch")()

            out = mojo_op.forward(x, residual, post, comb)
            ref_out = ref_op.forward(x, residual, post, comb)

            torch.testing.assert_close(out.float(), ref_out.float(), atol=1e-2, rtol=1e-2)

    @bypass_not_implemented
    def test_hc_post_register_call(self, dtype):
        bs = 8
        hc = 4
        d = 4096

        torch.manual_seed(0)
        x = torch.randn(bs, d, dtype=dtype)
        residual = torch.randn(bs, hc, d, dtype=dtype)
        post = torch.randn(bs, hc, dtype=dtype)
        comb = torch.randn(bs, hc, hc, dtype=dtype)

        ref_out = _hc_post_ref(x, residual, post, comb)

        mojo_op = MojoHcPost()
        mojo_out = mojo_op.forward(x, residual, post, comb)

        torch.testing.assert_close(mojo_out.float(), ref_out.float(), atol=1e-2, rtol=1e-2)