import numpy as np
import pytest
import torch

from mojo_opset import MojoHcPre
from mojo_opset.tests.utils import bypass_not_implemented


def _create_hc_pre_inputs(shape, hc_mix=24, hc_mult=4, seed=42):
    rng = np.random.default_rng(seed)
    hc_scale = torch.tensor(rng.uniform(0, 2, (3,)), dtype=torch.float32)
    hc_base = torch.tensor(rng.uniform(0, 2, (hc_mix,)), dtype=torch.float32)
    hc_fn = torch.tensor(rng.uniform(0, 2, (hc_mix, hc_mult * shape[-1])), dtype=torch.float32)
    x = torch.tensor(rng.uniform(0, 2, shape), dtype=torch.bfloat16)
    return x, hc_fn, hc_scale, hc_base


def _assert_hc_pre_close(out, ref_out):
    y, post, comb = out
    ref_y, ref_post, ref_comb = ref_out

    assert y.shape == ref_y.shape
    assert post.shape == ref_post.shape
    assert comb.shape == ref_comb.shape
    assert y.dtype == ref_y.dtype
    assert post.dtype == ref_post.dtype
    assert comb.dtype == ref_comb.dtype

    torch.testing.assert_close(y.float(), ref_y.float(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(post.float(), ref_post.float(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(comb.float(), ref_comb.float(), atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("d", [4096, 7168])
class TestHcPre:
    @bypass_not_implemented
    def test_hc_pre_2d(self, d):
        hc_mult = 4
        x, hc_fn, hc_scale, hc_base = _create_hc_pre_inputs((16, hc_mult, d), seed=d)

        mojo_op = MojoHcPre()
        ref_op = MojoHcPre._registry.get("torch")()

        out = mojo_op.forward(x, hc_fn, hc_scale, hc_base)
        ref_out = ref_op.forward(x, hc_fn, hc_scale, hc_base)

        _assert_hc_pre_close(out, ref_out)

    @bypass_not_implemented
    def test_hc_pre_3d(self, d):
        b = 1
        s = 16
        hc_mult = 4
        x, hc_fn, hc_scale, hc_base = _create_hc_pre_inputs((b, s, hc_mult, d), seed=d + s)

        mojo_op = MojoHcPre()
        ref_op = MojoHcPre._registry.get("torch")()

        out = mojo_op.forward(x, hc_fn, hc_scale, hc_base)
        ref_out = ref_op.forward(x, hc_fn, hc_scale, hc_base)

        _assert_hc_pre_close(out, ref_out)

    @bypass_not_implemented
    def test_hc_pre_register_call(self, d):
        hc_mult = 4
        x, hc_fn, hc_scale, hc_base = _create_hc_pre_inputs((8, hc_mult, d), seed=d + 8)

        mojo_op = MojoHcPre()
        ref_op = MojoHcPre._registry.get("torch")()
        print(MojoHcPre._registry._registry)
        print(type(MojoHcPre()), type(MojoHcPre._registry.get("torch")()))

        out = mojo_op.forward(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        ref_out = ref_op.forward(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )

        _assert_hc_pre_close(out, ref_out)
