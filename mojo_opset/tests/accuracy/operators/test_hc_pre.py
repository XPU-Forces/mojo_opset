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


@pytest.mark.parametrize(
    "shape",
    [
        (16, 4, 4096),
        (16, 4, 7168),
        (4, 16, 4, 4096),
        (4, 16, 4, 7168),
        (32, 4, 4, 1024),
        (32, 4, 4, 2048),
    ],
)
@bypass_not_implemented
def test_hc_pre(shape):
    x, hc_fn, hc_scale, hc_base = _create_hc_pre_inputs(shape, seed=sum(shape))

    mojo_op = MojoHcPre()
    ref_op = MojoHcPre._registry.get("torch")()

    y, post, comb = mojo_op.forward(x, hc_fn, hc_scale, hc_base)
    ref_y, ref_post, ref_comb = ref_op.forward(x, hc_fn, hc_scale, hc_base)

    assert y.shape == ref_y.shape
    assert y.dtype == ref_y.dtype
    torch.testing.assert_close(y.float(), ref_y.float(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(post.float(), ref_post.float(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(comb.float(), ref_comb.float(), atol=5e-3, rtol=5e-3)
