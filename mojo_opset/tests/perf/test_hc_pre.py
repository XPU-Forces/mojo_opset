import numpy as np
import pytest
import torch

from mojo_opset import MojoHcPre
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented
from mojo_opset.utils.platform import get_torch_device


def _create_hc_pre_inputs(shape, hc_mix=24, hc_mult=4, seed=42):
    rng = np.random.default_rng(seed)
    device = get_torch_device()
    x = torch.tensor(rng.uniform(0, 2, shape), dtype=torch.bfloat16, device=device)
    hc_fn = torch.tensor(rng.uniform(0, 2, (hc_mix, hc_mult * shape[-1])), dtype=torch.float32, device=device)
    hc_scale = torch.tensor(rng.uniform(0, 2, (3,)), dtype=torch.float32, device=device)
    hc_base = torch.tensor(rng.uniform(0, 2, (hc_mix,)), dtype=torch.float32, device=device)
    return x, hc_fn, hc_scale, hc_base


@pytest.mark.parametrize(
    "shape",
    [
        (16, 4, 4096),
        (16, 4, 7168),
        (32, 16, 4, 4096),
        (32, 16, 4, 7168),
        (64, 16, 4, 4096),
        (64, 16, 4, 7168),
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_hc_pre(shape):
    x, hc_fn, hc_scale, hc_base = _create_hc_pre_inputs(shape, seed=42)

    op = MojoHcPre()

    perf(lambda: op.forward(x, hc_fn, hc_scale, hc_base))  # noqa: F821
