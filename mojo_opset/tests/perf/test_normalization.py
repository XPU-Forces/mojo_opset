import os

import pytest
import torch

from mojo_opset import MojoResidualAddLayerNorm
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented


def _set_norm_tle_branch(norm_tle):
    os.environ["MOJO_TTX_NORM_TLE"] = norm_tle


@pytest.mark.parametrize("norm_tle", ["1", "0"], ids=["tle", "fallback"])
@pytest.mark.parametrize(
    "x, residual, weight, bias",
    [
        (
            torch.randn(size=(128, 128), dtype=dtype),
            torch.randn(size=(128, 128), dtype=dtype),
            torch.randn(size=(128,), dtype=dtype),
            torch.randn(size=(128,), dtype=dtype),
        )
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ],
)
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_residual_add_layernorm(x, residual, weight, bias, norm_pos, eps, norm_tle):
    _set_norm_tle_branch(norm_tle)
    add_norm = MojoResidualAddLayerNorm(
        norm_size=weight.size(0),
        eps=eps,
        norm_pos=norm_pos,
    )
    add_norm.weight = torch.nn.Parameter(weight)
    add_norm.bias = torch.nn.Parameter(bias)

    perf(lambda: add_norm(x, residual))  # noqa: F821
