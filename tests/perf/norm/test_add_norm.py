import pytest
import torch

from tests.utils import auto_switch_perf_fun
from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoResidualAddNorm


@pytest.mark.parametrize(
    "x, residual, gamma, beta",
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
@pytest.mark.parametrize("epsilon", [1e-5])
@pytest.mark.parametrize("norm_pos", ["pre", "post"])
@pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
@auto_switch_platform()
@auto_switch_perf_fun()
@bypass_not_implemented
def test_residual_add_norm(x, residual, gamma, beta, norm_type, norm_pos, epsilon):
    beta = beta if norm_type == "layernorm" else None

    add_norm = MojoResidualAddNorm(
        gamma=gamma,
        beta=beta,
        epsilon=epsilon,
        norm_pos=norm_pos,
        norm_type=norm_type,
    )

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 3e-2, 6e-3

    perf(add_norm(x, residual, atol=atol, rtol=rtol))  # noqa: F821
