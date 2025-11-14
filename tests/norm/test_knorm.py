import pytest
import torch

from tests.utils import auto_switch_platform, bypass_not_implemented

from mojo_opset import MojoNorm


@pytest.mark.parametrize(
    "x, gamma",
    [
        (
            torch.randn(size=(2, 256, 48, 128), dtype=dtype),
            torch.randn(size=(128,), dtype=torch.float32),
        )
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
    ],
)
@pytest.mark.parametrize("epsilon", [1e-5])
@auto_switch_platform()
@bypass_not_implemented
def test_k_rmsnorm(x, gamma, epsilon):
    krmsnorm = MojoNorm(
        epsilon=epsilon,
        norm_type="rmsnorm",
        gamma=gamma,
        is_varlen = False,
        only_k_norm = True,
        q_head_num = 32,
        kv_head_num = 8,
    ).to(x.device)

    with torch.no_grad():
        krmsnorm.gamma.copy_(gamma.to(torch.float32))

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 3e-2, 6e-3
    krmsnorm.forward_diff(x, atol=atol, rtol=rtol)