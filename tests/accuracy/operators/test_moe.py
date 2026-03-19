import pytest
import torch

from tests.utils import bypass_not_implemented

from mojo_opset import MojoMoeTopkGatingDispatchDynamicQuant


torch.manual_seed(43)

dtypes = [torch.float16, torch.bfloat16, torch.float32]


@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 6),
        (2, 4, 16),
        (3, 3)
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("k", [1])
@bypass_not_implemented
def test_moe_topk_gating(shape, dtype, k):
    x = torch.randn(size=shape, dtype=dtype)

    topk_gating = MojoMoeTopkGatingDispatchDynamicQuant().to(x.device)

    topk_gating_ref = MojoMoeTopkGatingDispatchDynamicQuant._registry.get("torch")().to(x.device)

    if x.dtype == torch.float32:
        atol, rtol = 1e-5, 1e-6
    else:
        atol, rtol = 3e-2, 6e-3
    topk_gating.forward_diff_with(topk_gating_ref, x, None, k, atol=atol, rtol=rtol)
