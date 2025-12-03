import pytest
import torch
import torch.nn.functional as F

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoGatedDeltaRule


@pytest.mark.parametrize(
    "q, k, v, beta, g, cu_seqlens",
    [
        (
            torch.randn(1, 512, 24, 128, dtype=torch.float16),
            torch.randn(1, 512, 4, 128, dtype=torch.float16),
            torch.randn(1, 512, 4, 256, dtype=torch.float16),
            torch.rand(1, 512, 4, dtype=torch.float32).sigmoid(),
            F.logsigmoid(torch.rand(1, 512, 4, dtype=torch.float16)),
            torch.tensor([0, 512 - 100, 512], dtype=torch.long),
        )
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_linear_attn(q, k, v, beta, g, cu_seqlens):
    op = MojoGatedDeltaRule()

    perf(lambda: op(q, k, v, g, beta, cu_seqlens, atol=1e-1, rtol=1e-1))  # noqa: F821
