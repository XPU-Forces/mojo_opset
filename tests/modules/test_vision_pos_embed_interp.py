import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import modules


@pytest.mark.api("modules.VisionPosEmbedInterp", ops=["vision_pos_embed_interp_batch"])
@pytest.mark.accuracy
def test_module(accuracy_backend):
    impl, _, device = accuracy_backend
    torch.manual_seed(42)
    weight = torch.randn(14 * 14, 1000, device=device, dtype=torch.bfloat16) * 0.25
    grids = [(1, 8, 16), (3, 16, 16)]
    op = modules.VisionPosEmbedInterp(14, 2, implementation=impl)
    actual = op(weight, grids)
    expected = F.vision_pos_embed_interp_batch(weight, grids, 14, 2, weight.dtype, implementation="torch")
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
