"""A5-only coverage for the three-axis 3D gate tile.

The tile needs a 3D gate, a row tile that is a multiple of the head count and at
least 32M elements, so this case is deliberately larger than the shared suite and
lives in its own module: it does not change the A2/A3 run counts.
"""

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset.kernels._layernorm_gated import calc_rows_per_block
from tests._checks import assert_close

pytest.importorskip("torch_npu")

M, N, HEADS = 262144, 128, 8
Z3D_TILE_MIN_ELEMENTS = 32 * 1024 * 1024


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_z3d_tile_matches_eager(accuracy_backend, dtype):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.a5"):
        pytest.skip("the three-axis gate tile is enabled on Ascend 950 only")
    rows_per_block = calc_rows_per_block(M, N)
    assert M * N >= Z3D_TILE_MIN_ELEMENTS, "the case must cross the tile threshold"
    assert rows_per_block % HEADS == 0, "the row tile must cover whole heads"

    torch.manual_seed(20261030)
    x = torch.randn((M, N), device=device, dtype=dtype)
    weight = torch.randn((N,), device=device, dtype=dtype)
    padded = torch.randn((M // HEADS, HEADS, N + 32), device=device, dtype=dtype)
    z = padded[..., :N]
    assert z.dim() == 3 and not z.is_contiguous()

    actual, mean, rstd = F.layer_norm_gated(
        x, weight, None, z=z, eps=1e-6, is_rms_norm=True, implementation=implementation
    )
    expected, mean_ref, rstd_ref = F.layer_norm_gated(
        x, weight, None, z=z, eps=1e-6, is_rms_norm=True, implementation="torch"
    )
    assert_close(actual, expected, dtype, name=f"z3d_tile_{target}")
    assert mean is None and mean_ref is None
    assert_close(rstd, rstd_ref, torch.float32, name=f"z3d_tile_rstd_{target}")
