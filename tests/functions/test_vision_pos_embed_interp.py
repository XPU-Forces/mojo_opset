"""Task-matrix development tests; the task's claimed attachment was not supplied.

Tolerances are copied unchanged from vLLM aa31fa0's upstream tests. These are
not presented as the missing official NPU acceptance suite.
"""

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset.kernels.torch.vision_pos_embed_interp import vision_pos_embed_interp_fwd as native

GRIDS = [(1, 4, 4), (1, 16, 16), (1, 32, 32), (1, 48, 48), (1, 8, 16), (1, 14, 20), (1, 32, 48), (1, 60, 80)]
BATCH = [(1, 8, 16), (1, 32, 48), (3, 16, 16)]


def make_weight(g, d, dtype, device):
    torch.manual_seed(42)
    return torch.randn(g * g, d, dtype=dtype, device=device) * 0.25


def check_close(actual, expected, dtype):
    atol = {torch.float32: 5e-5, torch.bfloat16: 1e-2}[dtype]
    rtol = {torch.float32: 1e-5, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("grid", GRIDS)
def test_upstream_matrix(accuracy_backend, dtype, grid):
    impl, _, device = accuracy_backend
    weight = make_weight(48, 1152, dtype, device)
    actual = F.vision_pos_embed_interp(weight, *grid, 48, 2, dtype, implementation=impl)
    check_close(actual, native(weight, *grid, 48, 2, dtype), dtype)


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "grid,g,d,m",
    [
        ((1, 16, 16), 14, 1152, 2),
        ((1, 16, 16), 48, 1000, 2),
        ((1, 32, 48), 48, 1000, 2),
        ((1, 1, 7), 14, 33, 1),
        ((1, 7, 1), 14, 33, 1),
        ((1, 1, 1), 1, 7, 1),
        ((1, 2, 16), 48, 65, 2),
        ((1, 16, 2), 48, 65, 2),
        ((1, 6, 9), 14, 127, 3),
        ((1, 6, 9), 14, 4097, 3),
        ((17, 4, 6), 14, 33, 2),
        ((16, 100, 168), 48, 1152, 2),
    ],
)
def test_boundaries_and_large_video(accuracy_backend, dtype, grid, g, d, m):
    impl, _, device = accuracy_backend
    weight = make_weight(g, d, dtype, device)
    actual = F.vision_pos_embed_interp(weight, *grid, g, m, dtype, implementation=impl)
    check_close(actual, native(weight, *grid, g, m, dtype), dtype)


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.bitwise
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("t,h,w", [(3, 16, 16), (16, 100, 168), (17, 4, 6)])
def test_temporal_repeat(accuracy_backend, dtype, t, h, w):
    impl, _, device = accuracy_backend
    weight = make_weight(48, 1152, dtype, device)
    one = F.vision_pos_embed_interp(weight, 1, h, w, 48, 2, dtype, implementation=impl)
    repeated = F.vision_pos_embed_interp(weight, t, h, w, 48, 2, dtype, implementation=impl)
    torch.testing.assert_close(repeated, one.repeat(t, 1), atol=0, rtol=0)


@pytest.mark.api(
    "functions.vision_pos_embed_interp_batch", ops=["vision_pos_embed_interp", "vision_pos_embed_interp_batch"]
)
@pytest.mark.bitwise
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "grids,g,d,m",
    [
        (BATCH, 48, 1152, 2),
        ([(1, 2, 6), (3, 6, 4), (1, 4, 10)], 14, 1000, 2),
        ([(1, 1, 7), (2, 7, 1), (1, 1, 1)], 14, 33, 1),
    ],
)
def test_fused_batch_bitwise(accuracy_backend, dtype, grids, g, d, m):
    impl, _, device = accuracy_backend
    weight = make_weight(g, d, dtype, device)
    actual = F.vision_pos_embed_interp_batch(weight, grids, g, m, dtype, implementation=impl)
    separate = torch.cat([F.vision_pos_embed_interp(weight, *grid, g, m, dtype, implementation=impl) for grid in grids])
    torch.testing.assert_close(actual, separate, atol=0, rtol=0)
    reference = torch.cat([native(weight, *grid, g, m, dtype) for grid in grids])
    check_close(actual, reference, dtype)


@pytest.mark.api(
    "functions.vision_pos_embed_interp_batch", ops=["vision_pos_embed_interp_batch", "vision_pos_embed_interp"]
)
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_strided_weight_and_empty(accuracy_backend, dtype):
    impl, _, device = accuracy_backend
    weight = make_weight(14, 2000, dtype, device)[:, ::2]
    result = F.vision_pos_embed_interp(weight, 1, 8, 16, 14, 2, dtype, implementation=impl)
    check_close(result, native(weight, 1, 8, 16, 14, 2, dtype), dtype)
    result = F.vision_pos_embed_interp_batch(weight, [], 14, 2, dtype, implementation=impl)
    assert result.shape == (0, 1000)


@pytest.mark.api("functions.vision_pos_embed_interp_batch")
def test_reject_invalid_grid(accuracy_backend):
    impl, _, device = accuracy_backend
    weight = make_weight(14, 33, torch.float32, device)
    with pytest.raises(AssertionError):
        F.vision_pos_embed_interp_batch(weight, [(1, 3, 4)], 14, 2, torch.float32, implementation=impl)
    with pytest.raises(ValueError):
        F.vision_pos_embed_interp_batch(weight, [(0, 4, 4)], 14, 2, torch.float32, implementation=impl)
