"""Accuracy tests for functions.vision_pos_embed_interp.

Covers the task-book shape matrix (eight single-image grids x two dtypes,
temporal repeat, 16x100x168 video, G=14 generalisation, hidden=1000 tail,
degenerate 2xN/Nx2 rows and merged multi-grid ordering).  The reference is
the fp32 eager torch kernel (``implementation="torch"``).
"""

import pytest
import torch

from mojo_opset import functions as F


def pos_embed_interpolate_native(embed_weight, t, h, w, num_grid_per_side, m_size, dtype):
    return F.vision_pos_embed_interp(
        embed_weight, t, h, w, num_grid_per_side, m_size, dtype, implementation="torch"
    )


def triton_pos_embed_interpolate(embed_weight, t, h, w, num_grid_per_side, m_size, dtype):
    return F.vision_pos_embed_interp(
        embed_weight, t, h, w, num_grid_per_side, m_size, dtype, implementation="triton"
    )


def vision_pos_embed_interp_grids(embed_weight, grid_thw, num_grid_per_side, m_size, dtype=None):
    return F.vision_pos_embed_interp_grids(
        embed_weight, grid_thw, num_grid_per_side, m_size, dtype, implementation="triton"
    )


def vision_pos_embed_interp_grids_native(embed_weight, grid_thw, num_grid_per_side, m_size, dtype=None):
    return F.vision_pos_embed_interp_grids(
        embed_weight, grid_thw, num_grid_per_side, m_size, dtype, implementation="torch"
    )

DTYPES = [torch.float32, torch.bfloat16]
NUM_GRID_PER_SIDE = 48
SPATIAL_MERGE_SIZE = 2
HIDDEN_DIM = 1152

SQUARE_GRIDS = [(1, 4, 4), (1, 16, 16), (1, 32, 32), (1, 48, 48)]
NON_SQUARE_GRIDS = [(1, 8, 16), (1, 14, 20), (1, 32, 48), (1, 60, 80)]
ALL_GRIDS = SQUARE_GRIDS + NON_SQUARE_GRIDS

ATOL = {torch.float32: 5e-5, torch.bfloat16: 1e-2}
RTOL = {torch.float32: 1e-5, torch.bfloat16: 1e-2}

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="NPU is required for this port",
)


def _weight(hidden, dtype, num_grid=NUM_GRID_PER_SIDE):
    torch.manual_seed(42)
    return torch.randn(num_grid * num_grid, hidden, device="npu", dtype=dtype) * 0.25


def _assert(actual, expected, dtype, *, exact=False):
    if exact:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    else:
        torch.testing.assert_close(actual, expected, atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("grid_thw", ALL_GRIDS, ids=[f"{t}x{h}x{w}" for t, h, w in ALL_GRIDS])
def test_triton_matches_native(grid_thw, dtype):
    t, h, w = grid_thw
    embed = _weight(HIDDEN_DIM, dtype)
    native = pos_embed_interpolate_native(
        embed, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    actual = triton_pos_embed_interpolate(
        embed, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    assert actual.shape == native.shape == (t * h * w, HIDDEN_DIM)
    _assert(actual, native, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_temporal_repeat(dtype):
    embed = _weight(HIDDEN_DIM, dtype)
    single = triton_pos_embed_interpolate(embed, 1, 16, 16, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype)
    multi = triton_pos_embed_interpolate(embed, 3, 16, 16, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype)
    _assert(multi, single.repeat(3, 1), dtype, exact=True)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_large_video_grid(dtype):
    t, h, w = 16, 100, 168
    embed = _weight(HIDDEN_DIM, dtype)
    native = pos_embed_interpolate_native(
        embed, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    actual = triton_pos_embed_interpolate(
        embed, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    _assert(actual, native, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("grid_thw", [(1, 16, 16), (1, 32, 48)], ids=["16x16", "32x48"])
def test_generalized_grid_side(grid_thw, dtype):
    num_grid = 14
    t, h, w = grid_thw
    embed = _weight(1152, dtype, num_grid=num_grid)
    native = pos_embed_interpolate_native(embed, t, h, w, num_grid, SPATIAL_MERGE_SIZE, dtype)
    actual = triton_pos_embed_interpolate(embed, t, h, w, num_grid, SPATIAL_MERGE_SIZE, dtype)
    _assert(actual, native, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("grid_thw", [(1, 16, 16), (1, 32, 48)], ids=["16x16", "32x48"])
def test_hidden_tail(grid_thw, dtype):
    t, h, w = grid_thw
    embed = _weight(1000, dtype)
    native = pos_embed_interpolate_native(
        embed, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    actual = triton_pos_embed_interpolate(
        embed, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    _assert(actual, native, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("grid_thw", [(1, 2, 2), (1, 2, 8), (1, 8, 2)], ids=["2x2", "2x8", "8x2"])
def test_degenerate_row_and_column(grid_thw, dtype):
    t, h, w = grid_thw
    embed = _weight(HIDDEN_DIM, dtype)
    native = pos_embed_interpolate_native(
        embed, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    actual = triton_pos_embed_interpolate(
        embed, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    _assert(actual, native, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_multi_grid_matches_per_grid(dtype):
    embed = _weight(HIDDEN_DIM, dtype)
    grids = [(1, 8, 16), (1, 32, 48), (3, 16, 16)]
    per_grid = torch.cat(
        [
            triton_pos_embed_interpolate(embed, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype)
            for t, h, w in grids
        ],
        dim=0,
    )
    merged = vision_pos_embed_interp_grids(embed, grids, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype)
    _assert(merged, per_grid, dtype, exact=True)

    native = vision_pos_embed_interp_grids_native(
        embed, grids, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype
    )
    _assert(merged, native, dtype)
