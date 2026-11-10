"""Accuracy tests for functions.vision_pos_embed_interp."""

import pytest
import torch
import torch_npu  # noqa: F401

from mojo_opset import functions as F

DTYPES = [torch.float32, torch.bfloat16]
NUM_GRID_PER_SIDE = 48
SPATIAL_MERGE_SIZE = 2
HIDDEN_DIM = 1152
ATOL = {torch.float32: 5e-5, torch.bfloat16: 1e-2}
RTOL = {torch.float32: 1e-5, torch.bfloat16: 1e-2}

SQUARE_GRIDS = [(1, 4, 4), (1, 16, 16), (1, 32, 32), (1, 48, 48)]
NON_SQUARE_GRIDS = [(1, 8, 16), (1, 14, 20), (1, 32, 48), (1, 60, 80)]
ALL_GRIDS = SQUARE_GRIDS + NON_SQUARE_GRIDS


def _make_weight(dtype, device, num_grid_per_side=NUM_GRID_PER_SIDE, hidden_dim=HIDDEN_DIM):
    torch.manual_seed(42)
    return torch.randn(num_grid_per_side * num_grid_per_side, hidden_dim, device=device, dtype=dtype) * 0.25


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("grid_thw", ALL_GRIDS, ids=[f"{t}x{h}x{w}" for t, h, w in ALL_GRIDS])
def test_triton_matches_reference(accuracy_backend, grid_thw, dtype):
    implementation, target, device = accuracy_backend
    t, h, w = grid_thw
    ew = _make_weight(dtype, device)
    expected = F.vision_pos_embed_interp(
        ew, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype, implementation="torch"
    )
    actual = F.vision_pos_embed_interp(
        ew, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype, implementation=implementation
    )
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_temporal_repeat(accuracy_backend, dtype):
    implementation, target, device = accuracy_backend
    ew = _make_weight(dtype, device)
    out_single = F.vision_pos_embed_interp(
        ew, 1, 16, 16, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype, implementation=implementation
    )
    out_multi = F.vision_pos_embed_interp(
        ew, 3, 16, 16, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype, implementation=implementation
    )
    torch.testing.assert_close(out_multi, out_single.repeat(3, 1), atol=0, rtol=0)


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_large_grid(accuracy_backend, dtype):
    implementation, target, device = accuracy_backend
    t, h, w = 16, 100, 168
    ew = _make_weight(dtype, device)
    expected = F.vision_pos_embed_interp(
        ew, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype, implementation="torch"
    )
    actual = F.vision_pos_embed_interp(
        ew, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype, implementation=implementation
    )
    assert actual.shape == (t * h * w, HIDDEN_DIM)
    torch.testing.assert_close(actual, expected, atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("grid_thw", [(1, 8, 16), (1, 14, 20)])
def test_generalized_num_grid(accuracy_backend, grid_thw, dtype):
    implementation, target, device = accuracy_backend
    g = 14
    t, h, w = grid_thw
    ew = _make_weight(dtype, device, num_grid_per_side=g)
    expected = F.vision_pos_embed_interp(ew, t, h, w, g, SPATIAL_MERGE_SIZE, dtype, implementation="torch")
    actual = F.vision_pos_embed_interp(ew, t, h, w, g, SPATIAL_MERGE_SIZE, dtype, implementation=implementation)
    torch.testing.assert_close(actual, expected, atol=ATOL[dtype], rtol=RTOL[dtype])


@pytest.mark.api("functions.vision_pos_embed_interp_multi")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_multi_grid_single_launch(accuracy_backend, dtype):
    implementation, target, device = accuracy_backend
    grids = [(1, 8, 16), (1, 32, 48), (3, 16, 16)]
    ew = _make_weight(dtype, device)
    expected = torch.cat(
        [
            F.vision_pos_embed_interp(
                ew, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype, implementation=implementation
            )
            for t, h, w in grids
        ],
        dim=0,
    )
    actual = F.vision_pos_embed_interp_multi(
        ew, grids, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype, implementation=implementation
    )
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("grid_thw", [(1, 16, 16), (1, 32, 48)])
def test_hidden_dim_tail(accuracy_backend, grid_thw, dtype):
    implementation, target, device = accuracy_backend
    hidden = 1000
    t, h, w = grid_thw
    ew = _make_weight(dtype, device, hidden_dim=hidden)
    expected = F.vision_pos_embed_interp(
        ew, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype, implementation="torch"
    )
    actual = F.vision_pos_embed_interp(
        ew, t, h, w, NUM_GRID_PER_SIDE, SPATIAL_MERGE_SIZE, dtype, implementation=implementation
    )
    assert actual.shape == (t * h * w, hidden)
    torch.testing.assert_close(actual, expected, atol=ATOL[dtype], rtol=RTOL[dtype])
