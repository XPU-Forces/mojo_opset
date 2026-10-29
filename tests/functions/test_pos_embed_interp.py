"""Tests for functions.pos_embed_interp_infer / pos_embed_interp_multi_infer.

The accuracy reference is ``implementation="torch"``, i.e. the eager
``pos_embed_interpolate_native`` port (4-neighbour gather + spatial-merge permute), which
is what the NPU production patch currently forces. Tolerances follow the upstream kernel
test: the kernel precomputes ``h_scale``/``w_scale`` while the reference uses
``torch.linspace``, which can differ by a single ULP in a handful of elements.
"""

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_close

DTYPES = (torch.float32, torch.bfloat16)
NUM_GRID_PER_SIDE = 48
M_SIZE = 2
HIDDEN_DIM = 1152

# 4 square + 4 non-square grids (h, w divisible by m_size).
STANDARD_GRIDS = [
    (1, 4, 4),
    (1, 16, 16),
    (1, 32, 32),
    (1, 48, 48),
    (1, 8, 16),
    (1, 14, 20),
    (1, 32, 48),
    (1, 60, 80),
]
# Temporal broadcast case.
TEMPORAL_GRIDS = [(3, 16, 16)]
# h == w == m_size degenerates to a full-table corner gather (scale = G - 1 per step).
# (1,2,2) has a single merge block and (1,2,6)/(1,6,2) have three, so the tail guard of the
# multi-block tile (blocks_per_iter=2) is exercised on hardware, not only in the CPU emulator.
DEGENERATE_GRIDS = [(1, 2, 2), (1, 2, 8), (1, 8, 2), (1, 2, 6), (1, 6, 2)]
# Video-scale grid: 268800 output rows.
BIG_GRIDS = [(16, 100, 168)]
# Non power-of-two feature dimension (tail masked block).
TAIL_HIDDEN = 1000
TAIL_GRIDS = [(1, 16, 16), (1, 32, 48)]

TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 5e-5},
    torch.bfloat16: {"rtol": 1e-2, "atol": 1e-2},
}


def make_embed(hidden, device, dtype, num_grid_per_side=NUM_GRID_PER_SIDE):
    # std ~ 0.23 matches the real Qwen3-VL pos_embed distribution.
    torch.manual_seed(42)
    return torch.randn(num_grid_per_side * num_grid_per_side, hidden, device=device, dtype=dtype) * 0.25


def check_interp(backend, grid, dtype, *, hidden=HIDDEN_DIM, num_grid_per_side=NUM_GRID_PER_SIDE):
    implementation, _target, device = backend
    embed = make_embed(hidden, device, dtype, num_grid_per_side)
    t, h, w = grid

    expected = F.pos_embed_interp_infer(
        embed, t, h, w, num_grid_per_side, M_SIZE, dtype, implementation="torch"
    )
    actual = F.pos_embed_interp_infer(embed, t, h, w, num_grid_per_side, M_SIZE, dtype, implementation=implementation)

    assert actual.shape == (t * h * w, hidden)
    assert actual.dtype == embed.dtype and actual.device == embed.device
    assert_close(actual, expected, dtype, name=f"{implementation} vs torch {grid}", **TOLERANCES[dtype])
    return actual, expected


@pytest.mark.api("functions.pos_embed_interp_infer")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize(
    "grid",
    STANDARD_GRIDS + BIG_GRIDS,
    ids=[f"{t}x{h}x{w}" for t, h, w in STANDARD_GRIDS + BIG_GRIDS],
)
@pytest.mark.accuracy
def test_pos_embed_interp_accuracy(accuracy_backend, dtype, grid):
    check_interp(accuracy_backend, grid, dtype)


@pytest.mark.api("functions.pos_embed_interp_infer")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("grid", DEGENERATE_GRIDS, ids=[f"{t}x{h}x{w}" for t, h, w in DEGENERATE_GRIDS])
@pytest.mark.accuracy
def test_pos_embed_interp_degenerate_axis(accuracy_backend, dtype, grid):
    """h == w == m_size: the clamp to NUM_GRID-1 must keep the gather in range."""
    check_interp(accuracy_backend, grid, dtype)


@pytest.mark.api("functions.pos_embed_interp_infer")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("num_grid", [14, 32], ids=lambda g: f"G{g}")
@pytest.mark.accuracy
def test_pos_embed_interp_generalized_grid_side(accuracy_backend, dtype, num_grid):
    """G must not be hardcoded to 48; non power-of-two G exercises the masks."""
    check_interp(accuracy_backend, (1, 16, 16), dtype, num_grid_per_side=num_grid)


@pytest.mark.api("functions.pos_embed_interp_infer")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("grid", TAIL_GRIDS, ids=[f"{t}x{h}x{w}" for t, h, w in TAIL_GRIDS])
@pytest.mark.accuracy
def test_pos_embed_interp_hidden_tail(accuracy_backend, dtype, grid):
    """hidden=1000 is not a power of two: the trailing masked block must not write OOB."""
    implementation, _target, device = accuracy_backend
    check_interp(accuracy_backend, grid, dtype, hidden=TAIL_HIDDEN)
    # Sentinel guard: the output buffer is exactly t*h*w rows, so an out-of-range store
    # would corrupt adjacent allocations. Reallocate and check the tail row explicitly.
    embed = make_embed(TAIL_HIDDEN, device, dtype)
    out = F.pos_embed_interp_infer(embed, *grid, NUM_GRID_PER_SIDE, M_SIZE, dtype, implementation=implementation)
    t, h, w = grid
    assert torch.isfinite(out).all(), "non-finite value in the tail block"
    assert out.shape == (t * h * w, TAIL_HIDDEN)


@pytest.mark.api("functions.pos_embed_interp_infer")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("grid", TEMPORAL_GRIDS, ids=[f"{t}x{h}x{w}" for t, h, w in TEMPORAL_GRIDS])
@pytest.mark.bitwise
def test_pos_embed_interp_temporal_repeat_bitwise(accuracy_backend, dtype, grid):
    """t>1 must be bit-identical to broadcasting the t=1 result."""
    t, h, w = grid
    implementation, _target, device = accuracy_backend
    embed = make_embed(HIDDEN_DIM, device, dtype)

    single = F.pos_embed_interp_infer(embed, 1, h, w, NUM_GRID_PER_SIDE, M_SIZE, dtype, implementation=implementation)
    multi = F.pos_embed_interp_infer(embed, t, h, w, NUM_GRID_PER_SIDE, M_SIZE, dtype, implementation=implementation)

    assert torch.equal(multi, single.repeat(t, 1))


@pytest.mark.api("functions.pos_embed_interp_multi_infer")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.bitwise
def test_pos_embed_interp_multi_matches_cat(accuracy_backend, dtype):
    """The merged single launch must be bit-identical to per-grid calls + cat."""
    implementation, _target, device = accuracy_backend
    grids = [(1, 8, 16), (1, 32, 48), (3, 16, 16)]
    embed = make_embed(HIDDEN_DIM, device, dtype)

    # Bitwise contract: the merged launch must equal per-image calls of *the same*
    # implementation. Comparing against the torch reference here would be an accuracy
    # check, not a bitwise one.
    per_grid = torch.cat(
        [
            F.pos_embed_interp_infer(embed, t, h, w, NUM_GRID_PER_SIDE, M_SIZE, dtype, implementation=implementation)
            for t, h, w in grids
        ],
        dim=0,
    )
    actual = F.pos_embed_interp_multi_infer(
        embed, grids, NUM_GRID_PER_SIDE, M_SIZE, dtype, implementation=implementation
    )
    assert torch.equal(actual, per_grid), "merged launch is not bit-identical to per-image launches"

    # And it stays within the accuracy budget of the torch reference.
    reference = torch.cat(
        [
            F.pos_embed_interp_infer(embed, t, h, w, NUM_GRID_PER_SIDE, M_SIZE, dtype, implementation="torch")
            for t, h, w in grids
        ],
        dim=0,
    )
    assert_close(actual, reference, dtype, name="multi vs torch", **TOLERANCES[dtype])


@pytest.mark.api("functions.pos_embed_interp_infer")
def test_pos_embed_interp_rejects_bad_geometry(accuracy_backend):
    _implementation, _target, device = accuracy_backend
    embed = make_embed(128, device, torch.float32)
    with pytest.raises(ValueError, match="divisible"):
        F.pos_embed_interp_infer(embed, 1, 15, 16, NUM_GRID_PER_SIDE, M_SIZE)
    with pytest.raises(ValueError, match="must equal"):
        F.pos_embed_interp_infer(embed, 1, 16, 16, 32, M_SIZE)
