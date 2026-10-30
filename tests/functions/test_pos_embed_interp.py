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
# (1,2,2) is a single merge block, while (1,2,8)/(1,8,2)/(1,2,6)/(1,6,2) split the output into
# several blocks and give the persistent grid more than one block per program, i.e. the
# block-chunk loop runs more than once.
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


def check_interp(
    backend,
    grid,
    dtype,
    *,
    hidden=HIDDEN_DIM,
    num_grid_per_side=NUM_GRID_PER_SIDE,
    m_size=M_SIZE,
    embed_dtype=None,
):
    implementation, _target, device = backend
    embed = make_embed(hidden, device, embed_dtype or dtype, num_grid_per_side)
    t, h, w = grid

    expected = F.pos_embed_interp_infer(
        embed, t, h, w, num_grid_per_side, m_size, dtype, implementation="torch"
    )
    actual = F.pos_embed_interp_infer(embed, t, h, w, num_grid_per_side, m_size, dtype, implementation=implementation)

    assert actual.shape == (t * h * w, hidden)
    out_dtype = embed.dtype if dtype is None else dtype
    assert actual.dtype == out_dtype and actual.device == embed.device
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
    """hidden=1000 is not a power of two, so the whole row is one masked tile.

    ``block_d`` is the next power of two above ``hidden`` (1024 here), which leaves 24 masked
    lanes at the end of every row. The tail columns are compared on their own so that a lane
    reading past ``hidden`` cannot hide behind the full-tensor tolerance.

    This is a read-side check: a *write* past the row would land in the next row (or past the
    buffer), which only a sanitizer run can see (``mssanitizer``-style oob detection), so no
    sentinel is claimed here.
    """
    implementation, _target, device = accuracy_backend
    actual, expected = check_interp(accuracy_backend, grid, dtype, hidden=TAIL_HIDDEN)
    assert torch.isfinite(actual).all(), "non-finite value in the tail tile"

    tail = actual[:, TAIL_HIDDEN - 8 :]
    assert_close(
        tail,
        expected[:, TAIL_HIDDEN - 8 :],
        dtype,
        name=f"{implementation} tail columns {grid}",
        **TOLERANCES[dtype],
    )


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
    # m_size is validated before the divisibility test, which divides by it.
    with pytest.raises(ValueError, match="m_size=0 must be positive"):
        F.pos_embed_interp_infer(embed, 1, 16, 16, NUM_GRID_PER_SIDE, 0)


@pytest.mark.api("functions.pos_embed_interp_infer")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("m_size", [1, 4], ids=lambda m: f"m{m}")
@pytest.mark.accuracy
def test_pos_embed_interp_other_merge_sizes(accuracy_backend, dtype, m_size):
    """m_size is a model config value, not a constant: 1 (no merge) and 4 must both work."""
    implementation, _target, device = accuracy_backend
    grid = (1, 16, 16) if m_size == 1 else (1, 8, 8)
    check_interp(accuracy_backend, grid, dtype, m_size=m_size)


@pytest.mark.api("functions.pos_embed_interp_infer")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.accuracy
def test_pos_embed_interp_grid32_size100(accuracy_backend, dtype):
    """G=32 with h=w=100: the one combination where the kernel and the reference floor differ.

    ``torch.linspace(0, G-1, 100)`` puts a sample exactly on an integer for some indices, and
    the kernel's precomputed ``scale`` can land an ULP below it, so ``floor`` picks the lower
    cell. Both answers are within tolerance because the weight of the far corner collapses to
    zero at that point -- this case pins that down instead of leaving it to chance.
    """
    check_interp(accuracy_backend, (1, 100, 100), dtype, num_grid_per_side=32)


@pytest.mark.api("functions.pos_embed_interp_infer")
@pytest.mark.accuracy
def test_pos_embed_interp_output_dtype_override(accuracy_backend):
    """``dtype`` selects the output dtype independently of the table dtype.

    Only the fp32-table / bf16-output direction is covered here: the eager reference casts the
    bilinear weights to the output dtype and accumulates in place, so a bf16 table with an fp32
    output cannot be expressed there (the multiply would have to up-cast in place).
    """
    check_interp(accuracy_backend, (1, 16, 16), torch.bfloat16, embed_dtype=torch.float32)


@pytest.mark.api("functions.pos_embed_interp_multi_infer")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.bitwise
def test_pos_embed_interp_multi_accepts_tensor_grids(accuracy_backend, dtype):
    """``grids`` may arrive as an int tensor (vllm keeps ``grid_thw`` as one); same result."""
    implementation, _target, device = accuracy_backend
    grids = [(1, 8, 16), (3, 16, 16)]
    embed = make_embed(HIDDEN_DIM, device, dtype)

    from_list = F.pos_embed_interp_multi_infer(
        embed, grids, NUM_GRID_PER_SIDE, M_SIZE, dtype, implementation=implementation
    )
    for tensor_dtype in (torch.int32, torch.int64):
        from_tensor = F.pos_embed_interp_multi_infer(
            embed,
            torch.tensor(grids, dtype=tensor_dtype, device=device),
            NUM_GRID_PER_SIDE,
            M_SIZE,
            dtype,
            implementation=implementation,
        )
        assert torch.equal(from_tensor, from_list), f"tensor grids ({tensor_dtype}) changed the result"
