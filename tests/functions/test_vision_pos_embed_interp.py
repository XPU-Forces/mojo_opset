"""vision_pos_embed_interp accuracy suite (task: bilinear pos-embed interpolation).

Reference: the eager ``torch`` provider (upstream pos_embed_interpolate_native).
Tolerances follow the CANN opbase mixed-tolerance standard; the temporal repeat
and chunk-boundary checks are bitwise.
"""

import pytest
import torch

from mojo_opset import functions as F

NUM_GRID_PER_SIDE = 48
SPATIAL_MERGE_SIZE = 2
HIDDEN_DIM = 1152

DTYPES = [torch.float32, torch.bfloat16]
SQUARE_GRIDS = [(1, 4, 4), (1, 16, 16), (1, 32, 32), (1, 48, 48)]
NON_SQUARE_GRIDS = [(1, 8, 16), (1, 14, 20), (1, 32, 48), (1, 60, 80)]
ALL_GRIDS = SQUARE_GRIDS + NON_SQUARE_GRIDS

# CANN opbase mixed tolerance standard: per-element |a-g| <= atol + rtol*|g|,
# matched_ratio >= 0.99 and max_abs_error <= max(floor, 32*ULP(golden)).
_TOL = {torch.float32: (2.0**-13, 2.0**-13, 1e-2), torch.bfloat16: (2.0**-7, 2.0**-7, 1e-1)}


def assert_mixed_tolerance(actual, golden, dtype):
    atol, rtol, floor = _TOL[dtype]
    err = (actual.float() - golden.float()).abs()
    assert (err <= atol + rtol * golden.float().abs()).float().mean().item() >= 0.99
    index = err.flatten().argmax()
    golden_low = golden.flatten()[index].to(dtype).cpu()
    ulp = torch.nextafter(golden_low, torch.full_like(golden_low, float("inf"))) - golden_low
    assert err.flatten()[index].item() <= max(floor, 32.0 * ulp.float().item())


def make_embed(device, num_grid=NUM_GRID_PER_SIDE, hidden=HIDDEN_DIM, dtype=torch.bfloat16):
    torch.manual_seed(42)
    return (torch.randn(num_grid * num_grid, hidden, device=device, dtype=dtype) * 0.25).contiguous()


def _interp(implementation, embed, t, h, w, dtype, num_grid=NUM_GRID_PER_SIDE):
    return F.vision_pos_embed_interp(embed, t, h, w, num_grid, SPATIAL_MERGE_SIZE, dtype, implementation=implementation)


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("grid_thw", ALL_GRIDS, ids=[f"{t}x{h}x{w}" for t, h, w in ALL_GRIDS])
def test_matches_reference(accuracy_backend, grid_thw, dtype):
    implementation, _, device = accuracy_backend
    t, h, w = grid_thw
    embed = make_embed(device, dtype=dtype)
    actual = _interp(implementation, embed, t, h, w, dtype)
    expected = _interp("torch", embed, t, h, w, dtype)
    assert actual.shape == expected.shape
    assert_mixed_tolerance(actual, expected, dtype)


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_temporal_repeat_bitwise(accuracy_backend, dtype):
    """t=3 output must equal the t=1 output repeated, bitwise."""
    implementation, _, device = accuracy_backend
    embed = make_embed(device, dtype=dtype)
    out_single = _interp(implementation, embed, 1, 16, 16, dtype)
    out_multi = _interp(implementation, embed, 3, 16, 16, dtype)
    assert torch.equal(out_multi, out_single.repeat(3, 1))


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_oversized_grid_chunked(accuracy_backend, dtype):
    """(16,100,168) = 268800 output rows exceeds the Ascend 65535 1D-grid cap;
    rows at the chunk boundary must match the single-frame repeat bitwise."""
    implementation, _, device = accuracy_backend
    t, h, w = 16, 100, 168
    embed = make_embed(device, dtype=dtype)
    actual = _interp(implementation, embed, t, h, w, dtype)
    assert_mixed_tolerance(actual, _interp("torch", embed, t, h, w, dtype), dtype)
    single = _interp(implementation, embed, 1, h, w, dtype)
    assert torch.equal(actual[65530:65540], single.repeat(t, 1)[65530:65540])


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("grid_thw", [(1, 16, 16), (1, 32, 48)], ids=lambda g: str(g))
def test_hidden_dim_tail(accuracy_backend, grid_thw, dtype):
    """hidden=1000 is not a power of two; the last BLOCK_D segment is partial and
    masked lanes must not write out of bounds."""
    implementation, _, device = accuracy_backend
    t, h, w = grid_thw
    embed = make_embed(device, hidden=1000, dtype=dtype)
    actual = _interp(implementation, embed, t, h, w, dtype)
    assert_mixed_tolerance(actual, _interp("torch", embed, t, h, w, dtype), dtype)


@pytest.mark.api("functions.vision_pos_embed_interp")
def test_general_num_grid(accuracy_backend):
    """G=14 (non power of two) guards against hard-coded grid geometry."""
    implementation, _, device = accuracy_backend
    g = 14
    torch.manual_seed(7)
    embed = (torch.randn(g * g, HIDDEN_DIM, device=device, dtype=torch.bfloat16) * 0.25).contiguous()
    for t, h, w in [(1, 6, 10), (2, 8, 8)]:
        actual = _interp(implementation, embed, t, h, w, torch.bfloat16, num_grid=g)
        assert_mixed_tolerance(actual, _interp("torch", embed, t, h, w, torch.bfloat16, num_grid=g), torch.bfloat16)


@pytest.mark.api("functions.vision_pos_embed_interp")
def test_multi_grid_cat_equivalence(accuracy_backend):
    """Per-image calls concatenated match the reference composition; h=m_size and
    w=m_size degenerate grids (scale=0) stay within bounds."""
    implementation, _, device = accuracy_backend
    embed = make_embed(device)
    grids = [(1, 8, 16), (1, 32, 48), (3, 16, 16)]
    actual = torch.cat([_interp(implementation, embed, t, h, w, torch.bfloat16) for t, h, w in grids])
    expected = torch.cat([_interp("torch", embed, t, h, w, torch.bfloat16) for t, h, w in grids])
    assert_mixed_tolerance(actual, expected, torch.bfloat16)
    for t, h, w in [(1, 2, 16), (1, 16, 2)]:
        actual = _interp(implementation, embed, t, h, w, torch.bfloat16)
        assert_mixed_tolerance(actual, _interp("torch", embed, t, h, w, torch.bfloat16), torch.bfloat16)


@pytest.mark.api("functions.vision_pos_embed_interp")
def test_divisibility_validation(accuracy_backend):
    _, _, device = accuracy_backend
    embed = make_embed(device)
    with pytest.raises(ValueError):
        _interp("torch", embed, 1, 15, 16, torch.bfloat16)
