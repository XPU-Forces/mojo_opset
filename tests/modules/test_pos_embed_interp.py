import pytest
import torch

from mojo_opset import functions
from mojo_opset import modules
from tests._checks import assert_close

NUM_GRID, M_SIZE, HIDDEN = 48, 2, 1152


def make_embed(device, dtype=torch.float32):
    torch.manual_seed(42)
    return torch.randn(NUM_GRID * NUM_GRID, HIDDEN, device=device, dtype=dtype) * 0.25


@pytest.mark.api("modules.PosEmbedInterp", ops=["pos_embed_interp_infer"])
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16), ids=lambda d: str(d).split(".")[-1])
@pytest.mark.accuracy
def test_pos_embed_module_accuracy(accuracy_backend, dtype):
    """The module is a thin wrapper: buffer weight and per-call weight must agree and match the reference."""
    implementation, _target, device = accuracy_backend
    embed = make_embed(device, dtype)
    expected = functions.pos_embed_interp_infer(embed, 1, 32, 48, NUM_GRID, M_SIZE, dtype, implementation="torch")

    module = modules.PosEmbedInterp(embed, NUM_GRID, M_SIZE, implementation=implementation)
    from_buffer = module(1, 32, 48, dtype=dtype)
    assert torch.equal(from_buffer, module(1, 32, 48, embed_weight=embed, dtype=dtype))

    assert from_buffer.shape == expected.shape and from_buffer.dtype == expected.dtype
    rtol, atol = (1e-5, 5e-5) if dtype == torch.float32 else (1e-2, 1e-2)
    assert_close(from_buffer, expected, dtype, name="module vs torch", rtol=rtol, atol=atol)


@pytest.mark.api("modules.PosEmbedInterp", ops=["pos_embed_interp_multi_infer"])
@pytest.mark.bitwise
def test_pos_embed_module_multi_matches_per_grid(accuracy_backend):
    implementation, _target, device = accuracy_backend
    embed = make_embed(device)
    grids = [(1, 8, 16), (1, 32, 48), (3, 16, 16)]
    module = modules.PosEmbedInterp(embed, NUM_GRID, M_SIZE, implementation=implementation)

    merged = module(grids=grids, dtype=torch.float32)
    per_grid = torch.cat([module(t, h, w, dtype=torch.float32) for t, h, w in grids], dim=0)
    assert torch.equal(merged, per_grid)


@pytest.mark.reference
def test_pos_embed_module_requires_weight():
    module = modules.PosEmbedInterp(None, NUM_GRID, M_SIZE)
    with pytest.raises(ValueError, match="embed_weight"):
        module(1, 16, 16)
