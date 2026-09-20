import pytest
import torch

from mojo_opset import modules
from tests._checks import assert_close

ROTARY_CASES = [
    (batch, dim, mode)
    for batch in (1, 6)
    for dim in (32, 48, 64, 88, 96, 128)
    for mode in ("padding_prefill", "varlen_prefill", "decode")
]


VISION_GRIDS = [((4, 4),), ((8, 6),), ((8, 8), (4, 6))]


def make_rotary_case(case, device):
    batch, dim, mode = case
    tokens, max_length = 2048, 32768
    options = {}
    if mode == "padding_prefill":
        x = torch.randn(batch, tokens, 256, device=device)
    elif mode == "decode":
        x = torch.randn(batch, 256, device=device)
        options["position_ids"] = torch.randint(max_length, (batch,), device=device, dtype=torch.int32)
    else:
        lengths = torch.randint((tokens + 1) // 2, tokens + 1, (batch,), device=device, dtype=torch.int32)
        options["cu_q_lens"] = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
        options["total_seq_lens"] = (
            torch.randint(0, max_length - tokens, (batch,), device=device, dtype=torch.int32) + lengths
        )
        x = torch.randn(int(lengths.sum()), 256, device=device)
    return x, options


def _rotary(dim, device, implementation, cached=True):
    table = modules.RotaryEmbedding(
        10000.0, dim, init_max_length=32768 if cached else None, implementation=implementation, device=device
    )
    return table


@pytest.mark.api("modules.RotaryEmbedding", ops=["rotary_embedding"])
@pytest.mark.parametrize("case", ROTARY_CASES)
@pytest.mark.accuracy
def test_rotary(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    x, options = make_rotary_case(case, device)
    actual = _rotary(case[1], device, implementation)(x, **options)
    expected = _rotary(case[1], device, "torch_reference")(x, **options)
    dynamic = _rotary(case[1], device, "torch_reference", cached=False)(x, **options)
    assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert_close(expected, dynamic, rtol=1e-5, atol=1e-5)


@pytest.mark.api("modules.VisionRotaryEmbedding2D", ops=["vision_rotary_embedding2d"])
@pytest.mark.parametrize("grid", VISION_GRIDS)
@pytest.mark.accuracy
def test_vision(accuracy_backend, grid):
    implementation, _, device = accuracy_backend
    grid_hw = torch.tensor(grid, device=device, dtype=torch.int32)
    table = modules.VisionRotaryEmbedding2D(
        rope_dim=64, adapooling_factor=2, device=device, implementation=implementation
    )
    actual = table(grid_hw)
    expected = modules.VisionRotaryEmbedding2D(
        rope_dim=64, adapooling_factor=2, device=device, implementation="torch_reference"
    )(grid_hw)
    assert_close(actual, expected, rtol=1e-5, atol=1e-5)
