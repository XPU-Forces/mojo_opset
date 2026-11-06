import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training

NUM_GRID, M_SIZE, HIDDEN = 48, 2, 1152
GRIDS = [(1, 8, 16), (1, 32, 48), (1, 60, 80), (16, 100, 168)]
MULTI_GRIDS = [(1, 8, 16), (1, 32, 48), (3, 16, 16)]


def make_embed(device):
    torch.manual_seed(42)
    # std ~ 0.23 matches the real Qwen3-VL pos_embed weight distribution.
    return torch.randn(NUM_GRID * NUM_GRID, HIDDEN, device=device, dtype=torch.bfloat16) * 0.25


@pytest.mark.api("functions.pos_embed_interp_infer")
@pytest.mark.parametrize("grid", GRIDS, ids=[f"{t}x{h}x{w}" for t, h, w in GRIDS])
@pytest.mark.parametrize("phase", ["forward"])
def test_pos_embed_interp(benchmark, perf_environment, grid, phase):
    _, device, _, implementation = perf_environment
    t, h, w = grid

    def factory():
        embed = make_embed(device)
        call = lambda: functions.pos_embed_interp_infer(
            embed, t, h, w, NUM_GRID, M_SIZE, torch.bfloat16, implementation=implementation
        )
        return training(call, (embed,), phase)

    benchmark(
        factory=factory,
        op="pos_embed_interp_infer",
        grid=f"{t}x{h}x{w}",
        rows=t * h * w,
        hidden=HIDDEN,
        dtype="bfloat16",
        phase=phase,
    )


@pytest.mark.api("functions.pos_embed_interp_multi_infer")
@pytest.mark.parametrize("phase", ["forward"])
def test_pos_embed_interp_multi(benchmark, perf_environment, phase):
    _, device, _, implementation = perf_environment

    def factory():
        embed = make_embed(device)
        call = lambda: functions.pos_embed_interp_multi_infer(
            embed, MULTI_GRIDS, NUM_GRID, M_SIZE, torch.bfloat16, implementation=implementation
        )
        return training(call, (embed,), phase)

    benchmark(
        factory=factory,
        op="pos_embed_interp_multi_infer",
        images=len(MULTI_GRIDS),
        rows=sum(t * h * w for t, h, w in MULTI_GRIDS),
        hidden=HIDDEN,
        dtype="bfloat16",
        phase=phase,
    )
