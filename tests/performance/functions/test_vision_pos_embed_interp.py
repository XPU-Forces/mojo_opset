"""vision_pos_embed_interp performance cases (task shape matrix, Qwen3-VL geometry)."""

import pytest
import torch

from mojo_opset import functions

MULTI_GRIDS = [(1, 8, 16), (1, 32, 48), (3, 16, 16)]


def _embed(device, hidden=1152):
    torch.manual_seed(42)
    return (torch.randn(48 * 48, hidden, dtype=torch.bfloat16, device=device) * 0.25).contiguous()


@pytest.mark.api("functions.vision_pos_embed_interp")
@pytest.mark.parametrize(
    "t,h,w,hidden",
    [
        (1, 8, 16, 1152),
        (1, 32, 48, 1152),
        (1, 60, 80, 1152),
        (3, 16, 16, 1152),
        (16, 100, 168, 1152),
        (3, 60, 80, 1000),
    ],
)
def test_vision_pos_embed_interp(benchmark, perf_environment, t, h, w, hidden):
    _, device, _, implementation = perf_environment

    def factory():
        embed = _embed(device, hidden)
        return lambda: functions.vision_pos_embed_interp(
            embed, t, h, w, 48, 2, torch.bfloat16, implementation=implementation
        )

    benchmark(
        factory=factory,
        op="vision_pos_embed_interp",
        t=t,
        h=h,
        w=w,
        num_grid_per_side=48,
        m_size=2,
        hidden=hidden,
        dtype="bfloat16",
        phase="forward",
    )


@pytest.mark.api("functions.vision_pos_embed_interp")
def test_vision_pos_embed_interp_multi(benchmark, perf_environment):
    """Multi-image baseline: per-image calls plus cat (a merged single launch
    arrives with the accelerated provider)."""
    _, device, _, implementation = perf_environment

    def factory():
        embed = _embed(device)
        return lambda: torch.cat(
            [
                functions.vision_pos_embed_interp(embed, t, h, w, 48, 2, torch.bfloat16, implementation=implementation)
                for t, h, w in MULTI_GRIDS
            ]
        )

    benchmark(
        factory=factory,
        op="vision_pos_embed_interp",
        grids=[list(g) for g in MULTI_GRIDS],
        num_grid_per_side=48,
        m_size=2,
        hidden=1152,
        dtype="bfloat16",
        phase="forward",
    )
