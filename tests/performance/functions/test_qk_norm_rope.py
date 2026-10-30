import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training

HEAD_DIM = 128
ROTARY_DIM = 96
EPS = 1e-6
SHAPES = [(32, 8), (320, 8), (320, 56), (1024, 56), (4096, 56), (4096, 57)]


def make_inputs(tokens, heads, device):
    gen = torch.Generator().manual_seed(0)
    half = ROTARY_DIM // 2
    theta = torch.rand(tokens, half, generator=gen, dtype=torch.float32) * 6.283185307179586
    rope = torch.cat((theta.cos(), theta.sin()), dim=-1).to(torch.bfloat16).to(device)
    q = (torch.randn(tokens, heads, HEAD_DIM, generator=gen) * 0.5).to(torch.bfloat16).to(device)
    k = (torch.randn(tokens, heads, HEAD_DIM, generator=gen) * 0.5).to(torch.bfloat16).to(device)
    qw = (torch.rand(HEAD_DIM, generator=gen) + 0.5).to(torch.bfloat16).to(device)
    kw = (torch.rand(HEAD_DIM, generator=gen) + 0.5).to(torch.bfloat16).to(device)
    return q, k, qw, kw, rope


@pytest.mark.api("functions.qk_norm_rope_infer")
@pytest.mark.parametrize("shape", SHAPES, ids=[f"{t}x{h}" for t, h in SHAPES])
@pytest.mark.parametrize("phase", ["forward"])
def test_qk_norm_rope(benchmark, perf_environment, shape, phase):
    _, device, _, implementation = perf_environment
    tokens, heads = shape
    q, k, qw, kw, rope = make_inputs(tokens, heads, device)

    def factory():
        call = lambda: functions.qk_norm_rope_infer(
            q, k, qw, kw, rope, EPS, implementation=implementation
        )
        return training(call, (q, k, qw, kw, rope), phase)

    benchmark(
        factory=factory,
        op="qk_norm_rope_infer",
        tokens=tokens,
        heads=heads,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        dtype="bfloat16",
        phase=phase,
    )
