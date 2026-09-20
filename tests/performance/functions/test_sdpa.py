import pytest
import torch

from mojo_opset import functions
from tests.functions.test_sdpa import generate_diffusion_attention_mask
from tests.performance._workload import training


@pytest.mark.api("functions.diffusion_attention")
@pytest.mark.parametrize("q_heads,kv_heads,seq_len", [(5, 1, 2048), (8, 2, 8192)])
@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_diffusion_attention(benchmark, perf_environment, q_heads, kv_heads, seq_len, phase):
    _, device, _, implementation = perf_environment

    def factory():
        q = torch.randn(1, q_heads, seq_len * 2, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
        k, v = [
            torch.randn(1, kv_heads, seq_len * 2, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
            for _ in range(2)
        ]
        mask = generate_diffusion_attention_mask(seq_len, 32).to(device)
        call = lambda: functions.diffusion_attention(
            q, k, v, mask, scale=128**-0.5, enable_gqa=True, implementation=implementation
        )
        return training(call, (q, k, v), phase)

    benchmark(
        factory=factory,
        op="diffusion_attention",
        q_heads=q_heads,
        kv_heads=kv_heads,
        seq_len=seq_len,
        block_size=32,
        dtype="bfloat16",
        phase=phase,
    )
