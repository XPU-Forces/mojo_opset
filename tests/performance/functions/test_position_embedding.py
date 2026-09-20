import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training


@pytest.mark.api("functions.apply_rope")
@pytest.mark.parametrize("batch,seq_len,q_heads,k_heads", [(1, 128, 8, 2), (32, 8192, 32, 8)])
@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_apply_rope(benchmark, perf_environment, batch, seq_len, q_heads, k_heads, phase):
    _, device, _, implementation = perf_environment

    def factory():
        q = torch.randn(batch, q_heads, seq_len, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
        k = torch.randn(batch, k_heads, seq_len, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
        cos = torch.randn(seq_len, 128, device=device)
        sin = torch.randn_like(cos)
        call = lambda: functions.apply_rope(q, k, cos, sin, unsqueeze_dim=0, implementation=implementation)
        return training(call, (q, k), phase)

    benchmark(
        factory=factory,
        op="apply_rope",
        batch=batch,
        seq_len=seq_len,
        q_heads=q_heads,
        k_heads=k_heads,
        head_dim=128,
        dtype="bfloat16",
        phase=phase,
    )
