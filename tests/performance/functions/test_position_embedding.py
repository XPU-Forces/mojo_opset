import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training


@pytest.mark.api("functions.rope")
@pytest.mark.parametrize(
    "batch,seq_len,q_heads,k_heads,rotary_dim",
    [(1, 128, 8, 2, 128), (32, 8192, 32, 8, 128), (1, 1024, 48, 8, 48), (1, 1024, 56, 8, 48)],
)
@pytest.mark.parametrize("phase", ["forward", "backward", "forward_backward"])
def test_rope(benchmark, perf_environment, batch, seq_len, q_heads, k_heads, rotary_dim, phase):
    _, device, _, implementation = perf_environment

    def factory():
        q = torch.randn(
            batch, q_heads, seq_len, 128, dtype=torch.bfloat16, device=device, requires_grad=phase != "forward"
        )
        k = torch.randn(
            batch, k_heads, seq_len, 128, dtype=torch.bfloat16, device=device, requires_grad=phase != "forward"
        )
        cos = torch.randn(seq_len, rotary_dim, device=device)
        sin = torch.randn_like(cos)
        call = lambda: functions.rope(q, k, cos, sin, unsqueeze_dim=0, implementation=implementation)
        return training(call, (q, k), phase)

    benchmark(
        factory=factory,
        op="rope",
        batch=batch,
        seq_len=seq_len,
        q_heads=q_heads,
        k_heads=k_heads,
        head_dim=128,
        rotary_dim=rotary_dim,
        dtype="bfloat16",
        phase=phase,
    )
