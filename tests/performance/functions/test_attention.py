import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training


@pytest.mark.api("functions.swa")
@pytest.mark.parametrize(
    "batch,q_heads,kv_heads,seq_len,window,interleave",
    [
        (1, 8, 2, 128, 31, True),
        (2, 16, 4, 4096, 1023, False),
        *[(1, 12, 4, 128 * 2**power, 1023, False) for power in range(3, 11)],
    ],
)
@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_swa(benchmark, perf_environment, batch, q_heads, kv_heads, seq_len, window, interleave, phase):
    _, device, _, implementation = perf_environment

    def factory():
        q = torch.randn(batch * seq_len, q_heads, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(batch * seq_len, kv_heads, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        cu = torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=device)
        call = lambda: functions.swa(
            q,
            k,
            v,
            cu,
            cu,
            is_causal=True,
            local_window_size=window,
            global_window_size=4,
            softmax_scale=128**-0.5,
            gqa_interleave=interleave,
            implementation=implementation,
        )
        return training(call, (q, k, v), phase)

    benchmark(
        factory=factory,
        op="swa",
        batch=batch,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seq_len=seq_len,
        window=window,
        interleave=interleave,
        dtype="bfloat16",
        phase=phase,
    )
