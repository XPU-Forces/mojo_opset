import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training


@pytest.mark.parametrize(
    "op,batch,q_heads,kv_heads,seq_len,window,interleave",
    [
        pytest.param(op, b, hq, hk, t, w, interleave, marks=pytest.mark.api("functions." + op))
        for op, b, hq, hk, t, w, interleave in [
            ("swa", 1, 8, 2, 128, 31, True),
            ("swa", 2, 16, 4, 4096, 1023, False),
            *[("swa", 1, 12, 4, 128 * 2**power, 1023, False) for power in range(3, 11)],
            # Real FA head counts; T=1024 is a CI truncation.
            ("flash_attention", 1, 48, 8, 1024, None, False),
            ("flash_attention", 1, 56, 8, 1024, None, False),
        ]
    ],
)
@pytest.mark.parametrize("phase", ["forward", "backward", "forward_backward"])
def test_attention(benchmark, perf_environment, op, batch, q_heads, kv_heads, seq_len, window, interleave, phase):
    _, device, _, implementation = perf_environment

    def factory():
        q = torch.randn(
            batch * seq_len, q_heads, 128, device=device, dtype=torch.bfloat16, requires_grad=phase != "forward"
        )
        k = torch.randn(
            batch * seq_len, kv_heads, 128, device=device, dtype=torch.bfloat16, requires_grad=phase != "forward"
        )
        v = torch.randn_like(k, requires_grad=phase != "forward")
        cu = torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=device)
        if op == "flash_attention":
            call = lambda: functions.flash_attention(
                q, k, v, cu, cu, seq_len, seq_len, causal=True, deterministic=True, implementation=implementation
            )
        else:
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
        op=op,
        batch=batch,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seq_len=seq_len,
        window=window,
        interleave=interleave,
        dtype="bfloat16",
        phase=phase,
    )
