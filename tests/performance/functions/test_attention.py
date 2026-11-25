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


@pytest.mark.api("functions.flash_attention_infer")
@pytest.mark.parametrize("batch,q_heads,kv_heads,seq_len", [(1, 48, 8, 1024), (1, 56, 8, 1024)])
def test_flash_attention_infer(benchmark, perf_environment, batch, q_heads, kv_heads, seq_len):
    _, device, _, implementation = perf_environment

    def factory():
        q = torch.randn(batch * seq_len, q_heads, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch * seq_len, kv_heads, 128, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cu = torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=device)
        return lambda: functions.flash_attention_infer(
            q, k, v, cu, cu, seq_len, seq_len, causal=True, implementation=implementation
        )

    benchmark(
        factory=factory,
        op="flash_attention_infer",
        batch=batch,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seq_len=seq_len,
        dtype="bfloat16",
        phase="forward",
    )


@pytest.mark.api("functions.native_swa_infer")
@pytest.mark.parametrize(
    "q_heads,interleave,window,q_lens,k_lens",
    [
        (16, False, 1023, [129, 257], [513, 1025]),
        (8, False, 1023, [1024], [4096]),
    ],
)
def test_native_swa_infer(benchmark, perf_environment, q_heads, interleave, window, q_lens, k_lens):
    _, device, _, implementation = perf_environment
    kv_heads = 4 if q_heads == 16 else 1

    def factory():
        cu_q = torch.tensor([0, *q_lens], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
        cu_k = torch.tensor([0, *k_lens], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
        q = torch.randn(sum(q_lens), q_heads, 128, dtype=torch.bfloat16, device=device)
        k = torch.randn(sum(k_lens), kv_heads, 128, dtype=torch.bfloat16, device=device)
        v = torch.randn_like(k)
        return lambda: functions.native_swa_infer(
            q,
            k,
            v,
            cu_q,
            cu_k,
            local_window_size=window,
            global_window_size=4,
            gqa_interleave=interleave,
            implementation=implementation,
        )

    benchmark(
        factory=factory,
        op="native_swa_infer",
        q_heads=q_heads,
        kv_heads=kv_heads,
        interleave=interleave,
        window=window,
        q_lens=q_lens,
        k_lens=k_lens,
        dtype="bfloat16",
        phase="forward",
    )

