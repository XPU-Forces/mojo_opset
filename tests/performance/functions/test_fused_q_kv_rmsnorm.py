"""fused_q_kv_rmsnorm / _quant performance cases (DSv4.1 production shapes)."""

import pytest
import torch

from mojo_opset import functions

Q_SIZE = 1280  # DSv4.1 TP4 q_lora_rank
KV_SIZE = 512  # DSv4.1 head_dim
TOKENS = [(1, "decode_1x"), (32, "decode_32x"), (256, "decode_256x"), (2048, "prefill_2k"), (8192, "prefill_8k")]


def _inputs(tokens, dtype, device):
    torch.manual_seed(0)
    q = torch.randn(tokens, Q_SIZE, dtype=torch.float32, device=device).to(dtype)
    kv = torch.randn(tokens, KV_SIZE, dtype=torch.float32, device=device).to(dtype)
    q_weight = torch.randn(Q_SIZE, dtype=torch.float32, device=device).to(dtype)
    kv_weight = torch.randn(KV_SIZE, dtype=torch.float32, device=device).to(dtype)
    return q, kv, q_weight, kv_weight


@pytest.mark.api("functions.fused_q_kv_rmsnorm")
@pytest.mark.parametrize("tokens,case", TOKENS, ids=[case for _, case in TOKENS])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_fused_q_kv_rmsnorm(benchmark, perf_environment, tokens, case, dtype):
    _, device, _, implementation = perf_environment
    inputs = _inputs(tokens, dtype, device)

    def factory():
        return lambda: functions.fused_q_kv_rmsnorm(
            *inputs, 1e-6, implementation=implementation
        )

    benchmark(
        factory=factory,
        op="fused_q_kv_rmsnorm",
        case=case,
        tokens=tokens,
        q_size=Q_SIZE,
        kv_size=KV_SIZE,
        dtype=str(dtype).split(".")[-1],
        phase="forward",
    )


@pytest.mark.api("functions.fused_q_kv_rmsnorm_quant")
@pytest.mark.parametrize("tokens,case", TOKENS, ids=[case for _, case in TOKENS])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_fused_q_kv_rmsnorm_quant(benchmark, perf_environment, tokens, case, dtype):
    _, device, target, implementation = perf_environment
    if str(target).startswith("npu.a2"):
        pytest.skip("910B (A2) has no FP8")
    inputs = _inputs(tokens, dtype, device)

    def factory():
        return lambda: functions.fused_q_kv_rmsnorm_quant(
            *inputs, 1e-20, implementation=implementation
        )

    benchmark(
        factory=factory,
        op="fused_q_kv_rmsnorm_quant",
        case=case,
        tokens=tokens,
        q_size=Q_SIZE,
        kv_size=KV_SIZE,
        dtype=str(dtype).split(".")[-1],
        phase="forward",
    )
