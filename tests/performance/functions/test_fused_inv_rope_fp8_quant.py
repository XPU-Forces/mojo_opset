"""fused_inv_rope_fp8_quant performance cases (DeepSeek V4.1 attention output)."""

import pytest
import torch

from mojo_opset import functions

N_GROUPS = 16
HEADS_PER_GROUP = 8
HEADS = N_GROUPS * HEADS_PER_GROUP
NOPE_DIM, ROPE_DIM, QUANT_GROUP_SIZE = 448, 64, 128
MAX_POS = 4096
# (tokens, case): decode batches and prefill chunks at the DSv4.1 production geometry.
TOKENS = [
    (1, "decode_1x"),
    (33, "decode_33x"),
    (256, "decode_256x"),
    (2048, "prefill_2k"),
    (8192, "prefill_8k"),
]


def _cache(device):
    half = ROPE_DIM // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    freqs = torch.outer(torch.arange(MAX_POS, dtype=torch.float32, device=device), inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


@pytest.mark.api("functions.fused_inv_rope_fp8_quant")
@pytest.mark.parametrize("tokens,case", TOKENS, ids=[case for _, case in TOKENS])
@pytest.mark.parametrize("quantize", [True, False])
def test_fused_inv_rope_fp8_quant(benchmark, perf_environment, tokens, case, quantize):
    _, device, target, implementation = perf_environment
    if quantize and str(target).startswith("npu.a2"):
        pytest.skip("910B (A2) has no FP8")
    torch.manual_seed(0)
    o = torch.randn(tokens, HEADS, NOPE_DIM + ROPE_DIM, dtype=torch.float32, device=device).to(torch.bfloat16)
    positions = torch.randint(0, MAX_POS, (tokens,), dtype=torch.int64, device=device)
    cache = _cache(device)

    def factory():
        return lambda: functions.fused_inv_rope_fp8_quant(
            o,
            positions,
            cache,
            N_GROUPS,
            HEADS_PER_GROUP,
            NOPE_DIM,
            ROPE_DIM,
            QUANT_GROUP_SIZE,
            False,
            quantize,
            implementation=implementation,
        )

    benchmark(
        factory=factory,
        op="fused_inv_rope_fp8_quant",
        case=case,
        tokens=tokens,
        heads=HEADS,
        quantize=quantize,
        dtype="bfloat16",
        phase="forward",
    )
