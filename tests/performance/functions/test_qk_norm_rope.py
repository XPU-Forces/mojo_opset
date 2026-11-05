"""qk_norm_rope performance cases (task shape matrix: 32..69888 tokens, 8/56/57 heads)."""

import pytest
import torch

from mojo_opset import functions


@pytest.mark.api("functions.qk_norm_rope")
@pytest.mark.parametrize(
    "tokens,heads",
    [(32, 8), (320, 8), (320, 56), (1024, 56), (4096, 56), (16384, 56), (69888, 56), (4096, 57)],
)
def test_qk_norm_rope(benchmark, perf_environment, tokens, heads):
    _, device, _, implementation = perf_environment

    def factory():
        torch.manual_seed(42)
        q = torch.randn(tokens, heads, 128, dtype=torch.bfloat16, device=device) * 0.3
        k = torch.randn_like(q)
        qw = torch.randn(128, dtype=torch.bfloat16, device=device) * 0.1 + 1
        kw = torch.randn_like(qw)
        cos = torch.rand(tokens, 48, dtype=torch.bfloat16, device=device) * 2 - 1
        sin = torch.sqrt((1 - cos.float() ** 2).clamp(0, 1)).to(torch.bfloat16)
        table = torch.cat((cos, sin), dim=-1).contiguous()
        return lambda: functions.qk_norm_rope(q, k, qw, kw, table, 1e-6, implementation=implementation)

    benchmark(
        factory=factory,
        op="qk_norm_rope",
        tokens=tokens,
        heads=heads,
        head_dim=128,
        rotary_dim=96,
        dtype="bfloat16",
        phase="forward",
    )
