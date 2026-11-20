import pytest
import torch

from mojo_opset import functions


@pytest.mark.api("functions.fused_qk_norm_rope")
@pytest.mark.parametrize(
    "tokens,heads", [(32, 8), (320, 8), (320, 56), (1024, 56), (4096, 56), (16384, 56), (69888, 56), (4096, 57)]
)
def test_fused_qk_norm_rope(benchmark, perf_environment, tokens, heads):
    _, device, _, implementation = perf_environment

    def factory():
        packed = torch.randn(tokens, heads * 3, 128, device=device, dtype=torch.bfloat16)
        q = packed[:, :heads]
        k = packed[:, heads : heads * 2]
        q_weight = torch.randn(128, device=device, dtype=torch.bfloat16)
        k_weight = torch.randn(128, device=device, dtype=torch.bfloat16)
        angles = torch.randn(tokens, 48, device=device)
        rope_table = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)

        def call():
            return functions.fused_qk_norm_rope(
                q,
                k,
                q_weight,
                k_weight,
                rope_table,
                1e-5,
                implementation=implementation,
            )

        return call

    benchmark(
        factory=factory,
        op="fused_qk_norm_rope",
        tokens=tokens,
        heads=heads,
        head_dim=128,
        rotary_dim=96,
        dtype="bfloat16",
        phase="forward",
    )
