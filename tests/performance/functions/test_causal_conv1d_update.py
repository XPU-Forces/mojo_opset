import pytest
import torch

from mojo_opset import functions

# Shapes from the A5/A2 profiling set; every case stays above the 15us floor on
# both platforms. ``cache`` is the rotation state ``[W, N, D]``.
CASES = [
    (4096, 1024, 4, torch.bfloat16),
    (2048, 2048, 4, torch.bfloat16),
    (2048, 2048, 4, torch.float16),
    (1024, 4096, 4, torch.bfloat16),
    (1024, 4096, 4, torch.float16),
    (512, 8192, 4, torch.float16),
    (4096, 2048, 2, torch.float16),
    (8192, 2048, 4, torch.bfloat16),
    (4096, 4096, 4, torch.float16),
    (2048, 8192, 4, torch.bfloat16),
    (8192, 2048, 2, torch.float16),
]

_DTYPE_NAME = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}


def _case_id(case):
    tokens, channels, width, dtype = case
    return f"N{tokens}_D{channels}_W{width}_{_DTYPE_NAME[dtype]}"


@pytest.mark.api("functions.causal_conv1d_update")
@pytest.mark.parametrize("tokens,channels,width,dtype", CASES, ids=[_case_id(c) for c in CASES])
def test_causal_conv1d_update(benchmark, perf_environment, tokens, channels, width, dtype):
    _, device, target, implementation = perf_environment
    if implementation not in (None, "triton") or not target.startswith("npu."):
        pytest.skip("causal_conv1d_update has only an NPU Triton provider")

    def factory():
        x = torch.randn(tokens, channels, dtype=dtype, device=device)
        cache = torch.randn(width, tokens, channels, dtype=dtype, device=device)
        weight = torch.randn(channels, width, dtype=dtype, device=device)
        bias = torch.randn(channels, dtype=dtype, device=device)

        def call():
            return functions.causal_conv1d_update(
                x,
                cache,
                weight=weight,
                bias=bias,
                activation="silu",
                implementation=implementation,
            )

        return call

    benchmark(
        factory=factory,
        op="causal_conv1d_update",
        shape=[tokens, channels],
        width=width,
        activation="silu",
        dtype=_DTYPE_NAME[dtype],
    )
