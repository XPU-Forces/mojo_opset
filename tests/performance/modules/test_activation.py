import pytest
import torch

from mojo_opset import modules


@pytest.mark.parametrize(
    "op,shape,dtype",
    [
        pytest.param(
            op,
            shape,
            dtype,
            marks=pytest.mark.api(
                {
                    "gelu": "modules.GELU",
                    "silu": "modules.SiLU",
                    "swiglu": "modules.SwiGLU",
                    "situ_and_mul": "modules.SituAndMul",
                }[op],
                ops=[op],
            ),
        )
        for op, shape, dtype in [
            ("gelu", (128, 128), torch.float32),
            ("gelu", (1024, 10240), torch.float32),
            *[("silu", shape, torch.float32) for shape in [(128, 128), (256, 128), (1024, 10240), (999, 9999)]],
            *[("swiglu", shape, torch.bfloat16) for shape in [(256, 128), (1024, 10240), (999, 9999)]],
            ("swiglu", (256, 128), torch.float32),
            # Kimi-K3 routed-expert prefill workload: 512 tokens * top-16 experts,
            # moe_intermediate_size=3072 -> concatenated [gate | up] width 6144.
            ("situ_and_mul", (8192, 6144), torch.bfloat16),
        ]
    ],
)
def test_activation(benchmark, perf_environment, op, shape, dtype):
    _, device, _, implementation = perf_environment
    # Kimi-K3 activation_situ_beta / activation_situ_linear_beta.
    options = {"beta": 4.0, "linear_beta": 25.0} if op == "situ_and_mul" else {}

    def factory():
        module = {
            "gelu": modules.GELU,
            "silu": modules.SiLU,
            "swiglu": modules.SwiGLU,
            "situ_and_mul": modules.SituAndMul,
        }[op](implementation=implementation, **options)
        x = torch.rand(shape, dtype=dtype, device=device)
        args = (x, torch.rand_like(x)) if op == "swiglu" else (x,)
        return lambda: module(*args)

    benchmark(factory=factory, op=op, shape=list(shape), dtype=str(dtype), phase="forward")
