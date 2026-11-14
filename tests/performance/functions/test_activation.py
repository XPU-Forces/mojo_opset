import pytest
import torch

from mojo_opset import functions
from tests.performance._workload import training


@pytest.mark.parametrize(
    "op,shape,scaled",
    [
        pytest.param(op, shape, scaled, marks=pytest.mark.api("functions." + op, ops=[op]))
        for op, shape in [
            *[(op, shape) for op in ("silu", "gelu") for shape in ((128, 1024), (1024, 4096))],
            # Real expert widths; 1024 CI tokens * top-8 routes.
            ("swiglu", (8192, 1536)),
            ("swiglu", (8192, 2048)),
        ]
        for scaled in ([False, True] if op == "swiglu" else [False])
    ],
)
@pytest.mark.parametrize("phase", ["forward", "backward", "forward_backward"])
def test_activation(benchmark, perf_environment, op, shape, phase, scaled):
    _, device, _, implementation = perf_environment
    function = getattr(functions, op)

    def factory():
        x = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=phase != "forward")
        inputs = (x, torch.randn_like(x, requires_grad=phase != "forward")) if op == "swiglu" else (x,)
        if scaled:
            inputs += (torch.randn(shape[:-1], device=device, dtype=torch.float32, requires_grad=phase != "forward"),)
        return training(lambda: function(*inputs, implementation=implementation), inputs, phase)

    benchmark(factory=factory, op=op, shape=list(shape), dtype="bfloat16", phase=phase, scaled=scaled)


# Original Mojo perf + perf_new function matrices; keep dtype/distribution distinct.
@pytest.mark.api("functions.silu")
@pytest.mark.parametrize(
    "shape,dtype,distribution",
    [
        ((128, 128), torch.float32, "rand"),
        ((999, 9999), torch.float32, "rand"),
        ((1024, 10240), torch.float32, "rand"),
        ((1024, 1024), torch.float16, "randn"),
        ((4096, 4096), torch.float16, "randn"),
    ],
)
@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_silu(benchmark, perf_environment, shape, dtype, distribution, phase):
    _, device, _, implementation = perf_environment

    def factory():
        x = getattr(torch, distribution)(shape, dtype=dtype, device=device, requires_grad=True)
        call = lambda: functions.silu(x, implementation=implementation)
        return training(call, (x,), phase, grad_factory=torch.rand_like if distribution == "rand" else torch.randn_like)

    benchmark(factory=factory, op="silu", shape=list(shape), dtype=str(dtype), distribution=distribution, phase=phase)


# Kimi-K3 SituAndMul MLP workloads with activation_situ_beta=4.0 and
# activation_situ_linear_beta=25.0. Routed experts see tokens * top-16 rows of
# moe_intermediate_size=3072 once grouped; shared experts use 2 * 3072; dense
# KimiMLP layers use intermediate_size=33792. The linear_beta=None case covers
# the other compiled kernel variant on the routed-expert workload.
@pytest.mark.api("functions.situ_and_mul", ops=["situ_and_mul"])
@pytest.mark.parametrize(
    "shape,linear_beta",
    [
        ((16, 6144), 25.0),  # moe decode: 1 token * top-16 experts
        ((8192, 6144), 25.0),  # moe prefill: 512 tokens * top-16 experts
        ((8192, 6144), None),
        ((1024, 12288), 25.0),  # shared experts: 2 * 3072
        ((256, 67584), 25.0),  # dense mlp: intermediate_size=33792
    ],
)
@pytest.mark.parametrize("phase", ["forward", "backward", "forward_backward"])
def test_situ_and_mul(benchmark, perf_environment, shape, linear_beta, phase):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=phase != "forward")
        call = lambda: functions.situ_and_mul(x, beta=4.0, linear_beta=linear_beta, implementation=implementation)
        return training(call, (x,), phase)

    benchmark(
        factory=factory,
        op="situ_and_mul",
        shape=list(shape),
        dtype="bfloat16",
        phase=phase,
        beta=4.0,
        linear_beta=linear_beta,
    )
