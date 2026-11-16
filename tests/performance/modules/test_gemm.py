import random

import pytest
import torch

from mojo_opset import Target
from mojo_opset import modules
from mojo_opset.functions._dispatch import load_impl
from mojo_opset.functions._dispatch import resolve_implementation
from tests.performance._workload import training


@pytest.mark.api("modules.GroupGemm", ops=["group_gemm"])
@pytest.mark.parametrize(
    "groups,rows,k,n,trans_weight,distribution",
    [
        (2, 64, 128, 128, False, "uniform"),
        (8, 20480, 4096, 4096, False, "uniform"),
        (8, 20480, 4096, 4096, False, "random"),
        (4, 4096, 2048, 1024, False, "random"),
        (6, 3072, 1024, 2048, True, "random"),
        (1, 256, 128, 64, False, "uniform"),
        (4, 192, 64, 96, False, "uneven"),
        (4, 256, 128, 96, True, "uneven"),
        (129, 2048, 128, 96, False, "empty"),
        (129, 2048, 128, 96, True, "empty"),
        # 384 experts / EP8 = 48, 1024 CI tokens * top-8 = 8192 rows.
        *[(48, 8192, k, n, True, "empty") for k, n in [(5120, 3072), (1536, 5120), (7168, 4096), (2048, 7168)]],
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("phase", ["forward", "backward", "forward_backward"])
def test_group_gemm(benchmark, perf_environment, groups, rows, k, n, trans_weight, distribution, dtype, phase):
    _, device, target, implementation = perf_environment
    backward = phase != "forward"
    if backward and load_impl("group_gemm", implementation, target=target)[1] is None:
        pytest.skip("group_gemm backward is not implemented by this backend")
    counts = [rows // groups] * groups
    counts[-1] += rows - sum(counts)
    if distribution == "random":
        rng = random.Random(42)
        values = [rng.randint(0, 2 * rows // groups) for _ in range(groups)]
        counts = [int(value * rows / sum(values)) for value in values]
        counts[-1] += rows - sum(counts)
    elif distribution == "uneven":
        counts = [48, 80, 64, 64] if trans_weight else [16, 64, 32, 80]

    if distribution == "empty":
        counts = [0] + [rows // (groups - 1)] * (groups - 1)
        counts[-1] += rows - sum(counts)

    def factory():
        weight = torch.randn(
            (groups, n, k) if trans_weight else (groups, k, n), dtype=dtype, device=device, requires_grad=backward
        )
        module = modules.GroupGemm(weight, trans_weight=trans_weight, implementation=implementation)
        x = torch.randn(rows, k, dtype=dtype, device=device, requires_grad=backward)
        group_sizes = torch.tensor(counts, dtype=torch.int32, device=device)
        return training(lambda: module(x, group_sizes), (x, weight), phase)

    benchmark(
        factory=factory,
        op="group_gemm",
        groups=groups,
        rows=rows,
        k=k,
        n=n,
        trans_weight=trans_weight,
        distribution=distribution,
        dtype=str(dtype),
        phase=phase,
    )


@pytest.mark.api("modules.QuantGemm", ops=["quant_gemm"])
@pytest.mark.parametrize(
    "m,k,n",
    [
        (16, 1024, 1024),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        *[(m, 4096, 4096) for m in [1, 32, 128, 256, 512, 1024, 2048]],
        (128, 4096, 11008),
        (1024, 8192, 4096),
        (4096, 8192, 4096),
    ],
)
@pytest.mark.parametrize("distribution", ["zeros", "quantized_normal"])
def test_quant_gemm(benchmark, perf_environment, m, k, n, distribution):
    _, device, _, implementation = perf_environment

    def factory():
        module = modules.QuantGemm(k, n, device=device, implementation=implementation)
        module.weight.zero_()
        module.weight_scale.fill_(1)
        x = torch.zeros(m, k, dtype=torch.int8, device=device)
        scale = torch.ones(m, dtype=torch.float32, device=device)
        if distribution == "quantized_normal":
            source = torch.randn(m, k, device=device)
            scale = source.abs().amax(dim=1).clamp_min(1e-12) / 127
            x = (source / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
            weight = torch.randn(n, k, device=device)
            weight_scale = weight.abs().amax(dim=1).clamp_min(1e-12) / 127
            module.weight.copy_((weight / weight_scale[:, None]).round().clamp(-128, 127).to(torch.int8).T)
            module.weight_scale.copy_(weight_scale)
        return lambda: module(x, scale)

    benchmark(
        factory=factory,
        op="quant_gemm",
        m=m,
        k=k,
        n=n,
        distribution=distribution,
        dtype="int8",
        output_dtype="bfloat16",
        phase="forward",
    )


@pytest.mark.api("modules.QuantBatchGemmReduceSum", ops=["quant_batch_gemm_reduce_sum"])
@pytest.mark.parametrize("batch,m,k,n", [(8, 512, 128, 256), (4, 1024, 128, 512)])
def test_quant_batch_gemm_reduce_sum(benchmark, perf_environment, batch, m, k, n):
    _, device, target, implementation = perf_environment
    selected = implementation or resolve_implementation("quant_batch_gemm_reduce_sum", Target.parse(target))
    if selected != "torch":
        pytest.skip("Original master NPU quantized batch reduction is disabled")

    def factory():
        weight = torch.randint(-128, 128, (batch, k, n), dtype=torch.int8, device=device)
        module = modules.QuantBatchGemmReduceSum(weight, implementation=implementation)
        x = torch.randint(-128, 128, (batch, m, k), dtype=torch.int8, device=device)
        xs = torch.rand(batch, m, dtype=torch.float32, device=device)
        ws = torch.rand(n, dtype=torch.bfloat16, device=device)
        return lambda: module(x, xs, ws)

    benchmark(
        factory=factory, op="quant_batch_gemm_reduce_sum", batch=batch, m=m, k=k, n=n, dtype="int8", phase="forward"
    )
