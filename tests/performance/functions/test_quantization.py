import pytest
import torch

from mojo_opset import functions as F


@pytest.mark.api("functions.dynamic_quant")
@pytest.mark.parametrize("shape", [(128, 1536), (1024, 4096), (4096, 7168)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dynamic_quant(benchmark, perf_environment, shape, dtype):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randn(shape, dtype=dtype, device=device)
        smooth = (torch.rand(shape[-1], device=device) + 0.1).reciprocal()
        return lambda: F.dynamic_quant(x, smooth, implementation=implementation)

    benchmark(
        factory=factory,
        op="dynamic_quant",
        shape=list(shape),
        dtype=str(dtype),
        phase="forward",
    )


@pytest.mark.api("functions.static_quant")
@pytest.mark.parametrize("shape", [(128, 1536), (1024, 4096), (4096, 7168)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_static_quant(benchmark, perf_environment, shape, dtype):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randn(shape, dtype=dtype, device=device)
        scale = (x.float().abs().amax(dim=0) / 127).clamp(min=1e-10)
        return lambda: F.static_quant(x, scale, implementation=implementation)

    benchmark(
        factory=factory,
        op="static_quant",
        shape=list(shape),
        dtype=str(dtype),
        phase="forward",
    )


@pytest.mark.api("functions.dequant")
@pytest.mark.parametrize("shape", [(128, 1536), (1024, 4096), (4096, 7168)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dequant(benchmark, perf_environment, shape, dtype):
    _, device, _, implementation = perf_environment

    def factory():
        q = torch.randint(-128, 127, shape, dtype=torch.int8, device=device)
        scale = torch.rand(shape[-1], dtype=torch.float32, device=device) + 1e-4
        return lambda: F.dequant(q, scale, output_dtype=dtype, implementation=implementation)

    benchmark(
        factory=factory,
        op="dequant",
        shape=list(shape),
        dtype=str(dtype),
        phase="forward",
    )


@pytest.mark.api("functions.moe_dynamic_quant")
@pytest.mark.parametrize(
    "tokens,width,counts",
    [
        (256, 2048, [64, 64, 64, 64]),
        (1024, 4096, [128] * 8),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_dynamic_quant(benchmark, perf_environment, tokens, width, counts, dtype):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randn(tokens, width, dtype=dtype, device=device)
        token_count = torch.tensor(counts, dtype=torch.int32, device=device)
        inv_smooth_scale = (torch.rand(len(counts), width, device=device) + 0.1).reciprocal()
        return lambda: F.moe_dynamic_quant(
            x, token_count, inv_smooth_scale=inv_smooth_scale, implementation=implementation
        )

    benchmark(
        factory=factory,
        op="moe_dynamic_quant",
        tokens=tokens,
        width=width,
        experts=len(counts),
        dtype=str(dtype),
        phase="forward",
    )


@pytest.mark.api("functions.dequant_swiglu_quant")
@pytest.mark.parametrize(
    "tokens,width,counts",
    [
        (256, 1024, [64, 64, 64, 64]),
        (1024, 2048, [128] * 8),
    ],
)
def test_dequant_swiglu_quant(benchmark, perf_environment, tokens, width, counts):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randint(-1024, 1024, (tokens, width * 2), dtype=torch.int32, device=device)
        token_count = torch.tensor(counts, dtype=torch.int64, device=device)
        activation_scale = torch.rand(tokens, device=device, dtype=torch.float32)
        weight_scale = torch.rand(len(counts), width * 2, device=device, dtype=torch.float32)
        quant_scale = torch.rand(len(counts), width, device=device, dtype=torch.float32)
        return lambda: F.dequant_swiglu_quant(
            x,
            weight_scale=weight_scale,
            quant_scale=quant_scale,
            activation_scale=activation_scale,
            token_count=token_count,
            implementation=implementation,
        )

    benchmark(
        factory=factory,
        op="dequant_swiglu_quant",
        tokens=tokens,
        width=width,
        experts=len(counts),
        dtype="int32",
        phase="forward",
    )


@pytest.mark.parametrize(
    "op,residual",
    [
        pytest.param("rms_norm_quant", False, marks=pytest.mark.api("functions.rms_norm_quant")),
        pytest.param("layer_norm_quant", False, marks=pytest.mark.api("functions.layer_norm_quant")),
        pytest.param(
            "residual_add_rms_norm_quant", True, marks=pytest.mark.api("functions.residual_add_rms_norm_quant")
        ),
        pytest.param(
            "residual_add_layer_norm_quant", True, marks=pytest.mark.api("functions.residual_add_layer_norm_quant")
        ),
    ],
)
@pytest.mark.parametrize("shape", [(128, 1024), (1024, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_norm_quant(benchmark, perf_environment, op, residual, shape, dtype):
    _, device, _, implementation = perf_environment

    def factory():
        x = torch.randn(shape, dtype=dtype, device=device)
        res = torch.randn_like(x) if residual else None
        weight = torch.randn(shape[-1], dtype=torch.float32, device=device)
        bias = torch.randn(shape[-1], dtype=torch.float32, device=device) if "layer" in op else None
        fn = getattr(F, op)
        if residual and "layer" in op:
            return lambda: fn(x, res, weight, bias, implementation=implementation)
        if residual:
            return lambda: fn(x, res, weight, implementation=implementation)
        if "layer" in op:
            return lambda: fn(x, weight, bias, implementation=implementation)
        return lambda: fn(x, weight, implementation=implementation)

    benchmark(
        factory=factory,
        op=op,
        shape=list(shape),
        dtype=str(dtype),
        phase="forward",
    )
