import pytest
import torch

from mojo_opset import MojoGroupedMatmul, MojoGroupQuantMatmulReduceSum
from tests.utils import auto_switch_platform, bypass_not_implemented


# -------------------------------
# Grouped Matmul Performance
# -------------------------------

def _gen_grouped_inputs(dtype: torch.dtype):
    inputs = [
        torch.randn(1024, 2048, dtype=dtype),
        torch.randn(2048, 1024, dtype=dtype),
    ]
    weights = [
        torch.randn(2048, 4096, dtype=dtype),
        torch.randn(1024, 2048, dtype=dtype),
    ]
    bias = [
        torch.randn(4096, dtype=dtype),
        torch.randn(2048, dtype=dtype),
    ]
    return inputs, weights, bias


@pytest.mark.parametrize(
    "inputs, weights, bias, dtype",
    [
        (*_gen_grouped_inputs(dtype), dtype)
        for dtype in [torch.bfloat16, torch.float16, torch.float32]
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_grouped_matmul_perf(inputs, weights, bias, dtype):
    grouped = MojoGroupedMatmul()

    def run():
        grouped(
            inputs,
            weights,
            bias=bias,
            output_dtype=dtype,
        )

    perf(run)  # noqa: F821


# -----------------------------------------------
# Group Quant Matmul ReduceSum Performance
# -----------------------------------------------

def _gen_group_quant_inputs(b: int, m: int, k: int, n: int):
    x1 = torch.randint(-128, 128, (b, m, k), dtype=torch.int8)
    x2 = torch.randint(-128, 128, (b, k, n), dtype=torch.int8)
    x1_scale = torch.rand(b, m, dtype=torch.float32)
    x2_scale = torch.rand(n, dtype=torch.bfloat16)
    return x1, x2, x1_scale, x2_scale


@pytest.mark.parametrize(
    "x1, x2, x1_scale, x2_scale",
    [
        _gen_group_quant_inputs(b, m, k, n)
        for b, m, k, n in [(8, 512, 128, 256), (4, 1024, 128, 512)]
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_group_quant_matmul_reduce_sum_perf(x1, x2, x1_scale, x2_scale):
    op = MojoGroupQuantMatmulReduceSum()

    def run():
        op(x1, x2, x1_scale, x2_scale)

    perf(run)  # noqa: F821