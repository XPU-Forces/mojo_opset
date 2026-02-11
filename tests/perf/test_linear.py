import random

import pytest
import torch

from mojo_opset import MojoGroupLinear
from mojo_opset import MojoQuantGroupLinearReduceSum
from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented


def generate_random_list(length, total_sum):
    avg = total_sum // length
    lst = [0] * length
    for i in range(length):
        lst[i] = random.randint(0, 2 * int(avg))
    ratio = total_sum / sum(lst)
    lst = [int(x * ratio) for x in lst]

    diff = total_sum - sum(lst)
    lst[-1] += diff
    return torch.Tensor(lst).to(torch.int64)


@pytest.mark.parametrize(
    "input, weight, group_list",
    [
        (
            torch.randn(size=(8 * 2560, 4096), dtype=dtype),
            torch.randn(size=(8, 4096, 4096), dtype=dtype),
            generate_random_list(8, 8 * 2560),
        )
        for dtype in [torch.float16, torch.bfloat16]
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_group_gemm(input, weight, group_list):
    group_gemm = MojoGroupLinear(
        trans_weight=False,
        weight=weight,
    )

    perf(lambda: group_gemm(input, group_list))  # noqa: F821


def generate_quant_group_linear_reduce_sum_perf_data(b: int, m: int, k: int, n: int):
    x1 = torch.randint(-128, 128, (b, m, k), dtype=torch.int8)
    weight = torch.randint(-128, 128, (b, k, n), dtype=torch.int8)
    x1_scale = torch.rand(b, m, dtype=torch.float32)
    x2_scale = torch.rand(n, dtype=torch.bfloat16)
    return x1, weight, x1_scale, x2_scale


@pytest.mark.parametrize(
    "x1, weight, x1_scale, x2_scale",
    [
        generate_quant_group_linear_reduce_sum_perf_data(b, m, k, n)
        for b, m, k, n in [(8, 512, 128, 256), (4, 1024, 128, 512)]
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_quant_group_linear_reduce_sum_perf(x1, weight, x1_scale, x2_scale):
    op = MojoQuantGroupLinearReduceSum(trans_weight=False, weight=weight)

    def run():
        op(x1, x1_scale, x2_scale)

    perf(run)  # noqa: F821
