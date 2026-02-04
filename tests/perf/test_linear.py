import random

import pytest
import torch

from mojo_opset import MojoDequantGroupLinear
from mojo_opset import MojoGroupLinear
from mojo_opset import MojoQuantGroupLinear
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


def generate_quant_group_linear_perf_data(b: int, m: int, k: int, n: int):
    x1 = torch.randint(-128, 128, (b, m, k), dtype=torch.int8)
    weight = torch.randint(-128, 128, (b, k, n), dtype=torch.int8)
    x1_scale = torch.rand(b, m, dtype=torch.float32)
    x2_scale = torch.rand(n, dtype=torch.bfloat16)
    return x1, weight, x1_scale, x2_scale


@pytest.mark.parametrize(
    "x1, weight, x1_scale, x2_scale",
    [
        generate_quant_group_linear_perf_data(b, m, k, n)
        for b, m, k, n in [(8, 512, 128, 256), (4, 1024, 128, 512)]
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_quant_group_linear_perf(x1, weight, x1_scale, x2_scale):
    op = MojoQuantGroupLinear(trans_weight=False, weight=weight)

    def run():
        op(x1, x1_scale, x2_scale)

    perf(run)  # noqa: F821


def generate_dequant_group_linear_perf_data(num_groups: int, total_m: int, k: int, n: int):
    input = torch.randn(total_m, k, dtype=torch.float16)
    weight = torch.randint(-128, 128, (num_groups, k, n), dtype=torch.int8)
    group_list = generate_random_list(num_groups, total_m)
    antiquant_scale = [torch.rand(k, n, dtype=torch.float32) for _ in range(num_groups)]
    antiquant_offset = [torch.rand(k, n, dtype=torch.float32) for _ in range(num_groups)]
    return input, weight, group_list, antiquant_scale, antiquant_offset


@pytest.mark.parametrize(
    "input, weight, group_list, antiquant_scale, antiquant_offset",
    [
        generate_dequant_group_linear_perf_data(num_groups=4, total_m=1024, k=128, n=256),
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_dequant_group_linear_perf(input, weight, group_list, antiquant_scale, antiquant_offset):
    op = MojoDequantGroupLinear(trans_weight=False, weight=weight)

    def run():
        op(input, group_list, antiquant_scale, antiquant_offset)

    perf(run)  # noqa: F821
