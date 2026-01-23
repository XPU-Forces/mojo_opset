import random

import pytest
import torch

from mojo_opset import MojoGroupLinear
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
