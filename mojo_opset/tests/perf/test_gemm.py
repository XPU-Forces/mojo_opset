import random

import pytest
import torch

from mojo_opset import MojoGroupGemm
from mojo_opset.tests.utils import auto_switch_platform
from mojo_opset.tests.utils import bypass_not_implemented


def generate_random_list(length, total_sum):
    avg = total_sum // length
    lst = [0] * length
    for i in range(length):
        lst[i] = random.randint(0, 2 * int(avg))
    ratio = total_sum / sum(lst)
    lst = [int(x * ratio) for x in lst]

    diff = total_sum - sum(lst)
    lst[-1] += diff
    return torch.Tensor(lst).to(torch.int32)


@pytest.mark.parametrize(
    "input, weight, group_list, trans_weight",
    [
        (
            torch.randn(size=(8 * 2560, 4096), dtype=dtype),
            torch.randn(size=(8, 4096, 4096), dtype=dtype),
            generate_random_list(8, 8 * 2560),
            False,
        )
        for dtype in [torch.float16, torch.bfloat16]
    ]
    + [
        (
            torch.randn(size=(4 * 1024, 2048), dtype=dtype),
            torch.randn(size=(4, 2048, 1024), dtype=dtype),
            generate_random_list(4, 4 * 1024),
            False,
        )
        for dtype in [torch.float16, torch.bfloat16]
    ]
    + [
        (
            torch.randn(size=(6 * 512, 1024), dtype=dtype),
            torch.randn(size=(6, 2048, 1024), dtype=dtype),
            generate_random_list(6, 6 * 512),
            True,
        )
        for dtype in [torch.float16, torch.bfloat16]
    ]
    + [
        pytest.param(
            torch.randn(size=(256, 128), dtype=dtype),
            torch.randn(size=(1, 128, 64), dtype=dtype),
            torch.tensor([256], dtype=torch.int32),
            False,
            id=f"single_group_fp={'bf16' if dtype is torch.bfloat16 else 'fp16'}",
        )
        for dtype in [torch.float16, torch.bfloat16]
    ]
    + [
        pytest.param(
            torch.randn(size=(192, 64), dtype=dtype),
            torch.randn(size=(4, 64, 96), dtype=dtype),
            torch.tensor([16, 64, 32, 80], dtype=torch.int32),
            False,
            id=f"uneven_groups_fp={'bf16' if dtype is torch.bfloat16 else 'fp16'}",
        )
        for dtype in [torch.float16, torch.bfloat16]
    ]
    + [
        pytest.param(
            torch.randn(size=(256, 128), dtype=dtype),
            torch.randn(size=(4, 96, 128), dtype=dtype),
            torch.tensor([48, 80, 64, 64], dtype=torch.int32),
            True,
            id=f"trans_weight_uneven_fp={'bf16' if dtype is torch.bfloat16 else 'fp16'}",
        )
        for dtype in [torch.float16, torch.bfloat16]
    ],
)
@bypass_not_implemented
@auto_switch_platform(set_perf=True)
def test_group_gemm(input, weight, group_list, trans_weight):
    group_gemm = MojoGroupGemm(
        trans_weight=trans_weight,
        weight=weight,
    )


    perf(lambda: group_gemm(input, group_list))  # noqa: F821