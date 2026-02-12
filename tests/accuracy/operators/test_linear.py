import random

import pytest
import torch

from tests.utils import bypass_not_implemented

from mojo_opset import MojoGroupGemm


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
    "group_size, token_num, hidden_dim",
    [
        (8, 8 * 2560, 4096),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
)
@bypass_not_implemented
def test_group_gemm(group_size, token_num, hidden_dim, dtype):
    input = torch.randn(size=(token_num, hidden_dim), dtype=dtype)
    weight = torch.randn(size=(group_size, hidden_dim, hidden_dim), dtype=dtype)
    group_list = generate_random_list(group_size, token_num).to(input.device)

    group_gemm = MojoGroupGemm(
        trans_weight=False,
        weight=weight,
    )

    group_gemm_ref = MojoGroupGemm._registry.get("torch")(
        trans_weight=False,
        weight=weight,
    )
    group_gemm.forward_diff_with(group_gemm_ref, input, group_list, mixed_tol=True)
