import random

import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoGroupLinear
from mojo_opset import MojoQuantGroupLinearReduceSum
from tests.utils import get_platform


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


def generate_quant_group_linear_data(b: int, m: int, k: int, n: int):
    x1 = torch.randint(-128, 128, (b, m, k), dtype=torch.int8)
    weight = torch.randint(-128, 128, (b, k, n), dtype=torch.int8)
    x1_scale = torch.randn(b, m, dtype=torch.float32)
    x2_scale = torch.randn(n, dtype=torch.bfloat16)
    return x1, weight, x1_scale, x2_scale


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
@auto_switch_platform()
@bypass_not_implemented
def test_group_gemm(input, weight, group_list):
    group_gemm = MojoGroupLinear(
        trans_weight=False,
        weight=weight,
    )

    group_gemm_ref = MojoGroupLinear._registry.get("torch")(
        trans_weight=False,
        weight=weight,
    )
    group_gemm.forward_diff_with(group_gemm_ref, input, group_list, mixed_tol=True)


@pytest.mark.parametrize(
    "x1, weight, x1_scale, x2_scale, atol, rtol",
    [
        pytest.param(
            *generate_quant_group_linear_data(b=4, m=7, k=128, n=256),
            1e-1,
            1e-2,
        )
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_quant_group_linear_reduce_sum(x1, weight, x1_scale, x2_scale, atol, rtol):
    quant_linear = MojoQuantGroupLinearReduceSum(
        trans_weight=False,
        weight=weight,
    )
    quant_linear_ref = MojoQuantGroupLinearReduceSum._registry.get("torch")(
        trans_weight=False,
        weight=weight,
    )
    quant_linear.forward_diff_with(quant_linear_ref, x1, x1_scale, x2_scale, atol=atol, rtol=rtol)


_test_grouped_matmul_cases = [
    (
        [torch.randn(16, 32), torch.randn(8, 16)],
        [torch.randn(32, 64), torch.randn(16, 32)],
        None,
        torch.float32,
    ),
    (
        [torch.randn(3, 4, dtype=torch.float16), torch.randn(5, 4, dtype=torch.float16)],
        [torch.randn(4, 6, dtype=torch.float16), torch.randn(4, 6, dtype=torch.float16)],
        None,
        torch.float16,
    ),
    (
        [torch.randn(10, 4, dtype=torch.bfloat16)],
        [torch.randn(4, 6, dtype=torch.bfloat16), torch.randn(4, 6, dtype=torch.bfloat16)],
        None,
        torch.bfloat16,
    ),
]


@pytest.mark.parametrize("inputs, weights, bias, dtype", _test_grouped_matmul_cases)
@auto_switch_platform()
@bypass_not_implemented
def test_grouped_matmul_cases_via_group_linear(inputs, weights, bias, dtype):
    device = get_platform()

    input_tensors = [t.to(device=device) for t in inputs]
    weight_tensors = [t.to(device=device) for t in weights]

    outputs = []
    for x, w in zip(input_tensors, weight_tensors):
        group_list = torch.tensor([x.shape[0]], device=device, dtype=torch.int64)
        weight_group = w.unsqueeze(0)
        op = MojoGroupLinear(weight=weight_group, trans_weight=False)
        out = op(x, group_list)
        outputs.append(out)

    for x, w, out in zip(input_tensors, weight_tensors, outputs):
        ref = x @ w
        torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=1e-3, rtol=1e-3)

