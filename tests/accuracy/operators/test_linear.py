import random

import pytest
import torch

from tests.utils import bypass_not_implemented
from tests.utils import get_platform
from tests.utils import auto_switch_platform

from mojo_opset import MojoGroupLinear
from mojo_opset import MojoQuantGroupLinearReduceSum

_SKIP_NPU_QUANT_GROUP_LINEAR = get_platform() == "npu"


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


def generate_quant_group_linear_data(
    b: int,
    m: int,
    k: int,
    n: int,
    trans_weight: bool = False,
    x2_scale_dtype: torch.dtype = torch.bfloat16,
):
    x1 = torch.randint(-128, 128, (b, m, k), dtype=torch.int8)
    if trans_weight:
        weight = torch.randint(-128, 128, (b, n, k), dtype=torch.int8)
    else:
        weight = torch.randint(-128, 128, (b, k, n), dtype=torch.int8)

    x1_scale = torch.randn(b, m, dtype=torch.float32)
    x2_scale = torch.randn(n, dtype=torch.float32).to(x2_scale_dtype)
    return x1, weight, x1_scale, x2_scale


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
            torch.tensor([256], dtype=torch.int64),
            False,
            id=f"single_group_fp={'bf16' if dtype is torch.bfloat16 else 'fp16'}",
        )
        for dtype in [torch.float16, torch.bfloat16]
    ]
    + [
        pytest.param(
            torch.randn(size=(192, 64), dtype=dtype),
            torch.randn(size=(4, 64, 96), dtype=dtype),
            torch.tensor([16, 64, 32, 80], dtype=torch.int64),
            False,
            id=f"uneven_groups_fp={'bf16' if dtype is torch.bfloat16 else 'fp16'}",
        )
        for dtype in [torch.float16, torch.bfloat16]
    ]
    + [
        pytest.param(
            torch.randn(size=(256, 128), dtype=dtype),
            torch.randn(size=(4, 96, 128), dtype=dtype),
            torch.tensor([48, 80, 64, 64], dtype=torch.int64),
            True,
            id=f"trans_weight_uneven_fp={'bf16' if dtype is torch.bfloat16 else 'fp16'}",
        )
        for dtype in [torch.float16, torch.bfloat16]
    ],
)
@auto_switch_platform()
def test_group_gemm(input, weight, group_list, trans_weight):
    group_gemm = MojoGroupLinear(
        trans_weight=trans_weight,
        weight=weight,
    )

    group_gemm_ref = MojoGroupLinear._registry.get("torch")(
        trans_weight=trans_weight,
        weight=weight,
    )
    group_gemm.forward_diff_with(group_gemm_ref, input, group_list, mixed_tol=True)
    group_gemm.forward_diff_with(group_gemm_ref, input, group_list, mixed_tol=True)


@pytest.mark.parametrize(
    "x1, weight, x1_scale, x2_scale, trans_weight, atol, rtol",
    [
        pytest.param(
            *generate_quant_group_linear_data(b=4, m=7, k=128, n=256, trans_weight=False),
            False,
            1e-1,
            1e-2,
            id="basic_b4_m7_k128_n256",
        ),
        pytest.param(
            *generate_quant_group_linear_data(b=1, m=16, k=64, n=128, trans_weight=False),
            False,
            1e-1,
            1e-2,
            id="basic_b1_m16_k64_n128",
        ),
        pytest.param(
            *generate_quant_group_linear_data(b=2, m=9, k=256, n=512, trans_weight=False),
            False,
            1e-1,
            1e-2,
            id="basic_b2_m9_k256_n512",
        ),
        pytest.param(
            *generate_quant_group_linear_data(b=4, m=31, k=128, n=256, trans_weight=True),
            True,
            1e-1,
            1e-2,
            id="trans_weight_b4_m31_k128_n256",
        ),
        pytest.param(
            *generate_quant_group_linear_data(b=8, m=1, k=128, n=128, trans_weight=False, x2_scale_dtype=torch.float16),
            False,
            1e-1,
            1e-2,
            id="x2_scale_fp16_cast",
        ),
    ],
)
@pytest.mark.skipif(_SKIP_NPU_QUANT_GROUP_LINEAR, reason="Skipped on NPU due to CANN 8.2 issue")
@auto_switch_platform()
@bypass_not_implemented
def test_quant_group_linear_reduce_sum(x1, weight, x1_scale, x2_scale, trans_weight, atol, rtol):
    quant_linear = MojoQuantGroupLinearReduceSum(
        trans_weight=trans_weight,
        weight=weight,
    )
    quant_linear_ref = MojoQuantGroupLinearReduceSum._registry.get("torch")(
        trans_weight=trans_weight,
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
    if device == "npu" and dtype == torch.float32:
        pytest.skip("NPU grouped matmul does not support float32")

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


@pytest.mark.parametrize(
    "dtype, trans_weight",
    [
        (torch.float16, False),
        (torch.bfloat16, False),
        (torch.float16, True),
        (torch.bfloat16, True),
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_group_linear_two_groups_single_call(dtype, trans_weight):
    device = get_platform()

    m0, m1 = 64, 128
    k, n = 128, 96

    x0 = torch.randn(m0, k, dtype=dtype, device=device)
    x1 = torch.randn(m1, k, dtype=dtype, device=device)
    x = torch.cat([x0, x1], dim=0)

    if trans_weight:
        w0 = torch.randn(n, k, dtype=dtype, device=device)
        w1 = torch.randn(n, k, dtype=dtype, device=device)
        weight = torch.stack([w0, w1], dim=0)
        ref = torch.cat([x0 @ w0.t(), x1 @ w1.t()], dim=0)
    else:
        w0 = torch.randn(k, n, dtype=dtype, device=device)
        w1 = torch.randn(k, n, dtype=dtype, device=device)
        weight = torch.stack([w0, w1], dim=0)
        ref = torch.cat([x0 @ w0, x1 @ w1], dim=0)

    group_list = torch.tensor([m0, m1], device=device, dtype=torch.int64)

    op = MojoGroupLinear(weight=weight, trans_weight=trans_weight)
    out = op(x, group_list)

    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), atol=1e-3, rtol=1e-3)
