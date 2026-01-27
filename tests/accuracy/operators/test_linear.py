import random
import os
import pytest
import torch

from mojo_opset.utils.platform import get_platform
from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented

from mojo_opset import MojoLinear, MojoGroupLinear

dtype_str_map = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}


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
    "batch_size, M, N, K, dtype",
    [
        (
            batch_size,
            M,
            N,
            K,
            dtype,
        )
        for M, N, K in [(1024, 128, 7168),]
        for batch_size in [1, 8]
        for dtype in ["bfloat16", "float16", "float32"]
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_gemm(batch_size, M, N, K, dtype):
    device = get_platform()
    map_tol = {
        "bfloat16": (1.6e-2, 1e-5, 1.0),
        "float16": (1e-3, 1e-5, 1.0),
        "float32": (1.3e-6, 1e-5, 1.0),
    }
    if device == 'npu':
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
    atol, rtol, ptol = map_tol[dtype]
    dtype = dtype_str_map[dtype]

    x = torch.randn(batch_size, M, K, device=device, dtype=dtype) # BNSD or TND
    weight = torch.randn(N, K, device=device, dtype=dtype)

    gemm = MojoLinear(
        weight=weight,
    )

    gemm_ref = MojoLinear._registry.get("torch")(
        weight=weight,
    )

    gemm.forward_diff_with(gemm_ref, x, atol=atol, rtol=rtol, ptol=ptol)


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


# if __name__ == "__main__":
#     pass
#     pytest.main(["-s", "-v", "tests/accuracy/operators/test_linear.py::test_gemm"])
