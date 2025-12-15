import pytest
import torch

from tests.utils import auto_switch_platform, bypass_not_implemented

from mojo_opset import MojoGroupQuantMatmulReduceSum


def generate_group_quant_matmul_reduce_sum_data(
    b: int, m: int, k: int, n: int,
):
    x1 = torch.randint(-128, 128, (b, m, k), dtype=torch.int8)
    x2 = torch.randint(-128, 128, (b, k, n), dtype=torch.int8)
    x1_scale = torch.randn(b, m, dtype=torch.float32)
    x2_scale = torch.randn(n, dtype=torch.bfloat16)

    return x1, x2, x1_scale, x2_scale

test_configs = [
    (4, 7, 2048, 5120),
]

@pytest.mark.parametrize(
    "x1, x2, x1_scale, x2_scale, atol, rtol",
    [
        pytest.param(
            *generate_group_quant_matmul_reduce_sum_data(b=B, m=M, k=K, n=N,),
            1e-1, 1e-2,
        )
        for B, M, K, N in test_configs
    ],
)
@auto_switch_platform()
@bypass_not_implemented
def test_group_quant_matmul_reduce_sum(x1, x2, x1_scale, x2_scale, atol, rtol):
    matmul = MojoGroupQuantMatmulReduceSum()

    matmul.forward_diff(x1, x2, x1_scale, x2_scale, atol=atol, rtol=rtol)
