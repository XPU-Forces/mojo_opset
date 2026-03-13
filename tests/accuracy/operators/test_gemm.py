import pytest
import torch
import torch.nn.functional as F

from tests.utils import bypass_not_implemented

from mojo_opset import MojoLinear


@pytest.mark.parametrize(
    "m, k, n",
    [
        (1024, 4096, 4096),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
)
@pytest.mark.parametrize(
    "bias",
    [True, False],
)
@bypass_not_implemented
def test_gemm(m, k, n, dtype, bias):
    input = torch.randn(size=(m, k), dtype=dtype)

    gemm = MojoLinear(k, n, bias=bias, dtype=dtype)
    gemm_ref = MojoLinear._registry.get("torch")(
        k,
        n,
        bias=bias,
        dtype=dtype,
    )
    gemm_ref.load_state_dict(gemm.state_dict())

    gemm.forward_diff_with(gemm_ref, input, mixed_tol=True)
    torch_out = F.linear(input, gemm.weight, gemm.bias)
    mojo_out = gemm(input)
    torch.testing.assert_close(mojo_out, torch_out)
