import pytest
import torch

from mojo_opset import MojoGemmDequant
from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented


def _make_gemm_dequant_perf_data(m, k, n, output_dtype, trans_weight, has_bias):
    x_fp = torch.randn(m, k)
    x_scale = (x_fp.abs().amax(dim=-1) / 127).clamp(min=1e-12)
    x_i8 = torch.clamp(torch.round(x_fp / x_scale.unsqueeze(-1)), -128, 127).to(torch.int8)

    w_fp_nk = torch.randn(n, k)
    w_scale = (w_fp_nk.abs().amax(dim=-1) / 127).clamp(min=1e-12)
    w_i8_nk = torch.clamp(torch.round(w_fp_nk / w_scale.unsqueeze(-1)), -128, 127).to(torch.int8)

    if trans_weight:
        w_i8 = w_i8_nk
    else:
        w_i8 = w_i8_nk.t().contiguous()

    bias = torch.randn(n, dtype=output_dtype) if has_bias else None

    return x_i8, w_i8, x_scale, w_scale, bias


@pytest.mark.parametrize(
    "x_i8, w_i8, x_scale, w_scale, bias, output_dtype, trans_weight",
    [
        (*_make_gemm_dequant_perf_data(m, k, n, torch.bfloat16, False, False), torch.bfloat16, False)
        for m, k, n in [
            (1, 4096, 4096),
            (32, 4096, 4096),
            (128, 4096, 4096),
            (256, 4096, 4096),
            (512, 4096, 4096),
            (1024, 4096, 4096),
            (2048, 4096, 4096),
            (4096, 4096, 4096),
            (128, 4096, 11008),
            (1024, 8192, 4096),
            (4096, 8192, 4096),
        ]
    ],
)
@auto_switch_platform(set_perf=True)
@bypass_not_implemented
def test_gemm_dequant_perf(x_i8, w_i8, x_scale, w_scale, bias, output_dtype, trans_weight):
    op = MojoGemmDequant(output_dtype=output_dtype, trans_weight=trans_weight)

    perf(lambda: op(x_i8, w_i8, x_scale, w_scale, bias))  # noqa: F821
