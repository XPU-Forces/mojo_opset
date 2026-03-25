import pytest
import torch
import torch.nn.functional as F

from tests.utils import auto_switch_platform, bypass_not_implemented

from mojo_opset import MojoGemmDequant, MojoLinear

torch.manual_seed(42)

dtypes = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize(
    "m, k, n",
    [
        (1024, 4096, 4096),
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("bias", [True, False])
@bypass_not_implemented
def test_gemm(m, k, n, dtype, bias):
    input = torch.randn(size=(m, k), dtype=dtype)

    gemm = MojoLinear(k, n, bias=bias, dtype=dtype)
    gemm_ref = MojoLinear._registry.get("torch")(k, n, bias=bias, dtype=dtype)
    gemm_ref.load_state_dict(gemm.state_dict())

    gemm.forward_diff_with(gemm_ref, input, mixed_tol=True)
    torch_out = F.linear(input, gemm.weight, gemm.bias)
    mojo_out = gemm(input)
    torch.testing.assert_close(mojo_out, torch_out)


# ===========================================================================
# MojoGemmDequant
# ===========================================================================

def _make_int8_gemm_data(m, k, n, output_dtype, trans_weight, has_bias):
    """Create quantised input/weight pairs with corresponding scales.

    Returns weight in (N, K) layout when trans_weight=True, (K, N) otherwise.
    All tensors are contiguous.
    """
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
    "m, k, n",
    [
        (1, 4096, 4096),
        (32, 4096, 11008),
        (128, 2048, 4096),
    ],
)
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("trans_weight", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@bypass_not_implemented
def test_gemm_dequant(m, k, n, output_dtype, trans_weight, has_bias):
    """Verify torch reference against manual calculation (exact match)."""
    x_i8, w_i8, x_scale, w_scale, bias = _make_int8_gemm_data(
        m, k, n, output_dtype, trans_weight, has_bias,
    )

    op = MojoGemmDequant._registry.get("torch")(
        output_dtype=output_dtype, trans_weight=trans_weight,
    )
    out = op(x_i8, w_i8, x_scale, w_scale, bias)

    w_for_mm = w_i8.t().contiguous() if trans_weight else w_i8
    ref = (x_i8.float() @ w_for_mm.float()) * x_scale.unsqueeze(-1) * w_scale.unsqueeze(0)
    if bias is not None:
        ref = ref + bias.float()
    ref = ref.to(output_dtype)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@bypass_not_implemented
def test_gemm_dequant_no_bias():
    """Verify torch reference output without bias (exact match)."""
    m, k, n = 64, 512, 256
    x_i8, w_i8, x_scale, w_scale, _ = _make_int8_gemm_data(
        m, k, n, torch.bfloat16, False, False,
    )

    op = MojoGemmDequant._registry.get("torch")(output_dtype=torch.bfloat16)
    out = op(x_i8, w_i8, x_scale, w_scale)

    ref = (x_i8.float() @ w_i8.float()) * x_scale.unsqueeze(-1) * w_scale.unsqueeze(0)
    ref = ref.to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


# ===========================================================================
# MojoGemmDequant — backend vs torch reference
# ===========================================================================

@pytest.mark.parametrize(
    "m, k, n",
    [
        (1, 4096, 4096),
        (32, 4096, 11008),
        (128, 2048, 4096),
    ],
)
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("trans_weight", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@bypass_not_implemented
@auto_switch_platform()
def test_gemm_dequant_backend(m, k, n, output_dtype, trans_weight, has_bias):
    """Compare active backend against the torch reference via forward_diff_with."""
    x_i8, w_i8, x_scale, w_scale, bias = _make_int8_gemm_data(
        m, k, n, output_dtype, trans_weight, has_bias,
    )

    op = MojoGemmDequant(output_dtype=output_dtype, trans_weight=trans_weight)
    op_ref = MojoGemmDequant._registry.get("torch")(
        output_dtype=output_dtype, trans_weight=trans_weight,
    )
    op.forward_diff_with(op_ref, x_i8, w_i8, x_scale, w_scale, bias, mixed_tol=True)
