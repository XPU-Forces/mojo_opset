import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_bitwise_equal
from tests._checks import assert_close

# (M, N, is_rms_norm, has_bias, z_kind, norm_before_gate, activation, group_size)
CASES = [
    (1, 128, True, False, "none", True, "swish", None),
    (128, 128, True, False, "2d", True, "swish", None),
    (128, 128, True, False, "3d", True, "swish", None),
    (128, 128, True, False, "3d", False, "sigmoid", None),
    (128, 128, False, True, "3d", True, "sigmoid", None),
    (132, 128, True, True, "3d", True, "swish", None),
    (132, 128, False, True, "2d", False, "sigmoid", None),
    (512, 1024, False, True, "none", True, "swish", None),
    (512, 1024, True, False, "2d", False, "silu", None),
    (128, 256, True, False, "2d", False, "sigmoid", 64),
    (128, 256, False, True, "2d", False, "swish", 64),
    (512, 256, True, False, "none", True, "swish", 64),
    (129, 192, True, False, "none", True, "swish", None),
    (129, 192, True, False, "2d", True, "swish", None),
]
CASE_IDS = [
    f"m{M}-n{N}-{'rms' if rms else 'ln'}{'-bias' if bias else ''}-z{z_kind}"
    f"-{'before' if before else 'after'}-{activation}{'-g' + str(group) if group else ''}"
    for M, N, rms, bias, z_kind, before, activation, group in CASES
]


def _make_inputs(case, dtype, device):
    M, N, _, has_bias, z_kind, _, _, _ = case
    torch.manual_seed(20261030 + M + N)
    x = torch.randn((M, N), device=device, dtype=dtype)
    weight = torch.randn((N,), device=device, dtype=dtype)
    bias = torch.randn((N,), device=device, dtype=dtype) if has_bias else None
    z = None
    if z_kind == "2d":
        z = torch.randn((M, N), device=device, dtype=dtype)
    elif z_kind == "3d":
        heads = 4
        padded = torch.randn((M // heads, heads, N + 32), device=device, dtype=dtype)
        z = padded[..., :N]
    return x, weight, bias, z


def _kwargs(case, z):
    _, _, rms, _, _, before, activation, group_size = case
    return {
        "z": z,
        "group_size": group_size,
        "norm_before_gate": before,
        "is_rms_norm": rms,
        "activation": activation,
    }


@pytest.mark.api("functions.layernorm_gated")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_layernorm_gated_accuracy(accuracy_backend, dtype, case):
    implementation, _, device = accuracy_backend
    actual_inputs = _make_inputs(case, dtype, device)
    reference_inputs = tuple(value.clone() if isinstance(value, torch.Tensor) else value for value in actual_inputs)
    x, weight, bias, z = actual_inputs

    actual, actual_mean, actual_rstd = F.layernorm_gated(
        x,
        weight,
        bias,
        1e-6,
        **_kwargs(case, z),
        implementation=implementation,
    )
    expected, expected_mean, expected_rstd = F.layernorm_gated(
        reference_inputs[0],
        reference_inputs[1],
        reference_inputs[2],
        1e-6,
        **_kwargs(case, reference_inputs[3]),
        implementation="torch",
    )

    assert actual.shape == expected.shape == x.shape
    assert actual.dtype == expected.dtype == dtype
    assert actual.device.type == expected.device.type == device
    assert_close(actual, expected, dtype, name="out")

    if case[2]:
        assert actual_mean is expected_mean is None
    else:
        stats_shape = torch.Size([(case[1] // (case[7] or case[1])) * case[0]])
        assert actual_mean.shape == expected_mean.shape == stats_shape
        assert actual_mean.dtype == expected_mean.dtype == torch.float32
        assert_close(actual_mean, expected_mean, rtol=1e-3, atol=1e-3, name="mean")
    assert actual_rstd.shape == expected_rstd.shape
    assert actual_rstd.dtype == expected_rstd.dtype == torch.float32
    assert_close(actual_rstd, expected_rstd, rtol=1e-3, atol=1e-3, name="rstd")


@pytest.mark.api("functions.layernorm_gated")
@pytest.mark.bitwise
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("before", [True, False])
def test_strided_gate_matches_contiguous_gate(accuracy_backend, dtype, before):
    implementation, _, device = accuracy_backend
    case = (132, 128, True, False, "3d", before, "swish", None)
    x, weight, bias, strided_z = _make_inputs(case, dtype, device)
    kwargs = {
        "group_size": None,
        "norm_before_gate": before,
        "is_rms_norm": True,
        "activation": "swish",
        "implementation": implementation,
    }
    strided = F.layernorm_gated(x, weight, bias, 1e-6, z=strided_z, **kwargs)
    contiguous = F.layernorm_gated(x, weight, bias, 1e-6, z=strided_z.contiguous(), **kwargs)
    assert_bitwise_equal(strided, contiguous, phase="strided_gate")


def _quant_reference(y, quant_heads):
    tokens, width = y.shape
    grouped = y.reshape(tokens // quant_heads, quant_heads * width).float()
    amax = torch.amax(torch.abs(grouped), dim=-1, keepdim=True)
    scale = torch.maximum(amax, torch.full_like(amax, 1e-10)) / 448.0
    quantized = torch.clamp(grouped * (1.0 / scale), -448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized, scale


@pytest.mark.api("functions.layernorm_gated")
@pytest.mark.bitwise
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("n", [128, 250])
def test_fused_quant_bit_exact(accuracy_backend, dtype, has_z, n):
    implementation, target, device = accuracy_backend
    if implementation == "torch":
        pytest.skip("FP8 bit-exact check targets the fused accelerator epilogue")
    if not target.startswith("npu.a5"):
        pytest.skip("FP8 quant is formally required only on Ascend 950")
    # n=250 rounds the column tile to 256, exercising the masked store path.
    case = (512, n, True, False, "2d" if has_z else "none", True, "swish", None)
    x, weight, bias, z = _make_inputs(case, dtype, device)
    fused, _, _ = F.layernorm_gated(
        x,
        weight,
        bias,
        1e-6,
        z=z,
        is_rms_norm=True,
        quant_heads=4,
        implementation=implementation,
    )
    normalized, _, _ = F.layernorm_gated(
        x,
        weight,
        bias,
        1e-6,
        z=z,
        is_rms_norm=True,
        implementation=implementation,
    )
    assert_bitwise_equal(fused, _quant_reference(normalized, 4), phase=f"fp8_quant_{target}")


@pytest.mark.api("functions.layernorm_gated")
@pytest.mark.bitwise
def test_batch_invariant_rows(accuracy_backend, monkeypatch):
    implementation, _, device = accuracy_backend
    monkeypatch.setenv("LN_GATED_BATCH_INVARIANT", "1")
    dtype = torch.bfloat16
    torch.manual_seed(17)
    x64 = torch.randn((64, 128), device=device, dtype=dtype)
    x128 = torch.cat((x64, torch.randn((64, 128), device=device, dtype=dtype)), dim=0)
    weight = torch.randn((128,), device=device, dtype=dtype)
    kwargs = {"is_rms_norm": True, "implementation": implementation}
    y64, _, r64 = F.layernorm_gated(x64, weight, None, 1e-6, **kwargs)
    y128, _, r128 = F.layernorm_gated(x128, weight, None, 1e-6, **kwargs)
    assert_bitwise_equal(y64, y128[:64], phase="batch_invariant_output")
    assert_bitwise_equal(r64, r128[:64], phase="batch_invariant_rstd")


@pytest.mark.api("functions.layernorm_gated")
def test_tail_block_keeps_output_in_bounds(accuracy_backend):
    implementation, _, device = accuracy_backend
    dtype = torch.bfloat16
    case = (129, 192, True, False, "2d", True, "swish", None)
    x, weight, bias, z = _make_inputs(case, dtype, device)
    guard = 256
    storage = torch.full((x.numel() + guard,), float("nan"), device=device, dtype=dtype)
    out = storage[: x.numel()].view(x.shape)
    F.layernorm_gated(
        x,
        weight,
        bias,
        1e-6,
        z=z,
        out=out,
        is_rms_norm=True,
        implementation=implementation,
    )
    assert torch.isnan(storage[x.numel() :]).all(), "masked stores wrote past the last row"

    expected, _, _ = F.layernorm_gated(
        x,
        weight,
        bias,
        1e-6,
        z=z,
        is_rms_norm=True,
        implementation="torch",
    )
    assert_close(out, expected, dtype, name="tail_out")


@pytest.mark.api("functions.layernorm_gated")
def test_sglang_compatible_wrapper(accuracy_backend):
    implementation, _, device = accuracy_backend
    dtype = torch.bfloat16
    case = (128, 128, True, False, "2d", True, "swish", None)
    x, weight, bias, z = _make_inputs(case, dtype, device)
    actual = F.rms_norm_gated(
        x=x,
        weight=weight,
        bias=bias,
        z=z,
        is_rms_norm=True,
        implementation=implementation,
    )
    expected = F.rms_norm_gated(
        x=x,
        weight=weight,
        bias=bias,
        z=z,
        is_rms_norm=True,
        implementation="torch",
    )
    assert_close(actual, expected, dtype, name="rms_norm_gated")
    forward, mean, rstd = F._layer_norm_fwd(
        x,
        weight,
        bias,
        1e-6,
        z=z,
        is_rms_norm=True,
    )
    assert_close(forward, expected, dtype, name="_layer_norm_fwd")
    assert mean is None
    assert rstd.shape == (x.shape[0],)
