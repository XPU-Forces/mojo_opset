"""Accuracy and bit-exactness coverage for the fused gated normalization.

This module complements the community acceptance suite
(``XPU-Forces/mojo_opset#525``) with the cases that suite does not cover: the
grouped-normalisation statistics layout, the row-strided input and output
views, and the out-of-bounds canary around the tail tile.  The rows of
``CASES`` are the matrix required by the task statement: LayerNorm and
RMSNorm, flat and 3-D strided gates, both gate orders, every activation, with
and without bias, and grouped normalisation.  The harness around that matrix
owns the input generation, so a case only has to describe *what* to exercise.
"""

from collections import namedtuple

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_bitwise_equal
from tests._checks import assert_close

Case = namedtuple(
    "Case",
    ("rows", "cols", "rms", "bias", "gate", "order", "activation", "group"),
)

CASES = [
    Case(1, 128, True, False, "none", True, "swish", None),
    Case(128, 128, True, False, "2d", True, "swish", None),
    Case(128, 128, True, False, "3d", True, "swish", None),
    Case(128, 128, True, False, "3d", False, "sigmoid", None),
    Case(128, 128, False, True, "3d", True, "sigmoid", None),
    Case(132, 128, True, True, "3d", True, "swish", None),
    Case(132, 128, False, True, "2d", False, "sigmoid", None),
    Case(512, 1024, False, True, "none", True, "swish", None),
    Case(512, 1024, True, False, "2d", False, "silu", None),
    Case(128, 256, True, False, "2d", False, "sigmoid", 64),
    Case(128, 256, False, True, "2d", False, "swish", 64),
    Case(512, 256, True, False, "none", True, "swish", 64),
    Case(129, 192, True, False, "none", True, "swish", None),
    Case(129, 192, True, False, "2d", True, "swish", None),
]

HEADS = 4
GATE_PADDING = 32


def _case_id(case):
    norm = "rms" if case.rms else "layernorm"
    bias = "-bias" if case.bias else ""
    order = "normfirst" if case.order else "gatefirst"
    group = f"-group{case.group}" if case.group else ""
    return f"rows{case.rows}-cols{case.cols}-{norm}{bias}-gate{case.gate}-{order}-{case.activation}{group}"


def _seed_for(case):
    return 700_003 + case.rows * 131 + case.cols


def _sample(case, dtype, device):
    """Draw the tensors a case needs, including its gate."""

    torch.manual_seed(_seed_for(case))
    x = torch.randn((case.rows, case.cols), device=device, dtype=dtype)
    weight = torch.randn((case.cols,), device=device, dtype=dtype)
    bias = torch.randn((case.cols,), device=device, dtype=dtype) if case.bias else None
    gate = None
    if case.gate == "2d":
        gate = torch.randn((case.rows, case.cols), device=device, dtype=dtype)
    elif case.gate == "3d":
        # A padded head dimension gives the gate the token/head strides that
        # only a strided view keeps.
        tokens = case.rows // HEADS
        padded = torch.randn((tokens, HEADS, case.cols + GATE_PADDING), device=device, dtype=dtype)
        gate = padded[..., : case.cols]
    return x, weight, bias, gate


def _options(case, gate):
    return {
        "z": gate,
        "group_size": case.group,
        "norm_before_gate": case.order,
        "is_rms_norm": case.rms,
        "activation": case.activation,
    }


def _expected_stats_shape(case):
    width = case.group if case.group else case.cols
    return torch.Size([(case.cols // width) * case.rows])


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_layernorm_gated_accuracy(accuracy_backend, dtype, case):
    implementation, _, device = accuracy_backend
    x, weight, bias, gate = _sample(case, dtype, device)
    reference_x = x.clone()
    reference_weight = weight.clone()
    reference_bias = None if bias is None else bias.clone()
    reference_gate = None if gate is None else gate.clone()

    actual, actual_mean, actual_rstd = F.layer_norm_gated(
        x,
        weight,
        bias,
        eps=1e-6,
        **_options(case, gate),
        implementation=implementation,
    )
    expected, expected_mean, expected_rstd = F.layer_norm_gated(
        reference_x,
        reference_weight,
        reference_bias,
        eps=1e-6,
        **_options(case, reference_gate),
        implementation="torch",
    )

    assert actual.shape == expected.shape == x.shape
    assert actual.dtype == expected.dtype == dtype
    assert actual.device.type == expected.device.type == device
    assert_close(actual, expected, dtype, name="out")

    if case.rms:
        assert actual_mean is expected_mean is None
    else:
        stats_shape = _expected_stats_shape(case)
        assert actual_mean.shape == expected_mean.shape == stats_shape
        assert actual_mean.dtype == expected_mean.dtype == torch.float32
        assert_close(actual_mean, expected_mean, rtol=1e-3, atol=1e-3, name="mean")
    assert actual_rstd.shape == expected_rstd.shape
    assert actual_rstd.dtype == expected_rstd.dtype == torch.float32
    assert_close(actual_rstd, expected_rstd, rtol=1e-3, atol=1e-3, name="rstd")


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.bitwise
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("order", [True, False])
def test_strided_gate_matches_contiguous_gate(accuracy_backend, dtype, order):
    implementation, _, device = accuracy_backend
    case = Case(132, 128, True, False, "3d", order, "swish", None)
    x, weight, bias, gate = _sample(case, dtype, device)
    options = {
        "group_size": None,
        "norm_before_gate": order,
        "is_rms_norm": True,
        "activation": "swish",
        "implementation": implementation,
    }
    strided = F.layer_norm_gated(x, weight, bias, eps=1e-6, z=gate, **options)
    contiguous = F.layer_norm_gated(x, weight, bias, eps=1e-6, z=gate.contiguous(), **options)
    assert_bitwise_equal(strided, contiguous, phase="strided_gate")


def _fp8_reference(y, quant_heads):
    """Per-token-group FP8 e4m3 quantisation of an already normalised tensor."""

    tokens, width = y.shape
    grouped = y.reshape(tokens // quant_heads, quant_heads * width).float()
    amax = torch.amax(torch.abs(grouped), dim=-1, keepdim=True)
    scale = torch.maximum(amax, torch.full_like(amax, 1e-10)) / 448.0
    quantized = torch.clamp(grouped * (1.0 / scale), -448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized, scale


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.bitwise
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_gate", [False, True])
@pytest.mark.parametrize("cols", [128, 250])
def test_fused_quant_bit_exact(accuracy_backend, dtype, has_gate, cols):
    implementation, target, device = accuracy_backend
    if implementation == "torch":
        pytest.skip("FP8 bit-exact check targets the fused accelerator epilogue")
    if not target.startswith("npu.a5"):
        pytest.skip("FP8 quant is formally required only on Ascend 950")
    # cols=250 rounds the column tile up to 256 and exercises the masked store.
    case = Case(512, cols, True, False, "2d" if has_gate else "none", True, "swish", None)
    x, weight, bias, gate = _sample(case, dtype, device)
    fused, _, _ = F.layer_norm_gated(
        x,
        weight,
        bias,
        eps=1e-6,
        z=gate,
        is_rms_norm=True,
        quant_heads=4,
        implementation=implementation,
    )
    normalized, _, _ = F.layer_norm_gated(
        x,
        weight,
        bias,
        eps=1e-6,
        z=gate,
        is_rms_norm=True,
        implementation=implementation,
    )
    assert_bitwise_equal(fused, _fp8_reference(normalized, 4), phase=f"fp8_quant_{target}")


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.bitwise
def test_batch_invariant_rows(accuracy_backend, monkeypatch):
    implementation, _, device = accuracy_backend
    monkeypatch.setenv("LN_GATED_BATCH_INVARIANT", "1")
    dtype = torch.bfloat16
    torch.manual_seed(17)
    x64 = torch.randn((64, 128), device=device, dtype=dtype)
    x128 = torch.cat((x64, torch.randn((64, 128), device=device, dtype=dtype)), dim=0)
    weight = torch.randn((128,), device=device, dtype=dtype)
    options = {"is_rms_norm": True, "implementation": implementation}
    y64, _, r64 = F.layer_norm_gated(x64, weight, None, eps=1e-6, **options)
    y128, _, r128 = F.layer_norm_gated(x128, weight, None, eps=1e-6, **options)
    assert_bitwise_equal(y64, y128[:64], phase="batch_invariant_output")
    assert_bitwise_equal(r64, r128[:64], phase="batch_invariant_rstd")


@pytest.mark.api("functions.layer_norm_gated")
def test_tail_block_keeps_output_in_bounds(accuracy_backend):
    implementation, _, device = accuracy_backend
    dtype = torch.bfloat16
    case = Case(129, 192, True, False, "2d", True, "swish", None)
    x, weight, bias, gate = _sample(case, dtype, device)
    guard = 256
    storage = torch.full((x.numel() + guard,), float("nan"), device=device, dtype=dtype)
    out = storage[: x.numel()].view(x.shape)
    F.layer_norm_gated(
        x,
        weight,
        bias,
        eps=1e-6,
        z=gate,
        out=out,
        is_rms_norm=True,
        implementation=implementation,
    )
    assert torch.isnan(storage[x.numel() :]).all(), "masked stores wrote past the last row"

    expected, _, _ = F.layer_norm_gated(
        x,
        weight,
        bias,
        eps=1e-6,
        z=gate,
        is_rms_norm=True,
        implementation="torch",
    )
    assert_close(out, expected, dtype, name="tail_out")
