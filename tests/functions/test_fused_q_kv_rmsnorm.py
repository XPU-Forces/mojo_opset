"""Tests for functions.fused_q_kv_rmsnorm / fused_q_kv_rmsnorm_quant.

Tolerances follow docs/operator_accuracy_standard.md §2.2 (all elements must
pass): bf16 2^-7, fp16 2^-10, fp32 2^-13 and fp8 E4M3 2^-3. The F8_128x4 scale
buffer is integer data, so it is compared byte-exactly.
"""

from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_bitwise_equal
from tests._checks import assert_close
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph

# docs/operator_accuracy_standard.md §2.2 (rtol, atol).
_TOL = {
    torch.float16: (2.0**-10, 2.0**-10),
    torch.bfloat16: (2.0**-7, 2.0**-7),
    torch.float32: (2.0**-13, 2.0**-13),
    torch.float8_e4m3fn: (2.0**-3, 2.0**-3),
}

MXFP8_MAX = 448.0
GROUP = 32

# DSv4.1 production (TP4 q_lora_rank=1280 / head_dim=512), the vllm kernel unit
# shape, a q < kv control pair and a non-power-of-two masked width.
SHAPES = [(1280, 512), (192, 576), (384, 512), (1344, 512)]
TOKENS = [1, 5, 127, 128, 129, 256, 1024, 2048]
EDGE_TOKENS = [130, 131]
DTYPES = [torch.bfloat16, torch.float16]


def _inputs(tokens, q_size, kv_size, dtype, device, seed=0, edge_rows=False):
    torch.manual_seed(seed)
    q = torch.randn(tokens, q_size, dtype=torch.float32, device=device)
    kv = torch.randn(tokens, kv_size, dtype=torch.float32, device=device)
    if edge_rows and tokens >= 3:
        # fp32 variance must not overflow: tiny drives the activation dtype into
        # subnormals, spike makes a single element dominate the group amax.
        tiny, spike = (1e-30, 1e18) if dtype == torch.bfloat16 else (1e-6, 300.0)
        q[0] = 0.0
        q[1] *= tiny
        q[2] = 0.0
        q[2, 0] = spike
    q_weight = torch.randn(q_size, dtype=torch.float32, device=device)
    kv_weight = torch.randn(kv_size, dtype=torch.float32, device=device)
    return (
        q.to(dtype),
        kv.to(dtype),
        q_weight.to(dtype),
        kv_weight.to(dtype),
    )


def _check_norm(actual, expected, name):
    dtype = expected.dtype
    rtol, atol = _TOL[dtype]
    assert_close(actual, expected, dtype, rtol=rtol, atol=atol, name=name)


def _check_quant(actual, expected):
    q8, scale, kv = actual
    ref_q8, ref_scale, ref_kv = expected
    assert q8.dtype == torch.float8_e4m3fn and q8.shape == ref_q8.shape
    assert scale.dtype == torch.uint8 and scale.shape == ref_scale.shape
    assert torch.equal(scale, ref_scale), "F8_128x4 scale bytes differ"
    rtol, atol = _TOL[torch.float8_e4m3fn]
    assert_close(q8.float(), ref_q8.float(), torch.float32, rtol=rtol, atol=atol, name="q_fp8")
    _check_norm(kv, ref_kv, "kv")


def _check_float_special(actual, expected, name):
    """NaN/+-inf outputs: the NaN positions must agree, finite lanes use §2.2."""
    nan_actual = torch.isnan(actual)
    nan_expected = torch.isnan(expected)
    assert torch.equal(nan_actual, nan_expected), f"{name}: NaN positions differ"
    finite = ~nan_actual
    if finite.any():
        dtype = expected.dtype
        rtol, atol = _TOL[dtype]
        assert_close(actual[finite], expected[finite], dtype, rtol=rtol, atol=atol, name=name)


def _run_norm(implementation, tokens, q_size, kv_size, dtype, device, seed=0, edge_rows=False):
    inputs = _inputs(tokens, q_size, kv_size, dtype, device, seed, edge_rows)
    actual = F.fused_q_kv_rmsnorm(*inputs, 1e-6, implementation=implementation)
    expected = F.fused_q_kv_rmsnorm(*inputs, 1e-6, implementation="torch")
    _check_norm(actual[0], expected[0], "q")
    _check_norm(actual[1], expected[1], "kv")
    return actual


def _run_quant(implementation, tokens, q_size, kv_size, dtype, device, seed=0, edge_rows=False):
    inputs = _inputs(tokens, q_size, kv_size, dtype, device, seed, edge_rows)
    actual = F.fused_q_kv_rmsnorm_quant(*inputs, 1e-20, implementation=implementation)
    expected = F.fused_q_kv_rmsnorm_quant(*inputs, 1e-20, implementation="torch")
    _check_quant(actual, expected)
    return actual


def _skip_without_fp8(target):
    if target.startswith("npu.a2"):
        pytest.skip("910B (A2) has no FP8: fused_q_kv_rmsnorm_quant fails fast there")


@pytest.mark.api("functions.fused_q_kv_rmsnorm")
@pytest.mark.accuracy
@pytest.mark.parametrize("q_size,kv_size", SHAPES)
@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_q_kv_rmsnorm_shapes(accuracy_backend, q_size, kv_size, tokens, dtype):
    implementation, _, device = accuracy_backend
    _run_norm(implementation, tokens, q_size, kv_size, dtype, device, seed=tokens + q_size)


@pytest.mark.api("functions.fused_q_kv_rmsnorm")
@pytest.mark.accuracy
@pytest.mark.parametrize("q_size,kv_size", SHAPES)
@pytest.mark.parametrize("tokens", EDGE_TOKENS)
def test_fused_q_kv_rmsnorm_edge_rows(accuracy_backend, q_size, kv_size, tokens):
    implementation, _, device = accuracy_backend
    _run_norm(implementation, tokens, q_size, kv_size, torch.bfloat16, device, seed=tokens, edge_rows=True)


@pytest.mark.api("functions.fused_q_kv_rmsnorm_quant")
@pytest.mark.accuracy
@pytest.mark.parametrize("q_size,kv_size", SHAPES)
@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_q_kv_rmsnorm_quant_shapes(accuracy_backend, q_size, kv_size, tokens, dtype):
    implementation, target, device = accuracy_backend
    _skip_without_fp8(target)
    _run_quant(implementation, tokens, q_size, kv_size, dtype, device, seed=tokens + q_size)


@pytest.mark.api("functions.fused_q_kv_rmsnorm_quant")
@pytest.mark.accuracy
@pytest.mark.parametrize("q_size,kv_size", SHAPES)
@pytest.mark.parametrize("tokens", EDGE_TOKENS)
def test_fused_q_kv_rmsnorm_quant_edge_rows(accuracy_backend, q_size, kv_size, tokens):
    implementation, target, device = accuracy_backend
    _skip_without_fp8(target)
    _run_quant(implementation, tokens, q_size, kv_size, torch.bfloat16, device, seed=tokens, edge_rows=True)


@pytest.mark.api("functions.fused_q_kv_rmsnorm_quant")
@pytest.mark.accuracy
def test_fused_q_kv_rmsnorm_quant_edge_row_semantics(accuracy_backend):
    """Zero row -> scale 0 and fp8 +-0; subnormal row -> non-zero scales; spike row -> large fp8."""
    implementation, target, device = accuracy_backend
    _skip_without_fp8(target)
    q8, scale, _ = _run_quant(implementation, 3, 1280, 512, torch.bfloat16, device, seed=7, edge_rows=True)
    raw = q8.view(torch.uint8)
    groups = 1280 // GROUP
    groups4 = (groups + 3) // 4
    rows = torch.arange(3, device=device)[:, None]
    group = torch.arange(groups, device=device)[None, :]
    offsets = (
        (rows // 128 * groups4 + group // 4) * 512
        + rows % 32 * 16
        + rows % 128 // 32 * 4
        + group % 4
    )
    scale_rows = scale[offsets]
    assert (scale_rows[0] == 0).all(), "zero row must produce scale 0"
    assert (raw[0] & 0x7F).eq(0).all(), "zero row must quantise to fp8 +-0"
    assert (scale_rows[1] > 0).all(), "subnormal row must keep normal scales"
    assert int((raw[1] & 0x7F).eq(0).sum()) == 0
    assert q8.float()[2].abs().max().item() >= 64.0


@pytest.mark.api("functions.fused_q_kv_rmsnorm")
@pytest.mark.accuracy
def test_fused_q_kv_rmsnorm_empty_batch(accuracy_backend):
    """Empty tensor (docs §1.4): empty outputs, no launch."""
    implementation, _, device = accuracy_backend
    empty_q = torch.empty(0, 1280, dtype=torch.bfloat16, device=device)
    empty_kv = torch.empty(0, 512, dtype=torch.bfloat16, device=device)
    q_weight = torch.ones(1280, dtype=torch.bfloat16, device=device)
    kv_weight = torch.ones(512, dtype=torch.bfloat16, device=device)
    q_out, kv_out = F.fused_q_kv_rmsnorm(
        empty_q, empty_kv, q_weight, kv_weight, 1e-6, implementation=implementation
    )
    assert q_out.shape == (0, 1280) and kv_out.shape == (0, 512)


@pytest.mark.api("functions.fused_q_kv_rmsnorm_quant")
@pytest.mark.accuracy
def test_fused_q_kv_rmsnorm_quant_empty_batch(accuracy_backend):
    """Empty tensor for the quantising entry: empty fp8/scales/kv outputs."""
    implementation, target, device = accuracy_backend
    _skip_without_fp8(target)
    empty_q = torch.empty(0, 1280, dtype=torch.bfloat16, device=device)
    empty_kv = torch.empty(0, 512, dtype=torch.bfloat16, device=device)
    q_weight = torch.ones(1280, dtype=torch.bfloat16, device=device)
    kv_weight = torch.ones(512, dtype=torch.bfloat16, device=device)
    q8, scale, kv_norm = F.fused_q_kv_rmsnorm_quant(
        empty_q, empty_kv, q_weight, kv_weight, 1e-20, implementation=implementation
    )
    assert q8.numel() == 0 and scale.numel() == 0 and kv_norm.numel() == 0


# The quantising half is guarded inline so this case still runs on A2 (where the
# quantising entry has no provider).
@pytest.mark.api("functions.fused_q_kv_rmsnorm")
@pytest.mark.accuracy
def test_fused_q_kv_rmsnorm_special_values(accuracy_backend):
    """NaN / +-inf inputs (docs §1.4): both sides agree bit-exactly."""
    implementation, target, device = accuracy_backend
    for dtype in DTYPES:
        q = torch.randn(6, 256, dtype=torch.float32, device=device)
        kv = torch.randn(6, 128, dtype=torch.float32, device=device)
        q[0, 0] = float("nan")
        q[1, 3] = float("inf")
        q[2, 5] = float("-inf")
        kv[3, 1] = float("nan")
        kv[4, 2] = float("inf")
        kv[5, 0] = float("-inf")
        q, kv = q.to(dtype), kv.to(dtype)
        q_weight = torch.randn(256, dtype=torch.float32, device=device).to(dtype)
        kv_weight = torch.randn(128, dtype=torch.float32, device=device).to(dtype)
        actual = F.fused_q_kv_rmsnorm(q, kv, q_weight, kv_weight, 1e-6, implementation=implementation)
        expected = F.fused_q_kv_rmsnorm(q, kv, q_weight, kv_weight, 1e-6, implementation="torch")
        _check_float_special(actual[0], expected[0], "q")
        _check_float_special(actual[1], expected[1], "kv")


@pytest.mark.api("functions.fused_q_kv_rmsnorm_quant")
@pytest.mark.accuracy
@pytest.mark.parametrize("tokens", [6, 128], ids=["fallback", "split"])
def test_fused_q_kv_rmsnorm_quant_special_values(accuracy_backend, tokens):
    """NaN / +-inf on the quantising entry, on both the fallback and split paths."""
    implementation, target, device = accuracy_backend
    _skip_without_fp8(target)
    for dtype in DTYPES:
        q = torch.randn(tokens, 1280, dtype=torch.float32, device=device)
        kv = torch.randn(tokens, 512, dtype=torch.float32, device=device)
        q[0, 0] = float("nan")
        q[1, 3] = float("inf")
        q[2, 5] = float("-inf")
        kv[3, 1] = float("nan")
        q, kv = q.to(dtype), kv.to(dtype)
        q_weight = torch.randn(1280, dtype=torch.float32, device=device).to(dtype)
        kv_weight = torch.randn(512, dtype=torch.float32, device=device).to(dtype)
        actual = F.fused_q_kv_rmsnorm_quant(q, kv, q_weight, kv_weight, 1e-20, implementation=implementation)
        expected = F.fused_q_kv_rmsnorm_quant(q, kv, q_weight, kv_weight, 1e-20, implementation="torch")
        assert_bitwise_equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        _check_float_special(actual[2], expected[2], "kv")


@pytest.mark.api("functions.fused_q_kv_rmsnorm")
@pytest.mark.bitwise
def test_fused_q_kv_rmsnorm_repeatable(accuracy_backend):
    implementation, _, device = accuracy_backend
    inputs = _inputs(1024, 1280, 512, torch.bfloat16, device, seed=11)
    assert_repeatable(
        partial(F.fused_q_kv_rmsnorm, eps=1e-6, implementation=implementation), inputs, backward=False
    )


@pytest.mark.api("functions.fused_q_kv_rmsnorm_quant")
@pytest.mark.bitwise
def test_fused_q_kv_rmsnorm_quant_repeatable(accuracy_backend):
    implementation, target, device = accuracy_backend
    _skip_without_fp8(target)
    inputs = _inputs(1024, 1280, 512, torch.bfloat16, device, seed=12)
    assert_repeatable(
        partial(F.fused_q_kv_rmsnorm_quant, eps=1e-20, implementation=implementation), inputs, backward=False
    )


@pytest.mark.api("functions.fused_q_kv_rmsnorm")
def test_fused_q_kv_rmsnorm_fake_metadata(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("NPU Triton fake metadata")
    preload("fused_q_kv_rmsnorm", implementation="triton", target=target)
    namespace = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    q = torch.randn(3, 1280, dtype=torch.bfloat16, device=device)
    kv = torch.randn(3, 512, dtype=torch.bfloat16, device=device)
    q_weight = torch.randn(1280, dtype=torch.bfloat16, device=device)
    kv_weight = torch.randn(512, dtype=torch.bfloat16, device=device)
    torch.library.opcheck(
        namespace.fused_q_kv_rmsnorm_fwd.default,
        (q, kv, q_weight, kv_weight, 1e-6),
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.api("functions.fused_q_kv_rmsnorm_quant")
def test_fused_q_kv_rmsnorm_quant_fake_metadata(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu.a5"):
        pytest.skip("MXFP8 quantisation needs the A5 Triton provider")
    preload("fused_q_kv_rmsnorm_quant", implementation="triton", target=target)
    namespace = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    q = torch.randn(129, 1280, dtype=torch.bfloat16, device=device)
    kv = torch.randn(129, 512, dtype=torch.bfloat16, device=device)
    q_weight = torch.randn(1280, dtype=torch.bfloat16, device=device)
    kv_weight = torch.randn(512, dtype=torch.bfloat16, device=device)
    torch.library.opcheck(
        namespace.fused_q_kv_rmsnorm_quant_fwd.default,
        (q, kv, q_weight, kv_weight, 1e-20),
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.api("functions.fused_q_kv_rmsnorm")
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_fused_q_kv_rmsnorm_compiled(accuracy_backend, backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation == "torch":
        pytest.skip("NPU Triton compile contract")
    implementation = implementation or "triton"
    preload("fused_q_kv_rmsnorm", implementation=implementation)
    inputs = _inputs(256, 1280, 512, torch.bfloat16, device, seed=13)

    def run(q, kv, q_weight, kv_weight):
        output = F.fused_q_kv_rmsnorm(
            q, kv, q_weight, kv_weight, 1e-6, implementation=implementation
        )
        return output

    try:
        eager = run(*inputs)
        actual = compile_fullgraph(run, backend=backend)(*inputs)
        assert all(left.is_contiguous() for left in actual)
        for left, right in zip(actual, eager):
            assert torch.equal(left, right)
        expected = F.fused_q_kv_rmsnorm(*inputs, 1e-6, implementation="torch")
        for left, right in zip(actual, expected):
            _check_norm(left, right, "compiled")
    finally:
        torch._dynamo.reset()


# --------------------------- reference-only checks ---------------------------


def _unswizzle(scales, tokens, width):
    groups = width // GROUP
    groups4 = (groups + 3) // 4
    rows = torch.arange(tokens)[:, None]
    group = torch.arange(groups)[None, :]
    offsets = (
        (rows // 128 * groups4 + group // 4) * 512
        + rows % 32 * 16
        + rows % 128 // 32 * 4
        + group % 4
    )
    return scales[offsets].to(torch.int64)


@pytest.mark.reference
@pytest.mark.parametrize("tokens,width", [(1, 1280), (129, 1280), (256, 192), (3, 1344)])
def test_scale_buffer_swizzle_roundtrip(tokens, width):
    """The F8_128x4 scatter must be invertible and leave padding rows zero."""
    groups = width // GROUP
    torch.manual_seed(3)
    scale = torch.randint(0, 255, (tokens, groups), dtype=torch.int64)
    from mojo_opset.kernels.torch.fused_q_kv_rmsnorm import _scale_bytes

    buffer = _scale_bytes(scale, tokens, width)
    total = ((tokens + 127) // 128) * 128 * ((groups + 3) // 4) * 4
    assert buffer.numel() == total
    assert torch.equal(_unswizzle(buffer, tokens, width), scale)
    padded = ((tokens + 127) // 128) * 128
    if padded > tokens:
        assert torch.equal(_unswizzle(buffer, padded, width)[tokens:], torch.zeros(padded - tokens, groups, dtype=torch.int64))


@pytest.mark.reference
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_reference_matches_manual_formula(dtype):
    """The torch provider must equal a direct FP32 RMSNorm + MXFP8 transcription on CPU."""
    torch.manual_seed(5)
    tokens, q_size, kv_size = 7, 128, 64
    q = torch.randn(tokens, q_size).to(dtype)
    kv = torch.randn(tokens, kv_size).to(dtype)
    q_weight = torch.randn(q_size).to(dtype)
    kv_weight = torch.randn(kv_size).to(dtype)
    eps = 1e-6

    def norm(x, weight):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(dtype)

    expected_q_norm, expected_kv = norm(q, q_weight), norm(kv, kv_weight)
    q_out, kv_out = F.fused_q_kv_rmsnorm(q, kv, q_weight, kv_weight, eps, implementation="torch")
    assert torch.equal(q_out, expected_q_norm)
    assert torch.equal(kv_out, expected_kv)

    q8, scale, kv_norm = F.fused_q_kv_rmsnorm_quant(
        q, kv, q_weight, kv_weight, eps, implementation="torch"
    )
    y = norm(q, q_weight).float().view(tokens, q_size // GROUP, GROUP)
    amax = y.abs().amax(-1)
    normalized = amax / MXFP8_MAX
    bits = normalized.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exponent, mantissa = (bits >> 23) & 255, bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = torch.where(normalized <= 0, torch.zeros_like(exponent), torch.minimum(exponent + bump.to(torch.int64), torch.tensor(254)))
    inv = torch.where(sf == 0, torch.zeros_like(amax), ((254 - sf) << 23).to(torch.int32).view(torch.float32))
    expected_q8 = (y * inv.unsqueeze(-1)).reshape(tokens, q_size).to(torch.float8_e4m3fn)
    assert torch.equal(q8.view(torch.uint8), expected_q8.view(torch.uint8))
    assert torch.equal(_unswizzle(scale, tokens, q_size), sf)
    assert torch.equal(kv_norm, expected_kv)
