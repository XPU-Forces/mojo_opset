import math

import pytest
import torch

from mojo_opset import functions as F

from .._checks import assert_bitwise_equal

Z_HEADS = 4  # H of [T, H, D] gate branches; T * H must equal M
Z_PAD = 32  # token-major strided-view padding, stride(-1) == 1
QUANT_HEADS = 4
GUARD_ELEMENTS = 64
FP8_MAX = 448.0
FP8_MIN_AMAX = 1e-10

# Mixed-tolerance thresholds per *actual output dtype* from the CANN ecosystem
# operator precision standard (opbase docs/zh/ops_precision_standard/
# mixed_tolerance_standard.md): (rtol, atol, max_abs_error floor). The hard cap
# is max(floor, 32 * ULP(g_low)) with g_low the golden value at the max-error
# point narrowed to the output dtype.
MIXED_TOLERANCE = {
    torch.float16: (2.0**-10, 2.0**-10, 1e-1),
    torch.bfloat16: (2.0**-7, 2.0**-7, 1e-1),
    torch.float32: (2.0**-13, 2.0**-13, 1e-2),
    torch.float8_e4m3fn: (2.0**-3, 2.0**-3, 1e-0),
}
REQUIRED_MATCHED_RATIO = 0.99
MAX_ABS_ERROR_ULP = 32

_MANTISSA_BITS = {
    torch.float16: 10,
    torch.bfloat16: 7,
    torch.float32: 23,
    torch.float8_e4m3fn: 3,
}
_MIN_SUBNORMAL = {
    torch.float16: 2.0**-24,
    torch.bfloat16: 2.0**-133,
    torch.float32: 2.0**-149,
    torch.float8_e4m3fn: 2.0**-9,
}

# (M, N, z_mode, activation, norm_before_gate, is_rms_norm, dtype, eps)
GATED_NORM_SHAPES = [
    (1, 128, "2d", "swish", True, True, torch.float16, 1e-5),
    (1, 128, "none", "sigmoid", False, False, torch.bfloat16, 1e-6),
    (1, 256, "2d", "swish", True, True, torch.float16, 1e-6),
    (1, 256, "none", "sigmoid", False, False, torch.bfloat16, 1e-5),
    (1, 1024, "2d", "swish", True, True, torch.float16, 1e-6),
    (1, 1024, "none", "sigmoid", False, False, torch.bfloat16, 1e-6),
    (128, 256, "3d", "swish", True, True, torch.float16, 1e-6),
    (128, 256, "2d", "sigmoid", False, False, torch.bfloat16, 1e-6),
    (128, 1024, "3d", "swish", True, True, torch.float16, 1e-5),
    (128, 1024, "2d", "sigmoid", False, False, torch.bfloat16, 1e-6),
    (132, 128, "3d", "swish", True, True, torch.float16, 1e-6),
    (132, 128, "2d", "sigmoid", False, False, torch.bfloat16, 1e-6),
    (132, 256, "3d", "swish", True, True, torch.float16, 1e-6),
    (132, 256, "2d", "sigmoid", False, False, torch.bfloat16, 1e-5),
    (132, 1024, "3d", "swish", True, True, torch.float16, 1e-6),
    (132, 1024, "2d", "sigmoid", False, False, torch.bfloat16, 1e-6),
    (512, 128, "3d", "swish", True, True, torch.float16, 1e-6),
    (512, 128, "2d", "sigmoid", False, False, torch.bfloat16, 1e-6),
    (512, 256, "3d", "swish", True, True, torch.float16, 1e-5),
    (512, 256, "2d", "sigmoid", False, False, torch.bfloat16, 1e-6),
    (512, 1024, "3d", "swish", True, True, torch.float16, 1e-6),
    (512, 1024, "2d", "sigmoid", False, False, torch.bfloat16, 1e-6),
]

# (M, is_rms_norm, z_mode, norm_before_gate); N=256 with group_size=64 (ngroups=4)
GROUP_NORM_CASES = [
    (1, True, "none", True),
    (1, True, "2d", False),
    (1, False, "2d", True),
    (128, True, "none", True),
    (128, True, "2d", False),
    (128, True, "3d", True),
    (128, False, "2d", True),
    (128, False, "none", False),
]

BITWISE_GATE_SHAPES = [(132, 128), (512, 1024)]


def _gate(z, activation):
    if activation in ("swish", "silu"):
        return z * torch.sigmoid(z)
    if activation == "sigmoid":
        return torch.sigmoid(z)
    raise ValueError(f"unknown activation {activation!r}")


def make_gate(mode, M, N, dtype, device, *, heads=Z_HEADS, pad=Z_PAD):
    """Build the gate branch: None, row-padded 2D view, or token-major 3D strided view."""
    if mode == "none":
        return None
    if mode == "2d":
        return torch.randn(M, N + pad, device=device, dtype=dtype)[:, :N]
    if mode == "3d":
        assert M % heads == 0, "3D gate requires T * H == M with H = heads"
        gate = torch.randn(M // heads, heads, N + pad, device=device, dtype=dtype)[:, :, :N]
        assert gate.stride(-1) == 1 and gate.stride(0) > gate.stride(1) > 1
        return gate
    raise ValueError(f"unknown gate mode {mode!r}")


def flatten_gate(z, M, N):
    """[M, N] value-equivalent of the gate branch.

    Mirrors the kernel row mapping row -> (row // Z_HEADS, row % Z_HEADS), i.e.
    the flattened [T*H, D] layout of a 3D gate.
    """
    if z is None:
        return None
    return z.reshape(M, N) if z.ndim == 3 else z


def gated_norm_reference(x, weight, bias=None, z=None, *, eps=1e-6, group_size=None,
                         norm_before_gate=True, is_rms_norm=False, activation="swish"):
    """Independent FP32 composition: norm statistics, affine, then/preceding gate.

    The output stays FP32 (the high-precision golden); mean is None for RMSNorm
    and mean/rstd use the kernel's group-major layout [ngroups * M] (index
    group * M + row).
    """
    M, N = x.shape
    group_size = N if group_size is None else group_size
    ngroups = N // group_size

    values = x.float()
    gate = z.float() if z is not None else None
    if gate is not None and not norm_before_gate:
        values = values * _gate(gate, activation)

    grouped = values.reshape(M, ngroups, group_size)
    if is_rms_norm:
        mean = None
        rstd = torch.rsqrt(grouped.square().mean(dim=-1) + eps)
        normalized = grouped * rstd.unsqueeze(-1)
    else:
        mean = grouped.mean(dim=-1)
        rstd = torch.rsqrt((grouped - mean.unsqueeze(-1)).square().mean(dim=-1) + eps)
        normalized = (grouped - mean.unsqueeze(-1)) * rstd.unsqueeze(-1)

    out = normalized * weight.float().reshape(1, ngroups, group_size)
    if bias is not None:
        out = out + bias.float().reshape(1, ngroups, group_size)
    if gate is not None and norm_before_gate:
        out = out * _gate(gate.reshape(M, ngroups, group_size), activation)
    return (
        out.reshape(M, N),
        None if mean is None else mean.transpose(0, 1).reshape(-1),
        rstd.transpose(0, 1).reshape(-1),
    )


def fp8_quant_reference(out, quant_heads, dtype=None):
    """Non-fused per-token FP8 oracle mirroring the kernel's rounding order.

    The golden is narrowed to the output dtype *first* (round-to-nearest-even),
    then amax/scale/clamp reuse the kernel arithmetic bit-for-bit (reciprocal
    multiply).
    """
    M, N = out.shape
    y = out.to(dtype or out.dtype).reshape(M // quant_heads, quant_heads * N).float()
    scale = torch.clamp(y.abs().amax(dim=-1, keepdim=True), min=FP8_MIN_AMAX) / FP8_MAX
    q = (y * (1.0 / scale)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def _ulp_spacing(value, dtype):
    """Spacing of ``dtype`` at ``value``: 2^(exp-1-mantissa) for normals, clamped
    to the smallest subnormal (which also covers subnormal and zero anchors)."""
    if value == 0.0:
        return _MIN_SUBNORMAL[dtype]
    exponent = math.frexp(abs(value))[1]
    return max(2.0 ** (exponent - 1 - _MANTISSA_BITS[dtype]), _MIN_SUBNORMAL[dtype])


def assert_mixed_tolerance(actual, golden, *, name="output", max_print=5):
    """CANN mixed tolerance: per-element ``atol + rtol * |golden|`` with a 0.99
    matched-ratio gate plus a ``max(floor, 32 * ULP(g_low))`` max-abs-error cap.

    Thresholds are keyed by the actual output dtype. Non-finite points use the
    narrowing rule: golden is rounded (round-to-nearest-even) to the output
    dtype and same-sign +/-inf (and NaN/NaN) pairs pass.
    """
    rtol, atol, max_abs_floor = MIXED_TOLERANCE[actual.dtype]
    values = actual.float()
    reference = golden.float()
    assert values.shape == reference.shape, f"{name}: shapes differ {values.shape} != {reference.shape}"

    finite = torch.isfinite(values) & torch.isfinite(reference)
    error = (values - reference).abs()
    threshold = atol + rtol * reference.abs()
    narrowed = reference.to(actual.dtype).float()
    nonfinite_match = (torch.isnan(values) & torch.isnan(narrowed)) | (
        ~torch.isfinite(values) & ~torch.isfinite(narrowed) & (values == narrowed)
    )
    matched = torch.where(finite, error <= threshold, nonfinite_match)

    problems = []
    matched_ratio = float(matched.sum()) / matched.numel()
    if matched_ratio < REQUIRED_MATCHED_RATIO:
        problems.append(f"matched_ratio={matched_ratio:.6f} < {REQUIRED_MATCHED_RATIO}")
    finite_error = torch.where(finite, error, torch.zeros_like(error))
    max_abs_error = float(finite_error.max())
    peak = int(finite_error.reshape(-1).argmax())
    g_low = float(reference.reshape(-1)[peak].to(actual.dtype))
    limit = max(max_abs_floor, MAX_ABS_ERROR_ULP * _ulp_spacing(g_low, actual.dtype))
    if max_abs_error > limit:
        problems.append(f"max_abs_error={max_abs_error:.8g} > limit={limit:.8g}")
    if not problems:
        return

    flat_actual = values.reshape(-1)
    flat_golden = reference.reshape(-1)
    flat_error = finite_error.reshape(-1)
    worst = torch.topk(flat_error, min(max_print, flat_error.numel())).indices.tolist()
    details = [
        f"{name}[{index}]: actual={flat_actual[index].item():.8g}, "
        f"golden={flat_golden[index].item():.8g}, abs_error={flat_error[index].item():.8g}, "
        f"allowed={threshold.reshape(-1)[index].item():.8g}"
        for index in worst
        if flat_error[index] > 0
    ]
    raise AssertionError(
        f"{name} (dtype={actual.dtype}, rtol={rtol:g}, atol={atol:g}): " + "; ".join(problems)
        + ("\n  " + "\n  ".join(details) if details else "")
    )


def _check_against_oracle(payload, mean, rstd, *, M, N, dtype, group_size, is_rms_norm,
                          quant_heads, out_ref, mean_ref, rstd_ref, preallocated=None):
    ngroups = N // (group_size or N)
    if quant_heads:
        q, s = payload
        assert q.dtype == torch.float8_e4m3fn and q.shape == (M // quant_heads, quant_heads * N)
        assert s.dtype == torch.float32 and s.shape == (M // quant_heads, 1)
        q_ref, s_ref = fp8_quant_reference(out_ref, quant_heads, dtype)
        assert_mixed_tolerance(s, s_ref, name="scale")
        assert_mixed_tolerance(q, q_ref, name="quantized")
    else:
        assert payload.shape == (M, N) and payload.dtype == dtype
        if preallocated is not None:
            assert payload is preallocated
        assert_mixed_tolerance(payload, out_ref, name="output")
    if is_rms_norm:
        assert mean is None
    else:
        assert mean.dtype == torch.float32 and mean.shape == (ngroups * M,)
        assert_mixed_tolerance(mean, mean_ref, name="mean")
    assert rstd.dtype == torch.float32 and rstd.shape == (ngroups * M,)
    assert_mixed_tolerance(rstd, rstd_ref, name="rstd")


def check_gated_norm(backend, *, M, N, z_mode, activation, norm_before_gate, is_rms_norm,
                     dtype, group_size=None, eps=1e-6, quant_heads=0, out=None):
    """Check the selected implementation and the registered torch reference
    (migrated from SGLang) against the independent FP32 oracle."""
    implementation, _, device = backend
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    bias = None if is_rms_norm else torch.randn(N, device=device, dtype=dtype)
    z = make_gate(z_mode, M, N, dtype, device)

    payload, mean, rstd = F.layer_norm_gated(
        x, weight, bias, z=z, eps=eps, out=out, group_size=group_size,
        norm_before_gate=norm_before_gate, is_rms_norm=is_rms_norm,
        activation=activation, quant_heads=quant_heads, implementation=implementation,
    )
    reference = F.layer_norm_gated(
        x, weight, bias, z=z, eps=eps, group_size=group_size,
        norm_before_gate=norm_before_gate, is_rms_norm=is_rms_norm,
        activation=activation, quant_heads=quant_heads, implementation="torch",
    )

    out_ref, mean_ref, rstd_ref = gated_norm_reference(
        x, weight, bias, flatten_gate(z, M, N), eps=eps, group_size=group_size,
        norm_before_gate=norm_before_gate, is_rms_norm=is_rms_norm, activation=activation,
    )
    checks = dict(
        M=M, N=N, dtype=dtype, group_size=group_size, is_rms_norm=is_rms_norm,
        quant_heads=quant_heads, out_ref=out_ref, mean_ref=mean_ref, rstd_ref=rstd_ref,
    )
    _check_against_oracle(payload, mean, rstd, preallocated=out, **checks)
    _check_against_oracle(*reference, **checks)
    return payload, mean, rstd


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("z_mode", ("none", "2d", "3d"))
@pytest.mark.parametrize("activation", ("swish", "sigmoid"))
@pytest.mark.parametrize("norm_before_gate", (True, False))
@pytest.mark.parametrize("is_rms_norm", (True, False))
@pytest.mark.accuracy
def test_gated_norm_specializations(accuracy_backend, dtype, z_mode, activation,
                                    norm_before_gate, is_rms_norm):
    """Every HAS_Z x Z_IS_3D x ACTIVATION x NORM_BEFORE_GATE x IS_RMS_NORM branch."""
    check_gated_norm(
        accuracy_backend,
        M=128, N=128, z_mode=z_mode, activation=activation,
        norm_before_gate=norm_before_gate, is_rms_norm=is_rms_norm, dtype=dtype,
    )


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.parametrize(
    "M,N,z_mode,activation,norm_before_gate,is_rms_norm,dtype,eps", GATED_NORM_SHAPES
)
@pytest.mark.accuracy
def test_gated_norm_shapes(accuracy_backend, M, N, z_mode, activation,
                           norm_before_gate, is_rms_norm, dtype, eps):
    """Shape coverage of the task matrix; M=1 decodes, M=132 is the odd-token 3D gate."""
    check_gated_norm(
        accuracy_backend,
        M=M, N=N, z_mode=z_mode, activation=activation,
        norm_before_gate=norm_before_gate, is_rms_norm=is_rms_norm, dtype=dtype, eps=eps,
    )


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("M,is_rms_norm,z_mode,norm_before_gate", GROUP_NORM_CASES)
@pytest.mark.accuracy
def test_gated_norm_groups(accuracy_backend, dtype, M, is_rms_norm, z_mode, norm_before_gate):
    """GroupNorm with group_size=64 (ngroups=4); mean/rstd use the [ngroups*M] layout."""
    check_gated_norm(
        accuracy_backend,
        M=M, N=256, z_mode=z_mode, activation="swish",
        norm_before_gate=norm_before_gate, is_rms_norm=is_rms_norm,
        dtype=dtype, group_size=64,
    )


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("z_mode", ("none", "2d"))
@pytest.mark.parametrize("norm_before_gate", (True, False))
@pytest.mark.accuracy
def test_gated_norm_tail_mask(accuracy_backend, dtype, z_mode, norm_before_gate):
    """M=129 leaves one row in the final rows tile; N=192 forces BLOCK_N=256 with 64
    masked tail columns. ``out`` heads a sentinel buffer so escaped stores corrupt
    the guard tail."""
    M, N = 129, 192
    _, _, device = accuracy_backend
    storage = torch.full((M * N + GUARD_ELEMENTS,), 123.0, dtype=dtype, device=device)
    out = storage[: M * N].view(M, N)

    check_gated_norm(
        accuracy_backend,
        M=M, N=N, z_mode=z_mode, activation="swish",
        norm_before_gate=norm_before_gate, is_rms_norm=True, dtype=dtype, out=out,
    )

    guard = storage[M * N :]
    assert_bitwise_equal(guard, torch.full_like(guard, 123.0), phase="out-of-bounds canary")


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.accuracy
def test_gated_norm_strided_rows(accuracy_backend, dtype):
    """Row-strided x/out views (stride(-1) == 1) with a token-major 3D gate and a
    preallocated output buffer."""
    M, N = 132, 128
    implementation, _, device = accuracy_backend
    x = torch.randn(M, N + Z_PAD, device=device, dtype=dtype)[:, :N]
    weight = torch.randn(N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)
    z = make_gate("3d", M, N, dtype, device)
    out = torch.empty(M, N + Z_PAD, device=device, dtype=dtype)[:, :N]

    payload, mean, rstd = F.layer_norm_gated(
        x, weight, bias, z=z, eps=1e-5, out=out,
        norm_before_gate=True, is_rms_norm=False, activation="swish",
        implementation=implementation,
    )
    assert payload is out

    out_ref, mean_ref, rstd_ref = gated_norm_reference(
        x, weight, bias, flatten_gate(z, M, N), eps=1e-5,
        norm_before_gate=True, is_rms_norm=False, activation="swish",
    )
    assert_mixed_tolerance(payload, out_ref, name="output")
    assert_mixed_tolerance(mean, mean_ref, name="mean")
    assert_mixed_tolerance(rstd, rstd_ref, name="rstd")

    torch_out = torch.empty(M, N + Z_PAD, device=device, dtype=dtype)[:, :N]
    torch_payload, torch_mean, torch_rstd = F.layer_norm_gated(
        x, weight, bias, z=z, eps=1e-5, out=torch_out,
        norm_before_gate=True, is_rms_norm=False, activation="swish", implementation="torch",
    )
    assert torch_payload is torch_out
    assert_mixed_tolerance(torch_payload, out_ref, name="torch output")
    assert_mixed_tolerance(torch_mean, mean_ref, name="torch mean")
    assert_mixed_tolerance(torch_rstd, rstd_ref, name="torch rstd")


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("z_mode", ("none", "2d"))
@pytest.mark.parametrize("norm_before_gate", (True, False))
@pytest.mark.accuracy
def test_gated_norm_quant(accuracy_backend, dtype, z_mode, norm_before_gate):
    """Fused per-token FP8 epilogue (quant_heads=4) with and without the gate."""
    check_gated_norm(
        accuracy_backend,
        M=512, N=128, z_mode=z_mode, activation="swish",
        norm_before_gate=norm_before_gate, is_rms_norm=True,
        dtype=dtype, quant_heads=QUANT_HEADS,
    )


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.accuracy
def test_gated_norm_quant_layernorm(accuracy_backend):
    """QUANT x IS_RMS_NORM=False x Z_IS_3D specialization keeps the FP32 mean."""
    check_gated_norm(
        accuracy_backend,
        M=512, N=128, z_mode="3d", activation="sigmoid",
        norm_before_gate=True, is_rms_norm=False,
        dtype=torch.bfloat16, quant_heads=QUANT_HEADS,
    )


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("M,N", BITWISE_GATE_SHAPES)
@pytest.mark.parametrize("z_mode", ("3d", "2d"))
@pytest.mark.parametrize("norm_before_gate", (True, False))
@pytest.mark.accuracy
def test_strided_gate_matches_contiguous_gate(accuracy_backend, dtype, M, N,
                                              z_mode, norm_before_gate):
    """A token-major strided gate view and its contiguous copy agree bit-for-bit."""
    implementation, _, device = accuracy_backend
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    z = make_gate(z_mode, M, N, dtype, device)

    def run(gate):
        return F.layer_norm_gated(
            x, weight, None, z=gate, norm_before_gate=norm_before_gate,
            is_rms_norm=True, activation="swish", implementation=implementation,
        )

    strided = run(z)
    contiguous = run(z.contiguous())
    assert_bitwise_equal(strided, contiguous, phase="strided vs contiguous gate")


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("z_mode", ("none", "2d"))
@pytest.mark.parametrize("norm_before_gate", (True, False))
@pytest.mark.accuracy
def test_fused_quant_bit_exact(accuracy_backend, dtype, z_mode, norm_before_gate):
    """Fused (q, s) matches non-fused quantization of the same kernel output
    bit-for-bit; the round-to-output-dtype boundary is not perturbed twice."""
    M, N = 512, 128
    implementation, _, device = accuracy_backend
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    z = make_gate(z_mode, M, N, dtype, device)

    (q, s), _, _ = F.layer_norm_gated(
        x, weight, None, z=z, norm_before_gate=norm_before_gate,
        is_rms_norm=True, activation="swish", quant_heads=QUANT_HEADS,
        implementation=implementation,
    )
    out, _, _ = F.layer_norm_gated(
        x, weight, None, z=z, norm_before_gate=norm_before_gate,
        is_rms_norm=True, activation="swish", implementation=implementation,
    )
    assert_bitwise_equal((q, s), fp8_quant_reference(out, QUANT_HEADS), phase="fused vs non-fused")


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.parametrize("m_long", (131, 132))
@pytest.mark.parametrize("norm_before_gate", (True, False))
@pytest.mark.accuracy
def test_batch_invariant_rows_bitwise(accuracy_backend, monkeypatch, m_long, norm_before_gate):
    """LN_GATED_BATCH_INVARIANT=1 pins ROWS_PER_BLOCK=MAX_ROWS_PER_BLOCK=4: shared
    rows stay bit-identical across batch sizes."""
    monkeypatch.setenv("LN_GATED_BATCH_INVARIANT", "1")
    implementation, _, device = accuracy_backend
    m_shared, N = 128, 256
    x = torch.randn(m_long, N, device=device, dtype=torch.bfloat16)
    z = torch.randn(m_long, N, device=device, dtype=torch.bfloat16)
    weight = torch.randn(N, device=device, dtype=torch.bfloat16)

    def run(rows):
        return F.layer_norm_gated(
            x[:rows], weight, None, z=z[:rows], norm_before_gate=norm_before_gate,
            is_rms_norm=True, activation="swish", implementation=implementation,
        )

    out_shared, _, rstd_shared = run(m_shared)
    out_long, _, rstd_long = run(m_long)
    assert_bitwise_equal(
        (out_long[:m_shared], rstd_long[:m_shared]),
        (out_shared, rstd_shared),
        phase="shared rows across M",
    )

    out_ref, _, rstd_ref = gated_norm_reference(
        x[:m_shared], weight, None, z[:m_shared],
        norm_before_gate=norm_before_gate, is_rms_norm=True, activation="swish",
    )
    assert_mixed_tolerance(out_shared, out_ref, name="output")
    assert_mixed_tolerance(rstd_shared, rstd_ref, name="rstd")
