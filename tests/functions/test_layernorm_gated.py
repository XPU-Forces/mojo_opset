"""Accuracy suite for the gated normalization operator (``layer_norm_gated``).

The oracle is the task-book FP32 reference (``gated_norm_reference``): fp32
statistics, swish/sigmoid gate, affine, one final cast. The FP8 checks use the
documented non-fused quantizer (round to the output dtype, then amax,
``scale = max(amax, 1e-10) / 448``, clamp +-448) so the two paths share the
same arithmetic order and can be compared bit for bit.
"""

import pytest
import torch

from mojo_opset import functions as F

from .._checks import assert_close
from tests._fixtures import check_implementation

FP8_MAX = 448.0
GATE_HEADS = 4
GATE_PAD = 32


def _backend(accuracy_backend, op_id="layer_norm_gated"):
    """Resolve the implementation and skip cleanly when it has no provider.

    CI runs the same suite for triton, torch_npu and native; only the triton
    provider exists for this operator, so the other jobs must skip, not fail.
    """
    implementation, target, device = accuracy_backend
    return check_implementation([op_id], implementation, target)[op_id], device


def _gate(z, activation):
    if activation in ("swish", "silu"):
        return z * torch.sigmoid(z)
    if activation == "sigmoid":
        return torch.sigmoid(z)
    raise ValueError(f"Unsupported activation: {activation}")


def gated_norm_reference(x, weight, bias, eps=1e-6, z=None, group_size=None,
                         norm_before_gate=True, is_rms_norm=False, activation="swish"):
    """Independent FP32 oracle: statistics accumulated in FP32, gate, affine."""
    M, N = x.shape
    group_size = N if group_size is None else group_size
    ngroups = N // group_size
    values = x.float()
    z2d = None if z is None else (z.reshape(M, N) if z.ndim == 3 else z).float()
    if z2d is not None and not norm_before_gate:
        values = values * _gate(z2d, activation)
    grouped = values.reshape(M, ngroups, group_size)
    if is_rms_norm:
        mean = None
        rstd = torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
        out = grouped * rstd
    else:
        mean = grouped.mean(-1, keepdim=True)
        rstd = torch.rsqrt((grouped - mean).square().mean(-1, keepdim=True) + eps)
        out = (grouped - mean) * rstd
    out = out * weight.float().reshape(ngroups, group_size)
    if bias is not None:
        out = out + bias.float().reshape(ngroups, group_size)
    out = out.reshape(M, N)
    if z2d is not None and norm_before_gate:
        out = out * _gate(z2d, activation)
    mean_flat = None if mean is None else mean.reshape(M, ngroups).transpose(0, 1).reshape(-1)
    rstd_flat = rstd.reshape(M, ngroups).transpose(0, 1).reshape(-1)
    return out.to(x.dtype), mean_flat, rstd_flat


def per_token_group_quant_reference(y, quant_heads):
    """Non-fused per-token-group FP8 quantizer with the reference arithmetic."""
    dtype = y.dtype
    M, N = y.shape
    rounded = y.to(dtype).float().reshape(M // quant_heads, quant_heads * N)
    amax = rounded.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax, min=1e-10) / FP8_MAX
    q = torch.clamp(rounded * (1.0 / scale), -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def make_inputs(device, dtype, M, N, z_mode, group_size=None, has_bias=False, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype) if has_bias else None
    if z_mode == "none":
        z = None
    elif z_mode == "2d":
        z = torch.randn(M, N, device=device, dtype=dtype)
    else:
        assert M % GATE_HEADS == 0
        # token-major strided view: last dim contiguous, 32-element row padding
        base = torch.randn(M // GATE_HEADS, GATE_HEADS, N + GATE_PAD, device=device, dtype=dtype)
        z = base[..., :N]
    return x, weight, bias, z


def run_op(implementation, x, weight, bias, z, group_size, norm_before_gate, is_rms_norm, activation, eps=1e-6):
    return F.layer_norm_gated(
        x,
        weight,
        bias,
        eps,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
        implementation=implementation,
    )


MATRIX_SHAPES = [(1, 128), (128, 128), (132, 128), (512, 128),
                 (128, 256), (512, 256), (128, 1024), (512, 1024)]


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("M,N", MATRIX_SHAPES)
@pytest.mark.parametrize("z_mode", ("none", "2d", "3d"))
def test_layer_norm_gated_accuracy(accuracy_backend, dtype, M, N, z_mode):
    implementation, device = _backend(accuracy_backend)
    if z_mode == "3d" and M % GATE_HEADS:
        pytest.skip("3D gate requires M divisible by Z_HEADS")
    x, weight, bias, z = make_inputs(device, dtype, M, N, z_mode)
    actual, mean, rstd = run_op(implementation, x, weight, bias, z, None, True, True, "swish")
    expected, mean_ref, rstd_ref = gated_norm_reference(x, weight, None, z=z, is_rms_norm=True)
    assert actual.shape == x.shape and actual.dtype == x.dtype
    assert_close(actual, expected, dtype, name=f"y {dtype} {M}x{N} z={z_mode}")
    assert mean is None
    assert_close(rstd, rstd_ref, torch.float32, name=f"rstd {dtype} {M}x{N} z={z_mode}")


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("activation", ("swish", "sigmoid"))
@pytest.mark.parametrize("norm_before_gate", (True, False))
@pytest.mark.parametrize("bias", (True, False))
@pytest.mark.parametrize("z_mode", ("none", "2d", "3d"))
def test_layer_norm_gated_variants(accuracy_backend, dtype, activation, norm_before_gate, bias, z_mode):
    implementation, device = _backend(accuracy_backend)
    M, N = 132, 256
    if z_mode == "none" and not norm_before_gate:
        pytest.skip("gate order is meaningless without z")
    x, weight, bias_t, z = make_inputs(device, dtype, M, N, z_mode, has_bias=bias, seed=7)
    actual, mean, rstd = run_op(implementation, x, weight, bias_t, z, None, norm_before_gate, False, activation)
    expected, mean_ref, rstd_ref = gated_norm_reference(
        x, weight, bias_t, z=z, norm_before_gate=norm_before_gate, is_rms_norm=False, activation=activation
    )
    name = f"{dtype} {activation} before={norm_before_gate} bias={bias} z={z_mode}"
    assert_close(actual, expected, dtype, name=f"y {name}")
    assert_close(mean, mean_ref, torch.float32, name=f"mean {name}")
    assert_close(rstd, rstd_ref, torch.float32, name=f"rstd {name}")


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("M", (128, 512))
@pytest.mark.parametrize("z_mode", ("none", "2d"))
@pytest.mark.parametrize("is_rms_norm", (True, False))
def test_group_norm(accuracy_backend, dtype, M, z_mode, is_rms_norm):
    implementation, device = _backend(accuracy_backend)
    N, group_size = 256, 64
    x, weight, bias, z = make_inputs(device, dtype, M, N, z_mode, has_bias=not is_rms_norm, seed=11)
    actual, mean, rstd = run_op(implementation, x, weight, bias, z, group_size, True, is_rms_norm, "swish")
    expected, mean_ref, rstd_ref = gated_norm_reference(
        x, weight, bias, group_size=group_size, z=z, is_rms_norm=is_rms_norm
    )
    name = f"{dtype} M{M} z={z_mode} rms={is_rms_norm}"
    assert_close(actual, expected, dtype, name=f"y {name}")
    assert_close(rstd, rstd_ref, torch.float32, name=f"rstd {name}")
    if not is_rms_norm:
        assert_close(mean, mean_ref, torch.float32, name=f"mean {name}")


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("z_mode", ("none", "2d"))
def test_tail_block(accuracy_backend, dtype, z_mode):
    """One row past the tile and 64 masked columns (BLOCK_N=256, valid 192)."""
    implementation, device = _backend(accuracy_backend)
    M, N = 129, 192
    x, weight, bias, z = make_inputs(device, dtype, M, N, z_mode, seed=13)
    actual, _, rstd = run_op(implementation, x, weight, bias, z, None, True, True, "swish")
    expected, _, rstd_ref = gated_norm_reference(x, weight, None, z=z, is_rms_norm=True)
    assert_close(actual, expected, dtype, name=f"tail y {dtype} z={z_mode}")
    assert_close(rstd, rstd_ref, torch.float32, name=f"tail rstd {dtype} z={z_mode}")


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.accuracy
def test_preallocated_out(accuracy_backend):
    implementation, device = _backend(accuracy_backend)
    dtype = torch.bfloat16
    x, weight, bias, z = make_inputs(device, dtype, 128, 256, "2d", seed=17)
    out = torch.empty_like(x)
    with torch.no_grad():
        actual, _, _ = F.layer_norm_gated(
            x, weight, None, 1e-6, z=z, out=out, is_rms_norm=True, implementation=implementation
        )
    expected, _, _ = gated_norm_reference(x, weight, None, z=z, is_rms_norm=True)
    assert actual.data_ptr() == out.data_ptr()
    assert_close(actual, expected, dtype, name="preallocated out")


def _quant_backend(accuracy_backend):
    implementation, device = _backend(accuracy_backend)
    target = accuracy_backend[1]
    if not target.startswith("npu.a5") and implementation != "torch":
        pytest.skip("fused FP8 epilogue is scoped to Ascend 950")
    return implementation, device


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.accuracy
@pytest.mark.parametrize("z_mode", ("none", "2d"))
def test_fp8_quant_bit_exact(accuracy_backend, z_mode):
    """Fused ``(q, s)`` must equal the non-fused quantizer byte for byte."""
    implementation, device = _quant_backend(accuracy_backend)
    dtype = torch.bfloat16
    M, N, quant_heads = 512, 128, 4
    x, weight, _, z = make_inputs(device, dtype, M, N, z_mode, seed=19)
    (q, s), _, _ = F.layer_norm_gated(
        x, weight, None, 1e-6, z=z, is_rms_norm=True, quant_heads=quant_heads, implementation=implementation
    )
    y, _, _ = F.layer_norm_gated(x, weight, None, 1e-6, z=z, is_rms_norm=True, implementation=implementation)
    q_ref, s_ref = per_token_group_quant_reference(y, quant_heads)
    assert q.shape == q_ref.shape and q.dtype == torch.float8_e4m3fn
    assert torch.equal(s, s_ref), "scales differ"
    if not torch.equal(q.view(torch.uint8).cpu(), q_ref.view(torch.uint8).cpu()):
        # Diagnose the cast mode instead of only reporting a byte count: the
        # contract assumes round-to-nearest-even on both sides.
        scaled = y.to(dtype).float().reshape(M // quant_heads, quant_heads * N) * (1.0 / s_ref)
        truncated = scaled.trunc().to(torch.float8_e4m3fn)
        rne_hits = int((q.view(torch.uint8) == q_ref.view(torch.uint8)).sum())
        trunc_hits = int((q.view(torch.uint8) == truncated.view(torch.uint8)).sum())
        raise AssertionError(
            f"fp8 bytes differ from the non-fused quantizer: rne_matches={rne_hits}, "
            f"truncation_matches={trunc_hits} of {q.numel()}; "
            "a truncation match means the device cast is not round-to-nearest"
        )


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.accuracy
@pytest.mark.parametrize("z_mode", ("none", "2d"))
def test_fp8_quant_accuracy(accuracy_backend, z_mode):
    implementation, device = _quant_backend(accuracy_backend)
    dtype = torch.bfloat16
    M, N, quant_heads = 512, 128, 4
    x, weight, _, z = make_inputs(device, dtype, M, N, z_mode, seed=23)
    (q, s), _, _ = F.layer_norm_gated(
        x, weight, None, 1e-6, z=z, is_rms_norm=True, quant_heads=quant_heads, implementation=implementation
    )
    expected, _, _ = gated_norm_reference(x, weight, None, z=z, is_rms_norm=True)
    # e4m3 has three mantissa bits: max relative step 1/16 after round-to-nearest.
    dequant = (q.float() * s).reshape(M, N)
    assert_close(dequant, expected.float(), torch.float32, rtol=6.25e-2, atol=1e-2, name="dequantized")


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("norm_before_gate", (True, False))
def test_strided_gate_matches_contiguous(accuracy_backend, dtype, norm_before_gate):
    """A token-major strided gate must be bit-identical to its contiguous view."""
    implementation, device = _backend(accuracy_backend)
    M, N = 132, 128
    x, weight, _, z = make_inputs(device, dtype, M, N, "3d", seed=29)
    assert z.stride(-1) == 1 and z.stride(1) == N + GATE_PAD
    actual, _, _ = run_op(implementation, x, weight, None, z, None, norm_before_gate, True, "swish")
    expected, _, _ = run_op(
        implementation, x, weight, None, z.contiguous().view(M, N), None, norm_before_gate, True, "swish"
    )
    assert torch.equal(actual, expected), "strided gate diverged from the contiguous gate"


@pytest.mark.api("functions.layer_norm_gated")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_batch_invariant(accuracy_backend, monkeypatch, dtype):
    """``LN_GATED_BATCH_INVARIANT=1`` makes shared rows independent of the batch size."""
    implementation, device = _backend(accuracy_backend)
    monkeypatch.setenv("LN_GATED_BATCH_INVARIANT", "1")
    x, weight, _, z = make_inputs(device, dtype, 512, 256, "2d", seed=31)
    full, _, full_rstd = run_op(implementation, x, weight, None, z, None, True, True, "swish")
    part, _, part_rstd = run_op(implementation, x[:128], weight, None, z[:128], None, True, True, "swish")
    assert torch.equal(full[:128], part), "shared rows changed with batch size"
    assert torch.equal(full_rstd[:128], part_rstd), "shared rstd changed with batch size"


@pytest.mark.api("functions.rms_norm_gated")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("z_mode", ("none", "2d", "3d"))
def test_rms_norm_gated_api(accuracy_backend, dtype, z_mode):
    """The ``rms_norm_gated`` convenience wrapper matches the FP32 oracle."""
    implementation, device = _backend(accuracy_backend)
    x, weight, _, z = make_inputs(device, dtype, 128, 128, z_mode, seed=23)
    with torch.no_grad():
        actual = F.rms_norm_gated(
            x=x, weight=weight, bias=None, z=z, eps=1e-6, is_rms_norm=True,
            implementation=implementation,
        )
    expected, _, _ = gated_norm_reference(x, weight, None, z=z, is_rms_norm=True)
    assert actual.shape == x.shape and actual.dtype == dtype
    assert_close(actual, expected, dtype, name="rms_norm_gated")


@pytest.mark.reference
def test_batch_invariant_tile_height():
    """Batch-invariant mode pins the row tile to MAX_ROWS_PER_BLOCK regardless of shape."""
    from mojo_opset.kernels._layernorm_gated_impl import MAX_ROWS_PER_BLOCK, calc_rows_per_block

    import os

    previous = os.environ.get("LN_GATED_BATCH_INVARIANT")
    os.environ["LN_GATED_BATCH_INVARIANT"] = "1"
    try:
        for M in (1, 129, 4096, 524288):
            assert calc_rows_per_block(M, 128) == MAX_ROWS_PER_BLOCK
    finally:
        if previous is None:
            os.environ.pop("LN_GATED_BATCH_INVARIANT", None)
        else:
            os.environ["LN_GATED_BATCH_INVARIANT"] = previous
