"""Tests for functions.layernorm_gated_infer.

The accuracy reference is ``implementation="torch"``, i.e. the eager operator chain
(FP32 reductions + ``silu``/``sigmoid`` gate + affine), which is the independent yardstick the
task statement names. The Triton kernel also accumulates in FP32 and rounds once at the store,
so the two may differ only by the final rounding -- hence a dtype-tiered tolerance, not an
exact match.

The ``bitwise`` cases cover the contracts that must hold exactly regardless of reduction
order, because they are the ones that catch real implementation bugs rather than rounding:

* a token-major strided 3-D gate must produce the same bits as the contiguous gate;
* the batch-invariant mode must produce the same bits for a shared row across different ``M``;
* the fused FP8 epilogue must equal the unfused per-token quantisation, including the
  "round to the output dtype before the amax" step.
"""

import os

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_close

EPS = 1e-6
DTYPES = (torch.bfloat16, torch.float16)
ACTIVATIONS = ("swish", "sigmoid")

# Task statement test matrix: single-row decode, GroupNorm batch, odd token count for the
# strided gate (33 x 4), and a regular prefill batch.
ROW_COUNTS = (1, 128, 132, 512)
NORM_SIZES = (128, 256, 1024)

QUANT_MAX = 448.0


def _strided_gate(t, h, d, device, dtype, pad=32, seed=7):
    """A token-major gate whose head stride is padded, so the view is genuinely strided.

    ``stride(-1) == 1`` still holds, which is the only layout guarantee the operator makes.
    """
    gen = torch.Generator().manual_seed(seed)
    base = torch.randn(t, h, d + pad, generator=gen).to(dtype).to(device)
    return base[:, :, :d]


def _make_case(backend, m, n, dtype, *, gate="none", activation="swish", norm_before_gate=True,
               is_rms_norm=False, has_bias=False, group_size=None, z_heads=4, seed=0):
    _impl, _target, device = backend
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(m, n, generator=gen) * 0.5).to(dtype).to(device)
    weight = (torch.rand(n, generator=gen) + 0.5).to(dtype).to(device)
    bias = (torch.randn(n, generator=gen) * 0.1).to(dtype).to(device) if has_bias else None
    z = None
    if gate == "2d":
        z = (torch.randn(m, n, generator=gen) * 0.5).to(dtype).to(device)
    elif gate == "3d":
        if m % z_heads:
            z_heads = 1
        z = _strided_gate(m // z_heads, z_heads, n, device, dtype, seed=seed + 7)
    return x, weight, bias, z, dict(
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation=activation,
    )


def _run(backend, x, weight, bias, z, kwargs, implementation):
    return F.layernorm_gated_infer(
        x, weight, bias, EPS, z=z, implementation=implementation, **kwargs
    )


def _check(backend, m, n, dtype, name, **kw):
    x, weight, bias, z, kwargs = _make_case(backend, m, n, dtype, **kw)
    expected = _run(backend, x, weight, bias, z, kwargs, "torch")
    actual = _run(backend, x, weight, bias, z, kwargs, None)

    assert_close(actual[0], expected[0], dtype, name=f"{name} out")
    if expected[1] is not None:
        assert actual[1] is not None, f"{name}: mean must be returned for LayerNorm"
        assert_close(actual[1], expected[1], torch.float32, name=f"{name} mean")
    else:
        assert actual[1] is None, f"{name}: RMSNorm must not return a mean"
    assert_close(actual[2], expected[2], torch.float32, name=f"{name} rstd")
    return actual, expected


@pytest.mark.api("functions.layernorm_gated_infer")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("m", ROW_COUNTS, ids=lambda v: f"m{v}")
@pytest.mark.parametrize("n", NORM_SIZES, ids=lambda v: f"n{v}")
@pytest.mark.accuracy
def test_layernorm_gated_plain(accuracy_backend, dtype, m, n):
    """No gate: LayerNorm and RMSNorm over the whole row, with and without bias."""
    _check(accuracy_backend, m, n, dtype, f"ln {m}x{n}")
    _check(accuracy_backend, m, n, dtype, f"ln+bias {m}x{n}", has_bias=True)
    _check(accuracy_backend, m, n, dtype, f"rms {m}x{n}", is_rms_norm=True)


@pytest.mark.api("functions.layernorm_gated_infer")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("norm_before_gate", (True, False))
@pytest.mark.accuracy
def test_layernorm_gated_2d_gate(accuracy_backend, dtype, activation, norm_before_gate):
    """2-D gate x activation x gate order, the GDN out-norm path."""
    _check(
        accuracy_backend,
        128,
        128,
        dtype,
        f"z2d {activation} nbg{norm_before_gate}",
        gate="2d",
        activation=activation,
        norm_before_gate=norm_before_gate,
        is_rms_norm=True,
    )


@pytest.mark.api("functions.layernorm_gated_infer")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.accuracy
def test_layernorm_gated_3d_strided_gate(accuracy_backend, dtype, activation):
    """3-D token-major gate with a padded head stride: Z_IS_3D x Z_HEADS addressing."""
    # 33 tokens x 4 heads = 132 rows, the odd-token case from the matrix.
    _check(
        accuracy_backend,
        132,
        128,
        dtype,
        f"z3d {activation}",
        gate="3d",
        activation=activation,
        z_heads=4,
    )


@pytest.mark.api("functions.layernorm_gated_infer")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.accuracy
def test_layernorm_gated_group_norm(accuracy_backend, dtype):
    """ngroups = 4: the reduction must stay inside each 64-wide group."""
    _check(accuracy_backend, 128, 256, dtype, "group", group_size=64, has_bias=True)
    _check(
        accuracy_backend, 128, 256, dtype, "group+gate", group_size=64, gate="2d", has_bias=True
    )


@pytest.mark.api("functions.layernorm_gated_infer")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.accuracy
def test_layernorm_gated_tail_block(accuracy_backend, dtype):
    """M=129 leaves one row past the tile; N=192 sits inside a 256-wide block.

    The mask therefore has to clip both axes, and the tail must not be written.
    """
    _check(accuracy_backend, 129, 192, dtype, "tail plain", is_rms_norm=True)
    _check(accuracy_backend, 129, 192, dtype, "tail gate", gate="2d", is_rms_norm=True)


@pytest.mark.api("functions.layernorm_gated_infer")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.bitwise
def test_layernorm_gated_strided_gate_bitwise(accuracy_backend, dtype):
    """Strided 3-D gate must equal its contiguous copy bit for bit.

    Zero tolerance: the values read are identical, so any difference is an addressing bug in
    the ``(row // Z_HEADS) * stride_z_token + (row % Z_HEADS) * stride_z_head`` decomposition.
    """
    _impl, _target, device = accuracy_backend
    gen = torch.Generator().manual_seed(11)
    m, n, heads = 132, 128, 4
    x = (torch.randn(m, n, generator=gen) * 0.5).to(dtype).to(device)
    weight = (torch.rand(n, generator=gen) + 0.5).to(dtype).to(device)
    z_strided = _strided_gate(m // heads, heads, n, device, dtype, seed=11)

    kwargs = dict(norm_before_gate=True, is_rms_norm=True, activation="swish")
    a = F.layernorm_gated_infer(x, weight, None, EPS, z=z_strided, **kwargs)[0]
    b = F.layernorm_gated_infer(x, weight, None, EPS, z=z_strided.contiguous(), **kwargs)[0]
    assert torch.equal(a, b), "strided 3-D gate diverged from its contiguous copy"


@pytest.mark.api("functions.layernorm_gated_infer")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.bitwise
def test_layernorm_gated_batch_invariant_bitwise(accuracy_backend, dtype):
    """A shared row must produce the same bits whatever the batch size is.

    ``batch_invariant=True`` pins the tile to ``MAX_ROWS_PER_BLOCK``, so the reduction order
    cannot depend on ``M``. Two prefixes of the same buffer are compared, which is exactly what
    a serving stack does when a request joins an existing batch.
    """
    _impl, _target, device = accuracy_backend
    gen = torch.Generator().manual_seed(3)
    n = 128
    big = (torch.randn(512, n, generator=gen) * 0.5).to(dtype).to(device)
    weight = (torch.rand(n, generator=gen) + 0.5).to(dtype).to(device)
    kwargs = dict(group_size=None, is_rms_norm=True, batch_invariant=True)

    short = F.layernorm_gated_infer(big[:128], weight, None, EPS, **kwargs)[0]
    long = F.layernorm_gated_infer(big, weight, None, EPS, **kwargs)[0]
    assert torch.equal(short, long[:128]), "batch-invariant mode leaked M into the numerics"


@pytest.mark.api("functions.layernorm_gated_infer")
@pytest.mark.accuracy
def test_layernorm_gated_quant(accuracy_backend):
    """Fused FP8 per-token quant, bit-exact against the unfused formula.

    Scoped to 950 by the task statement: the A2 part has no FP8 storage path, so the case is
    skipped there instead of being weakened.
    """
    _impl, target, device = accuracy_backend
    if "a5" not in str(target):
        pytest.skip("FP8 quant epilogue is required on 950 only")

    gen = torch.Generator().manual_seed(5)
    m, n, heads = 512, 128, 4
    x = (torch.randn(m, n, generator=gen) * 0.5).to(torch.bfloat16).to(device)
    weight = (torch.rand(n, generator=gen) + 0.5).to(torch.bfloat16).to(device)
    for gate in (None, "2d"):
        z = (torch.randn(m, n, generator=gen) * 0.5).to(torch.bfloat16).to(device) if gate else None
        kwargs = dict(is_rms_norm=True, activation="swish", quant_heads=heads)
        (q, s), _mean, _rstd = F.layernorm_gated_infer(x, weight, None, EPS, z=z, **kwargs)
        (q_ref, s_ref), _, _ = F.layernorm_gated_infer(
            x, weight, None, EPS, z=z, implementation="torch", **kwargs
        )
        assert torch.equal(q, q_ref), f"FP8 q mismatch (gate={gate})"
        assert torch.equal(s, s_ref), f"FP8 scale mismatch (gate={gate})"


@pytest.mark.api("functions.layernorm_gated_infer")
def test_layernorm_gated_rejects_bad_geometry(accuracy_backend):
    _impl, _target, device = accuracy_backend
    x = torch.randn(8, 128, device=device, dtype=torch.bfloat16)
    w = torch.randn(128, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="must divide"):
        F.layernorm_gated_infer(x, w, None, EPS, group_size=48)
    with pytest.raises(ValueError, match="must divide"):
        F.layernorm_gated_infer(x, w, None, EPS, quant_heads=3)
    with pytest.raises(ValueError, match="activation"):
        F.layernorm_gated_infer(x, w, None, EPS, activation="relu", z=x)
    with pytest.raises(ValueError, match="must match x dtype"):
        F.layernorm_gated_infer(x, w, None, EPS, z=x.float())
    with pytest.raises(ValueError, match="must be \\[128\\]"):
        F.layernorm_gated_infer(x, w[:64], None, EPS)


@pytest.mark.api("functions.layernorm_gated_infer")
@pytest.mark.accuracy
def test_layernorm_gated_out_buffer(accuracy_backend):
    """A caller-provided ``out`` is written in place and returned, as upstream does."""
    _impl, _target, device = accuracy_backend
    x = torch.randn(16, 128, device=device, dtype=torch.bfloat16)
    w = torch.rand(128, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    got = F.layernorm_gated_infer(x, w, None, EPS, out=out, is_rms_norm=True)[0]
    assert got.data_ptr() == out.data_ptr(), "out= must be returned, not reallocated"
