"""Tests for functions.qk_norm_rope_infer.

The accuracy reference is ``implementation="torch"``, i.e. this package's port of upstream's
``_eager_qk_norm_rope`` chain (``F.rms_norm`` + packed-table rotation in the activation dtype),
exactly the yardstick named by the task statement. The Triton kernel also rounds the normalized
value to BF16 *before* rotating, so the two differ only by BF16 rounding noise; the tolerance
therefore has to be a BF16 tolerance, not an exact match.

The ``bitwise`` cases target contracts that must hold exactly regardless of the reduction
order, because they are the ones that catch real implementation bugs:

* strided (packed-QKV slice) input must produce the same bits as a contiguous copy;
* a head's output must not depend on which token position it sits in;
* dimensions beyond ``rotary_dim`` must be the BF16-rounded norm value with no second
  perturbation (checked with a construction where that value is exactly representable).

Geometry coverage: the Triton kernel is measured on the MiniMax-H3 contract (``head_dim=128``,
``rotary_dim=96``) only. The ``reference``-marked cases below exercise other geometries against
the torch chain, so they pin the wrapper's contract checks rather than the kernel. The kernel's
``d_pad`` masking is written for a non-power-of-two ``head_dim`` but has not been measured on
hardware yet.
"""

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_close

HEAD_DIM = 128
ROTARY_DIM = 96
EPS = 1e-6
TAU = 6.283185307179586
# Activation dtype of the NPU contract; FP16/FP32 stay in the eager domain upstream.
DTYPE = torch.bfloat16
ROPE_DTYPES = (torch.bfloat16, torch.float32)

# (tokens, heads): upstream's own cases plus the MiniMax-H3 geometry and the long-video grid.
STANDARD_SHAPES = [(1, 8), (1, 56), (257, 8), (257, 56), (1024, 56), (4096, 56)]
LONG_SHAPES = [(16384, 56)]
# 57 heads: the 8-head tile leaves a single valid head in the last group.
HEAD_TAIL_SHAPES = [(257, 57), (4096, 57)]
# (head_dim, rotary_dim) beyond the MiniMax-H3 contract, reference-path only.
GEOMETRIES = [(128, 96), (128, 128), (128, 32), (64, 64), (100, 64)]

# Upstream's own hardware tolerance for this operator
# (vllm-omni tests/diffusion/layers/test_fused_qk_norm_rope_npu.py:193
#  ``assert_close(actual, expected, atol=2e-2, rtol=2e-2)``). The eager yardstick rotates in
# BF16 while the kernel rounds the norm value to BF16 and rotates in FP32, so a one-ULP
# difference in the rounded norm shows up as roughly 0.4% of the value -- well inside this
# band, and deliberately *not* tightened, because tightening it would force the kernel to
# copy the reference's rounding order instead of upstream's.
UPSTREAM_TOLERANCE = {"rtol": 2e-2, "atol": 2e-2}


def make_inputs(tokens, heads, device, rope_dtype=torch.bfloat16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    half = ROTARY_DIM // 2
    theta = torch.rand(tokens, half, generator=gen, dtype=torch.float32) * TAU
    rope = torch.cat((theta.cos(), theta.sin()), dim=-1).to(rope_dtype).to(device)
    q = (torch.randn(tokens, heads, HEAD_DIM, generator=gen) * 0.5).to(DTYPE).to(device)
    k = (torch.randn(tokens, heads, HEAD_DIM, generator=gen) * 0.5).to(DTYPE).to(device)
    qw = (torch.rand(HEAD_DIM, generator=gen) + 0.5).to(DTYPE).to(device)
    kw = (torch.rand(HEAD_DIM, generator=gen) + 0.5).to(DTYPE).to(device)
    return q, k, qw, kw, rope


def packed_qkv_view(tokens, heads, device, rope_dtype=torch.bfloat16, seed=0):
    """Q and K interleaved in one packed buffer, the way fused QKV storage keeps them.

    Interleaving rather than concatenating is what makes the head stride ``2 * head_dim``,
    i.e. a genuinely strided view even for a single token. A concatenated slice stays
    contiguous when ``tokens == 1`` because PyTorch ignores the stride of a size-1 dimension,
    so it would not exercise the strided path at all.
    """
    q, k, qw, kw, rope = make_inputs(tokens, heads, device, rope_dtype, seed)
    packed = torch.zeros(tokens, heads * 2, HEAD_DIM, dtype=DTYPE, device=device)
    packed[:, ::2] = q
    packed[:, 1::2] = k
    q_view = packed[:, ::2]
    k_view = packed[:, 1::2]
    assert q_view.stride(1) == 2 * HEAD_DIM and q_view.stride(2) == 1
    assert not q_view.is_contiguous()
    return q_view, k_view, qw, kw, rope


def check_accuracy(backend, shape, rope_dtype, strided):
    implementation, _target, device = backend
    tokens, heads = shape
    if strided:
        q, k, qw, kw, rope = packed_qkv_view(tokens, heads, device, rope_dtype)
    else:
        q, k, qw, kw, rope = make_inputs(tokens, heads, device, rope_dtype)

    expected = F.qk_norm_rope_infer(q, k, qw, kw, rope, EPS, implementation="torch")
    actual = F.qk_norm_rope_infer(q, k, qw, kw, rope, EPS, implementation=implementation)

    for name, got, ref in (("q", actual[0], expected[0]), ("k", actual[1], expected[1])):
        assert got.shape == ref.shape and got.dtype == ref.dtype
        assert_close(
            got,
            ref,
            DTYPE,
            name=f"{implementation} vs torch {name} {shape} rope={str(rope_dtype).split('.')[-1]} "
            f"strided={strided}",
            **UPSTREAM_TOLERANCE,
        )
    return actual


@pytest.mark.api("functions.qk_norm_rope_infer")
@pytest.mark.parametrize("rope_dtype", ROPE_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("strided", [False, True], ids=["contig", "strided"])
@pytest.mark.parametrize("shape", STANDARD_SHAPES, ids=[f"{t}x{h}" for t, h in STANDARD_SHAPES])
@pytest.mark.accuracy
def test_qk_norm_rope_accuracy(accuracy_backend, shape, strided, rope_dtype):
    check_accuracy(accuracy_backend, shape, strided, rope_dtype)


@pytest.mark.api("functions.qk_norm_rope_infer")
@pytest.mark.parametrize("shape", LONG_SHAPES, ids=[f"{t}x{h}" for t, h in LONG_SHAPES])
@pytest.mark.accuracy
def test_qk_norm_rope_long_sequence(accuracy_backend, shape):
    """16K tokens exercises the int64 token offset path upstream guards against."""
    check_accuracy(accuracy_backend, shape, torch.bfloat16, False)


@pytest.mark.api("functions.qk_norm_rope_infer")
@pytest.mark.parametrize("strided", [False, True], ids=["contig", "strided"])
@pytest.mark.parametrize("shape", HEAD_TAIL_SHAPES, ids=[f"{t}x{h}" for t, h in HEAD_TAIL_SHAPES])
@pytest.mark.accuracy
def test_qk_norm_rope_head_tail(accuracy_backend, shape, strided):
    """57 heads: the last 8-head tile has one valid head; the mask must not write past it."""
    actual = check_accuracy(accuracy_backend, shape, torch.bfloat16, strided)
    assert torch.isfinite(actual[0]).all() and torch.isfinite(actual[1]).all()


@pytest.mark.api("functions.qk_norm_rope_infer")
@pytest.mark.bitwise
def test_qk_norm_rope_strided_bitwise(accuracy_backend):
    """A packed-QKV slice must give bit-identical results to a contiguous copy."""
    implementation, _target, device = accuracy_backend
    tokens, heads = 1024, 56
    q_view, k_view, qw, kw, rope = packed_qkv_view(tokens, heads, device)
    q_dense, k_dense = q_view.contiguous(), k_view.contiguous()

    got = F.qk_norm_rope_infer(q_view, k_view, qw, kw, rope, EPS, implementation=implementation)
    ref = F.qk_norm_rope_infer(q_dense, k_dense, qw, kw, rope, EPS, implementation=implementation)
    assert torch.equal(got[0], ref[0]), "strided q differs from contiguous q"
    assert torch.equal(got[1], ref[1]), "strided k differs from contiguous k"


@pytest.mark.api("functions.qk_norm_rope_infer")
@pytest.mark.bitwise
def test_qk_norm_rope_noncontiguous_side_inputs(accuracy_backend):
    """A strided weight or rope_table must be materialised, not read at unit stride.

    The kernel loads both with unit stride, so the wrapper has to copy a non-contiguous view.
    Reading it in place stays shape-valid and only shows up as a wrong result, which is what
    this case catches.
    """
    implementation, _target, device = accuracy_backend
    tokens, heads = 64, 8
    q, k, qw, kw, rope = make_inputs(tokens, heads, device)

    wide_rope = torch.zeros(tokens, ROTARY_DIM * 2, dtype=rope.dtype, device=device)
    wide_rope[:, ::2] = rope
    rope_view = wide_rope[:, ::2]
    wide_weight = torch.zeros(HEAD_DIM * 2, dtype=qw.dtype, device=device)
    wide_weight[::2] = qw
    weight_view = wide_weight[::2]
    assert not rope_view.is_contiguous() and not weight_view.is_contiguous()

    got = F.qk_norm_rope_infer(q, k, weight_view, kw, rope_view, EPS, implementation=implementation)
    ref = F.qk_norm_rope_infer(q, k, qw, kw, rope, EPS, implementation=implementation)
    assert torch.equal(got[0], ref[0]), "q differs when the weight/table are strided"
    assert torch.equal(got[1], ref[1]), "k differs when the weight/table are strided"


@pytest.mark.api("functions.qk_norm_rope_infer")
@pytest.mark.bitwise
def test_qk_norm_rope_token_independence_bitwise(accuracy_backend):
    """One row's output must not depend on the token position it occupies.

    The kernel tiles several tokens into one program, so an indexing slip across the tile
    would show up here and nowhere else.
    """
    implementation, _target, device = accuracy_backend
    gen = torch.Generator().manual_seed(7)
    tokens, heads = 256, 56
    half = ROTARY_DIM // 2
    theta_single = torch.rand(1, half, generator=gen, dtype=torch.float32) * TAU
    rope_single = torch.cat((theta_single.cos(), theta_single.sin()), dim=-1).to(DTYPE).to(device)
    row_q = (torch.randn(1, heads, HEAD_DIM, generator=gen) * 0.5).to(DTYPE)
    qw = (torch.rand(HEAD_DIM, generator=gen) + 0.5).to(DTYPE).to(device)
    kw = qw.clone()

    single = F.qk_norm_rope_infer(
        row_q.to(device), row_q.to(device), qw, kw, rope_single, EPS, implementation=implementation
    )
    # Same row repeated: every token must reproduce the single-token result exactly.
    rope_many = rope_single.expand(tokens, -1).contiguous()
    row_many = row_q.expand(tokens, -1, -1).contiguous().to(device)
    many = F.qk_norm_rope_infer(row_many, row_many, qw, kw, rope_many, EPS, implementation=implementation)
    assert torch.equal(many[0], single[0].expand(tokens, -1, -1))


@pytest.mark.api("functions.qk_norm_rope_infer")
@pytest.mark.bitwise
def test_qk_norm_rope_pass_dims_bitwise(accuracy_backend):
    """Dimensions >= rotary_dim must be the BF16-rounded norm value, untouched.

    Construction: a constant activation ``a`` with unit weight and ``eps=0`` gives
    ``inv_rms = rsqrt(a^2) ``; with ``a = 1.0`` that is exactly 1.0, so the norm value is
    exactly ``1.0`` and every pass-through dimension must come out as exactly ``1.0``.
    A second perturbation (double rounding, stray scaling) would break this equality.
    """
    implementation, _target, device = accuracy_backend
    tokens, heads = 320, 8
    ones = torch.ones(tokens, heads, HEAD_DIM, dtype=DTYPE, device=device)
    weight = torch.ones(HEAD_DIM, dtype=DTYPE, device=device)
    half = ROTARY_DIM // 2
    rope = torch.cat(
        (torch.ones(tokens, half, device=device), torch.zeros(tokens, half, device=device)), dim=-1
    ).to(DTYPE)

    q_out, k_out = F.qk_norm_rope_infer(ones, ones, weight, weight, rope, 0.0, implementation=implementation)
    for name, out in (("q", q_out), ("k", k_out)):
        passthrough = out[..., ROTARY_DIM:]
        assert torch.equal(passthrough, torch.ones_like(passthrough)), (
            f"{name}: pass-through dims are not the exact BF16 norm value"
        )
        # cos=1, sin=0 leaves the rotary half at the norm value too.
        assert torch.equal(out[..., :ROTARY_DIM], ones[..., :ROTARY_DIM]), f"{name}: rotation not identity"


@pytest.mark.api("functions.qk_norm_rope_infer")
@pytest.mark.parametrize("head_dim, rotary_dim", GEOMETRIES, ids=[f"{d}_{r}" for d, r in GEOMETRIES])
@pytest.mark.reference
def test_qk_norm_rope_geometry_identity_rotation(head_dim, rotary_dim):
    """Every accepted geometry must be the identity for cos=1, sin=0, unit weight, x=1.

    Independent of both implementations: RMSNorm of a constant ``1.0`` with unit weight and
    ``eps=0`` is exactly ``1.0``, so all ``head_dim`` outputs (rotated or passed through) must
    be exactly ``1.0``. This covers a non-power-of-two ``head_dim`` and the no-tail geometry
    without hardware, and exercises the wrapper's geometry checks at the same time.
    """
    tokens, heads = 17, 8
    ones = torch.ones(tokens, heads, head_dim, dtype=DTYPE)
    weight = torch.ones(head_dim, dtype=DTYPE)
    half = rotary_dim // 2
    rope = torch.cat((torch.ones(tokens, half), torch.zeros(tokens, half)), dim=-1).to(DTYPE)

    q_out, k_out = F.qk_norm_rope_infer(ones, ones, weight, weight, rope, 0.0, implementation="torch")
    for name, out in (("q", q_out), ("k", k_out)):
        assert torch.equal(out, ones), f"{name} {head_dim}/{rotary_dim} is not the identity"


@pytest.mark.api("functions.qk_norm_rope_infer")
@pytest.mark.reference
def test_qk_norm_rope_accepts_float32_table():
    """A float32 table with BF16 activations is legal, as it is upstream."""
    q = torch.randn(4, 8, HEAD_DIM).to(DTYPE) * 0.5
    w = torch.ones(HEAD_DIM, dtype=DTYPE)
    rope = torch.rand(4, ROTARY_DIM, dtype=torch.float32)
    out = F.qk_norm_rope_infer(q, q, w, w, rope, EPS, implementation="torch")
    assert out[0].shape == q.shape and torch.isfinite(out[0]).all()


@pytest.mark.api("functions.qk_norm_rope_infer")
@pytest.mark.parametrize("implementation", ["torch", "triton"])
def test_qk_norm_rope_rejects_bad_shapes(implementation):
    """Contract checks must reject mismatched inputs instead of producing garbage.

    Everything here is rejected before ``load_impl``, so the triton case needs no device.
    """
    q = torch.zeros(4, 8, HEAD_DIM, dtype=DTYPE)
    k = torch.zeros(4, 8, HEAD_DIM, dtype=DTYPE)
    w = torch.ones(HEAD_DIM, dtype=DTYPE)
    rope = torch.zeros(4, ROTARY_DIM, dtype=DTYPE)
    # A token-count mismatch is illegal; a head-count mismatch is not (GQA is valid).
    with pytest.raises(ValueError):
        F.qk_norm_rope_infer(q, k[:2], w, w, rope, EPS, implementation=implementation)
    with pytest.raises(ValueError):
        F.qk_norm_rope_infer(q, k, torch.ones(HEAD_DIM // 2), w, rope, EPS, implementation=implementation)
    with pytest.raises(ValueError):
        F.qk_norm_rope_infer(q, k, w, w, torch.zeros(4, HEAD_DIM + 2), EPS, implementation=implementation)
    with pytest.raises(ValueError):
        F.qk_norm_rope_infer(q[..., :64], k[..., :64], w, w, rope, EPS, implementation=implementation)
    # An odd rotary_dim would make rope_table one element short of what it advertises.
    with pytest.raises(ValueError):
        F.qk_norm_rope_infer(
            q, k, w, w, torch.zeros(4, ROTARY_DIM - 1, dtype=DTYPE), EPS, implementation=implementation
        )


@pytest.mark.api("functions.qk_norm_rope_infer")
@pytest.mark.parametrize("implementation", ["torch", "triton"])
def test_qk_norm_rope_rejects_bad_dtypes_and_devices(implementation):
    """Unsupported dtypes, a foreign table dtype and a wrong device must not pass."""
    q = torch.zeros(4, 8, HEAD_DIM, dtype=DTYPE)
    k = torch.zeros(4, 8, HEAD_DIM, dtype=DTYPE)
    w = torch.ones(HEAD_DIM, dtype=DTYPE)
    rope = torch.zeros(4, ROTARY_DIM, dtype=DTYPE)

    # float64 is not in the supported set; int is not floating point at all.
    with pytest.raises(TypeError):
        F.qk_norm_rope_infer(
            q.double(), k.double(), w.double(), w.double(), rope.double(), EPS, implementation=implementation
        )
    with pytest.raises(ValueError):
        F.qk_norm_rope_infer(q.int(), k.int(), w, w, rope, EPS, implementation=implementation)
    # A table in a third precision is rejected; the activations' dtype or float32 are the
    # only two upstream accepts.
    with pytest.raises(ValueError):
        F.qk_norm_rope_infer(q, k, w, w, rope.half(), EPS, implementation=implementation)
    # A meta tensor is on a different device than cpu, which the device check must catch.
    with pytest.raises(ValueError):
        F.qk_norm_rope_infer(q, k, torch.empty(HEAD_DIM, dtype=DTYPE, device="meta"), w, rope, EPS,
                             implementation=implementation)
    with pytest.raises(ValueError):
        F.qk_norm_rope_infer(q, k, w, w, torch.empty(4, ROTARY_DIM, dtype=DTYPE, device="meta"), EPS,
                             implementation=implementation)


@pytest.mark.api("functions.qk_norm_rope_infer")
def test_qk_norm_rope_requires_no_grad():
    """Grad mode has to be rejected: the norm weights are normally ``nn.Parameter``s.

    ``rms_norm_infer`` has the same guard, so a caller that forgets ``torch.no_grad()`` gets a
    RuntimeError instead of a silently detached result.
    """
    q = torch.ones(4, 8, HEAD_DIM, dtype=DTYPE)
    w = torch.ones(HEAD_DIM, dtype=DTYPE, requires_grad=True)
    rope = torch.zeros(4, ROTARY_DIM, dtype=DTYPE)

    with pytest.raises(RuntimeError):
        F.qk_norm_rope_infer(q, q, w, w, rope, EPS, implementation="torch")
    with torch.no_grad():
        out = F.qk_norm_rope_infer(q, q, w, w, rope, EPS, implementation="torch")
    assert out[0].shape == q.shape and torch.isfinite(out[0]).all()
