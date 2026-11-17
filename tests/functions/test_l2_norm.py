"""Tests for functions.l2_norm."""

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_close
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph

# docs/operator_accuracy_standard.md §2.2: rtol and atol are both 2**-k for the
# computation dtype. The shared tests/_checks.py budget is looser for FP16/BF16,
# so every comparison here passes the standard value explicitly.
STANDARD_TOLERANCE = {
    torch.float16: 2.0**-10,
    torch.bfloat16: 2.0**-7,
    torch.float32: 2.0**-13,
}

DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# The kernel rejects a last dimension wider than 64 KiB of working set
# (max_fused_size = 65536 // element_size), so the widest legal row is dtype
# dependent. Wider axes therefore go in a leading dimension, and the last
# dimension stays at or below this bound.
MAX_LAST_DIM = {
    torch.float16: 32768,
    torch.bfloat16: 32768,
    torch.float32: 16384,
}

# Standard §1.2: axis values are drawn from [1, 2**20] and the element count of one
# tensor must not exceed 2**34.
MAX_AXIS = 2**20
MAX_ELEMENTS = 2**34

# Rows wider than this take the chunked kernel instead of the row-batched one.
#
# Both providers land on the same boundary, but for different reasons. A row
# needs BD * 4 * 3 bytes of FP32 working set, and the largest power-of-two BD
# that fits each unified buffer is 16384 in both cases:
#   A2  196608 / 12 = 16384.0    -> BD = 16384 fits exactly
#   A5  253952 / 12 = 21162.67   -> the next power of two down is 16384
# D = 16385 rounds up to BD = 32768, which overflows either buffer, so the
# chunked kernel takes over from that width on.
WHOLE_ROW_LIMIT = 16384


def _seed(*parts):
    """Deterministic seed so a failing case reproduces across processes."""
    acc = 2166136261
    for part in parts:
        for char in str(part):
            acc ^= ord(char)
            acc = (acc * 16777619) & 0xFFFFFFFF
    return acc


def _sample(shape, dtype, *, dist, seed):
    """Draw inputs from the standard's two distributions (§1.2).

    Half the regular cases are uniform on [-5, 5]; the other half use a normal
    distribution whose mean and sigma are themselves sampled from [-5, 5] and
    [0.1, 2].
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    if dist == "uniform":
        values = torch.empty(shape, dtype=torch.float32).uniform_(-5.0, 5.0, generator=generator)
    else:
        mean = torch.empty(1).uniform_(-5.0, 5.0, generator=generator).item()
        sigma = torch.empty(1).uniform_(0.1, 2.0, generator=generator).item()
        values = torch.randn(shape, dtype=torch.float32, generator=generator) * sigma + mean
    return values.to(dtype)


def _axis_pool(dtype, family):
    """Axis lengths from §1.2: powers of two, and one less than a power of two."""
    cap = MAX_LAST_DIM[dtype]
    values = []
    power = 1
    while power <= cap:
        values.append(power if family == "pow2" else power - 1)
        power *= 2
    return [value for value in values if value >= 1]


def _rank_shape(rank, family, dtype):
    """One shape per (rank, family): small leading axes, axis-valued last dim."""
    pool = _axis_pool(dtype, family)
    last = pool[min(len(pool) - 1, rank + 1)]
    if rank == 1:
        return (last,)
    lead = 2 if family == "pow2" else 3
    return (lead,) * (rank - 1) + (last,)


def assert_standard_close(actual, expected, dtype, *, name="output"):
    """Standard §2.1 mixed tolerance: |actual - golden| <= atol + rtol * |golden|."""
    tolerance = STANDARD_TOLERANCE[dtype]
    assert_close(actual, expected, dtype, rtol=tolerance, atol=tolerance, name=name)


def assert_nonfinite_match(actual, expected, dtype, *, name="output"):
    """Compare NaN/inf placement against the reference.

    The standard's §2.3 element rule fails any point where either side is NaN, so
    these inputs cannot be claimed as a tolerance pass. They still must not change
    where the result becomes non-finite, which is what this checks.
    """
    assert actual.shape == expected.shape, f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert actual.dtype == expected.dtype, f"{name}: dtype {actual.dtype} != {expected.dtype}"
    for check, label in (
        (torch.isnan, "NaN"),
        (torch.isposinf, "+inf"),
        (torch.isneginf, "-inf"),
    ):
        assert torch.equal(check(actual), check(expected)), f"{name}: {label} pattern differs from the reference"
    finite = torch.isfinite(expected)
    if finite.any():
        assert_standard_close(actual[finite], expected[finite], dtype, name=name)


# --------------------------------------------------------------------------- #
# Data type x distribution x dimension coverage (§1.1, §1.2)
# --------------------------------------------------------------------------- #


@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize("dist", ["uniform", "normal"])
@pytest.mark.parametrize("family", ["pow2", "pow2m1"])
@pytest.mark.parametrize("rank", range(1, 9))
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda value: str(value).split(".")[-1])
@pytest.mark.accuracy
def test_l2_norm_rank_coverage(accuracy_backend, dtype, rank, family, dist):
    """Ranks 1..8 with axis lengths from both families, both distributions."""
    implementation, _, device = accuracy_backend
    shape = _rank_shape(rank, family, dtype)
    assert 1 <= len(shape) <= 8
    assert torch.Size(shape).numel() <= MAX_ELEMENTS
    x = _sample(shape, dtype, dist=dist, seed=_seed(shape, dtype, dist)).to(device)

    actual = F.l2_norm(x, implementation=implementation)
    expected = F.l2_norm(x, implementation="torch")

    assert actual.shape == shape and actual.dtype == dtype
    assert actual.is_contiguous()
    assert_standard_close(actual, expected, dtype)


@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda value: str(value).split(".")[-1])
@pytest.mark.accuracy
def test_l2_norm_axis_values(accuracy_backend, dtype):
    """Sweep the last axis across every power of two and power of two minus one."""
    implementation, _, device = accuracy_backend
    for family in ("pow2", "pow2m1"):
        for dim in _axis_pool(dtype, family):
            shape = (8, dim) if dim > 1 else (16, dim)
            x = _sample(shape, dtype, dist="uniform", seed=_seed(shape, dtype)).to(device)

            actual = F.l2_norm(x, implementation=implementation)
            expected = F.l2_norm(x, implementation="torch")

            assert_standard_close(actual, expected, dtype, name=f"output[D={dim}]")


@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("dim", [WHOLE_ROW_LIMIT + 1, 2 * WHOLE_ROW_LIMIT], ids=["first_over", "wide"])
@pytest.mark.accuracy
def test_l2_norm_chunked_branch(accuracy_backend, dtype, dim):
    """Rows past the whole-row tile limit take the chunked kernel."""
    implementation, _, device = accuracy_backend
    shape = (4, dim)
    x = _sample(shape, dtype, dist="uniform", seed=_seed(shape, dtype, "chunked")).to(device)

    actual = F.l2_norm(x, implementation=implementation)
    expected = F.l2_norm(x, implementation="torch")

    assert_standard_close(actual, expected, dtype)


# --------------------------------------------------------------------------- #
# Attr coverage (§1.3): eps and output_dtype are the public non-tensor attributes
# --------------------------------------------------------------------------- #


@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize("eps", [1e-12, 1e-6, 1e-4, 1.0, 1e6], ids=lambda value: f"{value:g}")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda value: str(value).split(".")[-1])
@pytest.mark.accuracy
def test_l2_norm_eps(accuracy_backend, dtype, eps):
    """Every eps equivalence class: negligible, typical, and eps-dominated."""
    implementation, _, device = accuracy_backend
    shape = (4, 64)
    x = _sample(shape, dtype, dist="normal", seed=_seed(shape, dtype, eps)).to(device)

    actual = F.l2_norm(x, eps=eps, implementation=implementation)
    expected = F.l2_norm(x, eps=eps, implementation="torch")

    assert_standard_close(actual, expected, dtype)


@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize(
    "dtype,output_dtype",
    [
        pytest.param(
            input_dtype,
            out_dtype,
            id=f"{str(input_dtype).rsplit('.', 1)[-1]}-to-{str(out_dtype).rsplit('.', 1)[-1]}",
        )
        for input_dtype, out_dtype in [
            (torch.bfloat16, torch.float16),
            (torch.bfloat16, torch.float32),
            (torch.float16, torch.bfloat16),
            (torch.float16, torch.float32),
            (torch.float32, torch.bfloat16),
            (torch.float32, torch.float16),
        ]
    ],
)
@pytest.mark.accuracy
def test_l2_norm_output_dtype(accuracy_backend, dtype, output_dtype):
    """Cast on store: every input/output dtype pair is judged at the output precision.

    The kernel computes in FP32 and rounds once when storing, so the tolerance
    comes from output_dtype rather than the input dtype.
    """
    implementation, _, device = accuracy_backend
    shape = (8, 512)
    x = _sample(shape, dtype, dist="normal", seed=_seed(shape, dtype, output_dtype)).to(device)

    actual = F.l2_norm(x, output_dtype=output_dtype, implementation=implementation)
    expected = F.l2_norm(x, output_dtype=output_dtype, implementation="torch")

    assert actual.shape == shape
    assert actual.dtype == output_dtype
    assert actual.is_contiguous()
    assert_standard_close(actual, expected, output_dtype)


# --------------------------------------------------------------------------- #
# Special scenarios (§1.4)
# --------------------------------------------------------------------------- #


@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize("rank", range(1, 9))
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda value: str(value).split(".")[-1])
@pytest.mark.accuracy
def test_l2_norm_lower_bound(accuracy_backend, dtype, rank):
    """Lower boundary: every dimension of the shape is 1."""
    implementation, _, device = accuracy_backend
    shape = (1,) * rank
    x = _sample(shape, dtype, dist="uniform", seed=_seed(shape, dtype)).to(device)

    actual = F.l2_norm(x, implementation=implementation)
    expected = F.l2_norm(x, implementation="torch")

    assert actual.shape == shape
    assert_standard_close(actual, expected, dtype)


@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda value: str(value).split(".")[-1])
@pytest.mark.accuracy
def test_l2_norm_upper_bound(accuracy_backend, dtype):
    """Upper boundary: one axis takes the maximum value allowed for this dtype."""
    implementation, _, device = accuracy_backend
    cases = [(MAX_LAST_DIM[dtype],), (MAX_AXIS, 1)]
    for shape in cases:
        assert torch.Size(shape).numel() <= MAX_ELEMENTS
        x = _sample(shape, dtype, dist="uniform", seed=_seed(shape, dtype, "upper")).to(device)

        actual = F.l2_norm(x, implementation=implementation)
        expected = F.l2_norm(x, implementation="torch")

        assert_standard_close(actual, expected, dtype, name=f"output[{shape}]")


@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize("shape", [(0, 8), (2, 0, 8)], ids=["leading", "middle"])
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda value: str(value).split(".")[-1])
def test_l2_norm_empty(accuracy_backend, dtype, shape):
    """Empty tensors: a non-normalized dimension is zero, so there is nothing to compare.

    A zero last dimension is rejected by the API instead, because that is the axis
    being normalized.
    """
    implementation, _, device = accuracy_backend
    x = torch.empty(shape, dtype=dtype, device=device)

    actual = F.l2_norm(x, implementation=implementation)

    assert actual.shape == shape and actual.dtype == dtype


@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda value: str(value).split(".")[-1])
@pytest.mark.accuracy
def test_l2_norm_scalar(accuracy_backend, dtype):
    """Scalar tensor: shape [1]."""
    implementation, _, device = accuracy_backend
    x = _sample((1,), dtype, dist="uniform", seed=_seed(dtype, "scalar")).to(device)

    actual = F.l2_norm(x, implementation=implementation)
    expected = F.l2_norm(x, implementation="torch")

    assert actual.shape == (1,)
    assert_standard_close(actual, expected, dtype)


@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize("shape", [(16, 32), (4, 7), (1,)])
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda value: str(value).split(".")[-1])
@pytest.mark.accuracy
def test_l2_norm_all_zeros(accuracy_backend, dtype, shape):
    """All-zero rows: eps alone keeps the result finite."""
    implementation, _, device = accuracy_backend
    x = torch.zeros(shape, dtype=dtype, device=device)

    actual = F.l2_norm(x, implementation=implementation)
    expected = F.l2_norm(x, implementation="torch")

    assert torch.isfinite(actual).all()
    assert_standard_close(actual, expected, dtype)


@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize(
    "shape",
    [(1, 4), (5, 4), (2, 8), (3, 16), (4, 7), (2, 3, 6)],
    ids=lambda value: "x".join(str(dim) for dim in value),
)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda value: str(value).split(".")[-1])
def test_l2_norm_nonfinite(accuracy_backend, dtype, shape):
    """NaN, +inf, -inf and a mixed tensor keep the reference's non-finite pattern."""
    implementation, _, device = accuracy_backend
    values = [float("nan"), float("inf"), float("-inf"), 1.0]
    flat = torch.zeros(shape, dtype=dtype, device=device).reshape(-1)
    for index, value in enumerate(values):
        flat[index] = value

    actual = F.l2_norm(flat.reshape(shape), implementation=implementation)
    expected = F.l2_norm(flat.reshape(shape), implementation="torch")

    assert_nonfinite_match(actual, expected, dtype)


@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda value: str(value).split(".")[-1])
@pytest.mark.accuracy
def test_l2_norm_near_zero(accuracy_backend, dtype):
    """Values around zero stay inside the absolute tolerance."""
    implementation, _, device = accuracy_backend
    values = torch.tensor([[0.0, 1e-6, -1e-6, 1.0]], dtype=torch.float32).to(device=device, dtype=dtype)

    actual = F.l2_norm(values, implementation=implementation)
    expected = F.l2_norm(values, implementation="torch")

    assert_standard_close(actual, expected, dtype)


@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda value: str(value).split(".")[-1])
@pytest.mark.accuracy
def test_l2_norm_strided_input(accuracy_backend, dtype):
    """A transposed input is made contiguous before the launch."""
    implementation, _, device = accuracy_backend
    base = _sample((256, 64), dtype, dist="normal", seed=_seed("strided", dtype)).to(device)
    x = base.t()
    assert not x.is_contiguous()

    actual = F.l2_norm(x, implementation=implementation)
    expected = F.l2_norm(x, implementation="torch")

    assert_standard_close(actual, expected, dtype)


# --------------------------------------------------------------------------- #
# Repeatability, compile and error contracts
# --------------------------------------------------------------------------- #


@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda value: str(value).split(".")[-1])
@pytest.mark.bitwise
def test_l2_norm_bitwise(accuracy_backend, dtype):
    implementation, _, device = accuracy_backend
    x = _sample((7, 513), dtype, dist="normal", seed=_seed("bitwise", dtype)).to(device)

    # Forward-only op: assert_repeatable compares two forward results byte by byte.
    assert_repeatable(lambda: F.l2_norm(x, implementation=implementation), (), backward=False)


@pytest.mark.api("functions.l2_norm")
def test_l2_norm_fake_metadata(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith("npu."):
        pytest.skip("NPU Triton fake metadata")
    preload("l2_norm", implementation="triton", target=target)
    ops = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    x = torch.randn(17, 7, device=device, dtype=torch.bfloat16).T
    torch.library.opcheck(ops.l2_norm_fwd.default, (x, 1e-6, None), test_utils=("test_faketensor",))


@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_l2_norm_compiled_layout(accuracy_backend, backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation == "torch":
        pytest.skip("NPU contiguous-output contract")
    implementation = implementation or "triton"
    preload("l2_norm", implementation=implementation)
    x = torch.randn(17, 7, device=device, dtype=torch.bfloat16).T
    inputs = (x,)

    def run(*args):
        output = F.l2_norm(*args, implementation=implementation)
        return (output,), ((output.stride(), output.is_contiguous()),)

    try:
        _, metadata = run(*inputs)
        actual, captured_metadata = compile_fullgraph(run, backend=backend)(*inputs)
        assert captured_metadata == metadata
        assert all(y.is_contiguous() for y in actual)
        assert captured_metadata == ((actual[0].stride(), actual[0].is_contiguous()),)
        reference = F.l2_norm(*inputs, implementation="torch")
        tolerance = STANDARD_TOLERANCE[torch.bfloat16]
        torch.testing.assert_close(actual[0], reference, rtol=tolerance, atol=tolerance)
    finally:
        torch._dynamo.reset()


@pytest.mark.reference
@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize("eps", [0.0, -1e-6])
def test_l2_norm_rejects_nonpositive_eps(eps):
    x = torch.randn(2, 4)
    with pytest.raises(ValueError, match="eps must be positive"):
        F.l2_norm(x, eps=eps, implementation="torch")


@pytest.mark.reference
@pytest.mark.api("functions.l2_norm")
@pytest.mark.parametrize(
    "x",
    [
        pytest.param(torch.randint(0, 9, (2, 4), dtype=torch.int32), id="int"),
        pytest.param(torch.randn(2, 0), id="empty_last_dim"),
        pytest.param(torch.tensor(1.0), id="scalar_0d"),
    ],
)
def test_l2_norm_rejects_invalid_input(x):
    with pytest.raises(ValueError, match="floating input with a nonempty last dimension"):
        F.l2_norm(x, implementation="torch")


@pytest.mark.reference
@pytest.mark.api("functions.l2_norm")
def test_l2_norm_rejects_autograd():
    x = torch.randn(2, 4, requires_grad=True)
    with pytest.raises(RuntimeError, match="does not support autograd"):
        F.l2_norm(x, implementation="torch")
