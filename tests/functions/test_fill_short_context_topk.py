"""Tests for functions.fill_short_context_topk.

The op is integer-only (out of scope of docs/operator_accuracy_standard.md §1,
which covers floating-point compute), so every case pins the result bit-exactly
against the torch reference with ``torch.equal``.
"""

from functools import partial

import pytest
import torch

from mojo_opset import functions as F
from mojo_opset import preload
from tests._checks import assert_repeatable
from tests._compile import compile_fullgraph

# Production and boundary shapes (DeepSeek V4.1-Flash decode / DSv4 batch /
# chunked-prefill saturation) plus the dispatch-threshold boundary.
SHAPES = [
    (256, 512, 2),
    (1024, 2048, 4),
    (2048, 2048, 4),
    (16384, 2048, 4),
]

TINY_ROWS = [(1, 2048, 4), (2, 2048, 4), (5, 2048, 4)]

# compress_ratio spectrum used by the vllm configurations: 1/2 (V4.1 per-layer), 4
# (DSv4 C4A), 16/32 (tests/config) and 64/128 as regression guards.
PARAM_GRID = [(k, cr) for k in (2048, 1024) for cr in (1, 2, 4, 16, 32, 64, 128)]


def _positions(num_tokens, top_k, compress_ratio, device, seed=0, dtype=torch.int64):
    """Random positions with the boundary rows the kernel must get right."""
    torch.manual_seed(seed)
    max_position = top_k * compress_ratio - 1
    positions = torch.randint(0, max_position + 1, (num_tokens,), dtype=torch.int64, device=device)
    if num_tokens >= 5:
        positions[0] = 0  # zero candidates -> whole row -1
        positions[1] = compress_ratio - 1  # exactly one candidate
        positions[2] = 4 * compress_ratio - 1  # exact divide boundary
        positions[3] = max_position  # exactly top_k candidates, no -1
        positions[4] = max_position - compress_ratio + 1  # top_k - 1 candidates
    return positions.to(dtype)


def _run(implementation, num_tokens, top_k, compress_ratio, device, seed=0, positions_dtype=torch.int64):
    positions = _positions(num_tokens, top_k, compress_ratio, device, seed, dtype=positions_dtype)
    actual = torch.empty(num_tokens, top_k, dtype=torch.int32, device=device)
    expected = torch.empty_like(actual)
    F.fill_short_context_topk(actual, positions, top_k, compress_ratio, implementation=implementation)
    F.fill_short_context_topk(expected, positions, top_k, compress_ratio, implementation="torch")
    assert torch.equal(actual, expected), (
        f"mismatch: num_tokens={num_tokens} top_k={top_k} cr={compress_ratio}"
    )
    return actual


@pytest.mark.api("functions.fill_short_context_topk")
@pytest.mark.accuracy
@pytest.mark.parametrize("positions_dtype", [torch.int64, torch.int32], ids=["int64", "int32"])
@pytest.mark.parametrize("num_tokens,top_k,compress_ratio", SHAPES + TINY_ROWS)
def test_fill_short_context_topk_shapes(accuracy_backend, num_tokens, top_k, compress_ratio, positions_dtype):
    implementation, _, device = accuracy_backend
    _run(implementation, num_tokens, top_k, compress_ratio, device, positions_dtype=positions_dtype)


@pytest.mark.api("functions.fill_short_context_topk")
@pytest.mark.accuracy
@pytest.mark.parametrize("top_k,compress_ratio", PARAM_GRID)
def test_fill_short_context_topk_parameter_grid(accuracy_backend, top_k, compress_ratio):
    implementation, _, device = accuracy_backend
    _run(implementation, 1024, top_k, compress_ratio, device, seed=7)


@pytest.mark.api("functions.fill_short_context_topk")
@pytest.mark.accuracy
def test_fill_short_context_topk_exhaustive_small(accuracy_backend):
    """top_k=64, cr=8 with every position in [0, 63] exercised at least once."""
    implementation, _, device = accuracy_backend
    top_k, compress_ratio = 64, 8
    positions = torch.arange(40, dtype=torch.int64, device=device)
    actual = torch.empty(40, top_k, dtype=torch.int32, device=device)
    expected = torch.empty_like(actual)
    F.fill_short_context_topk(actual, positions, top_k, compress_ratio, implementation=implementation)
    F.fill_short_context_topk(expected, positions, top_k, compress_ratio, implementation="torch")
    assert torch.equal(actual, expected)


@pytest.mark.api("functions.fill_short_context_topk")
@pytest.mark.accuracy
def test_fill_short_context_topk_empty_batch(accuracy_backend):
    """Empty tensor (docs §1.4): no rows, no stores."""
    implementation, _, device = accuracy_backend
    positions = torch.empty(0, dtype=torch.int64, device=device)
    buffer = torch.empty(0, 2048, dtype=torch.int32, device=device)
    result = F.fill_short_context_topk(buffer, positions, 2048, 4, implementation=implementation)
    assert result.shape == (0, 2048) and result.numel() == 0


@pytest.mark.api("functions.fill_short_context_topk")
@pytest.mark.accuracy
@pytest.mark.parametrize("positions_dtype", [torch.int64, torch.int32], ids=["int64", "int32"])
def test_fill_short_context_topk_boundary_rows(accuracy_backend, positions_dtype):
    """Whole-row -1, single candidate, exactly-full and one-short rows."""
    implementation, _, device = accuracy_backend
    top_k, compress_ratio = 2048, 4
    positions = _positions(5, top_k, compress_ratio, device, dtype=positions_dtype)
    out = F.fill_short_context_topk(
        torch.empty(5, top_k, dtype=torch.int32, device=device),
        positions,
        top_k,
        compress_ratio,
        implementation=implementation,
    )
    assert (out[0] == -1).all()
    assert out[1, 0] == 0 and (out[1, 1:] == -1).all()
    assert torch.equal(out[3], torch.arange(top_k, dtype=torch.int32, device=device))
    assert out[4, -1] == -1 and (out[4, :-1] != -1).all()


@pytest.mark.api("functions.fill_short_context_topk")
@pytest.mark.bitwise
def test_fill_short_context_topk_repeatable(accuracy_backend):
    implementation, _, device = accuracy_backend
    positions = _positions(2048, 2048, 4, device)
    buffer = torch.empty(2048, 2048, dtype=torch.int32, device=device)
    assert_repeatable(
        partial(
            F.fill_short_context_topk,
            top_k=2048,
            compress_ratio=4,
            implementation=implementation,
        ),
        (buffer, positions),
        backward=False,
    )


@pytest.mark.api("functions.fill_short_context_topk")
def test_fill_short_context_topk_fake_metadata(accuracy_backend):
    implementation, target, device = accuracy_backend
    if implementation not in (None, "triton") or not target.startswith(("npu.a2", "npu.a5")):
        pytest.skip("NPU Triton fake metadata")
    preload("fill_short_context_topk", implementation="triton", target=target)
    namespace = getattr(torch.ops, f"mojo_npu_triton_{target.split('.')[1]}")
    buffer = torch.empty(4, 2048, dtype=torch.int32, device=device)
    positions = torch.randint(0, 8191, (4,), dtype=torch.int64, device=device)
    torch.library.opcheck(
        namespace.fill_short_context_topk_fwd.default,
        (buffer, positions, 2048, 4),
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.api("functions.fill_short_context_topk")
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_fill_short_context_topk_compiled(accuracy_backend, backend):
    implementation, target, device = accuracy_backend
    if not target.startswith("npu.") or implementation == "torch":
        pytest.skip("NPU Triton compile contract")
    implementation = implementation or "triton"
    preload("fill_short_context_topk", implementation=implementation)
    top_k, compress_ratio = 2048, 4
    positions = _positions(top_k, top_k, compress_ratio, device)

    def run(buffer, pos):
        output = F.fill_short_context_topk(
            buffer, pos, top_k, compress_ratio, implementation=implementation
        )
        return (output,)

    try:
        (eager,) = run(torch.empty(top_k, top_k, dtype=torch.int32, device=device), positions)
        (actual,) = compile_fullgraph(run, backend=backend)(
            torch.empty(top_k, top_k, dtype=torch.int32, device=device), positions
        )
        assert actual.is_contiguous()
        assert torch.equal(actual, eager)
        reference = torch.empty_like(actual)
        F.fill_short_context_topk(reference, positions, top_k, compress_ratio, implementation="torch")
        assert torch.equal(actual, reference)
    finally:
        torch._dynamo.reset()


@pytest.mark.api("functions.fill_short_context_topk")
@pytest.mark.accuracy
@pytest.mark.parametrize("top_k", [4097, 8192])
def test_fill_short_context_topk_wide_rows_fall_back(accuracy_backend, top_k):
    """top_k beyond the merged tile's 4096 columns runs one row per program."""
    implementation, _, device = accuracy_backend
    _run(implementation, 2048, top_k, 4, device, seed=top_k)


@pytest.mark.api("functions.fill_short_context_topk")
@pytest.mark.accuracy
@pytest.mark.parametrize("top_k", [49152, 65536])
def test_fill_short_context_topk_rejects_oversized_top_k(accuracy_backend, top_k):
    """From top_k = 49152 the single-row kernel exceeds the per-thread stack."""
    implementation, _, device = accuracy_backend
    positions = torch.zeros(4, dtype=torch.int64, device=device)
    buffer = torch.empty(4, top_k, dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="49151"):
        F.fill_short_context_topk(buffer, positions, top_k, 2, implementation=implementation)
