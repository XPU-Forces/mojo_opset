import pytest
import torch

from mojo_opset import functions
from tests._checks import assert_close
from tests._compile import compile_fullgraph

# Block ids, block counts and keep flags are integers and the only float path is
# an exact max reduction, so both operators are checked bit for bit. A tolerance
# here would hide the one failure mode that matters: a score tie at the top-k
# boundary silently selecting a different block set.
EXACT = {"rtol": 0, "atol": 0}


def _packed_case(width, block_size, topk_blocks, decode, device, rows=6):
    torch.manual_seed(42)
    packed = torch.randn(rows, width * 2, device=device)[:, ::2]
    packed[0, :16] = 0
    packed[2, 5] = float("nan")
    if decode:
        row_repeat = rows // 3
        row_ks = None
        row_ke = torch.tensor([0, 1, width], device=device, dtype=torch.int32)
    else:
        row_repeat = 1
        row_ks = torch.tensor([0, 5, 3, 0, 7, 1], device=device, dtype=torch.int32)
        row_ke = torch.tensor([0, width, width - 1, 1, 11, width], device=device, dtype=torch.int32)
    out = torch.empty(rows, topk_blocks * 2, device=device, dtype=torch.int32)[:, ::2]
    return packed, row_ks, row_ke, out, row_repeat


def _decode_case(rows, width, block_size, topk_blocks, next_n, device, dtype=torch.bfloat16):
    torch.manual_seed(7)
    logits = torch.randn(rows, width, device=device, dtype=dtype)
    row_ke = torch.randint(width // 2, width + 1, (rows // next_n,), device=device).to(torch.int32)
    out = torch.empty(rows, topk_blocks, device=device, dtype=torch.int32)
    return logits, row_ke, out, next_n


@pytest.mark.api("functions.select_candidate_blocks")
@pytest.mark.parametrize(
    "width,block_size,topk_blocks,decode",
    [
        (73, 8, 16, False),
        (97, 3, 7, False),
        (32768, 8, 2048, False),
        (32768, 8, 2048, True),
    ],
)
@pytest.mark.accuracy
def test_select_candidate_blocks_accuracy(accuracy_backend, width, block_size, topk_blocks, decode):
    implementation, _, device = accuracy_backend
    logits, row_ks, row_ke, out, row_repeat = _packed_case(
        width, block_size, topk_blocks, decode, device
    )
    expected = torch.empty_like(out)
    actual = functions.select_candidate_blocks(
        logits, row_ks, row_ke, topk_blocks, block_size, out, row_repeat, implementation=implementation
    )
    functions.select_candidate_blocks(
        logits, row_ks, row_ke, topk_blocks, block_size, expected, row_repeat, implementation="torch"
    )
    assert actual is out
    assert_close(actual, expected, **EXACT)


@pytest.mark.api("functions.select_candidate_blocks")
@pytest.mark.parametrize("next_n", [1, 4])
@pytest.mark.accuracy
def test_select_candidate_blocks_decode_accuracy(accuracy_backend, next_n):
    implementation, _, device = accuracy_backend
    logits, row_ke, out, row_repeat = _decode_case(64, 131072, 8, 2048, next_n, device)
    expected = torch.empty_like(out)
    functions.select_candidate_blocks(
        logits, None, row_ke, 2048, 8, out, row_repeat, implementation=implementation
    )
    functions.select_candidate_blocks(logits, None, row_ke, 2048, 8, expected, row_repeat, implementation="torch")
    assert_close(out, expected, **EXACT)


@pytest.mark.api("functions.select_candidate_blocks")
@pytest.mark.accuracy
def test_select_candidate_blocks_pins_newest_block(accuracy_backend):
    """Empty rows select nothing while real rows keep their newest block."""
    implementation, _, device = accuracy_backend
    logits = torch.zeros(3, 64, device=device)
    logits[0, 3] = 5.0
    logits[2, 9] = 5.0
    row_ks = torch.zeros(3, dtype=torch.int32, device=device)
    row_ke = torch.tensor([40, 0, 17], dtype=torch.int32, device=device)
    out = torch.empty(3, 3, dtype=torch.int32, device=device)
    functions.select_candidate_blocks(logits, row_ks, row_ke, 3, 8, out, implementation=implementation)
    assert out[1].tolist() == [-1, -1, -1]
    assert out[0, 0].item() == 4 and 0 in out[0].tolist()
    assert out[2, 0].item() == 2 and 1 in out[2].tolist()


@pytest.mark.api("functions.select_candidate_blocks")
@pytest.mark.parametrize("shape", [(3, 0), (0, 64)])
@pytest.mark.accuracy
def test_select_candidate_blocks_empty_accuracy(accuracy_backend, shape):
    implementation, _, device = accuracy_backend
    rows, width = shape
    logits = torch.zeros(shape, device=device)
    row_ks = torch.zeros(rows, dtype=torch.int32, device=device)
    row_ke = torch.zeros(rows, dtype=torch.int32, device=device)
    out = torch.empty(rows, 4, dtype=torch.int32, device=device)
    expected = torch.empty_like(out)
    functions.select_candidate_blocks(logits, row_ks, row_ke, 4, 8, out, implementation=implementation)
    functions.select_candidate_blocks(logits, row_ks, row_ke, 4, 8, expected, implementation="torch")
    assert_close(out, expected, **EXACT)


def _mask_cases(width, block_size, topk_blocks, decode, device, rows=6):
    logits, row_ks, row_ke, candidates, row_repeat = _packed_case(
        width, block_size, topk_blocks, decode, device, rows
    )
    # The shared builder plants a NaN, which the dedicated NaN test covers;
    # keep these cases finite because assert_close never matches NaN to NaN.
    logits[2, 5] = 0.0
    nblocks = -(-width // block_size)
    # Cover a clamped overflow id, a repeated valid id and padded -1 lanes.
    candidates.fill_(-1)
    candidates[:, 0] = nblocks + 10
    candidates[:, 1] = 0
    candidates[:, 2] = nblocks
    return logits, row_ks, row_ke, candidates.contiguous(), row_repeat


@pytest.mark.api("functions.apply_candidate_mask")
@pytest.mark.parametrize(
    "width,block_size,topk_blocks,decode",
    [
        (73, 8, 16, False),
        (97, 3, 7, False),
        (32768, 8, 2048, False),
        (32768, 8, 2048, True),
    ],
)
@pytest.mark.accuracy
def test_apply_candidate_mask_accuracy(accuracy_backend, width, block_size, topk_blocks, decode):
    implementation, _, device = accuracy_backend
    logits, row_ks, row_ke, candidates, row_repeat = _mask_cases(
        width, block_size, topk_blocks, decode, device
    )
    actual = logits.clone()
    expected = logits.clone()
    functions.apply_candidate_mask(
        actual, row_ks, row_ke, candidates, block_size, row_repeat, implementation=implementation
    )
    functions.apply_candidate_mask(
        expected, row_ks, row_ke, candidates, block_size, row_repeat, implementation="torch"
    )
    assert_close(actual, expected, **EXACT)


@pytest.mark.api("functions.apply_candidate_mask")
@pytest.mark.accuracy
def test_apply_candidate_mask_keeps_nan(accuracy_backend):
    """NaNs inside kept columns survive; masked columns become -inf on both sides."""
    implementation, _, device = accuracy_backend
    logits = torch.randn(2, 256, device=device)
    logits[0, 5] = float("nan")
    row_ks = torch.zeros(2, dtype=torch.int32, device=device)
    row_ke = torch.tensor([256, 200], dtype=torch.int32, device=device)
    candidates = torch.zeros(2, 4, dtype=torch.int32, device=device)
    actual = logits.clone()
    expected = logits.clone()
    functions.apply_candidate_mask(actual, row_ks, row_ke, candidates, 8, implementation=implementation)
    functions.apply_candidate_mask(expected, row_ks, row_ke, candidates, 8, implementation="torch")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.api("functions.apply_candidate_mask")
@pytest.mark.parametrize("shape", [(3, 0), (0, 64)])
@pytest.mark.accuracy
def test_apply_candidate_mask_empty_accuracy(accuracy_backend, shape):
    implementation, _, device = accuracy_backend
    rows, _ = shape
    logits = torch.zeros(shape, device=device)
    row_ks = torch.zeros(rows, dtype=torch.int32, device=device)
    row_ke = torch.zeros(rows, dtype=torch.int32, device=device)
    candidates = torch.zeros(rows, 4, dtype=torch.int32, device=device)
    actual = logits.clone()
    expected = logits.clone()
    functions.apply_candidate_mask(actual, row_ks, row_ke, candidates, 8, implementation=implementation)
    functions.apply_candidate_mask(expected, row_ks, row_ke, candidates, 8, implementation="torch")
    assert_close(actual, expected, **EXACT)


@pytest.mark.api("functions.select_candidate_blocks", ops=["select_candidate_blocks", "apply_candidate_mask"])
def test_candidate_blocks_compile_matches_eager(accuracy_backend):
    """The select -> mask pipeline stays one fullgraph and matches eager."""
    implementation, _, device = accuracy_backend
    logits, row_ks, row_ke, out, _ = _packed_case(256, 8, 32, False, device)
    logits[2, 5] = 0.0

    def run(logits, row_ks, row_ke, out):
        functions.select_candidate_blocks(logits, row_ks, row_ke, 32, 8, out, implementation=implementation)
        functions.apply_candidate_mask(logits, row_ks, row_ke, out, 8, implementation=implementation)
        return out, logits

    eager_logits = logits.clone()
    eager_out = torch.empty_like(out)
    try:
        compiled = compile_fullgraph(run)
        for _ in range(2):
            assert_close(compiled(logits, row_ks, row_ke, out), run(eager_logits, row_ks, row_ke, eager_out), **EXACT)
    finally:
        torch._dynamo.reset()
