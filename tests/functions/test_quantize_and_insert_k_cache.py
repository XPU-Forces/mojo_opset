"""Accuracy tests for functions.quantize_and_insert_k_cache.

Reference: the torch provider, plus an independent per-token/per-block loop
(``_naive_insert``) that catches scatter/indexing bugs in the torch reference.

Coverage
--------
* both page records: V4 (584 B/token) and V4.1 (528 B/token)
* block_size 16 / 32 / 64 / 128
* token counts around block boundaries (0, 1, 7, 8, 63, 64, 65, 127, 128, ...)
* slot patterns: sequential, cross-block, shuffled, sparse, with -1 padding
* page byte stride larger than the record (padding must stay untouched)
* k rows beyond ``slot_mapping.shape[0]`` (DP padding) must not be read
* non-contiguous ``k`` / ``slot_mapping`` inputs (the wrapper materializes them)
* special values: zeros (absmax floor), tiny, large (clamp), sparse
* bitwise repeatability of the in-place scatter
"""

import random

import pytest
import torch

from mojo_opset import functions as F
from tests._checks import assert_close
from tests._checks import assert_repeatable

V4 = 584
V41 = 528

BLOCK_SIZES = [16, 32, 64, 128]
N_EDGES = [0, 1, 2, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]


def _make_slots(num_tokens, block_size, mode, seed):
    if num_tokens == 0:
        return torch.empty(0, dtype=torch.int64)
    rng = random.Random(seed)
    if mode == "seq":
        return torch.arange(num_tokens, dtype=torch.int64)
    if mode == "cross":
        # Start every row near the end of a page so inserts straddle pages.
        return torch.tensor(
            [(i * block_size + block_size - 1) for i in range(num_tokens)],
            dtype=torch.int64,
        )
    if mode == "shuffle":
        slots = list(range(num_tokens))
        rng.shuffle(slots)
        return torch.tensor(slots, dtype=torch.int64)
    if mode == "sparse":
        top = max(num_tokens * 4, 8)
        slots = rng.sample(range(top), num_tokens)
        return torch.tensor(slots, dtype=torch.int64)
    if mode == "pad":
        slots = []
        for i in range(num_tokens):
            slots.append(-1 if i % 3 == 1 else i * 2)
        return torch.tensor(slots, dtype=torch.int64)
    raise ValueError(mode)


def _build(num_tokens, block_size, bytes_per_token, stride_pad, mode, seed=0, extra_k_rows=0):
    slots = _make_slots(num_tokens, block_size, mode, seed)
    max_slot = int(slots.max()) if slots.numel() else 0
    num_blocks = max_slot // block_size + 3
    stride = block_size * bytes_per_token + stride_pad
    k_cache = torch.full((num_blocks, stride), 0xAA, dtype=torch.uint8)
    k = torch.randn(num_tokens + extra_k_rows, 512, dtype=torch.bfloat16)
    return k, k_cache, slots


def _naive_insert(k, k_cache, slot_mapping, block_size, bytes_per_token):
    """Independent scalar loop; mirrors the record layout but not the vectorization."""
    out = k_cache.clone()
    flat = out.reshape(-1)
    stride = out.stride(0)
    fp8_max = 448.0
    for i, slot in enumerate(slot_mapping.tolist()):
        if slot == -1:
            continue
        block_idx, pos = divmod(slot, block_size)
        row = k[i].float()
        if bytes_per_token == V41:
            base = block_idx * stride + pos * (512 + 16)
            for j in range(16):
                seg = row[j * 32 : (j + 1) * 32]
                amax = max(float(seg.abs().max()), 1e-4)
                exponent = int(torch.ceil(torch.log2(torch.tensor(amax / fp8_max))))
                scale = 2.0**exponent
                q = (seg / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
                flat[base + j * 32 : base + (j + 1) * 32] = q.view(torch.uint8)
                flat[base + 512 + j] = min(max(exponent + 127, 0), 255)
        else:
            base = block_idx * stride + pos * 576
            for j in range(7):
                seg = row[j * 64 : (j + 1) * 64]
                amax = max(float(seg.abs().max()), 1e-4)
                exponent = int(torch.ceil(torch.log2(torch.tensor(amax / fp8_max))))
                scale = 2.0**exponent
                q = (seg / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
                flat[base + j * 64 : base + (j + 1) * 64] = q.view(torch.uint8)
                scale_base = block_idx * stride + block_size * 576 + pos * 8
                flat[scale_base + j] = min(max(exponent + 127, 0), 255)
            flat[base + 448 : base + 576] = k[i, 448:].contiguous().view(torch.uint8)
            flat[block_idx * stride + block_size * 576 + pos * 8 + 7] = 0
    return out


def _cases():
    cases = []
    for bytes_per_token in (V4, V41):
        for block_size in BLOCK_SIZES:
            for n in N_EDGES:
                for mode in ("seq", "cross", "shuffle", "sparse", "pad"):
                    cases.append((n, block_size, bytes_per_token, 0, mode, 0))
    # Padded page stride (padding bytes must not be touched).
    for bytes_per_token in (V4, V41):
        for n, block_size in ((64, 64), (257, 32), (33, 128)):
            for pad in (8, 64):
                cases.append((n, block_size, bytes_per_token, pad, "sparse", 1))
    return cases


CASES = _cases()


def _id(case):
    n, block_size, bpt, pad, mode, _ = case
    return f"n{n}-b{block_size}-r{bpt}-pad{pad}-{mode}"


def _make_inputs(case, extra_k_rows=0):
    n, block_size, bpt, pad, mode, seed = case
    return _build(n, block_size, bpt, pad, mode, seed, extra_k_rows)


@pytest.mark.api("functions.quantize_and_insert_k_cache")
@pytest.mark.accuracy
@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_quantize_and_insert_k_cache_accuracy(accuracy_backend, case):
    implementation, _, device = accuracy_backend
    n, block_size, bpt, _, _, _ = case
    k, k_cache, slots = _make_inputs(case)
    k, k_cache, slots = k.to(device), k_cache.to(device), slots.to(device)

    expected = F.quantize_and_insert_k_cache(
        k, k_cache.clone(), slots, block_size, bytes_per_token=bpt, implementation="torch"
    )
    actual = F.quantize_and_insert_k_cache(
        k, k_cache.clone(), slots, block_size, bytes_per_token=bpt, implementation=implementation
    )
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # Quantization is deterministic; the whole page (incl. padding) must match.
    assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.reference
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "case",
    [
        (5, 64, V4, 0, "cross", 0),
        (9, 32, V4, 16, "shuffle", 0),
        (7, 128, V4, 0, "pad", 0),
        (5, 64, V41, 0, "cross", 0),
        (9, 32, V41, 16, "shuffle", 0),
        (7, 128, V41, 0, "pad", 0),
    ],
)
def test_torch_reference_matches_naive_loop(accuracy_backend, case):
    _, _, device = accuracy_backend
    n, block_size, bpt, _, _, _ = case
    k, k_cache, slots = _make_inputs(case)
    k, k_cache, slots = k.to(device), k_cache.to(device), slots.to(device)
    expected = _naive_insert(k.cpu(), k_cache.cpu(), slots.cpu(), block_size, bpt).to(device)
    actual = F.quantize_and_insert_k_cache(
        k, k_cache.clone(), slots, block_size, bytes_per_token=bpt, implementation="torch"
    )
    assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.api("functions.quantize_and_insert_k_cache")
@pytest.mark.accuracy
@pytest.mark.parametrize("bytes_per_token", [V4, V41])
def test_dp_padding_rows_are_not_read(accuracy_backend, bytes_per_token):
    # k carries extra rows; only slot_mapping.shape[0] rows may be consumed.
    implementation, _, device = accuracy_backend
    n = 37
    k, k_cache, slots = _build(n, 64, bytes_per_token, 0, "sparse", 3, extra_k_rows=11)
    k, k_cache, slots = k.to(device), k_cache.to(device), slots.to(device)
    expected = F.quantize_and_insert_k_cache(
        k, k_cache.clone(), slots, 64, bytes_per_token=bytes_per_token, implementation="torch"
    )
    actual = F.quantize_and_insert_k_cache(
        k, k_cache.clone(), slots, 64, bytes_per_token=bytes_per_token, implementation=implementation
    )
    assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.api("functions.quantize_and_insert_k_cache")
@pytest.mark.accuracy
@pytest.mark.parametrize("bytes_per_token", [V4, V41])
@pytest.mark.parametrize("mode", ["zeros", "tiny", "large", "sparse"])
def test_special_values(accuracy_backend, bytes_per_token, mode):
    implementation, _, device = accuracy_backend
    n, block_size = 64, 64
    k, k_cache, slots = _build(n, block_size, bytes_per_token, 0, "seq", 5)
    if mode == "zeros":
        k = torch.zeros_like(k)
    elif mode == "tiny":
        k = k * 1e-3
    elif mode == "large":
        k = k * 1e4
    else:
        mask = (torch.rand_like(k.float()) > 0.9).to(k.dtype)
        k = k * mask
    k, k_cache, slots = k.to(device), k_cache.to(device), slots.to(device)
    expected = F.quantize_and_insert_k_cache(
        k, k_cache.clone(), slots, block_size, bytes_per_token=bytes_per_token, implementation="torch"
    )
    actual = F.quantize_and_insert_k_cache(
        k, k_cache.clone(), slots, block_size, bytes_per_token=bytes_per_token, implementation=implementation
    )
    assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.api("functions.quantize_and_insert_k_cache")
@pytest.mark.accuracy
@pytest.mark.parametrize("bytes_per_token", [V4, V41])
def test_strided_inputs(accuracy_backend, bytes_per_token):
    # Non-contiguous k (every other column) and slot_mapping (every other slot);
    # the public wrapper must materialize both before dispatch.
    implementation, _, device = accuracy_backend
    n, block_size = 64, 64
    # Build the strided views on the target device so ``.to`` does not
    # materialize them into contiguous copies.
    kb = torch.randn(n, 1024, dtype=torch.bfloat16, device=device)
    k = kb[:, ::2]
    sb = torch.arange(2 * n, dtype=torch.int64, device=device)
    slots = sb[::2]
    num_blocks = n // block_size + 3
    k_cache = torch.full(
        (num_blocks, block_size * bytes_per_token), 0xAA, dtype=torch.uint8, device=device
    )
    assert not k.is_contiguous()
    assert not slots.is_contiguous()

    expected = F.quantize_and_insert_k_cache(
        k, k_cache.clone(), slots, block_size, bytes_per_token=bytes_per_token, implementation="torch"
    )
    actual = F.quantize_and_insert_k_cache(
        k, k_cache.clone(), slots, block_size, bytes_per_token=bytes_per_token, implementation=implementation
    )
    assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.api("functions.quantize_and_insert_k_cache")
@pytest.mark.bitwise
@pytest.mark.parametrize("bytes_per_token", [V4, V41])
def test_repeatable(accuracy_backend, bytes_per_token):
    implementation, _, device = accuracy_backend
    k, k_cache, slots = _build(129, 64, bytes_per_token, 0, "shuffle", 7)
    k, k_cache, slots = k.to(device), k_cache.to(device), slots.to(device)

    assert_repeatable(
        lambda k, cache, slots: F.quantize_and_insert_k_cache(
            k,
            cache.clone(),
            slots,
            64,
            bytes_per_token=bytes_per_token,
            implementation=implementation,
        ),
        (k, k_cache, slots),
        backward=False,
    )


@pytest.mark.api("functions.quantize_and_insert_k_cache")
def test_rejects_bad_inputs():
    k = torch.zeros(4, 512, dtype=torch.bfloat16)
    cache = torch.zeros(4, 64 * V4, dtype=torch.uint8)
    slots = torch.zeros(4, dtype=torch.int64)
    with pytest.raises(ValueError):
        F.quantize_and_insert_k_cache(k.float(), cache, slots, implementation="torch")
    with pytest.raises(ValueError):
        F.quantize_and_insert_k_cache(k[:, :256], cache, slots, implementation="torch")
    with pytest.raises(ValueError):
        F.quantize_and_insert_k_cache(k, cache.int(), slots, implementation="torch")
    with pytest.raises(ValueError):
        F.quantize_and_insert_k_cache(k, cache, slots.int(), implementation="torch")
    with pytest.raises(ValueError):
        F.quantize_and_insert_k_cache(k, cache, slots, bytes_per_token=288, implementation="torch")
    with pytest.raises(ValueError):
        F.quantize_and_insert_k_cache(k, cache, slots, is_ue8m0=False, implementation="torch")
    # A strided page tensor cannot be scattered in place consistently.
    big = torch.zeros(8, 64 * V4, dtype=torch.uint8)
    with pytest.raises(ValueError):
        F.quantize_and_insert_k_cache(k, big[::2], slots, implementation="torch")
