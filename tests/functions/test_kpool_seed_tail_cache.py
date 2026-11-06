import pytest
import torch

from mojo_opset import functions as F


def make_case(n, ring, dtype, slot_dtype, device):
    blocks = max(1, (n + ring - 1) // ring)
    backing = torch.randn((blocks, 2, ring + 1, 128), device=device, dtype=dtype)
    key = torch.randn((n, 128), device=device, dtype=dtype)
    gate = torch.randn_like(key)
    slots = torch.arange(n, device=device, dtype=slot_dtype)
    slots[1::7] = torch.iinfo(slot_dtype).min
    return backing, key, gate, slots


@pytest.mark.reference
def test_kpool_seed_tail_cache_torch():
    backing = torch.full((1, 2, 8, 128), -7, dtype=torch.float16)
    key = torch.arange(6, dtype=torch.float16)[:, None].expand(-1, 128).contiguous()
    gate = key + 10
    slots = torch.tensor([0, -1, 2, 3, 4, 5], dtype=torch.int64)
    expected = backing.clone()
    for row in (2, 3, 4, 5):
        expected[0, 0, row] = key[row]
        expected[0, 1, row] = gate[row]
    assert F.kpool_seed_tail_cache(backing, key, gate, slots, implementation="torch") is None
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)


@pytest.mark.api("functions.kpool_seed_tail_cache")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype,slot_dtype", [(torch.bfloat16, torch.int64), (torch.float16, torch.int32)])
@pytest.mark.parametrize(
    "n,ring",
    [
        (0, 4),
        (1, 8),
        (4, 8),
        (5, 5),
        (9, 7),
        (10, 16),
        (33, 32),
        (255, 16),
        (256, 16),
        (257, 16),
        (259, 16),
        (16384, 16),
    ],
)
def test_kpool_seed_tail_cache_accuracy(accuracy_backend, dtype, slot_dtype, n, ring):
    implementation, _, device = accuracy_backend
    backing, key, gate, slots = make_case(n, ring, dtype, slot_dtype, "cpu")
    expected = backing.clone()
    F.kpool_seed_tail_cache(expected[:, :, :ring], key, gate, slots, implementation="torch")
    actual = backing.to(device)
    assert (
        F.kpool_seed_tail_cache(
            actual[:, :, :ring], key.to(device), gate.to(device), slots.to(device), implementation=implementation
        )
        is None
    )
    torch.testing.assert_close(actual.cpu().view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


@pytest.mark.api("functions.kpool_seed_tail_cache")
@pytest.mark.accuracy
@pytest.mark.parametrize("dtype,slot_dtype", [(torch.bfloat16, torch.int64), (torch.float16, torch.int32)])
@pytest.mark.parametrize("layout", ["regular", "shifted", "irregular", "negative_words"])
def test_kpool_seed_tail_cache_grouped_tail(accuracy_backend, dtype, slot_dtype, layout):
    """Exercise two dynamic four-row runs, fallback, and nonzero storage offsets."""
    implementation, _, device = accuracy_backend
    ring, blocks, offset = 16, 7, 3
    block_stride, channel_stride = 2 * (ring + 2) * 128, (ring + 1) * 128
    shape, strides = (blocks, 2, ring, 128), (block_stride, channel_stride, 128, 1)
    lengths = (96, 160) if layout == "shifted" else (128, 128)
    slots = torch.tensor(
        [
            ((2 * request + 1) * ring + (19 + request + row) % ring)
            for request, length in enumerate(lengths)
            for row in range(length)
        ],
        dtype=slot_dtype,
    )
    if layout == "irregular":
        slots[124] = -1
        slots[125:127] = slots[125:127].flip(0)
    if layout == "negative_words":
        slots[:4] = torch.tensor(
            [-(2**63), -(2**32) + 5, -(2**31), -1] if slot_dtype == torch.int64 else [-(2**31), -17, -2, -1],
            dtype=slot_dtype,
        )
    key = torch.randn((sum(lengths), 128), dtype=dtype)
    gate = torch.randn_like(key)
    key[0, 0], gate[-1, 1] = -0.0, torch.finfo(dtype).max
    backing = torch.full((offset + blocks * block_stride + 17,), -123.0, dtype=dtype)
    expected = backing.clone()
    expected_cache = expected.as_strided(shape, strides, offset)
    F.kpool_seed_tail_cache(expected_cache, key, gate, slots, implementation="torch")
    actual, actual_key, actual_gate, actual_slots = (x.to(device) for x in (backing, key, gate, slots))
    cache = actual.as_strided(shape, strides, offset)
    assert F.kpool_seed_tail_cache(cache, actual_key, actual_gate, actual_slots, implementation=implementation) is None
    for observed, wanted in zip((actual, actual_key, actual_gate, actual_slots), (expected, key, gate, slots)):
        assert torch.equal(observed.cpu().contiguous().view(torch.uint8), wanted.contiguous().view(torch.uint8))
