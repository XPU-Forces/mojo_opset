import pytest
import torch

from mojo_opset import functions as F


def _legacy_case(case_id, lengths, ring, slot_dtype, position_start, padded=False):
    requests = tuple((length, 2 * i + 1, position_start + i) for i, length in enumerate(lengths))
    return pytest.param(requests, ring, slot_dtype, 2 * len(lengths) + 3, padded, id=case_id)


# BF16 cases from the GPU comparison profile, preserving the original slot maps.
GPU_CASES = [
    _legacy_case("case02-length-1", (1,), 8, torch.int64, 11),
    _legacy_case("case05-length-4", (4,), 8, torch.int32, 11),
    _legacy_case("case06-length-5", (5,), 16, torch.int64, 19),
    _legacy_case("case10-length-9", (9,), 4, torch.int64, 7),
    _legacy_case("case13-length-33", (33,), 4, torch.int32, 7),
    _legacy_case("case16-length-129", (129,), 4, torch.int64, 7),
    _legacy_case("case17-mixed-requests", (0, 1, 2, 3, 4, 5, 8, 9, 31, 128, 129), 4, torch.int32, 7),
    _legacy_case("case19-padded-cache", (1, 4, 7, 32, 129), 16, torch.int32, 19, padded=True),
    _legacy_case("case21-batch-32", (33,) * 32, 8, torch.int32, 11),
    _legacy_case("case22-prefill-128x128", (128,) * 128, 16, torch.int64, 19),
    pytest.param(((16384, 4, 0),), 4, torch.int32, 5, False, id="case31-prefill-1x16384"),
    pytest.param(
        ((255, 5, 0), (256, 0, 7), (257, 7, 4), (256, 3, 1)),
        8,
        torch.int64,
        8,
        False,
        id="case32-prefill-1024",
    ),
]
GPU_CASE_ARGS = "requests,ring,slot_dtype,num_blocks,padded"


def make_case(requests, ring, slot_dtype, num_blocks, padded):
    slots = []
    for length, block, start in requests:
        slots.extend(block * ring + (start + row) % ring for row in range(length))
        if padded:
            slots.extend((-1, -ring - 1))

    shape = (num_blocks, 2, ring, 128)
    channel_stride = (ring + (2 if padded else 0)) * 128
    block_stride = 2 * channel_stride + (128 if padded else 0)
    strides = (block_stride, channel_stride, 128, 1)
    backing = torch.full((num_blocks * block_stride + 128,), -123, dtype=torch.bfloat16)
    backing.as_strided(shape, strides).copy_(torch.randn(shape, dtype=torch.bfloat16))
    key = torch.randn((len(slots), 128), dtype=torch.bfloat16)
    gate = torch.randn_like(key)
    return backing, shape, strides, key, gate, torch.tensor(slots, dtype=slot_dtype)


@pytest.mark.api("functions.kpool_seed_tail_cache")
@pytest.mark.accuracy
@pytest.mark.bitwise
@pytest.mark.parametrize(GPU_CASE_ARGS, GPU_CASES)
def test_kpool_seed_tail_cache_accuracy(accuracy_backend, requests, ring, slot_dtype, num_blocks, padded):
    implementation, _, device = accuracy_backend
    backing, shape, strides, key, gate, slots = make_case(requests, ring, slot_dtype, num_blocks, padded)
    expected = backing.clone()
    F.kpool_seed_tail_cache(expected.as_strided(shape, strides), key, gate, slots, implementation="torch")

    actual, actual_key, actual_gate, actual_slots = (x.to(device, copy=True) for x in (backing, key, gate, slots))
    cache = actual.as_strided(shape, strides)
    assert F.kpool_seed_tail_cache(cache, actual_key, actual_gate, actual_slots, implementation=implementation) is None
    for observed, wanted in zip((actual, actual_key, actual_gate, actual_slots), (expected, key, gate, slots)):
        assert torch.equal(observed.cpu().view(torch.uint8), wanted.view(torch.uint8))
