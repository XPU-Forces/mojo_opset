import pytest
import torch

from torch._subclasses.fake_tensor import FakeTensorMode

from mojo_opset import functions as F
from mojo_opset import preload
from mojo_opset.functions._dispatch import load_impl
from tests._compile import compile_fullgraph


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


def _bitwise(actual, expected):
    assert torch.equal(actual.cpu().contiguous().view(torch.uint8), expected.cpu().contiguous().view(torch.uint8))


@pytest.mark.api("functions.kpool_seed_tail_cache")
@pytest.mark.accuracy
@pytest.mark.bitwise
@pytest.mark.parametrize(GPU_CASE_ARGS, GPU_CASES)
@pytest.mark.parametrize("backend", ("eager", "aot_eager"))
def test_kpool_seed_tail_cache_fullgraph(accuracy_backend, requests, ring, slot_dtype, num_blocks, padded, backend):
    implementation, _, device = accuracy_backend
    preload("kpool_seed_tail_cache", implementation=implementation)
    backing, shape, strides, key, gate, slots = make_case(requests, ring, slot_dtype, num_blocks, padded)
    actual, actual_key, actual_gate, actual_slots = (x.to(device, copy=True) for x in (backing, key, gate, slots))
    cache = actual.as_strided(shape, strides)
    expected = backing.clone()

    def run(cache, key, gate, slots):
        # Returning None must not allow the compiler to discard the cache write.
        return F.kpool_seed_tail_cache(cache, key, gate, slots, implementation=implementation)

    try:
        compiled = compile_fullgraph(run, backend=backend)
        for step in (1, 2):
            # Reuse the compiled graph and cache, changing both input values and
            # which cache block is selected. The second call must execute again.
            actual_key.fill_(step)
            actual_gate.fill_(-step)
            changed_slots = torch.where(slots >= 0, (slots + ring) % (num_blocks * ring), slots) if step == 2 else slots
            actual_slots.copy_(changed_slots)
            F.kpool_seed_tail_cache(
                expected.as_strided(shape, strides),
                actual_key.cpu(),
                actual_gate.cpu(),
                changed_slots,
                implementation="torch",
            )
            assert compiled(cache, actual_key, actual_gate, actual_slots) is None
            _bitwise(actual, expected)
            _bitwise(actual_key, torch.full_like(key, step))
            _bitwise(actual_gate, torch.full_like(gate, -step))
            _bitwise(actual_slots, changed_slots)
    finally:
        torch._dynamo.reset()


@pytest.mark.api("functions.kpool_seed_tail_cache")
@pytest.mark.accuracy
def test_kpool_seed_tail_cache_schema_and_fake(accuracy_backend):
    implementation, _, device = accuracy_backend
    preload("kpool_seed_tail_cache", implementation=implementation)
    backing, shape, strides, key, gate, slots = make_case(*GPU_CASES[7].values)
    data = [x.to(device, copy=True) for x in (backing, key, gate, slots)]
    args = (data[0].as_strided(shape, strides), *data[1:])
    with FakeTensorMode() as mode:
        assert F.kpool_seed_tail_cache(*(mode.from_tensor(x) for x in args), implementation=implementation) is None
    forward, _ = load_impl("kpool_seed_tail_cache", implementation)
    from importlib import import_module

    leaf = import_module(forward.__module__)._kpool_seed_tail_cache_leaf
    schema = leaf._opoverload._schema
    assert schema.arguments[0].alias_info.is_write
    assert all(arg.alias_info is None for arg in schema.arguments[1:])
    assert len(schema.returns) == 0
    # Check mutation bitwise on CPU: the vendor's NPU allclose path used by
    # opcheck(test_schema) is unsuitable for these cache-copy checks.
    expected = data[0].cpu().clone()
    F.kpool_seed_tail_cache(expected.as_strided(shape, strides), key, gate, slots, implementation="torch")
    assert leaf(*args) is None
    for observed, wanted in zip(data, (expected, key, gate, slots)):
        _bitwise(observed, wanted)
    torch.library.opcheck(leaf, args, test_utils=("test_faketensor",))


@pytest.mark.api("functions.kpool_seed_tail_cache")
@pytest.mark.accuracy
@pytest.mark.bitwise
@pytest.mark.parametrize("mode", ("empty", "invalid", "offset", "stream"))
@pytest.mark.parametrize("slot_dtype", (torch.int32, torch.int64))
def test_kpool_seed_tail_cache_boundaries(accuracy_backend, mode, slot_dtype):
    implementation, _, device = accuracy_backend
    # Contract probes reuse the padded original case; they do not extend GPU_CASES.
    args = list(GPU_CASES[7].values)
    args[2] = slot_dtype
    backing, shape, strides, key, gate, slots = make_case(*args)
    offset = 1 if mode == "offset" else 0
    if offset:
        backing = torch.cat((torch.full((offset,), -77, dtype=backing.dtype), backing))
    if mode == "empty":
        key, gate, slots = key[:0], gate[:0], slots[:0]
    elif mode == "invalid":
        slots[::2] = torch.iinfo(slot_dtype).min
        slots[1::2] = -1
    expected = backing.clone()
    F.kpool_seed_tail_cache(
        expected.as_strided(shape, strides, offset),
        key,
        gate,
        slots,
        implementation="torch",
    )
    actual, actual_key, actual_gate, actual_slots = (x.to(device, copy=True) for x in (backing, key, gate, slots))
    cache = actual.as_strided(shape, strides, offset)
    if mode == "stream":
        torch.npu.synchronize()
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            # Queue input production on the same non-default stream as the write.
            actual_key.copy_(key.to(device))
            assert (
                F.kpool_seed_tail_cache(cache, actual_key, actual_gate, actual_slots, implementation=implementation)
                is None
            )
        stream.synchronize()
    else:
        assert (
            F.kpool_seed_tail_cache(cache, actual_key, actual_gate, actual_slots, implementation=implementation) is None
        )
    for observed, wanted in zip((actual, actual_key, actual_gate, actual_slots), (expected, key, gate, slots)):
        _bitwise(observed, wanted)


@pytest.mark.api("functions.kpool_seed_tail_cache")
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "fault", ("dtype", "gate_shape", "slot_dtype", "slot_length", "noncontiguous", "overlap", "device")
)
def test_kpool_seed_tail_cache_invalid_metadata(accuracy_backend, fault):
    implementation, _, device = accuracy_backend
    backing, shape, strides, key, gate, slots = make_case(*GPU_CASES[1].values)
    cache, key, gate, slots = (x.to(device) for x in (backing.as_strided(shape, strides), key, gate, slots))
    if fault == "dtype":
        key = key.to(torch.float16)
    elif fault == "gate_shape":
        gate = gate[:-1]
    elif fault == "slot_dtype":
        slots = slots.to(torch.float32)
    elif fault == "slot_length":
        slots = slots[:-1]
    elif fault == "noncontiguous":
        key = key.t().contiguous().t()
    elif fault == "overlap":
        cache = cache.as_strided(cache.shape, (strides[0], 0, 128, 1))
    elif fault == "device":
        slots = slots.cpu()
    with pytest.raises(ValueError):
        F.kpool_seed_tail_cache(cache, key, gate, slots, implementation=implementation)
