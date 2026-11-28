import pytest
import torch

from torch._subclasses.fake_tensor import FakeTensorMode

from mojo_opset import functions as F
from mojo_opset import preload
from mojo_opset.functions._dispatch import load_impl
from tests._compile import compile_fullgraph

# The BF16 comparison matrix pairs each request organization with R=4 and R=16.
_REQUEST_LENGTHS = (
    ("F01", (7,)),
    ("F02", (128,)),
    ("F03", (512,)),
    ("F04", (2048,)),
    ("F05", (128,) * 16),
    ("F06", (8192,)),
    ("F07", (512,) * 16),
    ("F08", (64,) * 128),
    ("F09", (16,) * 512),
    ("F10", (32768,)),
    ("F11", (256,) * 128),
    ("F12", (131072,)),
    ("F13", (1024,) * 128),
    ("F14", (129, 257, 513, 1149) * 4),
    ("F15", (257, 513, 1025, 2301) * 32),
    ("F16", (262144,)),
    ("F17", (524288,)),
    ("F18", (4096,) * 128),
    ("F20", (8192,) * 128),
)


def _matrix_case(case_id, lengths, ring):
    # Preserve the measured slot maps, including their wrapped starting positions.
    requests = tuple((length, (96 + 73 * i) % 512 + 1, 4096 + (3 + 7 * i) % ring) for i, length in enumerate(lengths))
    return pytest.param(requests, ring, torch.int64, 513, False, id=f"{case_id}_R{ring}")


KPOOL_CASES = [_matrix_case(case_id, lengths, ring) for case_id, lengths in _REQUEST_LENGTHS for ring in (4, 16)]
KPOOL_CASE_ARGS = "requests,ring,slot_dtype,num_blocks,padded"
_PADDED_PROBE = (((1, 1, 19), (4, 3, 20), (7, 5, 21), (32, 7, 22), (129, 9, 23)), 16, torch.int32, 13, True)
_METADATA_PROBE = (((4, 1, 11),), 8, torch.int32, 5, False)


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
@pytest.mark.parametrize(KPOOL_CASE_ARGS, KPOOL_CASES)
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
@pytest.mark.parametrize(KPOOL_CASE_ARGS, KPOOL_CASES)
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
    backing, shape, strides, key, gate, slots = make_case(*_PADDED_PROBE)
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
    args = list(_PADDED_PROBE)
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
@pytest.mark.bitwise
def test_kpool_seed_tail_cache_on_noncurrent_npu(accuracy_backend):
    implementation, _, device = accuracy_backend
    if device != "npu" or implementation not in (None, "triton") or torch.npu.device_count() < 2:
        pytest.skip("requires two visible NPUs and Triton")

    current = torch.device("npu", torch.npu.current_device())
    target = torch.device("npu", 1 if current.index == 0 else 0)
    backing, shape, strides, key, gate, slots = make_case(*_PADDED_PROBE)
    expected = backing.clone()
    converted_key = torch.full_like(key, 17)
    F.kpool_seed_tail_cache(expected.as_strided(shape, strides), converted_key, gate, slots, implementation="torch")
    actual, actual_key, actual_gate, actual_slots = (x.to(target, copy=True) for x in (backing, key, gate, slots))
    torch.npu.synchronize(target)
    target_stream = torch.npu.Stream(device=target)
    with torch.npu.stream(target_stream):
        # Keep the target stream selected while invoking the op from the other NPU.
        actual_key.fill_(17)
        with torch.npu.device(current):
            assert torch.npu.current_device() == current.index
            assert torch.npu.current_stream(target) == target_stream
            assert (
                F.kpool_seed_tail_cache(
                    actual.as_strided(shape, strides), actual_key, actual_gate, actual_slots, implementation="triton"
                )
                is None
            )
            assert torch.npu.current_device() == current.index
            assert torch.npu.current_stream(target) == target_stream
        target_stream.synchronize()

    for observed, wanted in zip(
        (actual, actual_key, actual_gate, actual_slots), (expected, converted_key, gate, slots)
    ):
        _bitwise(observed, wanted)


@pytest.mark.api("functions.kpool_seed_tail_cache")
@pytest.mark.accuracy
@pytest.mark.parametrize(
    "fault", ("dtype", "gate_shape", "slot_dtype", "slot_length", "noncontiguous", "overlap", "device")
)
def test_kpool_seed_tail_cache_invalid_metadata(accuracy_backend, fault):
    implementation, _, device = accuracy_backend
    backing, shape, strides, key, gate, slots = make_case(*_METADATA_PROBE)
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
