from __future__ import annotations

import pytest
import torch

from mojo_opset import functions as F

# (shape, axis) coverage: tail, stride-2 non-tail, wide-axis lanes and the
# generic transpose fallback.  Large cases enter the adaptive wide branches.
LAYOUT_CASES = [
    pytest.param((2, 64), -1, id="tail"),
    pytest.param((2, 512, 2), 1, id="stride2-fused"),
    pytest.param((2, 576, 2), 1, id="stride2"),
    pytest.param((256, 8), 0, id="width8"),
    pytest.param((256, 16), 0, id="width16"),
    pytest.param((64, 32), 0, id="width32"),
    pytest.param((64, 64), 0, id="width64"),
    pytest.param((64, 128), 0, id="width128"),
    pytest.param((2, 48, 3), 1, id="generic-axis"),
]

LARGE_LAYOUT_CASES = [
    pytest.param((128, 128, 32), id="width32-tile128"),
    pytest.param((64, 256, 32), id="width32-tile256"),
    pytest.param((32, 512, 32), id="width32-tile512"),
    pytest.param((128, 128, 64), id="width64-tile128"),
    pytest.param((64, 256, 64), id="width64-tile256"),
    pytest.param((128, 128, 128), id="width128-tile128"),
]


def _random_input(shape, axis, src):
    generator = torch.Generator().manual_seed(20260924)
    cpu = (torch.randn(shape, generator=generator) * 7).to(src)
    logical = cpu.movedim(axis, -1).contiguous().reshape(-1, shape[axis])
    logical[0, :32] = 0
    logical[0, 1:32:2] = -0.0
    if logical.numel() >= 40:
        logical.reshape(-1)[32:40] = torch.tensor(
            [float("inf"), -float("inf"), float("nan"), 0.5, -0.5, 1.125, -1.125, 0.0]
        ).to(src)
    return logical.reshape(cpu.movedim(axis, -1).shape).movedim(-1, axis).contiguous()


def _assert_bitwise(actual, expected):
    assert type(actual) is tuple and len(actual) == 2
    y, scale = actual
    ref_y, ref_scale = expected
    assert y.shape == ref_y.shape and y.dtype == ref_y.dtype
    assert scale.shape == ref_scale.shape and scale.dtype == ref_scale.dtype
    assert torch.equal(y.view(torch.uint8).cpu(), ref_y.view(torch.uint8).cpu())
    assert torch.equal(scale.view(torch.uint8).cpu(), ref_scale.view(torch.uint8).cpu())


def _pin_npu_rng(device):
    # The DSL kernel reserves (seed, block_offset) from the framework
    # generator. Pin it to (0, 0) so the run replays the golden's default
    # stream; the reservation advances the offset by 1024 on every call.
    if str(device) in ("npu", "privateuseone") or (isinstance(device, torch.device) and device.type in ("npu", "privateuseone")):
        torch.npu.manual_seed(0)


def _run_case(backend, shape, axis, src, dst_code=24, algorithm=1, bound=0.0, *, cpu=None):
    implementation, device = backend
    if cpu is None:
        cpu = _random_input(shape, axis, src)
    x = cpu.to(device)
    _pin_npu_rng(device)
    kwargs = dict(
        axis=axis, dst_type=dst_code, scale_alg=algorithm, max_low_bound=bound
    )
    actual = F.dynamic_mx_quant_sr(x, implementation=implementation, **kwargs)
    expected = F.dynamic_mx_quant_sr(cpu, implementation="torch_reference", **kwargs)
    _assert_bitwise(actual, expected)
    return actual


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize("shape,axis", LAYOUT_CASES)
@pytest.mark.parametrize("src", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dst_code", [23, 24])
@pytest.mark.parametrize("algorithm", [0, 1])
@pytest.mark.accuracy
def test_dynamic_mx_quant_sr_layouts(dynamic_mx_quant_sr_backend, shape, axis, src, dst_code, algorithm):
    _run_case(dynamic_mx_quant_sr_backend, shape, axis, src, dst_code, algorithm)


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize("shape", LARGE_LAYOUT_CASES)
@pytest.mark.parametrize("src", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dst_code", [23, 24])
@pytest.mark.parametrize("algorithm", [0, 1])
@pytest.mark.accuracy
def test_dynamic_mx_quant_sr_adaptive_tiles(dynamic_mx_quant_sr_backend, shape, src, dst_code, algorithm):
    _run_case(dynamic_mx_quant_sr_backend, shape, -2, src, dst_code, algorithm)


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize("shape,axis", LAYOUT_CASES + [
    pytest.param((4, 256), -1, id="tail-bound-aligned"),
    pytest.param((4, 80), -1, id="tail-bound-padded"),
])
@pytest.mark.parametrize("src", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dst_code", [23, 24])
@pytest.mark.accuracy
def test_dynamic_mx_quant_sr_max_low_bound(dynamic_mx_quant_sr_backend, shape, axis, src, dst_code):
    _run_case(dynamic_mx_quant_sr_backend, shape, axis, src, dst_code, 1, bound=32.0)


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"axis": True}, TypeError, "axis"),
        ({"axis": 2}, ValueError, "axis"),
        ({"round_mode": "rint"}, NotImplementedError, "round_mode"),
        ({"dst_type": 35}, NotImplementedError, "dst_type"),
        ({"block_size": 0}, ValueError, "block_size"),
        ({"block_size": 64}, NotImplementedError, "block_size"),
        ({"scale_alg": 2}, NotImplementedError, "scale_alg"),
        ({"dst_type_max": 448.0}, NotImplementedError, "dst_type_max"),
        ({"max_low_bound": -1.0}, ValueError, "max_low_bound"),
        ({"max_low_bound": float("nan")}, ValueError, "max_low_bound"),
        ({"max_low_bound": 1e100}, ValueError, "max_low_bound"),
        ({"scale_alg": 0, "max_low_bound": 1.0}, ValueError, "max_low_bound"),
    ],
)
@pytest.mark.accuracy
def test_dynamic_mx_quant_sr_rejects_attributes(kwargs, error, match):
    with pytest.raises(error, match=match):
        F.dynamic_mx_quant_sr(torch.ones((2, 64)), **kwargs)


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize("value,axis,error,match", [
    (None, -1, TypeError, "input"),
    (torch.empty(()), -1, ValueError, "nonempty"),
    (torch.empty((0, 64)), -1, ValueError, "nonempty"),
    (torch.empty((2, 17)), -1, ValueError, "multiple of 16"),
    (torch.empty((2, 16)), -1, ValueError, "at least 32"),
    (torch.empty((2, 17, 3)), 1, ValueError, "multiple of 16"),
    (torch.empty((2, 64)).t(), 0, ValueError, "contiguous"),
])
@pytest.mark.accuracy
def test_dynamic_mx_quant_sr_rejects_invalid_tensors(value, axis, error, match):
    with pytest.raises(error, match=match):
        F.dynamic_mx_quant_sr(value, axis=axis)


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
@pytest.mark.accuracy
def test_dynamic_mx_quant_sr_rejects_dtype(dtype):
    with pytest.raises(TypeError, match="float32.*bfloat16"):
        F.dynamic_mx_quant_sr(torch.ones((2, 64), dtype=dtype))


@pytest.fixture
def cannbotdsl_rng_device(dynamic_mx_quant_sr_backend):
    implementation, device = dynamic_mx_quant_sr_backend
    if implementation == "torch_reference":
        pytest.skip("device RNG probes require the cannbotdsl implementation")
    device = torch.device(device)
    if device.index is None:
        device = torch.device(device.type, torch.npu.current_device())
    saved = torch.npu.get_rng_state(device)
    try:
        yield device
    finally:
        torch.npu.synchronize(device)
        torch.npu.set_rng_state(saved, device)


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize("base", [0, 64, 127, 2**35 - 1, 2**39 - 1, 2**60 - 1])
@pytest.mark.parametrize("seed,offset", [
    (0, 0),
    (0xFEDCBA9876543210, 2**32 - 1),
    (2**64 - 1, 2**64 - 1 - 1024),
])
@pytest.mark.accuracy
def test_dynamic_mx_quant_sr_rng_2048_counters(cannbotdsl_rng_device, base, seed, offset):
    import numpy as np

    from mojo_opset.kernels.npu_a5_cannbotdsl._rng_state import _make_rng_state
    from mojo_opset.kernels.torch_reference.dynamic_mx_quant_sr_golden import _philox_blocks
    from tests.functions._dynamic_mx_quant_sr_rng import run_counter_words

    groups = [base + lane for lane in range(64)]
    counters = []
    for group in groups:
        row, column = divmod(group, 128)
        task = (row // 128) * 16 + column // 8
        subsequence = task * 128 + (row % 16) * 8 + column % 8
        counter = offset + (row // 16) % 8
        counters.append((counter & 0xFFFFFFFF, counter >> 32,
                         subsequence & 0xFFFFFFFF, subsequence >> 32))
    words = _philox_blocks(groups, quant_length=2048, seed=seed, block_offset=offset)
    expected = np.concatenate((np.asarray(counters, np.uint32).T, words.T))
    state = _make_rng_state(seed, offset, device=cannbotdsl_rng_device)
    output = torch.empty((8, 64), dtype=torch.uint32, device=cannbotdsl_rng_device)
    run_counter_words(state, output, base, 2048)
    np.testing.assert_array_equal(output.cpu().numpy(), expected)


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize("quant_length,base", [
    pytest.param(4080, 0, id="below-quad"),
    pytest.param(4096, 0, id="same-row"),
    pytest.param(4112, 1, id="unaligned-cross-row"),
    pytest.param(7168, 0, id="non-power-of-two"),
    pytest.param(8192, 0, id="two-quads"),
    pytest.param(16 * (2**32 + 511), 2**32 - 64, id="column-carry32"),
])
@pytest.mark.parametrize("seed,offset", [(0, 0), (0xFEDCBA9876543210, 2**32 - 1)])
@pytest.mark.accuracy
def test_dynamic_mx_quant_sr_rng_sixteen_batches(cannbotdsl_rng_device, quant_length, base, seed, offset):
    import numpy as np

    from mojo_opset.kernels.npu_a5_cannbotdsl._rng_state import _make_rng_state
    from mojo_opset.kernels.torch_reference.dynamic_mx_quant_sr_golden import _philox_blocks
    from tests.functions._dynamic_mx_quant_sr_rng import run_sixteen_batch_words

    groups = np.arange(16 * 64, dtype=np.uint64) + np.uint64(base)
    words = _philox_blocks(groups, quant_length=quant_length, seed=seed, block_offset=offset)
    output = torch.full((1, 16 * 256 + 64), -559038737, dtype=torch.int32, device=cannbotdsl_rng_device)
    expected = np.full(output.shape, 0xDEADBEEF, dtype=np.uint32)
    expected[0, :16 * 256] = words.reshape(-1)
    state = _make_rng_state(seed, offset, device=cannbotdsl_rng_device)
    run_sixteen_batch_words(state, output, base, quant_length)
    np.testing.assert_array_equal(output.cpu().numpy().view(np.uint32), expected)


@pytest.mark.api("functions.dynamic_mx_quant_sr")
@pytest.mark.parametrize("shape", [(128, 2048), (512, 4096), (512, 4112)])
@pytest.mark.parametrize("src", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dst_code", [23, 24])
@pytest.mark.parametrize("algorithm", [0, 1])
@pytest.mark.accuracy
def test_dynamic_mx_quant_sr_rng_replay(cannbotdsl_rng_device, shape, src, dst_code, algorithm):
    import numpy as np
    from ml_dtypes import bfloat16

    from mojo_opset.kernels.torch_reference.dynamic_mx_quant_sr_golden import dynamic_mx_quant_golden

    cpu = _random_input(shape, -1, src)
    values = cpu.float().numpy().astype(bfloat16 if src == torch.bfloat16 else np.float32)
    x = cpu.to(cannbotdsl_rng_device)
    generator = torch.npu.default_generators[cannbotdsl_rng_device.index]
    seed, offset = 0xFEDCBA9876543210, 2**32 - 4
    generator.manual_seed(seed)
    generator.set_offset(offset)
    saved = torch.npu.get_rng_state(cannbotdsl_rng_device)
    kwargs = dict(dst_type=dst_code, scale_alg=algorithm, implementation="cannbotdsl")
    first = None
    for call in range(2):
        actual = F.dynamic_mx_quant_sr(x, **kwargs)
        expected = dynamic_mx_quant_golden(
            values, "float8_e5m2" if dst_code == 23 else "float8_e4m3fn", algorithm,
            seed=seed, block_offset=offset + call * 1024,
        )
        for tensor, reference in zip(actual, expected):
            np.testing.assert_array_equal(tensor.view(torch.uint8).cpu().numpy(), reference)
        assert generator.get_offset() == offset + (call + 1) * 1024
        if first is None:
            first = tuple(tensor.clone() for tensor in actual)
    torch.npu.set_rng_state(saved, cannbotdsl_rng_device)
    replay = F.dynamic_mx_quant_sr(x, **kwargs)
    _assert_bitwise(replay, first)
    assert generator.get_offset() == offset + 1024
