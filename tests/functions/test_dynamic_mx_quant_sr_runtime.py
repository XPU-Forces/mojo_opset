"""SR runtime regressions adapted to the Mojo provider and capability gates."""

import pytest

pytest.importorskip("cannbotdsl", reason="SR provider tests require CANNBot-DSL 0.8")
import dataclasses
import gc
import hashlib
import json
import math

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from threading import local

import cannbotdsl
import numpy as np
import torch

from cannbotdsl import dtypes
from cannbotdsl import get_mem_size
from cannbotdsl.core.compiler.function import ProviderCallable
from cannbotdsl.core.toolchain.ascendc import AscendCCompiler
from ml_dtypes import bfloat16

from mojo_opset.kernels.npu_a5_cannbotdsl import dynamic_mx_quant_sr as mixed
from mojo_opset.kernels.torch import dynamic_mx_quant_sr_golden as golden

from . import _dynamic_mx_quant_sr_rng as utils
from ._dynamic_mx_quant_sr_rng import CARRY_BASE
from ._dynamic_mx_quant_sr_rng import DTYPE_PAIRS
from ._dynamic_mx_quant_sr_rng import INPUT_DTYPES
from ._dynamic_mx_quant_sr_rng import LARGE_LAYOUT_CASES
from ._dynamic_mx_quant_sr_rng import LAYOUT_CASES
from ._dynamic_mx_quant_sr_rng import OUTPUT_DTYPES
from ._dynamic_mx_quant_sr_rng import SHORT_CASES
from ._dynamic_mx_quant_sr_rng import WHITEBOX_CASES
from ._dynamic_mx_quant_sr_rng import _cpu
from ._dynamic_mx_quant_sr_rng import _review_run_fixed_fields
from ._dynamic_mx_quant_sr_rng import _rng_host
from ._dynamic_mx_quant_sr_rng import _state_words
from ._dynamic_mx_quant_sr_rng import axis_input
from ._dynamic_mx_quant_sr_rng import check
from ._dynamic_mx_quant_sr_rng import finite_scale_boundary_input
from ._dynamic_mx_quant_sr_rng import random_input
from ._dynamic_mx_quant_sr_rng import reject_materialization
from ._dynamic_mx_quant_sr_rng import rng_device as rng_device
from ._dynamic_mx_quant_sr_rng import run_check

pytestmark = [
    pytest.mark.api("functions.dynamic_mx_quant_sr"),
    pytest.mark.npu,
    pytest.mark.whitebox,
]


@pytest.fixture(autouse=True)
def require_cannbotdsl_runtime(dynamic_mx_quant_sr_backend, accuracy_backend):
    """These probes execute the DSL provider directly, even under a Torch override."""
    from mojo_opset import Target
    from mojo_opset.functions._dispatch import resolve_implementation

    implementation, _ = dynamic_mx_quant_sr_backend
    if implementation is None:
        implementation = resolve_implementation("dynamic_mx_quant_sr", Target.parse(accuracy_backend[1]))
    if implementation != "cannbotdsl":
        pytest.skip("SR device probes require the A5 cannbotdsl provider")


@pytest.mark.npu
@pytest.mark.parametrize("start", [0, (16384 - 64) * 32, (2**32 + 31) * 32])
def test_eight_field_rng_independent_workspace_matches_philox(start, rng_device):
    seed, offset = 2**63 + 17, 4 * (2**32 + 9)
    output = torch.empty((1, 512), dtype=torch.int32, device=rng_device)
    program = cannbotdsl.compile(
        utils.run_eight_field_words,
        cannbotdsl.TensorSpec((1, 512), dtypes.int32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
    )
    program(output, start, seed - 2**64, offset)
    groups = np.arange(start // 32, start // 32 + 128, dtype=np.uint64)
    expected = golden._philox_blocks(groups, quant_length=32, seed=seed, block_offset=offset)
    np.testing.assert_array_equal(output.cpu().numpy().view(np.uint32).reshape(128, 4), expected)


@pytest.mark.npu
def test_rng_linear_device_state_eager(rng_device):
    mixed._framework_rng._reserve_rng_carrier(rng_device, 1024)  # warm up before fixing the state
    torch.npu.synchronize(rng_device)
    generator = torch.npu.default_generators[rng_device.index]
    generator.manual_seed(2**64 - 1)
    generator.set_offset(4)
    first = mixed._framework_rng._reserve_rng_carrier(rng_device, 1024)
    assert utils.framework_state_words(first, rng_device) == [2**64 - 1, 4]
    assert generator.get_offset() == 1028
    second = mixed._framework_rng._reserve_rng_carrier(rng_device, 1024)
    assert utils.framework_state_words(second, rng_device) == [2**64 - 1, 1028]
    assert generator.get_offset() == 2052


@pytest.mark.npu
@pytest.mark.parametrize("block_offset", [4, 2**64 - 1028])
def test_rng_linear_device_state_explicit_does_not_reserve(block_offset, rng_device):
    generator = torch.npu.default_generators[rng_device.index]
    generator.manual_seed(29)
    generator.set_offset(4)
    before = torch.npu.get_rng_state(rng_device).clone()
    state = _rng_host()._make_rng_state(2**64 - 1, block_offset, device=rng_device)
    assert _state_words(state, rng_device) == [2**64 - 1, block_offset]
    assert torch.equal(before, torch.npu.get_rng_state(rng_device))
    assert generator.get_offset() == 4


@pytest.mark.npu
def test_rng_linear_graph_state_sequence(rng_device):
    mixed._framework_rng._reserve_rng_carrier(rng_device, 1024)
    torch.npu.synchronize(rng_device)
    generator = torch.npu.default_generators[rng_device.index]
    seed = 2**64 - 1
    generator.manual_seed(seed)
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            first = mixed._framework_rng._reserve_rng_carrier(rng_device, 1024)
            second = mixed._framework_rng._reserve_rng_carrier(rng_device, 1024)
        # Start both replays from an explicit raw framework position after
        # capture; capture-time reservation itself is not a replay baseline.
        generator.set_offset(4)
        saved = torch.npu.get_rng_state(rng_device).clone()
        sequences = []
        for _ in range(2):
            torch.npu.set_rng_state(saved, rng_device)
            sequence = []
            for expected_base in (4, 2052):
                graph.replay()
                first_words = utils.framework_state_words(first, rng_device)
                second_words = utils.framework_state_words(second, rng_device)
                assert first_words == [seed, expected_base]
                assert second_words == [seed, expected_base + 1024]
                assert second_words[1] - first_words[1] == 1024
                assert generator.get_offset() == expected_base + 2048
                sequence.append((first_words, second_words))
            assert sequence[1][0][1] - sequence[0][0][1] == 2048
            eager = mixed._framework_rng._reserve_rng_carrier(rng_device, 1024)
            eager_words = utils.framework_state_words(eager, rng_device)
            assert eager_words == [seed, 4100]
            assert generator.get_offset() == 5124
            graph.replay()
            first_words = utils.framework_state_words(first, rng_device)
            second_words = utils.framework_state_words(second, rng_device)
            assert first_words == [seed, 5124]
            assert second_words == [seed, 6148]
            assert generator.get_offset() == 7172
            sequence.append((eager_words, first_words, second_words))
            sequences.append(sequence)
        assert sequences[0] == sequences[1]
    finally:
        torch.npu.synchronize(rng_device)
        graph.reset()


@pytest.mark.npu
def test_rng_linear_device_state_eager_overflow_does_not_advance(rng_device):
    mixed._framework_rng._reserve_rng_carrier(rng_device, 1024)
    torch.npu.synchronize(rng_device)
    generator = torch.npu.default_generators[rng_device.index]
    generator.manual_seed(2**64 - 1)
    generator.set_offset(2**64 - 1024)
    before = torch.npu.get_rng_state(rng_device).clone()
    with pytest.raises(ValueError, match="overflow"):
        mixed._framework_rng._reserve_rng_carrier(rng_device, 1024)
    assert generator.get_offset() == 2**64 - 1024
    assert torch.equal(before, torch.npu.get_rng_state(rng_device))


@pytest.mark.npu
def test_rng_device_state_eager(rng_device):
    generator = torch.npu.default_generators[rng_device.index]
    generator.manual_seed(2**64 - 1)
    generator.set_offset(4 * (2**32 - 1))
    first = mixed._framework_rng._reserve_rng_carrier(rng_device, 4)
    second = mixed._framework_rng._reserve_rng_carrier(rng_device, 8)
    assert utils.framework_state_words(first, rng_device) == [2**64 - 1, 17179869180]
    assert utils.framework_state_words(second, rng_device) == [2**64 - 1, 17179869184]
    assert generator.get_offset() == 17179869192


@pytest.mark.npu
def test_rng_device_state_explicit_does_not_reserve(rng_device):
    rng = _rng_host()
    before = torch.npu.get_rng_state(rng_device).clone()
    state = rng._make_rng_state(2**64 - 1, 2**64 - 1028, device=rng_device)
    assert _state_words(state, rng_device) == [2**64 - 1, 2**64 - 1028]
    assert torch.equal(before, torch.npu.get_rng_state(rng_device))


@pytest.mark.npu
def test_rng_device_state_graph_replay(rng_device):
    mixed._framework_rng._reserve_rng_carrier(rng_device, 4)  # load the bridge before capture
    torch.npu.synchronize(rng_device)
    generator = torch.npu.default_generators[rng_device.index]
    generator.manual_seed(2**63 + 7)
    generator.set_offset(4 * (2**32 - 1))
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            first = mixed._framework_rng._reserve_rng_carrier(rng_device, 4)
            second = mixed._framework_rng._reserve_rng_carrier(rng_device, 8)
        saved = torch.npu.get_rng_state(rng_device).clone()
        start = generator.get_offset()
        for iteration in range(3):
            graph.replay()
            assert utils.framework_state_words(first, rng_device) == [2**63 + 7, start + 12 * iteration]
            assert utils.framework_state_words(second, rng_device) == [2**63 + 7, start + 12 * iteration + 4]
            assert generator.get_offset() == start + 12 * (iteration + 1)
        torch.npu.set_rng_state(saved, rng_device)
        graph.replay()
        assert utils.framework_state_words(first, rng_device) == [2**63 + 7, start]
        generator.manual_seed(17)
        graph.replay()
        assert utils.framework_state_words(first, rng_device) == [17, 0]
    finally:
        torch.npu.synchronize(rng_device)
        graph.reset()


@pytest.mark.npu
def test_rng_graph_sequential_eager_interleave(rng_device):
    mixed._framework_rng._reserve_rng_carrier(rng_device, 4)
    torch.npu.synchronize(rng_device)
    generator = torch.npu.default_generators[rng_device.index]
    generator.manual_seed(19)
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            captured = mixed._framework_rng._reserve_rng_carrier(rng_device, 8)
        generator.set_offset(100)
        graph.replay()
        assert utils.framework_state_words(captured, rng_device) == [19, 100]
        eager = mixed._framework_rng._reserve_rng_carrier(rng_device, 4)
        assert utils.framework_state_words(eager, rng_device) == [19, 108]
        graph.replay()
        assert utils.framework_state_words(captured, rng_device) == [19, 112]
        assert generator.get_offset() == 120
    finally:
        torch.npu.synchronize(rng_device)
        graph.reset()


@pytest.mark.npu
def test_rng_graph_python_exception_reset_and_recapture(rng_device):
    # Abandoning a capture must not strand our RNG state or prevent a new graph.
    # This is a Python-body exception, not a deliberately invalid device kernel.
    mixed._framework_rng._reserve_rng_carrier(rng_device, 4)
    torch.npu.synchronize(rng_device)
    generator = torch.npu.default_generators[rng_device.index]
    generator.manual_seed(83)
    graph = torch.npu.NPUGraph()
    try:
        with pytest.raises(RuntimeError, match="deliberate SR capture abort"):
            with torch.npu.graph(graph):
                mixed._framework_rng._reserve_rng_carrier(rng_device, 4)
                raise RuntimeError("deliberate SR capture abort")
        graph.reset()
        assert not torch.npu.is_current_stream_capturing()
        generator.set_offset(200)
        eager = mixed._framework_rng._reserve_rng_carrier(rng_device, 4)
        assert utils.framework_state_words(eager, rng_device) == [83, 200]
        with torch.npu.graph(graph):
            recovered = mixed._framework_rng._reserve_rng_carrier(rng_device, 8)
        generator.set_offset(400)
        graph.replay()
        assert utils.framework_state_words(recovered, rng_device) == [83, 400]
        assert generator.get_offset() == 408
    finally:
        torch.npu.synchronize(rng_device)
        graph.reset()


@pytest.mark.npu
def test_rng_input_device_selection(rng_device):
    if torch.npu.device_count() < 2:
        pytest.skip("two visible devices required to verify non-current input-device selection")
    other = torch.device("npu", 1)
    saved_other = torch.npu.get_rng_state(other)
    try:
        with torch.npu.device(other):
            torch.npu.default_generators[other.index].manual_seed(103)
        with torch.npu.device(rng_device):
            generator = torch.npu.default_generators[rng_device.index]
            generator.manual_seed(97)
            state = mixed._framework_rng._reserve_rng_carrier(other, 8)
            assert torch.npu.current_device() == rng_device.index
            assert utils.framework_state_words(state, other) == [103, 0]
            assert generator.get_offset() == 0
            assert torch.npu.default_generators[other.index].get_offset() == 8
    finally:
        torch.npu.synchronize(other)
        torch.npu.set_rng_state(saved_other, other)


@pytest.mark.npu
def test_rng_eager_overflow_does_not_advance(rng_device):
    generator = torch.npu.default_generators[rng_device.index]
    generator.set_offset(2**64 - 4)
    with pytest.raises(ValueError, match="overflow"):
        mixed._framework_rng._reserve_rng_carrier(rng_device, 4)
    assert generator.get_offset() == 2**64 - 4


@pytest.mark.npu
def test_rng_reservation_concurrent(rng_device):
    mixed._framework_rng._reserve_rng_carrier(rng_device, 4)
    torch.npu.synchronize(rng_device)
    generator = torch.npu.default_generators[rng_device.index]
    generator.manual_seed(11)
    with ThreadPoolExecutor(4) as executor:
        futures = [executor.submit(mixed._framework_rng._reserve_rng_carrier, rng_device, 8) for _ in range(16)]
        values = [utils.framework_state_words(future.result(30), rng_device) for future in futures]
    assert all(seed == 11 for seed, _ in values)
    assert sorted(offset for _, offset in values) == list(range(0, 128, 8))
    assert generator.get_offset() == 128


@pytest.mark.npu
def test_device_fixed_word_fields_match_required_lane_order():
    """Isolate the production field selector from Philox and quantization."""
    words = np.zeros((1, 256), dtype=np.uint32)
    for slot in range(4):
        words[0, slot * 64 : slot * 64 + 4] = 0x01234567 if slot % 2 == 0 else 0x89ABCDEF
    expected = np.tile(
        np.array(
            [0x0123, 0x6701, 0x4567, 0x2345, 0x89AB, 0xEF89, 0xCDEF, 0xABCD],
            dtype=np.uint32,
        ),
        8,
    ).reshape(1, 64)
    output = torch.empty((1, 64), dtype=torch.uint32, device="npu")
    _review_run_fixed_fields(
        torch.from_numpy(words.view(np.int32)).npu(),
        output,
    )
    np.testing.assert_array_equal(output.cpu().numpy(), expected)


@pytest.mark.npu
@pytest.mark.parametrize("shape,axis", LAYOUT_CASES)
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS)
@pytest.mark.parametrize("algorithm", [0, 1])
@pytest.mark.parametrize("block_offset", [0, 4, 1024, 2**32 - 4, 2**63])
def test_rng_all_layout_states(shape, axis, src, dst, algorithm, block_offset):
    # A fixed nonzero state must flow through every existing specialization.
    cpu = axis_input(shape, axis, src)
    run_check(cpu, dst, axis=axis, scale_alg=algorithm, rng_seed=0xFEDCBA9876543210, rng_block_offset=block_offset)


@pytest.mark.npu
def test_rng_launch_lifecycle_and_runtime_state_reuses_binary(rng_device):
    cpu = torch.randn((2, 1024), generator=torch.Generator().manual_seed(197))
    x = cpu.to(rng_device)
    prepared = mixed._prepare_launch(x, torch.float8_e4m3fn, 1, 1, 0)
    program = prepared.program
    digest = hashlib.sha256(Path(program.so_path).read_bytes()).hexdigest()
    generator = torch.npu.default_generators[rng_device.index]
    generator.manual_seed(37)
    for seed, offset in ((0, 0), (2**64 - 1, 4 * (2**32 - 1)), (2**63 + 1, 2**63)):
        state = _rng_host()._make_rng_state(seed, offset, device=rng_device)
        before = generator.get_offset()
        prepared.launch(rng_state=state)
        utils.assert_result(
            cpu, prepared.outputs(), -1, torch.float32, 24, 1, 0, rng_seed=seed, rng_block_offset=offset
        )
        assert generator.get_offset() == before
        assert prepared.program is program
        assert hashlib.sha256(Path(program.so_path).read_bytes()).hexdigest() == digest
    generator.manual_seed(41)
    increment = _rng_host()._rng_increment(x.numel())
    assert increment == 4
    for offset in (0, increment):
        prepared.launch()
        utils.assert_result(cpu, prepared.outputs(), -1, torch.float32, 24, 1, 0, rng_seed=41, rng_block_offset=offset)
    assert generator.get_offset() == 2 * increment


@pytest.mark.npu
@pytest.mark.parametrize("failure_stage", ["prepare", "closed"])
def test_rng_linear_prepare_failure_before_reservation(failure_stage, monkeypatch, rng_device):
    cpu = torch.randn((2, 1024), generator=torch.Generator().manual_seed(307))
    x = cpu.to(rng_device)
    prepared = mixed._prepare_launch(x, torch.float8_e4m3fn, 1, 1, 0)
    rng = _rng_host()
    prepared.launch(rng_state=rng._make_rng_state(0, 0, device=rng_device))
    mixed._framework_rng._load_bridge()
    torch.npu.synchronize(rng_device)
    generator = torch.npu.default_generators[rng_device.index]
    seed = 2**63 + 307
    generator.manual_seed(seed)
    before = torch.npu.get_rng_state(rng_device).clone()
    launches = []

    def fail_prepare(*args, **kwargs):
        raise RuntimeError("deliberate preparation failure")

    # Model a closed external Program without closing the shared cached one.
    class ClosedProgram:
        so_path = prepared.program.so_path
        closed = True

        def __call__(self, *args):
            launches.append(args)
            raise AssertionError("closed Program must not launch")

    if failure_stage == "prepare":
        with monkeypatch.context() as config:
            config.setattr(mixed, "_get_compiled_kernel", fail_prepare)
            with pytest.raises(RuntimeError, match="deliberate preparation failure"):
                mixed.dynamic_mx_quant_sr(x)
    else:
        with pytest.raises(RuntimeError, match="live executable AOT Program"):
            replace(prepared, program=ClosedProgram(), scalar_program=None).launch()
    assert launches == []
    assert generator.get_offset() == 0
    assert torch.equal(before, torch.npu.get_rng_state(rng_device))
    result = mixed.dynamic_mx_quant_sr(x)
    utils.assert_result(cpu, result, -1, torch.float32, 24, 1, 0, rng_seed=seed, rng_block_offset=0)
    assert generator.get_offset() == 4


@pytest.mark.npu
def test_rng_linear_launch_failure_keeps_reservation(rng_device):
    cpu = torch.randn((2, 1024), generator=torch.Generator().manual_seed(311))
    x = cpu.to(rng_device)
    prepared = mixed._prepare_launch(x, torch.float8_e4m3fn, 1, 1, 0)
    rng = _rng_host()
    prepared.launch(rng_state=rng._make_rng_state(0, 0, device=rng_device))
    mixed._framework_rng._load_bridge()
    torch.npu.synchronize(rng_device)
    generator = torch.npu.default_generators[rng_device.index]
    seed = 2**63 + 311
    generator.manual_seed(seed)
    launches = []

    # Inject a host-side launch exception, never an invalid device kernel.
    class FailingProgram:
        so_path = prepared.scalar_program.so_path
        closed = False

        def __call__(self, *args):
            launches.append(args[-2:])
            raise RuntimeError("deliberate post-reservation launch failure")

    with pytest.raises(RuntimeError, match="deliberate post-reservation launch failure"):
        replace(prepared, scalar_program=FailingProgram()).launch()
    assert len(launches) == 1
    assert launches[0] == (mixed._tail_rng_signed64(seed), 0)
    assert generator.get_offset() == 4
    result = mixed.dynamic_mx_quant_sr(x)
    utils.assert_result(cpu, result, -1, torch.float32, 24, 1, 0, rng_seed=seed, rng_block_offset=4)
    assert generator.get_offset() == 8


@pytest.mark.npu
@pytest.mark.parametrize("entry", ["public", "prepared"])
def test_rng_linear_concurrent_quantization_state_outputs(entry, monkeypatch, rng_device):
    cpus = [torch.randn((2, 1024), generator=torch.Generator().manual_seed(317 + index)) for index in range(4)]
    inputs = [cpu.to(rng_device) for cpu in cpus]
    prepared = [mixed._prepare_launch(x, torch.float8_e4m3fn, 1, 1, 0) for x in inputs]
    assert len({launch.encoded.data_ptr() for launch in prepared}) == 4
    assert len({launch.scales.data_ptr() for launch in prepared}) == 4
    rng = _rng_host()
    prepared[0].launch(rng_state=rng._make_rng_state(0, 0, device=rng_device))
    mixed._framework_rng._load_bridge()
    torch.npu.synchronize(rng_device)
    generator = torch.npu.default_generators[rng_device.index]
    seed = 2**64 - 1
    generator.manual_seed(seed)
    assert generator.get_offset() == 0
    reserve = mixed._framework_rng._reserve_rng_carrier
    context = local()
    barrier = Barrier(4)

    def capture_reserve(device, increment):
        assert increment == 4
        state = reserve(device, increment)
        context.reservations.append(state)
        return state

    def run_one(index):
        context.reservations = []
        with torch.npu.device(rng_device):
            barrier.wait(timeout=30)
            if entry == "public":
                output = mixed.dynamic_mx_quant_sr(inputs[index])
            else:
                prepared[index].launch()
                output = prepared[index].outputs()
            assert len(context.reservations) == 1
            captured, secondary, actual_seed, actual_offset, intragraph = context.reservations[0]
            assert not captured and not secondary and intragraph == 0
            utils.assert_result(
                cpus[index], output, -1, torch.float32, 24, 1, 0, rng_seed=actual_seed, rng_block_offset=actual_offset
            )
            return actual_seed, actual_offset, output

    monkeypatch.setattr(mixed._framework_rng, "_reserve_rng_carrier", capture_reserve)
    with ThreadPoolExecutor(4) as executor:
        futures = [executor.submit(run_one, index) for index in range(4)]
        results = [future.result(timeout=60) for future in futures]
    # Reservation order is deliberately not inferred from thread completion.
    assert all(actual_seed == seed for actual_seed, _, _ in results)
    assert sorted(offset for _, offset, _ in results) == [0, 4, 8, 12]
    assert len({output[0].data_ptr() for _, _, output in results}) == 4
    assert len({output[1].data_ptr() for _, _, output in results}) == 4
    assert generator.get_offset() == 16


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS)
def test_rng_public_resamples_and_restore_replays_sequence(src, dst, rng_device):
    # Missing default reservations or captured host constants break the
    # independently computed second result and the restored full sequence.
    cpu = torch.randn((2, 1024), generator=torch.Generator().manual_seed(821)).to(src)
    x = cpu.to(rng_device)
    code = 23 if dst == torch.float8_e5m2 else 24
    seed = 2**63 + 71
    torch.npu.manual_seed(seed)
    saved = torch.npu.get_rng_state(rng_device).clone()
    sequences = []
    for replay in range(2):
        torch.npu.set_rng_state(saved, rng_device)
        sequence = []
        for offset in (0, 4, 8):
            result = mixed.dynamic_mx_quant_sr(x, dst_type=code)
            actual = utils.assert_result(cpu, result, -1, src, code, 1, 0, rng_seed=seed, rng_block_offset=offset)
            sequence.append(actual)
        assert torch.npu.default_generators[rng_device.index].get_offset() == 12
        sequences.append(sequence)
    assert not np.array_equal(sequences[0][0][0], sequences[0][1][0])
    for first, restored in zip(*sequences):
        for original_bytes, restored_bytes in zip(first, restored):
            np.testing.assert_array_equal(original_bytes, restored_bytes)


@pytest.mark.npu
@pytest.mark.parametrize(
    "shape,axis,increment",
    [
        ((17749, 144), 1, 40),  # Non-128 C and the final padded RNG batch.
        ((32, 1024, 128), 1, 68),  # Wide lanes cross row/chunk boundaries.
        ((80, 16384, 2), 1, 44),  # Fused stride-2 large loop.
    ],
)
def test_rng_quant_period_boundary_across_core_budgets(shape, axis, increment, rng_device):
    cpu = torch.randn(shape, generator=torch.Generator().manual_seed(193))
    x = cpu.to(rng_device)
    seed, offset = 0xFEDCBA9876543210, 4 * (2**32 - 1)
    assert _rng_host()._rng_increment(x.numel()) == increment
    state = _rng_host()._make_rng_state(seed, offset, device=rng_device)
    reference = golden.dynamic_mx_quant_golden(cpu.numpy(), axis=axis, seed=seed, block_offset=offset)
    for budget in (1, 64):
        prepared = mixed._prepare_launch(x, torch.float8_e4m3fn, axis, 1, 0, block_dim=budget)
        prepared.launch(rng_state=state)
        for actual, expected in zip(prepared.outputs(), reference):
            np.testing.assert_array_equal(actual.view(torch.uint8).cpu().numpy(), expected)
    generator = torch.npu.default_generators[rng_device.index]
    generator.manual_seed(seed)
    generator.set_offset(offset)
    prepared.launch()
    for actual, expected in zip(prepared.outputs(), reference):
        np.testing.assert_array_equal(actual.view(torch.uint8).cpu().numpy(), expected)
    assert generator.get_offset() == offset + increment


@pytest.mark.npu
@pytest.mark.parametrize("shape,axis", [((2, 1024), 1), ((64, 32), 0), ((2, 48, 3), 1)])
def test_rng_quant_graph_replay(shape, axis, rng_device):
    # Public API and PreparedLaunch must both consume live device states in
    # the graph; offsets advance by two calls, not by capture-time constants.
    cpu = torch.randn(shape, generator=torch.Generator().manual_seed(927))
    x = cpu.to(rng_device)
    prepared = mixed._prepare_launch(x, torch.float8_e4m3fn, axis, 1, 0)
    prepared.launch()
    mixed.dynamic_mx_quant_sr(x, axis=axis)
    torch.npu.synchronize(rng_device)
    seed, start = 0xFEDCBA9876543210, 4 * (2**32 - 1)
    generator = torch.npu.default_generators[rng_device.index]
    generator.manual_seed(seed)
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            first = mixed.dynamic_mx_quant_sr(x, axis=axis)
            prepared.launch()
            second = prepared.outputs()
        generator.set_offset(start)
        saved = torch.npu.get_rng_state(rng_device).clone()
        for iteration in range(3):
            graph.replay()
            offset = start + iteration * 8
            utils.assert_result(cpu, first, axis, cpu.dtype, 24, 1, 0, rng_seed=seed, rng_block_offset=offset)
            utils.assert_result(cpu, second, axis, cpu.dtype, 24, 1, 0, rng_seed=seed, rng_block_offset=offset + 4)
            assert generator.get_offset() == offset + 8
        eager = mixed.dynamic_mx_quant_sr(x, axis=axis)
        utils.assert_result(cpu, eager, axis, cpu.dtype, 24, 1, 0, rng_seed=seed, rng_block_offset=start + 24)
        graph.replay()
        utils.assert_result(cpu, first, axis, cpu.dtype, 24, 1, 0, rng_seed=seed, rng_block_offset=start + 28)
        assert generator.get_offset() == start + 36
        torch.npu.set_rng_state(saved, rng_device)
        graph.replay()
        utils.assert_result(cpu, first, axis, cpu.dtype, 24, 1, 0, rng_seed=seed, rng_block_offset=start)
        utils.assert_result(cpu, second, axis, cpu.dtype, 24, 1, 0, rng_seed=seed, rng_block_offset=start + 4)
    finally:
        torch.npu.synchronize(rng_device)
        graph.reset()


@pytest.mark.npu
@pytest.mark.parametrize("src", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dst_code", [23, 24])
@pytest.mark.parametrize("algorithm", [0, 1])
def test_tail_precision_and_dynamic_binary_reuse(src, dst_code, algorithm, monkeypatch):
    # Current workload policies legitimately select distinct compiled schedules.
    # Runtime shapes must reuse each schedule's Program, rather than forcing
    # different schedules into one binary.
    mixed.clear_caches()
    programs = {}
    requested_keys = []
    get_or_build = mixed._PROGRAM_CACHE.get_or_build

    def track_program(key, builder, **options):
        requested_keys.append(key)
        program = get_or_build(key, builder, **options)
        if key in programs:
            assert program is programs[key]
        else:
            programs[key] = program
        return program

    monkeypatch.setattr(mixed._PROGRAM_CACHE, "get_or_build", track_program)
    dtype = mixed.dtypes.float32 if src == torch.float32 else mixed.dtypes.bfloat16
    tile = mixed._make_tiling(dtype, mixed.get_mem_size("ub")).tile_elements
    shapes = [(32,), (3, 48), (65, 32), (2, 3, 80), (1, tile + 16), (1, 2 * tile)]
    if src == torch.bfloat16:
        shapes.append((5, 16))
    cases = [(shape, None) for shape in shapes]
    # One core forces multiple UB iterations. Exact tiles use _run_original;
    # the extra 64 inputs select _run_coalesced and exercise its final tile.
    cases.extend([((1, 2 * tile), 1), ((1, 2 * tile + 64), 1)])
    for index, (shape, blocks) in enumerate(cases):
        check(shape, -1, src, dst_code, algorithm, offset=index % 2, block_dim=blocks)
        assert mixed.cache_info().misses == len(programs)
        if index < 4:
            # These different runtime shapes all use the baseline schedule.
            assert requested_keys[-1] == requested_keys[0]
    assert len(cases) > len(programs)


@pytest.mark.npu
@pytest.mark.parametrize("shape,axis", LAYOUT_CASES[1:])
@pytest.mark.parametrize("src", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dst_code", [23, 24])
@pytest.mark.parametrize("algorithm", [0, 1])
def test_non_tail_precision(shape, axis, src, dst_code, algorithm):
    check(shape, axis, src, dst_code, algorithm, offset=1)


@pytest.mark.npu
@pytest.mark.parametrize("src", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dst_code", [23, 24])
def test_max_low_bound_preserves_zero_groups(src, dst_code):
    # Exercise both packed-256 and short-row scale paths with a nonzero bound.
    check((4, 256), -1, src, dst_code, 1, bound=32.0)
    check((4, 80), -1, src, dst_code, 1, bound=32.0)


@pytest.mark.npu
def test_random_stream_is_independent_of_block_count():
    cpu = torch.randn((3, 256), generator=torch.Generator().manual_seed(20260924))
    x = cpu.npu()
    outputs = []
    for blocks in (1, 7, 64):
        result = mixed._dynamic_mx_quant_sr_impl(x, block_dim=blocks, rng_state=utils.replay_state(x))
        outputs.append(tuple(value.view(torch.uint8).cpu().numpy() for value in result))
    for current in outputs[1:]:
        for actual, reference in zip(current, outputs[0]):
            np.testing.assert_array_equal(actual, reference)


@pytest.mark.npu
@pytest.mark.parametrize("shape", LARGE_LAYOUT_CASES)
@pytest.mark.parametrize("src", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dst_code", [23, 24])
@pytest.mark.parametrize("algorithm", [0, 1])
def test_non_tail_adaptive_tiles(shape, src, dst_code, algorithm):
    check(shape, -2, src, dst_code, algorithm)


@pytest.mark.npu
@pytest.mark.parametrize("src", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dst_code", [23, 24])
@pytest.mark.parametrize("algorithm", [0, 1])
def test_finite_scale_underflow_boundaries(src, dst_code, algorithm):
    cpu = finite_scale_boundary_input(src, dst_code, algorithm)
    # 256 contiguous inputs exercise packed SR; 48-element rows exercise the
    # short-row implementation; both use independent finite scale groups.
    check((1, 256), -1, src, dst_code, algorithm, cpu=cpu.reshape(1, 256), block_dim=1)
    padded = torch.cat((cpu, cpu[:, :16]), dim=1)
    check((8, 48), -1, src, dst_code, algorithm, cpu=padded, block_dim=1)


@pytest.mark.npu
@pytest.mark.parametrize(
    "base,stride,quant_length",
    [
        # Exhaust every small-A reciprocal branch at a row boundary.
        *[(a - 1, 1, a * 16) for a in range(1, 64)],
        (63, 1, 1024),
        (64, 1, 1040),
        (255, 1, 4096),
        (0, 1, 2048),
        (64, 1, 2048),
        (127, 1, 2048),
        (2**60 - 1, 1, 2048),
        (447, 1, 7168),
        (128 * 448 - 1, 1, 7168),
        (2**32 - 1, 1, 16),
        (2**48 + 19, 1, 7168),
        (2**60 - 1, 1, 48),
        (2**60 - 1, 1, 7168),
        (2**32 + 12, 1, 16 * (2**32 + 13)),
        (2**48 + 2**32 - 1, 1, 2**40),
        (2**60 - 1, 1, 2**60),
        (0, 32, 512),
        (15 * 32 + 7, 32, 512),
        (127 * 448 + 7, 448, 7168),
        (2**32 + 12, 2**32 + 13, 16 * (2**32 + 13)),
    ],
)
@pytest.mark.parametrize(
    "seed,block_offset",
    [
        (0, 4),
        (0xFEDCBA9876543210, 2**32 - 4),
        (2**64 - 1, 2**64 - 1028),
    ],
)
def test_rng_linear_device_counter_and_words(base, stride, quant_length, seed, block_offset, rng_device):
    groups = [base + lane * stride for lane in range(64)]
    counters, words = [], []
    # Plain Python coordinates and scalar Philox, independent of SIMD pairs.
    for group in groups:
        subsequence = group % 16384
        counter = block_offset // 4 + group // 16384
        counters.append((0, subsequence & 0xFFFFFFFF, counter & 0xFFFFFFFF, counter >> 32))
        words.append(utils.scalar_philox(counter, subsequence=subsequence, seed=seed))
    state = _rng_host()._make_rng_state(seed, block_offset, device=rng_device)
    output = torch.empty((8, 64), dtype=torch.uint32, device=rng_device)
    utils.run_runtime_state_words(state, output, base, stride, quant_length)
    expected = np.concatenate((np.asarray(counters, np.uint32).T, np.asarray(words, np.uint32).T))
    np.testing.assert_array_equal(output.cpu().numpy(), expected)


@pytest.mark.npu
def test_device_philox_words():
    rng = np.random.default_rng(101)
    counters = rng.integers(0, 2**32, size=(4, 64), dtype=np.uint32)
    counters[:, 0] = 0
    counters[:, 1] = 4294967295
    expected = np.stack(golden._philox4x32_10(*counters))
    np.testing.assert_array_equal(expected[:, 0], [1713891541, 3781805453, 3159862348, 2600524760])
    src = torch.from_numpy(counters).npu()
    out = torch.empty_like(src)
    utils.run_counter_words(src, out)
    np.testing.assert_array_equal(out.cpu().numpy(), expected)


@pytest.mark.npu
@pytest.mark.parametrize("dst", ["float8_e4m3fn", "float8_e5m2"])
def test_device_sr_explicit_random_fields(dst):
    bias, mantissa, _ = golden._FP8_SR_PARAMS[dst]
    emin = 1 - bias
    vals = np.array(
        [
            1 + 2.0 ** (-mantissa - 1),
            2 - 2.0 ** (-mantissa - 1),
            2.0 ** (emin - mantissa - 1),
            2.0**emin - 2.0 ** (emin - mantissa - 1),
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
            -np.nan,
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
            np.float32(2.0 ** (-149)),
            -np.float32(2.0 ** (-149)),
            1 + 2.0 ** (-23),
            -(1 + 2.0 ** (-mantissa - 1)),
        ],
        dtype=np.float32,
    )
    x_np = np.repeat(vals, 4).reshape(1, 64)
    random_np = np.tile(np.array([0, 32767, 2882371584, 65535], dtype=np.uint32), 16).reshape(1, 64)
    expected = golden._sr_cast_fp32_to_fp8(x_np, random_np, dst)
    x = torch.from_numpy(x_np).npu()
    random = torch.from_numpy(random_np).npu()
    out = torch.empty((1, 64), dtype=torch.uint8, device="npu")
    utils.ExplicitSRLaunch(dst).run(x, random, out)
    np.testing.assert_array_equal(out.cpu().numpy(), expected)


@pytest.mark.npu
@pytest.mark.parametrize(
    "base,count",
    [
        (0, 1024),
        (1, 1024),
        (4294967280, 1024),
        (CARRY_BASE, 1024),
        *[(CARRY_BASE, count) for count in (16, 32, 48, 80, 1008)],
    ],
)
@pytest.mark.parametrize("quant_length", [16, 48, 128, 7168, 16 * (2**32 + 13)])
def test_device_batched_random_words_and_fields(base, count, quant_length):
    assert callable(getattr(mixed, "_random_fields_from_words", None)), "batched field consumer missing"
    counters = np.arange(64, dtype=np.uint64) + np.uint64(base)
    expected_words = golden._philox_blocks(counters, quant_length=quant_length).T
    stream = expected_words.T.reshape(-1)
    high, low = (stream >> np.uint32(16), stream & np.uint32(65535))
    rotate = (stream << np.uint32(8)) | (stream >> np.uint32(24))
    expected_fields = np.stack((high, rotate & np.uint32(65535), low, rotate >> np.uint32(16)), axis=-1).reshape(-1)
    initial = np.full((16, 64), 3735928559, dtype=np.uint32)
    expected = initial.copy()
    expected.reshape(-1)[:count] = expected_fields[:count]
    words_out = torch.empty((4, 64), dtype=torch.uint32, device="npu")
    fields_out = torch.from_numpy(initial).npu()
    utils.run_batched_fields(words_out, fields_out, base & 4294967295, base >> 32, count, quant_length)
    np.testing.assert_array_equal(words_out.cpu().numpy(), expected_words)
    np.testing.assert_array_equal(fields_out.cpu().numpy(), expected)


@pytest.mark.npu
@pytest.mark.parametrize("aos", [False, True])
@pytest.mark.parametrize("base", [0, 4294967280, CARRY_BASE, (1 << 48) + 19, (1 << 63) - 64])
@pytest.mark.parametrize("quant_length", [16, 48, 128, 7168, 16 * (2**32 + 13)])
def test_mixed_philox_counter_specialization(aos, base, quant_length):
    counters = np.arange(64, dtype=np.uint64) + np.uint64(base)
    words = golden._philox_blocks(counters, quant_length=quant_length).T
    expected = (words.T if aos else words).reshape(1, 256)
    out = torch.empty((1, 256), dtype=torch.int32, device="npu")
    (utils.run_words_aos if aos else utils.run_words_soa)(out, base, quant_length)
    np.testing.assert_array_equal(out.cpu().numpy().view(np.uint32), expected)


@pytest.mark.npu
@pytest.mark.parametrize("base", [0, 4294967280, CARRY_BASE, (1 << 48) + 19, (1 << 63) - 128])
@pytest.mark.parametrize("quant_length", [16, 48, 128, 7168, 16 * (2**32 + 13)])
def test_mixed_paired_philox_counter_carry(base, quant_length):
    counters = np.arange(128, dtype=np.uint64) + np.uint64(base)
    words = golden._philox_blocks(counters, quant_length=quant_length).T
    out = torch.empty((1, 512), dtype=torch.int32, device="npu")
    utils.run_paired_words(out, base, quant_length)
    np.testing.assert_array_equal(out.cpu().numpy().view(np.uint32), words.T.reshape(1, 512))


@pytest.mark.npu
@pytest.mark.parametrize("aos", [False, True], ids=["soa", "aos"])
@pytest.mark.parametrize("batches", [1, 2, 3, 5, 7, 12])
@pytest.mark.parametrize(
    "quant_length,base",
    [
        pytest.param(columns, base, id=f"C{columns}-{name}")
        for columns in (16, 32, 48, 80, 128, 144, 512, 1024, 7168, 16 * (2**32 + 13))
        for name, base in (("row-end", columns // 16 - 1), ("high", 2**48 + 19))
    ]
    + [
        pytest.param(16, 2**32 - 20, id="row-low32-carry"),
        pytest.param(16 * (2**32 + 13), 2**32 - 20, id="col-low32-carry"),
    ],
)
def test_rng_linear_multibatch_words(aos, batches, quant_length, base, rng_device):
    groups = np.arange(batches * 64, dtype=np.uint64) + np.uint64(base)
    words = golden._philox_blocks(groups, quant_length=quant_length).reshape(batches, 64, 4)
    # SoA is local to each 64-group batch, not one transpose of all batches.
    packed = words if aos else words.transpose(0, 2, 1)
    initial = np.full((1, 3072), 0xDEADBEEF, dtype=np.uint32)
    expected = initial.copy()
    expected[0, : batches * 256] = packed.reshape(-1)
    out = torch.from_numpy(initial.view(np.int32)).to(rng_device)
    utils.MultiBatchWordsLaunch(aos).run(out, base, quant_length, batches)
    # Compare the complete output, including every untouched sentinel word.
    np.testing.assert_array_equal(out.cpu().numpy().view(np.uint32), expected)


@pytest.mark.npu
@pytest.mark.parametrize(
    "quant_length,base",
    [
        pytest.param(255 * 16, 0, id="below-one-row-quad"),
        pytest.param(256 * 16, 0, id="one-row-quad"),
        pytest.param(257 * 16, 1, id="unaligned-cross-row"),
        pytest.param(448 * 16, 0, id="non-power-of-two-row"),
        pytest.param(512 * 16, 0, id="two-quads-per-row"),
        pytest.param(16 * (2**32 + 511), 2**32 - 64, id="column-low32-carry"),
    ],
)
def test_rng_linear_sixteen_batch_words(quant_length, base, rng_device):
    batches = 16
    groups = np.arange(batches * 64, dtype=np.uint64) + np.uint64(base)
    expected = golden._philox_blocks(groups, quant_length=quant_length)
    out = torch.empty((1, batches * 256), dtype=torch.int32, device=rng_device)
    utils.MultiBatchWordsLaunch(True, batches * 256).run(out, base, quant_length, batches)
    np.testing.assert_array_equal(out.cpu().numpy().view(np.uint32), expected.reshape(1, batches * 256))


@pytest.mark.npu
@pytest.mark.parametrize("dst", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("family", ["exponent_edges", "bf16_all_bits", "random_bits"])
def test_q16_sr_matches_integer_reference(family, dst):
    assert callable(getattr(mixed, "_sr_fp32_to_fp8_q16", None)), "Q16 SR implementation missing"
    values, fields = utils.q16_inputs(family)
    expected = golden._sr_cast_fp32_to_fp8(values, fields, dst)
    x, random = (torch.from_numpy(values).npu(), torch.from_numpy(fields).npu())
    out = torch.empty(values.shape, dtype=torch.uint8, device="npu")
    utils.Q16Launch(dst).run(x, random, out)
    np.testing.assert_array_equal(out.cpu().numpy(), expected)


@pytest.mark.npu
@pytest.mark.parametrize("dst,inv_bits", [("float8_e4m3fn", 991054117), ("float8_e5m2", 932333861)])
def test_integer_scale_product_rounding_boundaries(dst, inv_bits):
    assert callable(getattr(mixed, "_scale_alg1_integer", None)), "integer scale implementation missing"
    bits = utils.amax_bits(inv_bits)
    expected, recip16 = golden._mx_scale_alg1_kernel(bits.view(np.float32), dst)
    x = torch.from_numpy(bits).npu()
    codes, recip = (torch.empty_like(x), torch.empty_like(x))
    utils.IntegerScaleLaunch(inv_bits).run(x, codes, recip)
    np.testing.assert_array_equal(codes.cpu().numpy(), expected)
    expected_recip = recip16.astype(np.uint32) << 16
    expected_recip[bits == 0] = np.uint32(0x7F000000)
    np.testing.assert_array_equal(recip.cpu().numpy(), expected_recip)


@pytest.mark.npu
def test_short_rows_helper_uses_logical_rng_across_padded_rows():
    assert callable(mixed.quantize_short_rows), "short-row helper is missing"
    rows, columns, prefix_rows, pitch_shift = (5, 48, 3, 6)
    start = prefix_rows * columns
    cpu = torch.linspace(-9.0, 11.0, rows * columns).reshape(rows, columns)
    cpu[0, 1::2] = -0.0
    cpu[1, 7], cpu[2, 33] = (float("nan"), -float("inf"))
    whole = torch.cat((torch.zeros((prefix_rows, columns)), cpu))
    expected = golden.dynamic_mx_quant_golden(utils.input_numpy(whole), utils.dtype_name(torch.float8_e4m3fn))
    x = cpu.npu().view(1, -1)
    y = torch.empty((1, rows * columns), dtype=torch.uint8, device="npu")
    scales = torch.empty((1, rows * 2), dtype=torch.uint8, device="npu")
    utils.ShortRowsLaunch(dtypes.float32, False, rows, columns, pitch_shift).run(x, y, scales, start)
    np.testing.assert_array_equal(y.cpu().numpy().reshape(rows, columns), expected[0][prefix_rows:])
    np.testing.assert_array_equal(scales.cpu().numpy().reshape(rows, 1, 2), expected[1][prefix_rows:])


@pytest.mark.npu
def test_short_rows_helper_preserves_high_counter_and_low32_carry(monkeypatch):
    assert callable(mixed.quantize_short_rows)
    rows, columns, start = (2, 48, ((17 << 32) + 4294967294) * 16)
    cpu = torch.linspace(-3.25, 5.75, rows * columns).reshape(rows, columns)

    def fields_at_start(size, *, quant_length, seed, block_offset):
        assert quant_length == columns
        assert seed == 0 and block_offset == 0
        return utils.random_bits_at(start, size, quant_length=quant_length)

    monkeypatch.setattr(golden, "_random_bits", fields_at_start)
    expected = golden.dynamic_mx_quant_golden(utils.input_numpy(cpu), "float8_e4m3fn")
    x = cpu.npu().view(1, -1)
    y = torch.empty((1, cpu.numel()), dtype=torch.uint8, device="npu")
    scales = torch.empty((1, rows * 2), dtype=torch.uint8, device="npu")
    utils.ShortRowsLaunch(dtypes.float32, False, rows, columns, 6).run(x, y, scales, start)
    np.testing.assert_array_equal(y.cpu().numpy().reshape(cpu.shape), expected[0])
    np.testing.assert_array_equal(scales.cpu().numpy().reshape(expected[1].shape), expected[1])


@pytest.mark.npu
@pytest.mark.parametrize("base", [2**48 + 19, 3 * (2**32 - 20)], ids=["high-row", "row-low32-carry"])
def test_short_rows_rng_scratch_preserves_full_coordinates(base):
    # Inspect exact words: FP8 bytes alone can hide an incorrect random field.
    rows, columns = 48, 48
    batches = (rows * columns + 1023) // 1024
    cpu = torch.linspace(-3.25, 5.75, rows * columns).reshape(rows, columns)
    x = cpu.npu().view(1, -1)
    y = torch.empty((1, cpu.numel()), dtype=torch.uint8, device="npu")
    scales = torch.empty((1, rows * 2), dtype=torch.uint8, device="npu")
    words = torch.empty((1, batches * 256), dtype=torch.int32, device="npu")
    utils.ShortRowsWordsLaunch(dtypes.float32, False, rows, columns, 6).run(x, y, scales, words, base * 16)
    groups = np.arange(batches * 64, dtype=np.uint64) + np.uint64(base)
    expected = golden._philox_blocks(groups, quant_length=columns).reshape(1, -1)
    np.testing.assert_array_equal(words.cpu().numpy().view(np.uint32), expected)


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS, ids=["fp32-e4", "bf16-e4", "fp32-e5", "bf16-e5"])
def test_one_binary_for_shapes_ranks_and_blocks(src, dst, monkeypatch, record_property, tmp_path):
    sample = mixed
    # Isolate the baseline's dynamic call contract. Automatic workload-based
    # schedules have separate CompileKeys and are covered by the tail reuse test.
    for selector in (
        "_select_fp32_four_pair_policy",
        "_select_bf16_four_panel_policy",
        "_select_bf16_lower_wave_four_panel_policy",
    ):
        monkeypatch.setattr(sample, selector, lambda *args, **kwargs: False)
    dtype = dtypes.float32 if src == torch.float32 else dtypes.bfloat16
    ub = cannbotdsl.get_mem_size("ub")
    tile = sample._make_tiling(dtype, ub).tile_elements
    minimum = 16 if src == torch.bfloat16 else 32
    variants = [
        ((minimum,), 64),
        ((2, 64), 32),
        ((3, 80), 1),
        ((2, 3, 96), 7),
        ((2, 3, 4, 112), 31),
        ((1, 2, 3, 4, 128), 64),
        ((65, 80), 64),
        ((3, 1008), 1),
        ((3, 1024), 7),
        ((3, 1040), 33),
        ((1, tile - 16), 1),
        ((1, tile), 1),
        ((1, tile + 16), 1),
        ((1, 2 * tile + 16), 1),
        ((5, 32), 1),
        ((65, 32), 7),
        ((2049, 32), 64),
        ((8, 4096), None),
        ((3, 80), None),
        ((minimum,), None),
        ((3, 48), 63),
        ((2, 64), 1),
        ((3, 80), 32),
        ((2, 64), 64),
        ((3, 496), 1),
        ((3, 512), 1),
        ((3, 528), 1),
        ((3, 128), 1),
        ((17, 48), 1),
        ((9, 80), 1),
        ((1, 112), 1),
        ((2, 256), 1),
    ]
    # Obtain a fresh cache reference and publish its artifact before other
    # preparations or SDK cache pressure can reclaim the temporary .so.
    sample.clear_caches()
    before = sample.cache_info()
    device = torch.device("npu", torch.npu.current_device())
    baseline_program = sample._get_compiled_kernel(dtype, dst == torch.float8_e5m2, ub, device=device)
    saved = Path(baseline_program.save(str(tmp_path / "dynamic.so")))
    binary_sha256 = hashlib.sha256(saved.read_bytes()).hexdigest()
    invoke = ProviderCallable.__call__
    executions = []

    def observe(program, *args, **kwargs):
        assert program is baseline_program
        assert not program.closed
        executions.append(
            dict(
                path=program.so_path,
                identity=id(program),
                blocks=args[-2],
            )
        )
        return invoke(program, *args, **kwargs)

    monkeypatch.setattr(ProviderCallable, "__call__", observe)
    for index, (shape, blocks) in enumerate(variants):
        run_check(random_input(shape, src), dst, blocks)
        after = sample.cache_info()
        if index == 0:
            first_misses = after.misses
            assert first_misses - before.misses in (0, 1)
        else:
            assert after.misses == first_misses, "A shape/rank/core change triggered compilation"
        assert len(executions) == index + 1
        assert len({(e["path"], e["identity"]) for e in executions}) == 1
        budget = blocks
        if budget is None:
            budget = min(64, cannbotdsl.get_platform_info(stream=torch.npu.current_stream()).vector_core_num)
            if shape[-1] % 64 == 0 or shape[-1] == 32:
                budget = min(budget, (math.prod(shape) + 1023) // 1024)
        rows = 1
        for n in shape[:-1]:
            rows *= n
        assert executions[-1]["blocks"] == min(budget, rows * ((shape[-1] + 63) // 64))
    # Execution retains the loaded image, not the SDK temporary pathname.
    # Inspect the independently saved artifact and its published contract.
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == binary_sha256
    assert baseline_program.so_path == executions[0]["path"]
    assert not baseline_program.closed
    meta = json.loads(Path(str(saved) + ".meta.json").read_text())["call_contract"]
    assert {d["name"] for d in meta["dimensions"]} == {"rows", "columns", "scale_columns"}
    assert all(d["max"] is None for d in meta["dimensions"])
    tensor_shapes = []
    for parameter in meta["parameters"]:
        assert parameter["role"] == "runtime"
        argument = parameter["argument"]
        if argument["kind"] == "tensor":
            tensor_shapes.append(argument["shape"])
    assert len(tensor_shapes) == 4
    assert all(dim["kind"] == "dimension" for shape in tensor_shapes[:3] for dim in shape)
    assert tensor_shapes[-1] == [{"kind": "constant", "value": 2}]
    assert len(executions) == len(variants)
    record_property("unique_executed_binaries", 1)
    record_property("binary_sha256", binary_sha256)
    record_property("binary_path", str(saved))
    record_property("compile_misses", sample.cache_info().misses - before.misses)
    record_property("shape_rank_core_variants", json.dumps(variants))
    record_property("ub_bytes", ub)
    record_property("max_tile_elements", tile)


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS)
def test_public_ocp_quantization_literal_scales(src, dst):
    cpu = torch.full((3, 80), 1.9375, dtype=src)
    code = 23 if dst == torch.float8_e5m2 else 24
    result = utils.public_replay(cpu.npu(), dst_type=code, scale_alg=0)
    utils.assert_result(cpu, result, -1, src, code, 0, 0)
    shift = 15 if code == 23 else 8
    np.testing.assert_array_equal(
        result[1].view(torch.uint8).cpu().numpy().reshape(3, 4), np.array([[127 - shift] * 3 + [0]] * 3, np.uint8)
    )


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS)
@pytest.mark.parametrize("shape", [(64,), (2, 32), (1, 256)])
def test_public_bound_zero_exception_literal_bytes(src, dst, shape):
    # Preserve old independent bound anchors in both packed and general paths.
    groups = torch.ones((math.prod(shape) // 32, 32), dtype=src)
    groups[1::2] = 0
    cpu = groups.reshape(shape)
    code = 23 if dst == torch.float8_e5m2 else 24
    bound, y_code = (57344.0, 0x3C) if code == 23 else (448.0, 0x38)
    result = mixed.dynamic_mx_quant_sr(cpu.npu(), dst_type=code, max_low_bound=bound)
    want_y = np.tile(np.array([y_code] * 32 + [0] * 32, np.uint8), math.prod(shape) // 64).reshape(shape)
    np.testing.assert_array_equal(result[0].view(torch.uint8).cpu().numpy(), want_y)
    if shape[-1] == 32:
        want_scale = np.array([[[127, 0]], [[0, 0]]], np.uint8)
    else:
        want_scale = np.tile(np.array([127, 0], np.uint8), math.prod(shape) // 64)
        want_scale = want_scale.reshape(utils.expected_scale_shape(shape))
    np.testing.assert_array_equal(result[1].view(torch.uint8).cpu().numpy(), want_scale)


@pytest.mark.npu
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(64,), (3, 80), (2, 3, 64)])
def test_sr_defaults_and_input_keyword_match_existing_e4m3_golden(input_dtype, shape):
    cpu = random_input(shape, input_dtype)
    y, scale = utils.public_replay(input=cpu.npu())
    assert y.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float8_e8m0fnu
    expected = golden.dynamic_mx_quant_golden(utils.input_numpy(cpu), "float8_e4m3fn")
    for actual, reference in zip((y, scale), expected):
        np.testing.assert_array_equal(actual.view(torch.uint8).cpu().numpy(), reference)


@pytest.mark.npu
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("code,dtype", [(23, torch.float8_e5m2), (24, torch.float8_e4m3fn)])
@pytest.mark.parametrize("shape", [(64,), (3, 80), (2, 3, 64)])
def test_sr_explicit_torch_attributes_and_positive_tail_axis(input_dtype, code, dtype, shape):
    cpu = random_input(shape, input_dtype)
    x = cpu.npu()
    y, scale = utils.public_replay(
        x, axis=x.ndim - 1, round_mode="stochastic", dst_type=code, block_size=32, scale_alg=1, dst_type_max=0.0
    )
    assert y.dtype == dtype and y.shape == x.shape and y.device == x.device
    assert scale.shape == (*shape[:-1], (shape[-1] + 63) // 64, 2)
    assert scale.dtype == torch.float8_e8m0fnu and scale.device == x.device
    expected = golden.dynamic_mx_quant_golden(utils.input_numpy(cpu), str(dtype).removeprefix("torch."))
    for actual, reference in zip((y, scale), expected):
        np.testing.assert_array_equal(actual.view(torch.uint8).cpu().numpy(), reference)


@pytest.mark.npu
@pytest.mark.parametrize("input_dtype", INPUT_DTYPES)
@pytest.mark.parametrize("dst_dtype", OUTPUT_DTYPES)
def test_vector256_distinct_scales_and_normalization_underflow(input_dtype, dst_dtype):
    # Wrong reciprocal packing or broadcasting destroys the large values;
    # lost signs in native normalization change the tiny values' zero bytes.
    maximum = 57344.0 if dst_dtype == torch.float8_e5m2 else 448.0
    max_code = 0x7B if dst_dtype == torch.float8_e5m2 else 0x7E
    minimum_exponent = -133 if input_dtype == torch.bfloat16 else -149
    exponents = np.arange(104, 112)
    values = np.zeros((8, 32), dtype=np.float32)
    ties = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    for block, exponent in enumerate(exponents):
        values[block, :16] = np.ldexp(np.linspace(-maximum / 2, maximum / 2, 16, dtype=np.float32), int(exponent))
        values[block, 16:20] = np.ldexp(ties, int(exponent) + minimum_exponent)
        values[block, 20:24] = -values[block, 16:20]
        values[block, 24:28] = np.array([0.0, -0.0, 0.0, -0.0], dtype=np.float32)
        # Move the exact maximum across all four SR streams.
        values[block, 28 + block % 4] = np.ldexp(np.float32(maximum), int(exponent))
    x = torch.from_numpy(values.reshape(1, 256)).to(input_dtype)
    actual = utils.input_numpy(x).astype(np.float32).reshape(8, 32)
    assert np.isfinite(actual).all()
    assert np.unique(np.max(np.abs(actual), axis=1)).size == 8
    # Verify that the fixture really reaches tie-to-even subnormal rounding,
    # rather than rounding the probes to zero when constructing the input.
    with np.errstate(under="ignore"):
        normalized = actual * np.ldexp(np.ones((8, 1), dtype=np.float32), -exponents[:, None])
        if input_dtype == torch.bfloat16:
            rounded_bits = normalized.astype(bfloat16).view(np.uint16)
        else:
            rounded_bits = normalized.view(np.uint32)
    np.testing.assert_array_equal(rounded_bits[:, 16:20], np.tile([0, 2, 2, 4], (8, 1)))

    # K256 with one core selects the complete-microblock production path.
    codes, scales = run_check(x, dst_dtype, blocks=1)
    np.testing.assert_array_equal(scales.reshape(-1), exponents + 127)
    codes = codes.reshape(8, 32)
    np.testing.assert_array_equal(codes[np.arange(8), 28 + np.arange(8) % 4], np.full(8, max_code))
    np.testing.assert_array_equal(codes[:, 16:24], np.tile([0, 0, 0, 0, 128, 128, 128, 128], (8, 1)))
    np.testing.assert_array_equal(codes[:, 24:28], np.tile([0, 128, 0, 128], (8, 1)))


@pytest.mark.npu
@pytest.mark.parametrize("input_dtype", INPUT_DTYPES)
@pytest.mark.parametrize("dst_dtype", OUTPUT_DTYPES)
def test_vector256_half_subnormal_scale_boundary(input_dtype, dst_dtype):
    # A generic log2 ceil misses the exceptional product==half-subnormal
    # rule; adjacent input values must survive the packed maxima reduction.
    scalar_type = bfloat16 if input_dtype == torch.bfloat16 else np.float32
    maximum = 57344.0 if dst_dtype == torch.float8_e5m2 else 448.0
    threshold = scalar_type(maximum * 2.0**-127)
    below = np.nextafter(threshold, scalar_type(0))
    above = np.nextafter(threshold, scalar_type(np.inf))
    above2 = np.nextafter(above, scalar_type(np.inf))
    smallest = np.nextafter(scalar_type(0), scalar_type(1))
    info = torch.finfo(input_dtype)
    amplitudes = np.array(
        [below, threshold, above, above2, smallest, info.tiny / 2, info.tiny, info.max], dtype=np.float32
    )
    values = amplitudes[:, None] * np.linspace(-1, 1, 32, dtype=np.float32)[None, :]
    values[:, 1], values[:, 2] = 0.0, -0.0
    x = torch.from_numpy(values.reshape(1, 256)).to(input_dtype)
    maxima = np.max(np.abs(utils.input_numpy(x).astype(np.float32).reshape(8, 32)), axis=1)
    assert np.unique(maxima).size == 8
    codes, scales = run_check(x, dst_dtype, blocks=1)
    expected_boundary = [0, 0, 1, 1] if input_dtype == torch.bfloat16 else [0, 0, 0, 1]
    np.testing.assert_array_equal(scales.reshape(-1)[:4], expected_boundary)
    np.testing.assert_array_equal(codes.reshape(8, 32)[:, 1:3], np.tile([0, 128], (8, 1)))


@pytest.mark.npu
@pytest.mark.parametrize("input_dtype", INPUT_DTYPES)
@pytest.mark.parametrize("dst_dtype", OUTPUT_DTYPES)
def test_vector256_sr_boundaries_in_every_stream(input_dtype, dst_dtype):
    # Each block has a different power-of-two scale. Midpoints, binade carry,
    # and FP8 subnormal transitions must retain their four Philox lane fields.
    e5m2 = dst_dtype == torch.float8_e5m2
    maximum, mantissa, emin = (57344.0, 2, -14) if e5m2 else (448.0, 3, -6)
    scalar_type = bfloat16 if input_dtype == torch.bfloat16 else np.float32
    centers = [2.0 ** (emin - mantissa - 1), 2.0**emin, 1 + 2.0 ** (-mantissa - 1), 2 - 2.0 ** (-mantissa - 1)]
    probes = []
    for value in centers:
        center = scalar_type(value)
        neighborhood = [np.nextafter(center, scalar_type(0)), center, np.nextafter(center, scalar_type(np.inf))]
        probes.extend(neighborhood)
        probes.extend(-float(item) for item in neighborhood)
    probes.extend([0.0, -0.0, 1.0, -1.0, maximum, -maximum, maximum / 2, -maximum / 2])
    values = np.zeros((8, 32), dtype=np.float32)
    exponents = np.arange(-28, 29, 8)
    for block, exponent in enumerate(exponents):
        values[block] = np.ldexp(np.roll(np.asarray(probes, dtype=np.float32), block), int(exponent))
    x = torch.from_numpy(values.reshape(1, 256)).to(input_dtype)
    _, scales = run_check(x, dst_dtype, blocks=1)
    np.testing.assert_array_equal(scales.reshape(-1), exponents + 127)


@pytest.mark.npu
@pytest.mark.parametrize("input_dtype", INPUT_DTYPES)
@pytest.mark.parametrize("dst_dtype", OUTPUT_DTYPES)
def test_vector256_nonfinite_blocks_do_not_poison_neighbors(input_dtype, dst_dtype):
    # Adjacent exceptional maxima must produce only their own all-NaN block,
    # including special values located in different deinterleaved streams.
    maximum = 57344.0 if dst_dtype == torch.float8_e5m2 else 448.0
    exponents = np.arange(-21, 22, 6)
    values = np.ldexp(np.full((8, 32), maximum, dtype=np.float32), exponents[:, None])
    values[:, 1::2] *= -1
    values[0] = 0.0
    values[0, 1::2] = -0.0
    values[1, 5] = np.nan
    values[3, 14] = np.inf
    values[5, 23] = -np.inf
    x = torch.from_numpy(values.reshape(1, 256)).to(input_dtype)
    codes, scales = run_check(x, dst_dtype, blocks=1)
    codes = codes.reshape(8, 32)
    np.testing.assert_array_equal(codes[0], np.tile([0, 128], 16))
    np.testing.assert_array_equal(codes[[1, 3, 5]], np.full((3, 32), 0x7F))
    expected_scales = exponents + 127
    expected_scales[0] = 0
    expected_scales[[1, 3, 5]] = 255
    np.testing.assert_array_equal(scales.reshape(-1), expected_scales)


@pytest.mark.npu
@pytest.mark.parametrize("input_dtype", INPUT_DTYPES)
@pytest.mark.parametrize("dst_dtype", OUTPUT_DTYPES)
@pytest.mark.parametrize("columns", [496, 512, 528, 1008, 1024, 1040])
def test_independent_block_scales_across_tile_and_tail(columns, input_dtype, dst_dtype):
    x = random_input((3, columns), torch.float32)
    for row in range(3):
        for block in range((columns + 31) // 32):
            start, end = block * 32, min(columns, (block + 1) * 32)
            exponent = ((block * 17 + row * 31) % 241) - 120
            x[row, start:end] *= 2.0**exponent
    x[0, :32] = 0.0
    x[0, 1:32:2] = -0.0
    x[1, 31] = float("nan")
    x[1, 32] = float("inf")
    x[2, -1] = -float("inf")
    run_check(x.to(input_dtype), dst_dtype, blocks=1)


@pytest.mark.npu
@pytest.mark.parametrize("input_dtype", INPUT_DTYPES)
@pytest.mark.parametrize("dst_dtype", OUTPUT_DTYPES)
@pytest.mark.parametrize("columns", [240, 256, 272, 768, 1008, 1024, 1040])
def test_microblock_scale_rng_and_tail(columns, input_dtype, dst_dtype):
    x = random_input((3, columns), torch.float32)
    for row in range(3):
        for block in range((columns + 31) // 32):
            x[row, block * 32 : min(columns, (block + 1) * 32)] *= 2.0 ** (((block * 17 + row * 31) % 241) - 120)
    x[0, :32] = 0.0
    x[0, 1:32:2] = -0.0
    x[1, 255 if columns > 255 else 63] = float("nan")
    x[2, -1] = -float("inf")
    run_check(x.to(input_dtype), dst_dtype, blocks=1)


@pytest.mark.npu
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "dst_dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2],
    ids=["e4m3fn", "e5m2"],
)
def test_short_rows_scale_half_subnormal_boundary(input_dtype, dst_dtype):
    scalar_type = bfloat16 if input_dtype == torch.bfloat16 else np.float32
    maximum = 57344.0 if dst_dtype == torch.float8_e5m2 else 448.0
    threshold = scalar_type(maximum * 2.0**-127)
    below = np.nextafter(threshold, scalar_type(0))
    above = np.nextafter(threshold, scalar_type(np.inf))
    above2 = np.nextafter(above, scalar_type(np.inf))
    info = torch.finfo(input_dtype)
    amplitudes = np.array(
        [
            below,
            threshold,
            above,
            above2,
            np.nextafter(scalar_type(0), scalar_type(1)),
            info.tiny / 2,
            info.tiny,
            info.max,
        ],
        dtype=np.float32,
    )
    values = np.zeros((8, 48), dtype=np.float32)
    values[:, :32] = amplitudes[:, None] * np.linspace(-1, 1, 32, dtype=np.float32)
    values[:, 1], values[:, 2] = 0.0, -0.0
    cpu = torch.from_numpy(values).to(input_dtype)
    maxima = np.max(np.abs(utils.input_numpy(cpu).astype(np.float32)[:, :32]), axis=1)
    assert np.unique(maxima).size == 8

    codes, scales = run_check(cpu, dst_dtype, blocks=1)
    expected = [0, 0, 1, 1] if input_dtype == torch.bfloat16 else [0, 0, 0, 1]
    np.testing.assert_array_equal(scales.reshape(8, 2)[:4, 0], expected)
    np.testing.assert_array_equal(codes[:, 1:3], np.tile([0, 128], (8, 1)))


@pytest.mark.npu
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "dst_dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2],
    ids=["e4m3fn", "e5m2"],
)
@pytest.mark.parametrize("columns", [80, 144], ids=["short-rows", "four-field"])
@pytest.mark.parametrize("algorithm", [0, 1])
def test_normalization_underflow_preserves_sr_bytes_and_signed_zero(input_dtype, dst_dtype, columns, algorithm):
    maximum = 57344.0 if dst_dtype == torch.float8_e5m2 else 448.0
    max_code = 0x7B if dst_dtype == torch.float8_e5m2 else 0x7E
    minimum_exponent = -133 if input_dtype == torch.bfloat16 else -149
    exponents = np.arange(104, 112)
    values = np.zeros((8, columns), dtype=np.float32)
    ties = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    for row, exponent in enumerate(exponents):
        values[row, :16] = np.ldexp(
            np.linspace(-maximum / 2, maximum / 2, 16, dtype=np.float32),
            int(exponent),
        )
        values[row, 16:20] = np.ldexp(ties, int(exponent) + minimum_exponent)
        values[row, 20:24] = -values[row, 16:20]
        values[row, 24:28] = [0.0, -0.0, 0.0, -0.0]
        values[row, 28 + row % 4] = np.ldexp(np.float32(maximum), int(exponent))
    cpu = torch.from_numpy(values).to(input_dtype)
    actual = utils.input_numpy(cpu).astype(np.float32)
    with np.errstate(under="ignore"):
        normalized = actual[:, :32] * np.ldexp(
            np.ones((8, 1), dtype=np.float32),
            -exponents[:, None],
        )
        rounded = (
            normalized.astype(bfloat16).view(np.uint16) if input_dtype == torch.bfloat16 else normalized.view(np.uint32)
        )
    np.testing.assert_array_equal(rounded[:, 16:20], np.tile([0, 2, 2, 4], (8, 1)))

    codes, scales = run_check(cpu, dst_dtype, blocks=1, scale_alg=algorithm)
    np.testing.assert_array_equal(scales.reshape(8, -1)[:, 0], exponents + 127)
    np.testing.assert_array_equal(codes[np.arange(8), 28 + np.arange(8) % 4], max_code)
    np.testing.assert_array_equal(
        codes[:, 16:28],
        np.tile([0, 0, 0, 0, 128, 128, 128, 128, 0, 128, 0, 128], (8, 1)),
    )


@pytest.mark.npu
@pytest.mark.parametrize("input_dtype", INPUT_DTYPES)
@pytest.mark.parametrize("dst_dtype", OUTPUT_DTYPES)
def test_npu_special_values_and_scale_boundaries(input_dtype, dst_dtype):
    x = np.zeros((10, 64), dtype=np.float32)
    x[0, 1::2] = -0.0
    x[1] = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    x[1, 3], x[1, 35] = np.nan, -np.nan
    x[2] = x[1]
    x[2, 3], x[2, 35] = np.inf, -np.inf
    max_value = 448.0 if dst_dtype == torch.float8_e4m3fn else 57344.0
    scalar_type = np.float32 if input_dtype == torch.float32 else bfloat16
    lower = np.nextafter(scalar_type(max_value), scalar_type(-np.inf))
    upper = np.nextafter(scalar_type(max_value), scalar_type(np.inf))
    for row, amplitude in enumerate((lower, max_value, upper), start=3):
        x[row] = np.linspace(-float(amplitude), float(amplitude), 64, dtype=np.float32)
    info = torch.finfo(input_dtype)
    x[6, ::2], x[6, 1::2] = info.max, -info.max
    x[7, ::2], x[7, 1::2] = info.tiny, -info.tiny
    x[8, ::2], x[8, 1::2] = info.tiny / 2, -info.tiny / 2
    x[9] = np.linspace(-2.0, 2.0, 64, dtype=np.float32)
    x[9, 0], x[9, 32] = info.max, -info.max
    run_check(torch.from_numpy(x).to(input_dtype), dst_dtype)


@pytest.mark.npu
@pytest.mark.parametrize("input_dtype", INPUT_DTYPES)
@pytest.mark.parametrize("dst_dtype", OUTPUT_DTYPES)
def test_npu_subnormal_scale_product_threshold(input_dtype, dst_dtype):
    # At amax = max_fp8 * 2^-127 the scale product has FP32 fraction
    # 0x400000. The immediate FP32 successor rounds to that same product;
    # its second successor crosses the strict subnormal ceil threshold.
    scalar_type = np.float32 if input_dtype == torch.float32 else bfloat16
    max_value = 448.0 if dst_dtype == torch.float8_e4m3fn else 57344.0
    threshold = scalar_type(max_value * 2.0**-127)
    predecessor = np.nextafter(threshold, scalar_type(0.0))
    successor = np.nextafter(threshold, scalar_type(np.inf))
    second_successor = np.nextafter(successor, scalar_type(np.inf))
    smallest = np.nextafter(scalar_type(0.0), scalar_type(1.0))
    tiny = torch.finfo(input_dtype).tiny
    values = np.zeros((10, 64), dtype=np.float32)
    for row, amplitude in enumerate((predecessor, threshold, successor)):
        values[row, ::2], values[row, 1::2] = float(amplitude), -float(amplitude)
    values[3] = float(smallest)
    values[4] = -float(smallest)
    values[5, ::2], values[5, 1::2] = float(smallest), -float(smallest)
    values[6, ::2], values[6, 1::2] = tiny / 2, -tiny / 2
    values[7, ::2], values[7, 1::2] = tiny, -tiny
    values[8] = values[5]
    values[8, 0], values[8, 32] = float(predecessor), -float(second_successor)
    values[9, 1::2] = -0.0
    x = torch.from_numpy(values).to(input_dtype)
    actual_input = utils.input_numpy(x).astype(np.float32)
    assert actual_input[0, 0] < actual_input[1, 0] < actual_input[2, 0]
    assert 0 < actual_input[3, 0] < tiny
    # run_check derives both expected byte arrays from utils.input_numpy(x),
    # including the actual BF16 rounding of the input rather than its source.
    run_check(x, dst_dtype)


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS, ids=["fp32-e4", "bf16-e4", "fp32-e5", "bf16-e5"])
@pytest.mark.parametrize("columns", (240, 256, 272, 768, 1024, 1280, 2048))
def test_four_way_sr_and_philox_batch_edges(src, dst, columns):
    # One core keeps the exact 256-element path visible; 1280/2048 need
    # distinct Philox batches rather than repeatedly using the first 64 counters.
    run_check(random_input((1, columns), src), dst, blocks=1)


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS, ids=["fp32-e4", "bf16-e4", "fp32-e5", "bf16-e5"])
@pytest.mark.parametrize("columns", (48, 80, 112))
def test_single_row_to_compact_multirow_dispatch(src, dst, columns):
    # A single row uses the general tail path; adding a row enables short rows.
    for rows in (1, 2):
        run_check(random_input((rows, columns), src), dst, blocks=1, offset=1)


@pytest.mark.npu
@pytest.mark.parametrize("src,dst,columns,multiple,delta", SHORT_CASES)
def test_short_rows_capacity_compact_rng_and_tail(src, dst, columns, multiple, delta):
    pitch = 64 if columns < 64 else 128
    max_rows = utils.tile_elements(src) // pitch
    rows = multiple * max_rows + delta
    # K48/K80/K112 place adjacent rows at different positions in the Philox
    # counter stream; padding must affect reductions only, never random indices.
    run_check(random_input((rows, columns), src), dst, blocks=1)


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS, ids=["fp32-e4", "bf16-e4", "fp32-e5", "bf16-e5"])
@pytest.mark.parametrize("extra", (0, 256, 272))
def test_large_aos_tiles_and_last_partial_tile(src, dst, extra):
    # More than one UB tile exercises nonzero logical RNG starts and scale
    # writeback beyond the first 64-lane scale register.
    run_check(random_input((1, 2 * utils.tile_elements(src) + extra), src), dst, blocks=1)


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS, ids=["fp32-e4", "bf16-e4", "fp32-e5", "bf16-e5"])
@pytest.mark.parametrize("pattern", ("scale_thresholds", "sr_boundaries", "raw_bits"))
def test_four_way_sr_numerical_boundaries(src, dst, pattern):
    # Existing numerical cases have K96 and now exercise short rows. These
    # K256 rows coalesce into a full four-way tile, including native BF16 math.
    cpu = utils.numerical_input(pattern, (16, 256), src, dst)
    run_check(cpu, dst, blocks=1)


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS, ids=["fp32-e4", "bf16-e4", "fp32-e5", "bf16-e5"])
def test_four_way_reduction_keeps_eight_blocks_independent(src, dst):
    maximum = 57344.0 if dst == torch.float8_e5m2 else 448.0
    cpu = torch.empty((8, 256), dtype=torch.float32)
    signs = torch.where(torch.arange(32) % 2 == 0, 1.0, -1.0)
    for group in range(8):
        cpu[:, group * 32 : (group + 1) * 32] = signs * maximum * 2.0 ** (group - 4)
    # Put a special value at a different reduction boundary in each row.
    # Unaffected groups have manually known scale bytes 123 through 130.
    for row in range(8):
        cpu[row, row * 32 + (31 if row % 2 else 0)] = float("nan") if row % 2 else -float("inf")
    codes, scales = run_check(cpu.to(src), dst, blocks=1)
    expected = np.tile(np.arange(123, 131, dtype=np.uint8), (8, 1))
    expected[np.arange(8), np.arange(8)] = 255
    np.testing.assert_array_equal(scales.reshape(8, 8), expected)
    blocks = codes.reshape(8, 8, 32)
    np.testing.assert_array_equal(blocks[np.arange(8), np.arange(8)], np.full((8, 32), 0x7F, np.uint8))


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS, ids=["fp32-e4", "bf16-e4", "fp32-e5", "bf16-e5"])
@pytest.mark.parametrize("rows", (31, 32, 33, 65))
def test_k32_padding_crosses_scale_register_boundary(src, dst, rows):
    _, scales = run_check(random_input((rows, 32), src), dst, blocks=1)
    np.testing.assert_array_equal(scales.reshape(rows, 2)[:, 1], np.zeros(rows, np.uint8))


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS, ids=["fp32-e4", "bf16-e4", "fp32-e5", "bf16-e5"])
def test_rng_stays_identical_when_core_count_changes_compute_path(src, dst):
    cpu = random_input((3, 256), src)
    # One core processes 768 values through four-way SR. Other core counts
    # split the same logical stream into 64/128/384-element general tiles.
    expected = run_check(cpu, dst, blocks=1)
    for blocks in (2, 7, 64):
        run_check(cpu, dst, blocks=blocks, expected=expected)


@pytest.mark.npu
@pytest.mark.parametrize("axis", [1, -2])
@pytest.mark.parametrize(
    "scale_alg,scale_code,y_code,bound",
    [
        (0, 119, 0x78, 0.0),
        (1, 127, 0x38, 448.0),
    ],
)
def test_non_tail_axis_scale_interleaves_pairs(axis, scale_alg, scale_code, y_code, bound):
    x = torch.zeros((2, 64, 3), dtype=torch.float32)
    x[:, :32, :] = 1
    y, scale = mixed.dynamic_mx_quant_sr(x.npu(), axis=axis, scale_alg=scale_alg, max_low_bound=bound)
    assert y.shape == x.shape
    assert scale.shape == (2, 1, 3, 2)
    want_y = np.zeros(x.shape, dtype=np.uint8)
    want_y[:, :32, :] = y_code
    want_scale = np.zeros((2, 1, 3, 2), dtype=np.uint8)
    want_scale[..., 0] = scale_code
    np.testing.assert_array_equal(y.view(torch.uint8).cpu().numpy(), want_y)
    np.testing.assert_array_equal(scale.view(torch.uint8).cpu().numpy(), want_scale)


@pytest.mark.npu
@pytest.mark.parametrize(
    "src,dst,code,alg,bound",
    [
        (torch.float32, "float8_e4m3fn", 24, 0, 0.0),
        (torch.float32, "float8_e4m3fn", 24, 1, 3.25),
        (torch.bfloat16, "float8_e5m2", 23, 1, 0.75),
    ],
)
def test_non_tail_random_values_match_cpu_golden(src, dst, code, alg, bound):
    cpu = (torch.randn((2, 48, 5), generator=torch.Generator().manual_seed(4201)) * 5).to(src)
    y, scale = utils.public_replay(cpu.npu(), axis=1, dst_type=code, scale_alg=alg, max_low_bound=bound)
    from ml_dtypes import bfloat16

    numpy_x = cpu.float().numpy().astype(bfloat16 if src == torch.bfloat16 else np.float32)
    want_y, want_scale = golden.dynamic_mx_quant_golden(numpy_x, dst, scale_alg=alg, max_low_bound=bound, axis=1)
    np.testing.assert_array_equal(y.view(torch.uint8).cpu().numpy(), want_y)
    np.testing.assert_array_equal(scale.view(torch.uint8).cpu().numpy(), want_scale)


@pytest.mark.npu
def test_non_tail_e8m0_move_handles_512_axis():
    # The FP8 E8M0 Transpose AICPU path fails for this otherwise valid shape.
    x = torch.ones((8, 512, 2), dtype=torch.float32).npu()
    y, scale = mixed.dynamic_mx_quant_sr(x, axis=-2, scale_alg=0)
    np.testing.assert_array_equal(y.view(torch.uint8).cpu().numpy(), np.full((8, 512, 2), 0x78, dtype=np.uint8))
    np.testing.assert_array_equal(scale.view(torch.uint8).cpu().numpy(), np.full((8, 8, 2, 2), 119, dtype=np.uint8))


@pytest.mark.npu
@pytest.mark.parametrize(
    "shape,axis,src,dst,alg,bound",
    [
        ((16, 3), 0, torch.bfloat16, "float8_e4m3fn", 0, 0.0),
        ((48, 2), -2, torch.bfloat16, "float8_e5m2", 1, 0.75),
        ((2, 32, 5), 1, torch.float32, "float8_e5m2", 0, 0.0),
        ((2, 80, 3), -2, torch.bfloat16, "float8_e4m3fn", 1, 3.25),
        ((2, 5, 96, 3), 2, torch.float32, "float8_e4m3fn", 0, 0.0),
        ((2, 64, 3, 4), 1, torch.bfloat16, "float8_e5m2", 1, 0.75),
        ((1, 512, 2), -2, torch.float32, "float8_e4m3fn", 1, 448.0),
        ((2, 1024, 3), 1, torch.float32, "float8_e5m2", 0, 0.0),
        ((8, 512, 2), 1, torch.float32, "float8_e4m3fn", 0, 0.0),
        ((2, 1024, 2), -2, torch.float32, "float8_e5m2", 0, 0.0),
        ((2, 512, 2), 1, torch.bfloat16, "float8_e5m2", 1, 0.75),
        ((2, 512, 2), -2, torch.float32, "float8_e4m3fn", 1, 448.0),
        ((512, 8), 0, torch.float32, "float8_e4m3fn", 0, 0.0),
        ((64, 128), 0, torch.bfloat16, "float8_e5m2", 1, 0.75),
        ((1, 2, 128, 64), -1, torch.float32, "float8_e4m3fn", 0, 0.0),
        ((1, 2, 128, 64), -2, torch.float32, "float8_e4m3fn", 0, 0.0),
        ((1, 2, 64, 128), -1, torch.bfloat16, "float8_e5m2", 1, 0.75),
        ((1, 2, 64, 128), -2, torch.bfloat16, "float8_e5m2", 1, 0.75),
        ((2, 128, 4), 1, torch.float32, "float8_e4m3fn", 0, 0.0),
        ((1, 64, 16), -2, torch.bfloat16, "float8_e5m2", 1, 0.75),
        ((1, 64, 128), 1, torch.float32, "float8_e4m3fn", 0, 0.0),
        ((512, 1), 0, torch.float32, "float8_e4m3fn", 0, 0.0),
        ((2, 4096, 2), 1, torch.float32, "float8_e4m3fn", 0, 0.0),
        ((8, 4096, 2), -2, torch.float32, "float8_e5m2", 1, 448.0),
        ((1, 1536, 2), 1, torch.bfloat16, "float8_e4m3fn", 1, 0.75),
        ((2, 16384, 2), 1, torch.float32, "float8_e5m2", 0, 0.0),
        ((2, 256, 2), 1, torch.float32, "float8_e4m3fn", 0, 0.0),
        ((2, 256, 2), -2, torch.bfloat16, "float8_e5m2", 1, 3.25),
    ],
)
def test_axis_generalized_shapes_match_cpu_golden(shape, axis, src, dst, alg, bound):
    code = {"float8_e4m3fn": 24, "float8_e5m2": 23}[dst]
    cpu = (torch.randn(shape, generator=torch.Generator().manual_seed(4172)) * 5).to(src)
    # Exercise the all-zero block exception and nonzero data in the same tensor.
    if shape[axis] >= 48:
        slices = [slice(None)] * len(shape)
        slices[axis] = slice(32, 64)
        cpu[tuple(slices)] = 0
    y, scale = utils.public_replay(cpu.npu(), axis=axis, dst_type=code, scale_alg=alg, max_low_bound=bound)
    from ml_dtypes import bfloat16

    numpy_x = cpu.float().numpy().astype(bfloat16 if src == torch.bfloat16 else np.float32)
    want_y, want_scale = golden.dynamic_mx_quant_golden(numpy_x, dst, scale_alg=alg, max_low_bound=bound, axis=axis)
    np.testing.assert_array_equal(y.view(torch.uint8).cpu().numpy(), want_y)
    np.testing.assert_array_equal(scale.view(torch.uint8).cpu().numpy(), want_scale)


@pytest.mark.npu
def test_mixed_non_tail_fast_path_reuses_one_binary_for_dynamic_k():
    api = mixed.dynamic_mx_quant_sr
    before = mixed.cache_info()
    for columns in (512, 1024):
        x = torch.ones((2, columns, 2), dtype=torch.float32).npu()
        y, scale = api(x, axis=-2, scale_alg=0)
        assert y.is_contiguous() and scale.is_contiguous()
        np.testing.assert_array_equal(y.view(torch.uint8).cpu().numpy(), np.full(x.shape, 0x78, dtype=np.uint8))
        np.testing.assert_array_equal(
            scale.view(torch.uint8).cpu().numpy(), np.full((2, columns // 64, 2, 2), 119, dtype=np.uint8)
        )
    after = mixed.cache_info()
    assert after.misses - before.misses <= 1
    assert after.hits - before.hits >= 1


@pytest.mark.npu
@pytest.mark.parametrize("alg", [0, 1])
@pytest.mark.parametrize(
    "shape,axis,src,dst",
    [
        ((2, 96, 4), -2, torch.float32, "float8_e4m3fn"),
        ((2, 128, 8), -2, torch.bfloat16, "float8_e5m2"),
        ((2, 256, 16), -2, torch.float32, "float8_e5m2"),
        ((2, 512, 32), -2, torch.float32, "float8_e4m3fn"),
        ((2, 512, 64), -2, torch.bfloat16, "float8_e4m3fn"),
        ((1, 512, 128), -2, torch.float32, "float8_e4m3fn"),
        ((2, 4096, 16), -2, torch.bfloat16, "float8_e5m2"),
        ((512, 8), 0, torch.float32, "float8_e4m3fn"),
        ((512, 64), 0, torch.bfloat16, "float8_e5m2"),
        ((512, 128), 0, torch.float32, "float8_e5m2"),
        ((1, 2, 128, 64), -1, torch.float32, "float8_e4m3fn"),  # FA Q/K: D axis
        ((1, 2, 128, 64), -2, torch.bfloat16, "float8_e5m2"),  # FA V: S axis
        ((1, 8, 512, 64), -1, torch.float32, "float8_e4m3fn"),  # FA Q/K: D axis
        ((1, 8, 512, 64), -2, torch.float32, "float8_e4m3fn"),  # FA V: S axis
        ((2, 2, 96, 128), -1, torch.bfloat16, "float8_e4m3fn"),
        ((2, 2, 96, 128), -2, torch.float32, "float8_e5m2"),
    ],
)
def test_width_and_fa_layouts_match_cpu_golden(alg, shape, axis, src, dst):
    """Vary physical tail width independently of the quantization axis."""
    code = {"float8_e4m3fn": 24, "float8_e5m2": 23}[dst]
    x = (torch.randn(shape, generator=torch.Generator().manual_seed(16341)) * 5).to(src)
    slices = [slice(None)] * len(shape)
    slices[axis] = slice(32, 64)
    x[tuple(slices)] = 0
    bound = 3.25 if alg == 1 else 0.0
    y, scale = utils.public_replay(x.npu(), axis=axis, dst_type=code, scale_alg=alg, max_low_bound=bound)
    from ml_dtypes import bfloat16

    numpy_x = x.float().numpy().astype(bfloat16 if src == torch.bfloat16 else np.float32)
    want_y, want_scale = golden.dynamic_mx_quant_golden(numpy_x, dst, scale_alg=alg, max_low_bound=bound, axis=axis)
    np.testing.assert_array_equal(y.view(torch.uint8).cpu().numpy(), want_y)
    np.testing.assert_array_equal(scale.view(torch.uint8).cpu().numpy(), want_scale)


@pytest.mark.npu
def test_mixed_large_width128_scale_pairs_match_cpu_golden():
    """A second D chunk must not overwrite the first chunk's next S pair."""
    shape = (16, 1024, 128)  # selects the 128-row wide tile
    x = (torch.randn(shape, generator=torch.Generator().manual_seed(2468)) * 5).float()
    x[:, 32:64, :] = 0
    y, scale = utils.public_replay(x.npu(), axis=-2, scale_alg=1)
    want_y, want_scale = golden.dynamic_mx_quant_golden(x.numpy(), "float8_e4m3fn", scale_alg=1, axis=-2)
    np.testing.assert_array_equal(y.view(torch.uint8).cpu().numpy(), want_y)
    np.testing.assert_array_equal(scale.view(torch.uint8).cpu().numpy(), want_scale)


@pytest.mark.npu
@pytest.mark.parametrize("shape", [(16, 1024, 32), (16, 1024, 64), (16, 1024, 128)])
@pytest.mark.parametrize(
    "src,dst",
    [
        (torch.float32, "float8_e4m3fn"),
        (torch.bfloat16, "float8_e5m2"),
    ],
)
@pytest.mark.parametrize("alg", [0, 1])
def test_mixed_wide_large_tiles_match_cpu_golden(shape, src, dst, alg):
    """Exercise the 256-row and 128-row tiles for both scale algorithms."""
    x = (torch.randn(shape, generator=torch.Generator().manual_seed(7293)) * 5).to(src)
    x[:, 32:64, :] = 0
    code = {"float8_e4m3fn": 24, "float8_e5m2": 23}[dst]
    bound = 3.25 if alg == 1 else 0.0
    y, scale = utils.public_replay(x.npu(), axis=-2, dst_type=code, scale_alg=alg, max_low_bound=bound)
    from ml_dtypes import bfloat16

    numpy_x = x.float().numpy().astype(bfloat16 if src == torch.bfloat16 else np.float32)
    want_y, want_scale = golden.dynamic_mx_quant_golden(numpy_x, dst, scale_alg=alg, max_low_bound=bound, axis=-2)
    np.testing.assert_array_equal(y.view(torch.uint8).cpu().numpy(), want_y)
    np.testing.assert_array_equal(scale.view(torch.uint8).cpu().numpy(), want_scale)


@pytest.mark.npu
@pytest.mark.parametrize("alg", [0, 1])
def test_mixed_wide32_two_dimensional_first_axis_matches_cpu_golden(alg):
    x = (torch.randn((512, 32), generator=torch.Generator().manual_seed(3460)) * 5).float()
    x[32:64, :] = 0
    bound = 3.25 if alg == 1 else 0.0
    y, scale = utils.public_replay(x.npu(), axis=0, scale_alg=alg, max_low_bound=bound)
    want_y, want_scale = golden.dynamic_mx_quant_golden(
        x.numpy(), "float8_e4m3fn", axis=0, scale_alg=alg, max_low_bound=bound
    )
    np.testing.assert_array_equal(y.view(torch.uint8).cpu().numpy(), want_y)
    np.testing.assert_array_equal(scale.view(torch.uint8).cpu().numpy(), want_scale)


@pytest.mark.npu
def test_mixed_wide32_reuses_binary_across_dynamic_s():
    api = mixed.dynamic_mx_quant_sr
    before = mixed.cache_info()
    for shape in ((1, 64, 32), (2, 512, 32), (16, 1024, 32)):
        x = torch.ones(shape, dtype=torch.float32, device="npu")
        y, scale = api(x, axis=-2, scale_alg=1)
        assert y.shape == x.shape and y.is_contiguous()
        assert scale.shape == (shape[0], shape[1] // 64, 32, 2)
        assert scale.is_contiguous()
        torch.npu.synchronize()
    after = mixed.cache_info()
    assert after.misses - before.misses <= 1
    assert after.hits - before.hits >= 2


@pytest.mark.npu
def test_mixed_wide_reuses_binary_across_dynamic_s_and_d():
    api = mixed.dynamic_mx_quant_sr
    before = mixed.cache_info()
    for shape in ((8, 512, 64), (16, 1024, 64), (16, 1024, 128)):
        x = torch.ones(shape, dtype=torch.float32, device="npu")
        y, scale = api(x, axis=-2, scale_alg=1)
        assert y.shape == x.shape and y.is_contiguous()
        assert scale.shape == (shape[0], shape[1] // 64, shape[2], 2)
        assert scale.is_contiguous()
        torch.npu.synchronize()
    after = mixed.cache_info()
    assert after.misses - before.misses <= 1
    assert after.hits - before.hits >= 2


@pytest.mark.npu
@pytest.mark.parametrize(
    "alg,dst,code,bound",
    [
        (0, "float8_e4m3fn", 24, 0.0),
        (1, "float8_e4m3fn", 24, 3.25),
        (0, "float8_e5m2", 23, 0.0),
        (1, "float8_e5m2", 23, 3.25),
    ],
)
def test_mixed_wide8_direct_dynamic_shapes_match_cpu_golden(alg, dst, code, bound, monkeypatch):
    """The width-8 direct binary must serve multiple non-tail shapes exactly."""
    import struct

    from cannbotdsl import dtypes
    from cannbotdsl import get_mem_size

    api = mixed.dynamic_mx_quant_sr
    compile_wide8 = api.__globals__["_get_compiled_kernel"]
    bound_bits = struct.unpack("<I", struct.pack("<f", bound))[0]
    first = compile_wide8(
        dtypes.float32,
        code == 23,
        get_mem_size("ub"),
        alg,
        bound_bits,
        layout=mixed.Layout.WIDE8,
        device=torch.device("npu", 0),
    )
    assert first is compile_wide8(
        dtypes.float32,
        code == 23,
        get_mem_size("ub"),
        alg,
        bound_bits,
        layout=mixed.Layout.WIDE8,
        device=torch.device("npu", 0),
    )

    invoke = ProviderCallable.__call__
    fingerprint = hashlib.sha256(Path(first.so_path).read_bytes()).hexdigest()
    executions = []

    def observe(program, *args, **kwargs):
        assert program is first
        assert hashlib.sha256(Path(program.so_path).read_bytes()).hexdigest() == fingerprint
        executions.append(id(program))
        return invoke(program, *args, **kwargs)

    monkeypatch.setattr(ProviderCallable, "__call__", observe)

    for shape in ((256, 8), (512, 8), (3, 768, 8), (2, 1024, 8), (1, 2, 1536, 8), (8, 2048, 8)):
        cpu = torch.randn(shape, generator=torch.Generator().manual_seed(20260923)) * 5
        cpu[..., 32:64, :] = 0
        y, scale = utils.launch(cpu.npu(), getattr(torch, dst), axis=-2, scale_alg=alg, max_low_bound=bound)
        want_y, want_scale = golden.dynamic_mx_quant_golden(
            cpu.numpy(), dst, axis=-2, scale_alg=alg, max_low_bound=bound
        )
        np.testing.assert_array_equal(y.view(torch.uint8).cpu().numpy(), want_y)
        np.testing.assert_array_equal(scale.view(torch.uint8).cpu().numpy(), want_scale)
    assert len(executions) == 6


@pytest.mark.npu
@pytest.mark.parametrize(
    "alg,dst,code,bound",
    [
        (0, "float8_e4m3fn", 24, 0.0),
        (1, "float8_e4m3fn", 24, 3.25),
        (0, "float8_e5m2", 23, 0.0),
        (1, "float8_e5m2", 23, 3.25),
    ],
)
def test_mixed_wide8_direct_special_values_match_cpu_golden(alg, dst, code, bound):
    """Check special FP32 values in the physical width-8 non-tail path."""

    x = np.zeros((2, 512, 8), dtype=np.float32)
    x[:, 32:64, :] = -0.0
    x[:, 64:96, :] = np.float32(2.0**-24)
    x[:, 96:128, :] = np.float32(-(2.0**-24))
    x[:, 128:160, :] = np.float32(448.0)
    x[:, 160:192, :] = np.float32(-448.0)
    x[:, 192:224, :] = np.inf
    x[:, 224:256, :] = -np.inf
    x[:, 256:288, :] = np.nan
    x[:, 288:, :] = np.random.default_rng(20260923).standard_normal((2, 224, 8), dtype=np.float32)

    y, scale = utils.public_replay(
        torch.from_numpy(x).npu(), axis=-2, dst_type=code, scale_alg=alg, max_low_bound=bound
    )
    want_y, want_scale = golden.dynamic_mx_quant_golden(x, dst, axis=-2, scale_alg=alg, max_low_bound=bound)
    np.testing.assert_array_equal(y.view(torch.uint8).cpu().numpy(), want_y)
    np.testing.assert_array_equal(scale.view(torch.uint8).cpu().numpy(), want_scale)


@pytest.mark.npu
@pytest.mark.parametrize(
    "alg,dst,code,bound",
    [
        (0, "float8_e4m3fn", 24, 0.0),
        (1, "float8_e4m3fn", 24, 3.25),
        (0, "float8_e5m2", 23, 0.0),
        (1, "float8_e5m2", 23, 3.25),
    ],
)
def test_mixed_wide16_direct_dynamic_shapes_match_cpu_golden(alg, dst, code, bound):
    """Small width-16 inputs share a binary and preserve special-value SR."""
    before = mixed.cache_info()
    for shape in ((1, 256, 16), (3, 768, 16), (8, 2048, 16)):
        x = np.random.default_rng(20260923).standard_normal(shape, dtype=np.float32) * 5
        x[:, 32:64, :] = -0.0
        x[:, 64:96, :] = np.float32(2.0**-24)
        x[:, 96:128, :] = np.nan
        x[:, 128:160, :] = np.inf
        y, scale = utils.public_replay(
            torch.from_numpy(x).npu(), axis=-2, dst_type=code, scale_alg=alg, max_low_bound=bound
        )
        want_y, want_scale = golden.dynamic_mx_quant_golden(x, dst, axis=-2, scale_alg=alg, max_low_bound=bound)
        np.testing.assert_array_equal(y.view(torch.uint8).cpu().numpy(), want_y)
        np.testing.assert_array_equal(scale.view(torch.uint8).cpu().numpy(), want_scale)
    after = mixed.cache_info()
    assert after.misses - before.misses <= 1
    assert after.hits - before.hits >= 2


@pytest.mark.npu
def test_mixed_wide16_large_input_uses_existing_layout_path():
    api = mixed.dynamic_mx_quant_sr
    x = torch.ones((64, 4096, 16), dtype=torch.float32, device="npu")
    assert api.__globals__["_select_layout"](x, 1) == "transpose"
    api(x, axis=-2, scale_alg=0)  # Prime the binary when this test runs alone.
    before = mixed.cache_info()
    y, scale = api(x, axis=-2, scale_alg=0)
    after = mixed.cache_info()
    # The transpose fallback reuses the already compiled tail binary.
    assert after.misses == before.misses and after.hits == before.hits + 1
    np.testing.assert_array_equal(y.view(torch.uint8).cpu().numpy(), np.full(x.shape, 0x78, dtype=np.uint8))
    np.testing.assert_array_equal(
        scale.view(torch.uint8).cpu().numpy(), np.full((64, 4096 // 64, 16, 2), 119, dtype=np.uint8)
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "shape,axis,alg,dst,code,bound",
    [
        ((512, 8), 0, 0, "float8_e4m3fn", 24, 0.0),
        ((8, 2048, 8), -2, 1, "float8_e5m2", 23, 3.25),
        ((1, 2, 512, 8), -2, 0, "float8_e5m2", 23, 0.0),
    ],
)
def test_mixed_bf16_wide8_direct_preserves_bytes(shape, axis, alg, dst, code, bound):
    """Narrow BF16 layouts use one direct kernel and retain exact SR bytes."""
    api = mixed.dynamic_mx_quant_sr
    x = torch.randn(shape, generator=torch.Generator().manual_seed(20260925)) * 5
    x = x.to(torch.bfloat16)
    slices = [slice(None)] * len(shape)
    slices[axis] = slice(32, 64)
    x[tuple(slices)] = 0
    npu_x = x.npu()
    assert api.__globals__["_select_layout"](npu_x, axis % len(shape)) == "wide8_non_tail"
    y, scale = utils.public_replay(npu_x, axis=axis, dst_type=code, scale_alg=alg, max_low_bound=bound)
    from ml_dtypes import bfloat16

    want_y, want_scale = golden.dynamic_mx_quant_golden(
        x.float().numpy().astype(bfloat16), dst, axis=axis, scale_alg=alg, max_low_bound=bound
    )
    np.testing.assert_array_equal(y.view(torch.uint8).cpu().numpy(), want_y)
    np.testing.assert_array_equal(scale.view(torch.uint8).cpu().numpy(), want_scale)


@pytest.mark.npu
def test_mixed_bf16_wide8_direct_reuses_dynamic_binary():
    """Changing BF16 quantization length keeps the direct program reusable."""
    api = mixed.dynamic_mx_quant_sr
    before = mixed.cache_info()
    for k in (512, 768):
        x = torch.ones((2, k, 8), dtype=torch.bfloat16, device="npu")
        assert api.__globals__["_select_layout"](x, 1) == "wide8_non_tail"
        y, scale = api(x, axis=-2, scale_alg=0)
        np.testing.assert_array_equal(y.view(torch.uint8).cpu().numpy(), np.full(x.shape, 0x78, dtype=np.uint8))
        np.testing.assert_array_equal(
            scale.view(torch.uint8).cpu().numpy(), np.full((2, k // 64, 8, 2), 119, dtype=np.uint8)
        )
    after = mixed.cache_info()
    assert after.misses - before.misses <= 1
    assert after.hits - before.hits >= 1


@pytest.mark.npu
def test_real_aot_references_and_diagnostic_bypass(monkeypatch, tmp_path):
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    x_cpu = torch.randn((3, 80), generator=torch.Generator().manual_seed(279))
    x = x_cpu.npu()
    aot_calls, backend_calls = [], []
    original_aot = mixed.cannbotdsl.compile
    original_backend = AscendCCompiler.compile_ascendc_to_so

    def aot(*args, **kwargs):
        aot_calls.append(1)
        return original_aot(*args, **kwargs)

    def backend(self, *args, **kwargs):
        backend_calls.append(1)
        return original_backend(self, *args, **kwargs)

    monkeypatch.setattr(mixed.cannbotdsl, "compile", aot)
    monkeypatch.setattr(AscendCCompiler, "compile_ascendc_to_so", backend)
    arguments = (dtypes.float32, False, get_mem_size("ub"))
    kwargs = dict(device=x.device)
    mixed.clear_caches()
    first = mixed._get_compiled_kernel(*arguments, **kwargs)
    assert first.so_path and Path(first.so_path).is_file() and not first.closed
    # Keep artifact inspection independent of the SDK cache file lifetime.
    first_saved = Path(first.save(str(tmp_path / "first.so")))
    first_hash = hashlib.sha256(first_saved.read_bytes()).hexdigest()
    assert mixed._get_compiled_kernel(*arguments, **kwargs) is first
    assert len(aot_calls) == 1
    first_backend_count = len(backend_calls)
    mixed.clear_caches()
    second = mixed._get_compiled_kernel(*arguments, **kwargs)
    assert second is not first and not first.closed
    assert len(aot_calls) == 2 and len(backend_calls) == first_backend_count
    assert first_saved.is_file() and hashlib.sha256(first_saved.read_bytes()).hexdigest() == first_hash
    prepared = mixed._prepare_launch(x, torch.float8_e4m3fn, 1, 1, 0, block_dim=1)
    # Execute the external old-generation reference, not only the replacement.
    first(*prepared.arguments, utils.replay_state(x))
    utils.assert_result(x_cpu, prepared.outputs(), -1, torch.float32, 24, 1, 0.0)

    # Exercise real references through local LRU eviction as well as clear.
    cache = mixed._ProgramCache(1)
    key = mixed.capture_compile_context("fp32", False, arguments[2], 1, 0, mixed.Layout.TAIL, x.device).key
    assert cache.get_or_build(key, lambda: first) is first
    assert cache.get_or_build("another-specialization", lambda: second) is second
    assert not first.closed
    cache.clear()
    first(*prepared.arguments, utils.replay_state(x))
    utils.assert_result(x_cpu, prepared.outputs(), -1, torch.float32, 24, 1, 0.0)

    # Cold preparation may compile scalar and Graph RNG entries. Only
    # the following diagnostic-only calls must avoid backend compilation.
    count_before_diagnostic = len(aot_calls)
    backend_before_diagnostic = len(backend_calls)
    with monkeypatch.context() as config:
        config.setenv("CANNBOTDSL_PIPE_STAGE", "translate")
        config.setenv("CANNBOTDSL_DUMP_ASCENDC", "1")
        config.setenv("CANNBOTDSL_DUMP_DIR", str(tmp_path))
        for _ in range(2):
            stub = mixed._get_compiled_kernel(*arguments, **kwargs)
            assert not stub.so_path
        assert list(tmp_path.glob("*.asc"))
    assert len(aot_calls) == count_before_diagnostic + 2
    assert len(backend_calls) == backend_before_diagnostic
    assert mixed._get_compiled_kernel(*arguments, **kwargs) is second
    assert hashlib.sha256(first_saved.read_bytes()).hexdigest() == first_hash
    assert not first.closed


@pytest.mark.npu
@pytest.mark.whitebox
@pytest.mark.parametrize("algorithm", [0, 1])
@pytest.mark.parametrize("src,dst,case", WHITEBOX_CASES)
def test_whitebox_matrix_and_scale_generalization(src, dst, case, algorithm, record_property):
    tile = utils.tile_elements(src)
    shape = case.resolve_shape(tile)
    cpu = utils.numerical_input(case.pattern, shape, src, dst)
    actual = run_check(cpu, dst, case.blocks, case.offset, scale_alg=algorithm)
    if case.pattern == "special_isolation" and algorithm == 1:
        np.testing.assert_array_equal(
            actual[1].reshape(6, 4),
            np.array(
                [
                    [0, 0, 0, 0],
                    [63, 127, 190, 0],
                    [255, 127, 135, 0],
                    [119, 255, 135, 0],
                    [119, 127, 255, 0],
                    [127, 127, 127, 0],
                ],
                dtype=np.uint8,
            ),
        )
        np.testing.assert_array_equal(actual[0][0], np.tile(np.array([0, 0x80], np.uint8), 48))
        for row, group in ((2, 0), (3, 1), (4, 2)):
            np.testing.assert_array_equal(actual[0][row, group * 32 : (group + 1) * 32], np.full(32, 0x7F, np.uint8))
    if case.name == "generated-0":
        misses = mixed.cache_info().misses
        for blocks in (1, 7, 64):
            repeated = run_check(cpu.reshape(1, -1), dst, blocks, scale_alg=algorithm)
            for output, reference in zip(repeated, actual):
                np.testing.assert_array_equal(output.reshape(-1), reference.reshape(-1))
        assert mixed.cache_info().misses == misses
    record_property(
        "whitebox_case",
        json.dumps(
            dict(
                category=case.category,
                shape=shape,
                blocks=case.blocks,
                pattern=case.pattern,
                storage_offset=case.offset,
                algorithm=algorithm,
                max_tile=tile,
            )
        ),
    )


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS)
def test_dirty_large_short_zero_replay(src, dst):
    tile = utils.tile_elements(src)
    dirty = random_input((3, tile + 16), src)
    dirty[0, 1023] = float("nan")
    dirty[1, 1024] = float("inf")
    dirty[2, -1] = -float("inf")
    run_check(dirty, dst, 1)
    run_check(random_input((3, 80), src), dst, 7)
    minimum = 16 if src == torch.bfloat16 else 32
    zero = torch.zeros((3, minimum), dtype=src)
    zero[:, 1::2] = -0.0
    run_check(zero, dst, 1)
    run_check(random_input((3, 1040), src), dst, 1)


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS)
def test_rng_invariant_for_tail_rows_and_repeated_calls(src, dst):
    for shape, budgets in (((3, 272), (32, 32, 1)), ((65, 80), (1, 2, 7, 31, 32, 33, 63, 64))):
        cpu = random_input(shape, src)
        expected = run_check(cpu, dst, budgets[0])
        for blocks in budgets[1:]:
            run_check(cpu, dst, blocks, expected=expected)


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS)
def test_channel_slots_wrap_with_changed_input_and_replay(src, dst):
    tile = utils.tile_elements(src)
    shapes = [(3 * (tile // 128) + 1, 80), (1, 3 * tile + 64), (5, 1040)]
    if src == torch.bfloat16:
        shapes.append((3 * (tile // 64) + 1, 16))
    for shape in shapes:
        first = random_input(shape, src)
        changed = (-first.float() / 32 + 1 / 128).to(src)
        expected = run_check(first, dst, 1)
        run_check(changed, dst, 1)
        run_check(first, dst, 1, expected=expected)


@pytest.mark.npu
@pytest.mark.whitebox
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS)
@pytest.mark.parametrize("algorithm", [0, 1])
@pytest.mark.parametrize("blocks", [1, 7, 64])
def test_core_partition_full_tiles_and_uneven_tail(src, dst, algorithm, blocks):
    tile = utils.tile_elements(src)
    count = blocks * (tile + 256) - 64
    shape = (count // 32, 32) if blocks == 1 else (1, count)
    cpu = random_input(shape, src)
    special = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1e-38], dtype=src)
    for start in (0, tile - 3, tile + 253, count - 6):
        if start + 6 <= count:
            cpu.reshape(-1)[start : start + 6] = special
    assert count > blocks * tile and count % (blocks * tile) != 0 and count % 64 == 0
    run_check(cpu, dst, blocks, 7 if blocks == 7 else 0, scale_alg=algorithm)


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS)
@pytest.mark.parametrize(
    "rows,blocks",
    [
        (1, 1),
        (2, 1),
        (3, 1),
        (31, 1),
        (32, 1),
        (33, 1),
        (63, 64),
        (64, 64),
        (65, 64),
        (127, 7),
        (128, 7),
        (129, 7),
        (2047, 64),
        (2048, 64),
        (2049, 64),
        (4097, 64),
    ],
)
def test_k32_logical_rows_padding_and_output_guards(rows, blocks, src, dst):
    cpu = random_input((rows, 32), torch.float32)
    for row in range(rows):
        cpu[row] *= 2.0 ** ((row * 17) % 241 - 120)
    cpu[0] = 0.0
    cpu[0, 1::2] = -0.0
    if rows > 1:
        cpu[1, 0] = float("nan")
        cpu[-1, -1] = -float("inf")
    cpu = cpu.to(src)
    run_check(cpu, dst, blocks)
    utils.guarded_launch(cpu, dst, blocks, 32)


@pytest.mark.npu
@pytest.mark.parametrize(
    "src,dst,rows,columns,offset",
    [
        (torch.float32, torch.float8_e4m3fn, 17, 48, 1),
        (torch.bfloat16, torch.float8_e5m2, 9, 80, 16),
    ],
)
def test_short_rows_unaligned_input_output_and_scale_guards(src, dst, rows, columns, offset):
    utils.guarded_launch(utils.challenging_input(rows, columns, src), dst, 1, offset)


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS)
@pytest.mark.parametrize("columns", [32, 64, 80])
def test_storage_offset_on_device_preserves_input_and_bytes(src, dst, columns):
    run_check(random_input((3, columns), src), dst, 1, 16)


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS)
def test_wide_exponents_raw_special_values_and_large_32784_tail(src, dst):
    run_check(random_input((2, 32784), src), dst, 32)
    generator = torch.Generator().manual_seed(532)
    values = torch.ldexp(
        0.5 + torch.rand((4, 4096), generator=generator) * 0.5, torch.randint(-140, 120, (4, 4096), generator=generator)
    )
    values *= torch.randint(0, 2, values.shape, generator=generator) * 2 - 1
    values[0, :8] = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1e-40, -1e-40, 1.0])
    run_check(values.to(src), dst, 32)


@pytest.mark.npu
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS)
@pytest.mark.parametrize("rows,columns,blocks", [(7, 96, 1), (65, 112, 64)])
def test_short_row_scheduler_multiple_aivs_and_row_tail(src, dst, rows, columns, blocks):
    run_check(utils.challenging_input(rows, columns, src), dst, blocks)


@pytest.mark.npu
@pytest.mark.whitebox
@pytest.mark.parametrize("shape", [(64, 4096), (128, 4096), (512, 1024), (512, 1536)])
@pytest.mark.parametrize("dst", [24, 23])
@pytest.mark.parametrize("algorithm", [0, 1])
def test_bf16_align4k_medium_partitions(shape, dst, algorithm):
    cpu = (torch.randn(shape, generator=torch.Generator().manual_seed(749)) * 5).to(torch.bfloat16)
    cpu.reshape(-1)[:9] = torch.tensor(
        [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 2**-30, -(2**-30), 1.0, -1.0]
    ).to(torch.bfloat16)
    dtype = torch.float8_e4m3fn if dst == 24 else torch.float8_e5m2
    utils.run_check(cpu, dtype, blocks=56, scale_alg=algorithm, rng_seed=2**64 - 1, rng_block_offset=4 * (2**32 - 1))


@pytest.mark.npu
@pytest.mark.whitebox
@pytest.mark.parametrize("dst,mantissa,bias,shift", [(24, 3, 7, 8), (23, 2, 15, 15)])
@pytest.mark.parametrize("algorithm", [0, 1])
@pytest.mark.parametrize("family", ["lower_edge", "small_range", "mixed_normal", "below_lower"])
def test_bf16_middle_range_signed_two_tiles(dst, mantissa, bias, shift, algorithm, family):
    lower = (123 - bias - mantissa) * 128
    normal = (128 - bias) * 128
    patterns = {
        "lower_edge": [lower, lower + 1, normal - 1, normal],
        "small_range": np.arange(lower, normal, dtype=np.uint16),
        "mixed_normal": [lower, normal - 1, normal, normal + 127, (127 + shift) * 128 - 1],
        "below_lower": [lower - 1, lower, normal - 1],
    }
    # Anchor each block at exactly1. Both scale algorithms then choose the
    # same power of two, so these raw values reach the intended SR boundaries.
    count = 2 * utils.tile_elements(torch.bfloat16)
    normalized = np.resize(np.asarray(patterns[family], dtype=np.uint16), count)
    bits = normalized - np.uint16(shift * 128)
    bits |= (np.arange(count, dtype=np.uint16) & 1) * np.uint16(32768)
    bits[31::32] = 0x3F80
    cpu = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16).reshape(1, -1)
    dtype = torch.float8_e4m3fn if dst == 24 else torch.float8_e5m2
    _, scales = utils.run_check(
        cpu,
        dtype,
        blocks=1,
        scale_alg=algorithm,
        rng_seed=2**64 - 1,
        rng_block_offset=4 * (2**32 - 1),
    )
    np.testing.assert_array_equal(scales, np.full_like(scales, 127 - shift))


@pytest.mark.npu
@pytest.mark.whitebox
@pytest.mark.parametrize("dst,mantissa,bias,shift", [(24, 3, 7, 8), (23, 2, 15, 15)])
@pytest.mark.parametrize("algorithm", [0, 1])
@pytest.mark.parametrize("family", ["lower_edge", "small_range", "mixed_normal", "below_lower"])
def test_bf16_scale16_k32_signed_two_tiles(dst, mantissa, bias, shift, algorithm, family):
    lower = (123 - bias - mantissa) * 128
    normal = (128 - bias) * 128
    patterns = {
        "lower_edge": [lower, lower + 1, normal - 1, normal],
        "small_range": np.arange(lower, normal, dtype=np.uint16),
        "mixed_normal": [lower, normal - 1, normal, normal + 127, (127 + shift) * 128 - 1],
        "below_lower": [lower - 1, lower, normal - 1],
    }
    # Anchor each block at exactly1. Both scale algorithms then choose the
    # same power of two, so these raw values reach the intended SR boundaries.
    count = 2 * utils.tile_elements(torch.bfloat16)
    normalized = np.resize(np.asarray(patterns[family], dtype=np.uint16), count)
    bits = normalized - np.uint16(shift * 128)
    bits |= (np.arange(count, dtype=np.uint16) & 1) * np.uint16(32768)
    bits[31::32] = 0x3F80
    cpu = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16).reshape(-1, 32)
    dtype = torch.float8_e4m3fn if dst == 24 else torch.float8_e5m2
    _, scales = utils.run_check(
        cpu,
        dtype,
        blocks=1,
        scale_alg=algorithm,
        rng_seed=2**64 - 1,
        rng_block_offset=4 * (2**32 - 1),
    )
    expected_scales = np.full_like(scales, 127 - shift)
    expected_scales[..., 1::2] = 0
    np.testing.assert_array_equal(scales, expected_scales)


@pytest.mark.npu
@pytest.mark.parametrize("rows,alg", [(32767, 0), (32768, 0), (32768, 1)])
def test_half_q7_boundary_state_restore(rows, alg):
    import torch_npu  # noqa: F401

    op = utils.sample()
    cpu = _cpu(rows)
    x = cpu.npu()
    gen = torch.npu.default_generators[0]
    saved = torch.npu.get_rng_state().clone()
    seed, offset = 0xFEDCBA9876543210, 4 * ((1 << 32) - 1)
    inc = op._rng_increment(x.numel())
    try:
        gen.manual_seed(seed)
        gen.set_offset(offset)
        checkpoint = torch.npu.get_rng_state().clone()
        first = op.dynamic_mx_quant_sr(x, dst_type=24, scale_alg=alg)
        expected = utils.assert_result(cpu, first, 1, cpu.dtype, 24, alg, 0.0, rng_seed=seed, rng_block_offset=offset)
        assert gen.get_offset() == offset + inc
        second = op.dynamic_mx_quant_sr(x, dst_type=24, scale_alg=alg)
        utils.assert_result(cpu, second, 1, cpu.dtype, 24, alg, 0.0, rng_seed=seed, rng_block_offset=offset + inc)
        assert gen.get_offset() == offset + 2 * inc
        torch.npu.set_rng_state(checkpoint)
        restored = op.dynamic_mx_quant_sr(x, dst_type=24, scale_alg=alg)
        for a, b in zip(restored, expected):
            np.testing.assert_array_equal(a.view(torch.uint8).cpu().numpy(), b)
        assert gen.get_offset() == offset + inc
        p = op._prepare_launch(x, torch.float8_e4m3fn, 1, alg, 0)
        assert op._select_bf16_four_panel_policy(
            x.numel(),
            op.dtypes.bfloat16,
            torch.float8_e4m3fn,
            alg,
            0,
            p.plan.layout,
            p.plan.quant_length,
            p.plan.blocks,
        )
        carrier = utils.replay_state(x, seed, offset)
        before = carrier.clone()
        p.launch(rng_state=carrier)
        for a, b in zip(p.outputs(), expected):
            np.testing.assert_array_equal(a.view(torch.uint8).cpu().numpy(), b)
        assert torch.equal(carrier, before) and gen.get_offset() == offset + inc
        np.testing.assert_array_equal(x.view(torch.uint8).cpu().numpy(), cpu.view(torch.uint8).numpy())
    finally:
        torch.npu.synchronize()
        torch.npu.set_rng_state(saved)


@pytest.mark.npu
@pytest.mark.parametrize("alg,ub,rows", [(0, 98304, 7), (1, 65536, 33)])
def test_forced_low_ub_partial(alg, ub, rows):
    import torch_npu  # noqa: F401

    op = utils.sample()
    cpu = _cpu(64)[:rows].clone()
    x = cpu.npu()
    p = op._prepare_launch(x, torch.float8_e4m3fn, 1, alg, 0, block_dim=7)
    state = op._get_compiled_kernel(op.dtypes.bfloat16, False, ub, alg, 0, device=x.device)
    scalar = op._get_compiled_tail_scalar_kernel(op.dtypes.bfloat16, False, ub, alg, 0, x.device)
    p = dataclasses.replace(p, program=state, scalar_program=scalar)
    seed, offset = 2**64 - 1, 2**63 + 16
    p.launch(rng_state=utils.replay_state(x, seed, offset))
    utils.assert_result(cpu, p.outputs(), 1, cpu.dtype, 24, alg, 0.0, rng_seed=seed, rng_block_offset=offset)
    scalar(*p.arguments, op._tail_rng_signed64(seed), op._tail_rng_signed64(offset))
    utils.assert_result(cpu, p.outputs(), 1, cpu.dtype, 24, alg, 0.0, rng_seed=seed, rng_block_offset=offset)


@pytest.mark.npu
@pytest.mark.parametrize("dst,bound", [(23, 0.0), (24, 4.0)])
def test_half_q7_large_format_bound_fallback(dst, bound):
    import torch_npu  # noqa: F401

    cpu = _cpu(32768)
    x = cpu.npu()
    seed, offset = 2**64 - 1, 2**63 + 16
    result = utils.public_replay(
        x, rng_seed=seed, rng_block_offset=offset, dst_type=dst, scale_alg=1, max_low_bound=bound
    )
    utils.assert_result(cpu, result, 1, cpu.dtype, dst, 1, bound, rng_seed=seed, rng_block_offset=offset)


@pytest.mark.npu
def test_half_q7_graph_optimized_algorithms_restore():
    import torch_npu  # noqa: F401

    op = utils.sample()
    cpu = _cpu(32768)
    x = cpu.npu()
    for alg in (0, 1):
        op.dynamic_mx_quant_sr(x, dst_type=24, scale_alg=alg)
    torch.npu.synchronize()
    gen = torch.npu.default_generators[0]
    saved = torch.npu.get_rng_state().clone()
    graph = torch.npu.NPUGraph()
    seed, offset = 0xFEDCBA9876543210, 4 * ((1 << 32) - 1)
    inc = op._rng_increment(x.numel())
    try:
        gen.manual_seed(seed)
        gen.set_offset(offset)
        with torch.npu.graph(graph):
            outputs = [op.dynamic_mx_quant_sr(x, dst_type=24, scale_alg=alg) for alg in (0, 1)]
        checkpoint = torch.npu.get_rng_state().clone()
        first = None
        for replay in range(3):
            if replay == 2:
                torch.npu.set_rng_state(checkpoint)
            cursor = gen.get_offset()
            graph.replay()
            torch.npu.synchronize()
            current = []
            for alg, result in enumerate(outputs):
                current.append(
                    utils.assert_result(cpu, result, 1, cpu.dtype, 24, alg, 0.0, rng_seed=seed, rng_block_offset=cursor)
                )
                cursor += inc
            assert gen.get_offset() == cursor
            if replay == 0:
                first = current
            if replay == 2:
                for actual, expected in zip(current, first):
                    for a, b in zip(actual, expected):
                        np.testing.assert_array_equal(a, b)
    finally:
        torch.npu.synchronize()
        graph.reset()
        torch.npu.set_rng_state(saved)


@pytest.mark.npu
@pytest.mark.parametrize(
    "src,rows,columns,dst,alg",
    [
        (torch.float32, 8192, 256, 24, 0),
        (torch.float32, 8192, 256, 23, 1),
        (torch.float32, 8191, 256, 24, 0),
        (torch.float32, 8192, 272, 23, 1),
        (torch.bfloat16, 8192, 256, 24, 0),
    ],
)
def test_fp32_mapping_value_and_state(src, rows, columns, dst, alg):
    import torch_npu  # noqa: F401

    op = utils.sample()
    cpu = torch.randn((rows, columns), generator=torch.Generator().manual_seed(52), dtype=torch.float32).to(src)
    x = cpu.npu()
    seed, offset = 0xFEDCBA9876543210, 2**32 - 4
    gen = torch.npu.default_generators[0]
    saved = torch.npu.get_rng_state().clone()
    try:
        gen.manual_seed(seed)
        gen.set_offset(offset)
        before = torch.npu.get_rng_state().clone()
        first = op.dynamic_mx_quant_sr(x, dst_type=dst, scale_alg=alg)
        expected = utils.assert_result(cpu, first, 1, src, dst, alg, 0.0, rng_seed=seed, rng_block_offset=offset)
        inc = op._rng_increment(x.numel())
        assert gen.get_offset() == offset + inc
        second = op.dynamic_mx_quant_sr(x, dst_type=dst, scale_alg=alg)
        utils.assert_result(cpu, second, 1, src, dst, alg, 0.0, rng_seed=seed, rng_block_offset=offset + inc)
        assert gen.get_offset() == offset + 2 * inc
        torch.npu.set_rng_state(before)
        restored = op.dynamic_mx_quant_sr(x, dst_type=dst, scale_alg=alg)
        for actual, want in zip(restored, expected):
            np.testing.assert_array_equal(actual.view(torch.uint8).cpu().numpy(), want)
        assert gen.get_offset() == offset + inc
        explicit = utils.public_replay(x, rng_seed=seed, rng_block_offset=offset, dst_type=dst, scale_alg=alg)
        utils.assert_result(cpu, explicit, 1, src, dst, alg, 0.0, rng_seed=seed, rng_block_offset=offset)
        assert gen.get_offset() == offset + inc
    finally:
        torch.npu.synchronize()
        torch.npu.set_rng_state(saved)


@pytest.mark.npu
def test_fp32_mapping_graph_optimized_and_fallback_restore():
    import torch_npu  # noqa: F401

    op = utils.sample()
    cases = [(8192, 256, 24, 0), (8191, 256, 23, 1)]
    prepared = []
    for rows, columns, dst, alg in cases:
        cpu = torch.randn((rows, columns), generator=torch.Generator().manual_seed(rows), dtype=torch.float32)
        x = cpu.npu()
        op.dynamic_mx_quant_sr(x, dst_type=dst, scale_alg=alg)
        prepared.append((cpu, x, dst, alg, op._rng_increment(x.numel())))
    torch.npu.synchronize()
    gen = torch.npu.default_generators[0]
    saved = torch.npu.get_rng_state().clone()
    graph = torch.npu.NPUGraph()
    seed, offset = 0xFEDCBA9876543210, 2**32 - 4
    try:
        gen.manual_seed(seed)
        gen.set_offset(offset)
        with torch.npu.graph(graph):
            outputs = [op.dynamic_mx_quant_sr(x, dst_type=dst, scale_alg=alg) for _, x, dst, alg, _ in prepared]
        checkpoint = torch.npu.get_rng_state().clone()
        first = None
        for i in range(3):
            if i == 2:
                torch.npu.set_rng_state(checkpoint)
            cursor = gen.get_offset()
            graph.replay()
            torch.npu.synchronize()
            current = []
            for (cpu, _, dst, alg, inc), result in zip(prepared, outputs):
                current.append(
                    utils.assert_result(
                        cpu, result, 1, cpu.dtype, dst, alg, 0.0, rng_seed=seed, rng_block_offset=cursor
                    )
                )
                cursor += inc
            assert gen.get_offset() == cursor
            if i == 0:
                first = current
            if i == 2:
                for actual, previous in zip(current, first):
                    for a, b in zip(actual, previous):
                        np.testing.assert_array_equal(a, b)
    finally:
        torch.npu.synchronize()
        graph.reset()
        torch.npu.set_rng_state(saved)


@pytest.mark.npu
def test_formal_public_optimized_and_fallback():
    import torch_npu  # noqa: F401

    op = utils.sample()
    saved = torch.npu.get_rng_state().clone()
    seed, offset = 0xFEDCBA9876543210, 2**32 - 4
    try:
        for src, shape, dst, alg, alias in (
            (torch.float32, (8192, 256), 24, 0, True),
            (torch.float32, (8192, 256), 23, 1, True),
            (torch.float32, (8191, 256), 24, 0, False),
            (torch.float32, (8192, 272), 23, 1, False),
            (torch.bfloat16, (8192, 256), 24, 0, False),
        ):
            cpu = torch.randn(shape, generator=torch.Generator().manual_seed(60), dtype=src)
            x = cpu.npu()
            p = op._prepare_launch(x, torch.float8_e4m3fn if dst == 24 else torch.float8_e5m2, 1, alg, 0)
            assert p.program is not None
            assert (
                op._select_fp32_eight_policy(
                    x.numel(),
                    op.dtypes.float32 if src == torch.float32 else op.dtypes.bfloat16,
                    p.plan.compile_layout,
                    p.plan.quant_length,
                )
                is alias
            )
            result = utils.public_replay(x, rng_seed=seed, rng_block_offset=offset, dst_type=dst, scale_alg=alg)
            utils.assert_result(cpu, result, 1, src, dst, alg, 0.0, rng_seed=seed, rng_block_offset=offset)
    finally:
        torch.npu.synchronize()
        torch.npu.set_rng_state(saved)


@pytest.mark.npu
@pytest.mark.parametrize(
    "shape,dst,alg",
    [
        ((2048, 16384), 24, 0),
        ((8192, 8192), 24, 1),
        ((8192, 7168), 24, 0),
        ((2048, 16384), 23, 0),
    ],
)
def test_bf16_compact_formal_bf16_value_state(shape, dst, alg):
    import torch_npu  # noqa: F401

    op = utils.sample()
    cpu = torch.randn(shape, generator=torch.Generator().manual_seed(76), dtype=torch.bfloat16)
    x = cpu.npu()
    seed, offset = 0xFEDCBA9876543210, 2**32 - 4
    saved = torch.npu.get_rng_state().clone()
    try:
        explicit = utils.public_replay(x, rng_seed=seed, rng_block_offset=offset, dst_type=dst, scale_alg=alg)
        utils.assert_result(cpu, explicit, 1, cpu.dtype, dst, alg, 0.0, rng_seed=seed, rng_block_offset=offset)
        gen = torch.npu.default_generators[0]
        gen.manual_seed(seed)
        gen.set_offset(offset)
        checkpoint = torch.npu.get_rng_state().clone()
        first = op.dynamic_mx_quant_sr(x, dst_type=dst, scale_alg=alg)
        utils.assert_result(cpu, first, 1, cpu.dtype, dst, alg, 0.0, rng_seed=seed, rng_block_offset=offset)
        assert gen.get_offset() == offset + op._rng_increment(x.numel())
        torch.npu.set_rng_state(checkpoint)
        restored = op.dynamic_mx_quant_sr(x, dst_type=dst, scale_alg=alg)
        for actual, want in zip(restored, first):
            np.testing.assert_array_equal(actual.view(torch.uint8).cpu().numpy(), want.view(torch.uint8).cpu().numpy())
    finally:
        torch.npu.synchronize()
        torch.npu.set_rng_state(saved)


@pytest.mark.npu
def test_bf16_compact_formal_bf16_graph_restore():
    import torch_npu  # noqa: F401

    op = utils.sample()
    cpu = torch.randn((2048, 16384), generator=torch.Generator().manual_seed(176), dtype=torch.bfloat16)
    x = cpu.npu()
    op.dynamic_mx_quant_sr(x, dst_type=24, scale_alg=0)
    torch.npu.synchronize()
    gen = torch.npu.default_generators[0]
    saved = torch.npu.get_rng_state().clone()
    seed, offset = 17, 2**32 - 4
    try:
        gen.manual_seed(seed)
        gen.set_offset(offset)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = op.dynamic_mx_quant_sr(x, dst_type=24, scale_alg=0)
        checkpoint = torch.npu.get_rng_state().clone()
        seen = None
        for step in range(3):
            if step == 2:
                torch.npu.set_rng_state(checkpoint)
            cursor = gen.get_offset()
            graph.replay()
            torch.npu.synchronize()
            utils.assert_result(cpu, output, 1, cpu.dtype, 24, 0, 0.0, rng_seed=seed, rng_block_offset=cursor)
            now = tuple(t.view(torch.uint8).clone() for t in output)
            if step == 0:
                seen = now
            if step == 2:
                assert all(torch.equal(a, b) for a, b in zip(seen, now))
            assert gen.get_offset() == cursor + op._rng_increment(x.numel())
    finally:
        torch.npu.synchronize()
        torch.npu.set_rng_state(saved)


@pytest.mark.npu
@pytest.mark.parametrize(
    "src,shape,axis",
    [
        (torch.bfloat16, (3, 256), 1),
        (torch.bfloat16, (2, 80), 1),
        (torch.float32, (511, 4096), 1),
        (torch.float32, (512, 4096), 1),
        (torch.float32, (683, 1024, 3), 1),
    ],
)
@pytest.mark.parametrize("dst,alg", [(24, 0), (23, 1)])
def test_rng_seed_offset_restore(src, shape, axis, dst, alg):
    import torch_npu  # noqa: F401

    op = utils.sample()
    cpu = torch.randn(shape, generator=torch.Generator().manual_seed(7493)).to(src)
    cpu.reshape(-1)[:5] = torch.tensor([0.0, -0.0, float("nan"), float("inf"), -float("inf")]).to(src)
    x = cpu.npu()
    gen = torch.npu.default_generators[0]
    saved = torch.npu.get_rng_state().clone()
    seed, offset = 0xFEDCBA9876543210, 4 * ((1 << 32) - 1)
    increment = op._rng_increment(x.numel())
    try:
        gen.manual_seed(seed)
        gen.set_offset(offset)
        checkpoint = torch.npu.get_rng_state().clone()
        outputs = []
        for step in range(2):
            result = op.dynamic_mx_quant_sr(x, axis=axis, dst_type=dst, scale_alg=alg)
            outputs.append(
                utils.assert_result(
                    cpu, result, axis, src, dst, alg, 0.0, rng_seed=seed, rng_block_offset=offset + step * increment
                )
            )
            assert gen.get_offset() == offset + (step + 1) * increment
        torch.npu.set_rng_state(checkpoint)
        repeated = op.dynamic_mx_quant_sr(x, axis=axis, dst_type=dst, scale_alg=alg)
        for actual, expected in zip(repeated, outputs[0]):
            np.testing.assert_array_equal(actual.view(torch.uint8).cpu().numpy(), expected)
        assert gen.get_offset() == offset + increment
        current = gen.get_offset()
        prepared = op._prepare_launch(x, torch.float8_e4m3fn if dst == 24 else torch.float8_e5m2, axis, alg, 0)
        prepared.launch(rng_state=utils.replay_state(x, seed, offset))
        for actual, expected in zip(prepared.outputs(), outputs[0]):
            np.testing.assert_array_equal(actual.view(torch.uint8).cpu().numpy(), expected)
        assert gen.get_offset() == current
        np.testing.assert_array_equal(x.view(torch.uint8).cpu().numpy(), cpu.view(torch.uint8).numpy())
    finally:
        torch.npu.synchronize()
        torch.npu.set_rng_state(saved)


@pytest.mark.npu
def test_rng_mixed_policy_graph_replay_restore():
    import torch_npu  # noqa: F401

    op = utils.sample()
    configs = [
        (torch.bfloat16, (3, 256), 24, 0),
        (torch.float32, (511, 4096), 23, 0),
        (torch.float32, (512, 4096), 24, 1),
    ]
    fixtures = []
    for src, shape, dst, alg in configs:
        cpu = torch.randn(shape, generator=torch.Generator().manual_seed(17493)).to(src)
        x = cpu.npu()
        for _ in range(2):
            op.dynamic_mx_quant_sr(x, dst_type=dst, scale_alg=alg)
        fixtures.append((cpu, x, dst, alg))
    torch.npu.synchronize()
    gen = torch.npu.default_generators[0]
    saved = torch.npu.get_rng_state().clone()
    seed, offset = 0xFEDCBA9876543210, 4 * ((1 << 32) - 1)
    graph = torch.npu.NPUGraph()
    outputs = []
    try:
        gen.manual_seed(seed)
        gen.set_offset(offset)
        with torch.npu.graph(graph):
            for cpu, x, dst, alg in fixtures:
                outputs.append(op.dynamic_mx_quant_sr(x, dst_type=dst, scale_alg=alg))
        checkpoint = torch.npu.get_rng_state().clone()
        first = None
        for replay in range(3):
            if replay == 2:
                torch.npu.set_rng_state(checkpoint)
            cursor = gen.get_offset()
            graph.replay()
            torch.npu.synchronize()
            current = []
            for (cpu, x, dst, alg), result in zip(fixtures, outputs):
                current.append(
                    utils.assert_result(
                        cpu, result, -1, cpu.dtype, dst, alg, 0.0, rng_seed=seed, rng_block_offset=cursor
                    )
                )
                cursor += op._rng_increment(x.numel())
            assert gen.get_offset() == cursor
            if replay == 0:
                first = current
            if replay == 2:
                for actual, expected in zip(current, first):
                    for a, b in zip(actual, expected):
                        np.testing.assert_array_equal(a, b)
    finally:
        torch.npu.synchronize()
        graph.reset()
        torch.npu.set_rng_state(saved)


@pytest.mark.npu
@pytest.mark.parametrize(
    "shape,axis,src,dst,alg",
    [
        ((1, 512), -1, torch.float32, 24, 0),
        ((2, 48, 3), 1, torch.float32, 23, 1),
        ((17, 4096, 16), 1, torch.bfloat16, 24, 0),
        ((1, 768, 2), 1, torch.bfloat16, 23, 1),
        ((1, 576, 2), 1, torch.float32, 24, 0),
        ((2, 256, 8), 1, torch.bfloat16, 23, 0),
        ((1, 256, 16), 1, torch.float32, 23, 1),
        ((1, 256, 32), 1, torch.float32, 24, 1),
        ((1, 256, 64), 1, torch.bfloat16, 24, 0),
    ],
)
def test_public_non_tail_avoids_state_materialization(monkeypatch, rng_device, shape, axis, src, dst, alg):
    op = mixed
    cpu = utils.axis_input(shape, axis % len(shape), src)
    x = cpu.to(rng_device)
    op.dynamic_mx_quant_sr(x, axis=axis, dst_type=dst, scale_alg=alg)
    torch.npu.synchronize(rng_device)
    for name in ("prepare_unpack", "unpack_rng_state", "_reserve_rng_state"):
        monkeypatch.setattr(op, name, reject_materialization, raising=False)
    generator = torch.npu.default_generators[rng_device.index]
    seed, offset = 0xFEDCBA9876543210, 4 * (2**32 - 1)
    generator.manual_seed(seed)
    generator.set_offset(offset)
    result = op.dynamic_mx_quant_sr(x, axis=axis, dst_type=dst, scale_alg=alg)
    utils.assert_result(cpu, result, axis, src, dst, alg, 0, rng_seed=seed, rng_block_offset=offset)
    assert generator.get_offset() == offset + op._rng_increment(x.numel())


@pytest.mark.npu
@pytest.mark.parametrize(
    "shape,axis,src,dst,alg",
    [
        ((2, 48, 3), 1, torch.float32, 24, 1),
        ((512, 7168), 1, torch.float32, 23, 0),
        ((512, 7168), 1, torch.bfloat16, 24, 0),
    ],
)
def test_public_graph_uses_fused_rng_state(monkeypatch, rng_device, shape, axis, src, dst, alg):
    op = mixed
    cpu = utils.axis_input(shape, axis, src)
    x = cpu.to(rng_device)
    op.dynamic_mx_quant_sr(x, axis=axis, dst_type=dst, scale_alg=alg)
    torch.npu.synchronize(rng_device)
    for name in ("prepare_unpack", "unpack_rng_state", "_reserve_rng_state"):
        monkeypatch.setattr(op, name, reject_materialization, raising=False)
    seed, start = 0xFEDCBA9876543210, 4 * (2**32 - 1)
    generator = torch.npu.default_generators[rng_device.index]
    generator.manual_seed(seed)
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            first = op.dynamic_mx_quant_sr(x, axis=axis, dst_type=dst, scale_alg=alg)
            second = op.dynamic_mx_quant_sr(x, axis=axis, dst_type=dst, scale_alg=alg)
        increment = op._rng_increment(x.numel())
        generator.set_offset(start)
        saved = torch.npu.get_rng_state(rng_device).clone()
        sequences = []
        for _ in range(2):
            torch.npu.set_rng_state(saved, rng_device)
            sequence = []
            for iteration in range(2):
                graph.replay()
                offset = start + iteration * 2 * increment
                pair = []
                for result, expected_offset in ((first, offset), (second, offset + increment)):
                    actual = utils.assert_result(
                        cpu, result, axis, src, dst, alg, 0, rng_seed=seed, rng_block_offset=expected_offset
                    )
                    pair.append(tuple(a.copy() for a in actual))
                assert generator.get_offset() == offset + 2 * increment
                sequence.append(pair)
            sequences.append(sequence)
        import numpy as np

        for original, restored in zip(*sequences):
            for a, b in zip(original, restored):
                for av, bv in zip(a, b):
                    np.testing.assert_array_equal(av, bv)
    finally:
        torch.npu.synchronize(rng_device)
        graph.reset()


@pytest.mark.npu
@pytest.mark.parametrize("offset", [2**32 - 4, 2**63])
def test_explicit_state_does_not_consume_generator(rng_device, offset):
    op = mixed
    cpu = utils.axis_input((1, 48), 1, torch.bfloat16)
    x = cpu.to(rng_device)
    prepared = op._prepare_launch(x, torch.float8_e5m2, 1, 1, 0)
    seed = 2**64 - 1
    state = op._framework_rng._make_rng_state(seed, offset, device=rng_device)
    before = torch.npu.get_rng_state(rng_device).clone()
    prepared.launch(rng_state=state)
    utils.assert_result(cpu, prepared.outputs(), -1, torch.bfloat16, 23, 1, 0, rng_seed=seed, rng_block_offset=offset)
    assert torch.equal(before, torch.npu.get_rng_state(rng_device))


@pytest.mark.npu
def test_graph_carrier_lifetime_reseed_and_eager_interleave(rng_device):
    op = mixed
    cpus = [utils.axis_input((1, 512), 1, torch.float32), utils.axis_input((1, 256, 8), 1, torch.bfloat16)]
    xs = [cpu.to(rng_device) for cpu in cpus]
    attributes = [dict(axis=1, dst_type=24, scale_alg=1), dict(axis=1, dst_type=23, scale_alg=1)]
    for x, attrs in zip(xs, attributes):
        op.dynamic_mx_quant_sr(x, **attrs)
    torch.npu.synchronize(rng_device)
    generator = torch.npu.default_generators[rng_device.index]
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            outputs = [op.dynamic_mx_quant_sr(x, **attrs) for x, attrs in zip(xs, attributes)]
        op.clear_caches()
        gc.collect()
        # Exercise allocation after dropping every cached Program reference.
        junk = [torch.empty((32768,), dtype=torch.uint8, device=rng_device) for _ in range(8)]
        increments = [op._rng_increment(x.numel()) for x in xs]
        for seed in (2**64 - 1, 2**63 + 19):
            generator.manual_seed(seed)
            generator.set_offset(2**32 - 4)
            offset = generator.get_offset()
            for replay in range(2):
                graph.replay()
                for cpu, result, attrs, increment in zip(cpus, outputs, attributes, increments):
                    utils.assert_result(
                        cpu,
                        result,
                        attrs["axis"],
                        cpu.dtype,
                        attrs["dst_type"],
                        attrs["scale_alg"],
                        0,
                        rng_seed=seed,
                        rng_block_offset=offset,
                    )
                    offset += increment
                assert generator.get_offset() == offset
                if replay == 0:
                    eager = op.dynamic_mx_quant_sr(xs[0], **attributes[0])
                    utils.assert_result(
                        cpus[0], eager, 1, torch.float32, 24, 1, 0, rng_seed=seed, rng_block_offset=offset
                    )
                    offset += increments[0]
                    assert generator.get_offset() == offset
        del junk
    finally:
        torch.npu.synchronize(rng_device)
        graph.reset()


@pytest.mark.npu
def test_public_framework_rng_on_nondefault_stream(rng_device):
    cpu = utils.axis_input((2, 48, 3), 1, torch.float32)
    x = cpu.to(rng_device)
    mixed.dynamic_mx_quant_sr(x, axis=1)
    torch.npu.synchronize(rng_device)
    generator = torch.npu.default_generators[rng_device.index]
    seed = 2**63 + 29
    generator.manual_seed(seed)
    stream = torch.npu.Stream(device=rng_device)
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph, stream=stream):
            captured = mixed.dynamic_mx_quant_sr(x, axis=1)
        generator.set_offset(100)
        with torch.npu.stream(stream):
            graph.replay()
            eager = mixed.dynamic_mx_quant_sr(x, axis=1)
        torch.npu.current_stream(rng_device).wait_stream(stream)
        utils.assert_result(cpu, captured, 1, torch.float32, 24, 1, 0, rng_seed=seed, rng_block_offset=100)
        utils.assert_result(cpu, eager, 1, torch.float32, 24, 1, 0, rng_seed=seed, rng_block_offset=104)
        assert generator.get_offset() == 108
    finally:
        torch.npu.synchronize(rng_device)
        graph.reset()


@pytest.mark.npu
@pytest.mark.parametrize("dst", [24, 23], ids=["e4m3", "e5m2"])
@pytest.mark.parametrize("algorithm", [0, 1])
def test_bf16_width16_route_changes_preserve_bytes_and_rng(rng_device, monkeypatch, dst, algorithm):
    """Quota and UB fallbacks consume the same public stream as direct SIMD."""
    cpu = random_input((17, 4096, 16), torch.bfloat16)
    x = cpu.to(rng_device)
    quota = min(56, cannbotdsl.get_platform_info(stream=torch.npu.current_stream(rng_device)).vector_core_num)
    full_ub = mixed.get_mem_size("ub")
    direct = mixed._make_execution_plan(x, 1, block_dim=quota, ub_bytes=full_ub)
    if direct.layout is not mixed.Layout.WIDE16:
        pytest.skip("available AIV quota cannot select direct width16 for this workload")
    routes = [
        (quota, full_ub, mixed.Layout.WIDE16),
        (1, full_ub, mixed.Layout.TRANSPOSE),
        (quota, 65536, mixed.Layout.TRANSPOSE),
    ]
    generator = torch.npu.default_generators[rng_device.index]
    # Compile both entries and the RNG bridge before fixing the tested state.
    for blocks, ub_bytes, layout in routes:
        monkeypatch.setattr(mixed, "get_mem_size", lambda name, value=ub_bytes: value)
        monkeypatch.setattr(mixed, "_resolve_block_dim", lambda x, block_dim, quota=blocks, **kwargs: quota)
        assert mixed._make_execution_plan(x, 1).layout is layout
        mixed.dynamic_mx_quant_sr(x, dst_type=dst, axis=1, scale_alg=algorithm)
    seed, offset = 2**63 + 97, 4 * (2**32 + 7)
    generator.manual_seed(seed)
    generator.set_offset(offset)
    initial = torch.npu.get_rng_state(rng_device).clone()
    reference = golden.dynamic_mx_quant_golden(
        utils.input_numpy(cpu),
        "float8_e5m2" if dst == 23 else "float8_e4m3fn",
        scale_alg=algorithm,
        axis=1,
        seed=seed,
        block_offset=offset,
    )
    next_state = None
    for blocks, ub_bytes, layout in routes:
        monkeypatch.setattr(mixed, "get_mem_size", lambda name, value=ub_bytes: value)
        monkeypatch.setattr(mixed, "_resolve_block_dim", lambda x, block_dim, quota=blocks, **kwargs: quota)
        assert mixed._make_execution_plan(x, 1).layout is layout
        torch.npu.set_rng_state(initial, rng_device)
        actual = mixed.dynamic_mx_quant_sr(x, dst_type=dst, axis=1, scale_alg=algorithm)
        for tensor, expected in zip(actual, reference):
            np.testing.assert_array_equal(tensor.view(torch.uint8).cpu().numpy(), expected)
        assert generator.get_offset() == offset + mixed._rng_increment(x.numel())
        state = torch.npu.get_rng_state(rng_device)
        if next_state is None:
            next_state = state.clone()
        else:
            assert torch.equal(state, next_state)
        # Graph replay must retain the same mapping and state reservation too.
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                captured = mixed.dynamic_mx_quant_sr(x, dst_type=dst, axis=1, scale_alg=algorithm)
            torch.npu.set_rng_state(initial, rng_device)
            graph.replay()
            for tensor, expected in zip(captured, reference):
                np.testing.assert_array_equal(tensor.view(torch.uint8).cpu().numpy(), expected)
            assert torch.equal(torch.npu.get_rng_state(rng_device), next_state)
        finally:
            graph.reset()


@pytest.mark.npu
@pytest.mark.parametrize("offset", [2**32 - 4, 4 * (2**32 - 1)], ids=["word-carry", "counter-carry"])
@pytest.mark.parametrize(
    "route,dst,algorithm",
    [
        ("fp32-four-pair", 24, 0),
        ("fp32-four-pair", 23, 0),
        ("fp32-four-pair", 24, 1),
        ("fp32-four-pair", 23, 1),
        ("fp32-output-alias", 24, 0),
        ("fp32-output-alias", 23, 0),
        ("fp32-output-alias", 24, 1),
        ("fp32-output-alias", 23, 1),
        ("bf16-crossdraw", 24, 0),
        ("bf16-crossdraw", 23, 0),
        ("bf16-batched", 24, 0),
        ("fp32-width16", 24, 0),
        ("fp32-width16", 23, 1),
    ],
)
def test_latest_optimized_routes_preserve_bytes_and_framework_state(route, dst, algorithm, offset, rng_device):
    op = mixed
    cores = int(cannbotdsl.get_platform_info(stream=torch.npu.current_stream(rng_device)).vector_core_num)
    ub = op.get_mem_size("ub")
    src = torch.bfloat16 if route.startswith("bf16") else torch.float32
    dtype = op.dtypes.bfloat16 if src == torch.bfloat16 else op.dtypes.float32
    tile = op._make_tiling(dtype, ub, output_rng_alias=route == "fp32-output-alias").tile_elements
    axis = 1
    if route == "fp32-width16":
        shape = (cores * 5, 256, 16)
    elif route == "bf16-crossdraw":
        draw = op.FOUR_FIELD_DRAW_ELEMENTS
        # One extra draw makes per-core tails cross a virtual-stream boundary.
        elements = ((cores * tile + draw - 1) // draw + 1) * draw
        shape = (elements // 64, 64)
    else:
        tiles = 1 if route == "fp32-four-pair" else 2
        shape = (cores * (tiles * tile + 256) // 256, 256)
    cpu = axis_input(shape, axis, src)
    x = cpu.to(rng_device)
    target = torch.float8_e5m2 if dst == 23 else torch.float8_e4m3fn
    prepared = op._prepare_launch(x, target, axis, algorithm, 0)
    plan = prepared.plan
    if route == "fp32-four-pair":
        assert op._select_fp32_four_pair_policy(
            x.numel(), dtype, plan.compile_layout, plan.quant_length, plan.blocks, ub
        )
    elif route == "fp32-output-alias":
        assert op._select_fp32_eight_policy(x.numel(), dtype, plan.compile_layout, plan.quant_length)
    elif route == "bf16-crossdraw":
        assert op._select_bf16_crossdraw_prefix3_policy(x.numel(), dtype, algorithm, plan, ub)
    elif route == "bf16-batched":
        assert op._select_bf16_four_panel_policy(
            x.numel(), dtype, target, algorithm, 0, plan.layout, plan.quant_length, plan.blocks
        )
    else:
        assert plan.layout is op.Layout.TRANSPOSE
    # Warm up before fixing a seed/offset that crosses the low-word boundary.
    attrs = dict(axis=axis, dst_type=dst, scale_alg=algorithm)
    op.dynamic_mx_quant_sr(x, **attrs)
    generator = torch.npu.default_generators[rng_device.index]
    seed = 2**63 + 97
    generator.manual_seed(seed)
    generator.set_offset(offset)
    checkpoint = torch.npu.get_rng_state(rng_device).clone()
    increment = op._rng_increment(x.numel())
    for call in range(2):
        result = op.dynamic_mx_quant_sr(x, **attrs)
        utils.assert_result(
            cpu, result, axis, src, dst, algorithm, 0, rng_seed=seed, rng_block_offset=offset + call * increment
        )
        assert generator.get_offset() == offset + (call + 1) * increment
    torch.npu.set_rng_state(checkpoint, rng_device)
    restored = op.dynamic_mx_quant_sr(x, **attrs)
    utils.assert_result(cpu, restored, axis, src, dst, algorithm, 0, rng_seed=seed, rng_block_offset=offset)
    assert generator.get_offset() == offset + increment
    assert torch.equal(x.cpu().view(torch.uint8), cpu.view(torch.uint8))
    if route in ("fp32-four-pair", "bf16-crossdraw"):
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                captured = op.dynamic_mx_quant_sr(x, **attrs)
            torch.npu.set_rng_state(checkpoint, rng_device)
            graph.replay()
            utils.assert_result(cpu, captured, axis, src, dst, algorithm, 0, rng_seed=seed, rng_block_offset=offset)
            assert generator.get_offset() == offset + increment
        finally:
            torch.npu.synchronize(rng_device)
            graph.reset()


@pytest.mark.npu
@pytest.mark.parametrize("dst,algorithm", [(24, 0), (24, 1), (23, 0), (23, 1)])
@pytest.mark.parametrize("waves", [1, 2])
@pytest.mark.parametrize("offset", [2**32 - 4, 4 * (2**32 - 1)], ids=["word-carry", "counter-carry"])
def test_complete_wave_bf16_mixed_classes_rng_and_graph(dst, algorithm, waves, offset, monkeypatch, rng_device):
    op = mixed
    cores = int(cannbotdsl.get_platform_info(stream=torch.npu.current_stream(rng_device)).vector_core_num)
    tile = 32768
    cpu = torch.ones((cores * tile * waves // 512, 512), dtype=torch.bfloat16)
    bits = cpu.view(torch.int16).reshape(-1)
    markers = (0x7F80, 0xFF80, 0x7FC1, 0xFFC1, 0x0001, 0x8001, 0x0080, 0x3700, 0x3500, 0x0000, 0x8000)
    for index in range(cores * waves):
        # The first four panels stay ordinary; exercise the second half of
        # eight-panel classes in each of the four 32-panel bundles.
        position = index * tile + (index % 4) * 8192 + 4 * 256
        marker = markers[index % len(markers)]
        bits[position] = marker if marker < 32768 else marker - 65536
        bits[position + 1] = -16512  # BF16 -1, a finite companion in the same scale group.
    x = cpu.to(rng_device)
    target = torch.float8_e5m2 if dst == 23 else torch.float8_e4m3fn
    observed = []
    original = op._get_compiled_kernel

    def record(*args, **kwargs):
        observed.append(kwargs)
        return original(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(op, "_get_compiled_kernel", record)
        prepared = op._prepare_launch(x, target, 1, algorithm, 0)
    assert len(observed) == 1 and observed[0]["bf16_full_tiles"] and observed[0]["bf16_four_panel"]
    assert prepared.plan.blocks == cores
    attrs = dict(dst_type=dst, scale_alg=algorithm)
    op.dynamic_mx_quant_sr(x, **attrs)  # Warm up the bridge outside capture.
    generator = torch.npu.default_generators[rng_device.index]
    seed = 0xFEDCBA9876543210
    generator.manual_seed(seed)
    generator.set_offset(offset)
    checkpoint = torch.npu.get_rng_state(rng_device).clone()
    increment = op._rng_increment(x.numel())
    state = op._framework_rng._make_rng_state(seed, offset, device=rng_device)
    prepared.launch(rng_state=state)
    utils.assert_result(
        cpu, prepared.outputs(), 1, cpu.dtype, dst, algorithm, 0, rng_seed=seed, rng_block_offset=offset
    )
    assert generator.get_offset() == offset
    for call in range(2):
        result = op.dynamic_mx_quant_sr(x, **attrs)
        utils.assert_result(
            cpu, result, 1, cpu.dtype, dst, algorithm, 0, rng_seed=seed, rng_block_offset=offset + call * increment
        )
        assert generator.get_offset() == offset + (call + 1) * increment
    torch.npu.set_rng_state(checkpoint, rng_device)
    restored = op.dynamic_mx_quant_sr(x, **attrs)
    utils.assert_result(cpu, restored, 1, cpu.dtype, dst, algorithm, 0, rng_seed=seed, rng_block_offset=offset)
    assert generator.get_offset() == offset + increment
    assert torch.equal(x.cpu().view(torch.uint8), cpu.view(torch.uint8))
    if waves == 2:
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph):
                captured = op.dynamic_mx_quant_sr(x, **attrs)
            for _ in range(2):
                torch.npu.set_rng_state(checkpoint, rng_device)
                graph.replay()
                utils.assert_result(
                    cpu, captured, 1, cpu.dtype, dst, algorithm, 0, rng_seed=seed, rng_block_offset=offset
                )
                assert generator.get_offset() == offset + increment
        finally:
            torch.npu.synchronize(rng_device)
            graph.reset()
    elif offset == 2**32 - 4:
        # Adding a partial wave must retain a correct fallback for every format/algorithm.
        partial_cpu = torch.cat((cpu.reshape(-1), torch.ones(256, dtype=cpu.dtype))).reshape(-1, 256)
        partial_x = partial_cpu.to(rng_device)
        observed.clear()
        with monkeypatch.context() as scoped:
            scoped.setattr(op, "_get_compiled_kernel", record)
            partial = op._prepare_launch(partial_x, target, 1, algorithm, 0)
        assert len(observed) == 1 and not observed[0]["bf16_full_tiles"]
        partial.launch(rng_state=state)
        utils.assert_result(
            partial_cpu, partial.outputs(), 1, cpu.dtype, dst, algorithm, 0, rng_seed=seed, rng_block_offset=offset
        )


@pytest.mark.npu
@pytest.mark.parametrize(
    "shape,axis,src,dst,algorithm",
    [
        ((2, 80), 1, torch.bfloat16, 24, 0),
        ((2, 80), 1, torch.float32, 23, 1),
        ((3, 176), 1, torch.bfloat16, 23, 0),
        ((3, 176), 1, torch.float32, 24, 1),
        ((3, 512), 1, torch.bfloat16, 24, 1),
        ((3, 512), 1, torch.float32, 23, 0),
        ((64, 64), 0, torch.bfloat16, 24, 1),
        ((64, 128), 0, torch.float32, 23, 0),
        ((64, 32), 0, torch.bfloat16, 23, 0),
        ((256, 8), 0, torch.float32, 24, 1),
        ((256, 16), 0, torch.bfloat16, 23, 1),
        ((2, 512, 2), 1, torch.float32, 24, 0),
        ((2, 576, 2), 1, torch.bfloat16, 23, 1),
        ((2, 80, 3), 1, torch.bfloat16, 24, 0),
        ((8193, 256), 1, torch.float32, 24, 1),
        ((512, 256), 1, torch.bfloat16, 24, 0),
        ((512, 256), 1, torch.bfloat16, 24, 1),
        ((512, 256), 1, torch.bfloat16, 23, 0),
        ((512, 256), 1, torch.bfloat16, 23, 1),
    ],
    ids=[
        "short-bf16",
        "short-fp32",
        "tail-bf16",
        "tail-fp32",
        "coalesced-bf16",
        "coalesced-fp32",
        "wide64",
        "wide128",
        "wide32",
        "wide8",
        "wide16",
        "stride2-fused",
        "stride2",
        "transpose",
        "fp32-alias-tail",
        "full-e4-alg0",
        "full-e4-alg1",
        "full-e5-alg0",
        "full-e5-alg1",
    ],
)
def test_output_dma_preserves_guards_across_layouts(shape, axis, src, dst, algorithm, rng_device):
    op = mixed
    cpu = axis_input(shape, axis, src)
    x = cpu.to(rng_device)
    target = torch.float8_e5m2 if dst == 23 else torch.float8_e4m3fn
    prepared = op._prepare_launch(x, target, axis, algorithm, 0, block_dim=4)
    guard = 64
    # Own the entire historical erroneous DMA span, including a full source
    # slot after the logical output, so a regression cannot corrupt other allocations.
    storage = [
        torch.full((guard + value.numel() + 65536,), 0xA5, dtype=torch.uint8, device=rng_device)
        for value in (prepared.encoded, prepared.scales)
    ]
    encoded, scales = [
        backing[guard : guard + value.numel()].view(value.shape)
        for backing, value in zip(storage, (prepared.encoded, prepared.scales))
    ]
    arguments = list(prepared.arguments)
    arguments[2 if prepared.plan.layout in (op.Layout.NON_TAIL, op.Layout.NON_TAIL_FUSED) else 1] = encoded.view(
        prepared.plan.kernel_shape
    )
    arguments[3 if prepared.plan.layout in (op.Layout.NON_TAIL, op.Layout.NON_TAIL_FUSED) else 2] = scales.view(
        prepared.plan.kernel_scale_shape
    )
    prepared = replace(prepared, arguments=tuple(arguments), encoded=encoded, scales=scales)
    for offset in (0, 4, 2**32 - 4):
        for backing in storage:
            backing.fill_(0xA5)
        prepared.launch(rng_state=op._framework_rng._make_rng_state(17, offset, device=rng_device))
        torch.npu.synchronize(rng_device)
        for backing, value in zip(storage, (encoded, scales)):
            assert torch.all(backing[:guard] == 0xA5).item()
            assert torch.all(backing[guard + value.numel() :] == 0xA5).item()
        utils.assert_result(cpu, prepared.outputs(), axis, src, dst, algorithm, 0, rng_seed=17, rng_block_offset=offset)
    assert torch.equal(x.cpu().view(torch.uint8), cpu.view(torch.uint8))
