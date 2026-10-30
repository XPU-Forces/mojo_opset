"""SR policy regressions adapted to the Mojo provider and capability gates."""

import pytest

pytest.importorskip("cannbotdsl", reason="SR provider tests require CANNBot-DSL 0.8")
import dataclasses
import gc
import inspect
import itertools
import math
import shutil
import subprocess
import sys
import weakref

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from threading import Event
from types import SimpleNamespace

import cannbotdsl
import numpy as np
import torch

from cannbotdsl import dtypes
from ml_dtypes import bfloat16

from mojo_opset.kernels.npu_a5_cannbotdsl import _rng_state as rng
from mojo_opset.kernels.npu_a5_cannbotdsl import dynamic_mx_quant_sr as mixed
from mojo_opset.kernels.torch import dynamic_mx_quant_sr_golden as golden

from . import _dynamic_mx_quant_sr_rng as utils
from ._dynamic_mx_quant_sr_rng import INPUT_DTYPES
from ._dynamic_mx_quant_sr_rng import LAYOUT_CASES
from ._dynamic_mx_quant_sr_rng import THREADS
from ._dynamic_mx_quant_sr_rng import TensorGeometry
from ._dynamic_mx_quant_sr_rng import _cache_key
from ._dynamic_mx_quant_sr_rng import _compile_context
from ._dynamic_mx_quant_sr_rng import _fake_program
from ._dynamic_mx_quant_sr_rng import _observe_cache_waiter
from ._dynamic_mx_quant_sr_rng import _platform
from ._dynamic_mx_quant_sr_rng import _resolver
from ._dynamic_mx_quant_sr_rng import _rng_host
from ._dynamic_mx_quant_sr_rng import _scalar_philox_words as scalar_philox
from ._dynamic_mx_quant_sr_rng import _unexpected_device_work
from ._dynamic_mx_quant_sr_rng import finite_scale_boundary_input
from ._dynamic_mx_quant_sr_rng import quota as quota

pytestmark = pytest.mark.reference


@pytest.fixture(autouse=True)
def host_npu_interface(monkeypatch):
    """Expose only host queries to monkeypatch; never load/init torch_npu."""
    if not hasattr(torch, "npu"):

        def unavailable(*args, **kwargs):
            raise AssertionError("Host regression attempted unmocked NPU work")

        monkeypatch.setattr(
            torch,
            "npu",
            SimpleNamespace(
                device=unavailable,
                current_stream=unavailable,
                current_device=lambda: 0,
                is_current_stream_capturing=lambda: False,
            ),
            raising=False,
        )
    if torch._C._get_privateuse1_backend_name() == "privateuseone":
        torch.utils.rename_privateuse1_backend("npu")


@pytest.mark.parametrize("member", ["_create_simt_cast", "_wrap_simt_scalar"])
def test_missing_unsigned_simt_capability_has_actionable_error(monkeypatch, member):
    monkeypatch.delattr(mixed.simt, member)
    with pytest.raises(RuntimeError, match=rf"CANNBotDSL.*{member}"):
        mixed._simt_umulhi(0, mixed.PHILOX_M0)


@pytest.mark.parametrize("member", ["_runtime_identity", "_toolchain_version", "_FORMAT_VERSION_TAG"])
def test_missing_sdk_identity_capability_has_actionable_error(monkeypatch, member):
    monkeypatch.delattr(mixed.cache_key, member)
    with pytest.raises(RuntimeError, match=rf"CANNBotDSL.*{member}"):
        mixed._build_identity()


@pytest.mark.parametrize("shape,axis", [((2, 512), 1), ((2, 1024, 2), 1), ((2, 48, 3), 1), ((256, 16), 0)])
@pytest.mark.parametrize("change", ["runtime", "compiler", "configuration", "diagnostics", "concurrency"])
def test_prepare_discovers_identity_once_per_call_and_refreshes(monkeypatch, shape, axis, change):
    # Keep real planning, context construction and Program caches; replace device/AOT work.
    _compile_context(monkeypatch)
    stamp = ["runtime-a", "compiler-a"]
    discoveries = []
    from threading import local

    thread_state = local()

    def identity():
        current = getattr(thread_state, "stamp", stamp)
        discoveries.append(tuple(current))
        return ("test-wheel", *current, 9, "test-format")

    monkeypatch.setattr(mixed, "_build_identity", identity)
    capture = mixed._capture_runtime_identity
    monkeypatch.setattr(
        mixed, "_capture_runtime_identity", lambda device: capture(SimpleNamespace(type="npu", index=0))
    )
    monkeypatch.setattr(torch.npu, "device", lambda device: nullcontext())
    monkeypatch.setattr(mixed, "get_mem_size", lambda name: 253952)
    monkeypatch.setattr(mixed, "_resolve_block_dim", lambda *args, **kwargs: 4)
    monkeypatch.setattr(mixed.cannbotdsl, "compile", lambda *args: _fake_program())
    for name, capacity in (
        ("_PROGRAM_CACHE", 64),
        ("_TAIL_SCALAR_PROGRAM_CACHE", 64),
        ("_FRAMEWORK_PROGRAM_CACHE", 128),
    ):
        monkeypatch.setattr(mixed, name, mixed._ProgramCache(capacity))
    monkeypatch.delenv("CANNBOTDSL_BUFFER_READ_SHARING", raising=False)
    monkeypatch.delenv("CANNBOTDSL_DUMP_FRONTEND", raising=False)
    x = torch.empty(shape, dtype=torch.float32)
    first = mixed._prepare_launch(x, torch.float8_e4m3fn, axis, 1, 0)
    assert len(discoveries) == 1
    second = mixed._prepare_launch(x, torch.float8_e4m3fn, axis, 1, 0)
    assert len(discoveries) == 2
    assert first.program is second.program
    assert all(a is b for a, b in zip(first.framework_programs, second.framework_programs))
    if change == "concurrency":
        barrier = Barrier(2)

        def prepare(native):
            thread_state.stamp = [native, "compiler-a"]
            barrier.wait(timeout=10)
            return mixed._prepare_launch(x, torch.float8_e4m3fn, axis, 1, 0)

        with ThreadPoolExecutor(max_workers=2) as executor:
            a, b = list(executor.map(prepare, ("runtime-b", "runtime-c")))
        assert len(discoveries) == 4
        assert {entry[0] for entry in discoveries[2:]} == {"runtime-b", "runtime-c"}
        assert a.program is not b.program
        assert all(p is not q for p, q in zip(a.framework_programs, b.framework_programs))
        return
    if change == "runtime":
        stamp[0] = "runtime-b"
    elif change == "compiler":
        stamp[1] = "compiler-b"
    elif change == "configuration":
        monkeypatch.setenv("CANNBOTDSL_BUFFER_READ_SHARING", "0")
    else:
        monkeypatch.setenv("CANNBOTDSL_DUMP_FRONTEND", "source")
    third = mixed._prepare_launch(x, torch.float8_e4m3fn, axis, 1, 0)
    assert len(discoveries) == 3
    assert third.program is not first.program
    assert all(a is not b for a, b in zip(first.framework_programs, third.framework_programs))
    if change == "diagnostics":
        fourth = mixed._prepare_launch(x, torch.float8_e4m3fn, axis, 1, 0)
        assert len(discoveries) == 4
        assert fourth.program is not third.program
        assert all(a is not b for a, b in zip(third.framework_programs, fourth.framework_programs))


def test_framework_rng_reservation_is_independent_of_operator_compilation(monkeypatch):
    import builtins

    from contextlib import nullcontext

    rng = _rng_host()
    device = SimpleNamespace(type="npu", index=2)
    state = (True, False, object(), object(), 2**63)
    reservations = []

    def reserve(index, increment):
        reservations.append((index, increment))
        return state

    monkeypatch.setattr(rng, "_resolved_device", lambda value: device)
    monkeypatch.setattr(rng, "_load_bridge", lambda: SimpleNamespace(reserve_philox_state=reserve))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=SimpleNamespace(device=lambda value: nullcontext())))
    original_import = builtins.__import__

    def framework_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("cannbotdsl") or (level and name == "dynamic_mx_quant_sr"):
            raise AssertionError("framework RNG reservation must not depend on DSL compilation")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", framework_import)
    assert rng._reserve_rng_carrier(device, 4) is state
    assert reservations == [(2, 4)]


@pytest.mark.parametrize(
    "groups,error",
    [
        (-1, ValueError),
        ([0, -1], ValueError),
        (2**64, ValueError),
        (True, TypeError),
        ([0, True], TypeError),
        (1.0, TypeError),
        ([0, 1.5], TypeError),
        ("1", TypeError),
    ],
)
def test_rng_linear_golden_rejects_invalid_groups(groups, error):
    with pytest.raises(error):
        golden._philox_blocks(groups, quant_length=128)


@pytest.mark.parametrize(
    "quant_length,error",
    [
        (True, TypeError),
        (16.0, TypeError),
        (0, ValueError),
        (-16, ValueError),
        (15, ValueError),
        (2**64, ValueError),
    ],
)
def test_rng_linear_golden_rejects_invalid_axis(quant_length, error):
    with pytest.raises(error):
        golden._philox_blocks(0, quant_length=quant_length)


@pytest.mark.parametrize(
    "numel,quant_length,error",
    [
        (True, 16, TypeError),
        (16, True, TypeError),
        (16.0, 16, TypeError),
        (16, 16.0, TypeError),
        (0, 16, ValueError),
        (-16, 16, ValueError),
        (16, 0, ValueError),
        (16, -16, ValueError),
        (16, 15, ValueError),
        (48, 32, ValueError),
        (2**64, 16, ValueError),
        (16, 2**64, ValueError),
    ],
)
def test_rng_linear_geometry_rejects_invalid(numel, quant_length, error):
    with pytest.raises(error):
        _rng_host()._validate_rng_geometry(numel, quant_length)


@pytest.mark.parametrize(
    "numel,quant_length",
    [
        (16, 16),
        (17749 * 144, 144),
        (2**36, 16),
        (2**40, 2**40),
        (2**64 - 16, 2**64 - 16),
    ],
)
def test_rng_linear_geometry_accepts_full_width(numel, quant_length):
    assert _rng_host()._validate_rng_geometry(numel, quant_length) is None


@pytest.mark.parametrize(
    "function,args,kwargs",
    [
        (golden._philox_blocks, (0,), {"quant_length": 16}),
        (golden._philox_word_stream, (4,), {"quant_length": 16}),
        (golden._random_bits, (16,), {"quant_length": 16}),
        (golden.dynamic_mx_quant_golden, (np.ones((1, 16), np.float32),), {}),
    ],
)
def test_rng_linear_rejects_old_word_offset(function, args, kwargs):
    with pytest.raises(TypeError):
        function(*args, **kwargs, word_offset=4)


@pytest.mark.parametrize(
    "function,count",
    [
        (golden._philox_blocks, 0),
        (golden._philox_word_stream, 4),
        (golden._random_bits, 16),
    ],
)
def test_rng_linear_requires_original_axis(function, count):
    with pytest.raises(TypeError, match="quant_length"):
        function(count)


@pytest.mark.parametrize("shape,axis", [((2, 64), -1), ((2, 64, 3), 1)])
def test_rng_linear_golden_uses_original_axis(shape, axis, monkeypatch):
    calls = []
    original = golden._random_bits

    def random_fields(numel, *, quant_length, seed, block_offset):
        calls.append((numel, quant_length, seed, block_offset))
        return original(numel, quant_length=quant_length, seed=seed, block_offset=block_offset)

    monkeypatch.setattr(golden, "_random_bits", random_fields)
    x = np.linspace(-2, 3, math.prod(shape), dtype=np.float32).reshape(shape)
    golden.dynamic_mx_quant_golden(x, axis=axis, seed=17, block_offset=4)
    assert calls == [(math.prod(shape), 64, 17, 4)]


@pytest.mark.parametrize(
    "numel, error",
    [
        (0, ValueError),
        (-16, ValueError),
        (15, ValueError),
        (17, ValueError),
        (True, TypeError),
        (16.0, TypeError),
        (2**64, ValueError),
    ],
)
def test_rng_increment_rejects_invalid_geometry(numel, error):
    with pytest.raises(error):
        _rng_host()._rng_increment(numel)


@pytest.mark.parametrize(
    "increment,error",
    [
        (True, TypeError),
        (4.0, TypeError),
        (0, ValueError),
        (2, ValueError),
        (-4, ValueError),
        (2**64, ValueError),
    ],
)
def test_rng_framework_reservation_rejects_invalid_increment(increment, error):
    # These fail before resolving the device or initializing any NPU runtime.
    with pytest.raises(error):
        mixed._framework_rng._reserve_rng_carrier(None, increment)


def test_rng_word_stream_and_fields_forward_state():
    seed, offset = 0x1122334455667788, 4 * 2**32
    words = np.asarray(
        [golden._philox4x32_10(0, sequence, 0, 1, 0x55667788, 0x11223344) for sequence in (0, 1)], np.uint32
    ).reshape(-1)
    np.testing.assert_array_equal(
        golden._philox_word_stream(7, quant_length=32, seed=seed, block_offset=offset),
        words[:7],
    )
    fields = []
    for value in words[:7]:
        b3, b2, b1, b0 = int(value).to_bytes(4, "big")
        fields.extend(((b3 << 8) | b2, (b0 << 8) | b3, (b1 << 8) | b0, (b2 << 8) | b1))
    np.testing.assert_array_equal(
        golden._random_bits(25, quant_length=32, seed=seed, block_offset=offset),
        np.asarray(fields[:25], np.uint32),
    )


def test_rng_golden_forwards_state_through_non_tail_axis():
    x = np.random.default_rng(19).normal(size=(2, 64, 3)).astype(np.float32)
    state = dict(seed=2**63 + 7, block_offset=4 * 2**32)
    moved = np.ascontiguousarray(np.moveaxis(x, 1, -1))
    tail = golden.dynamic_mx_quant_golden(moved, **state)
    actual = golden.dynamic_mx_quant_golden(x, axis=1, **state)
    for output, reference, tail_axis in zip(actual, tail, (-1, -2)):
        np.testing.assert_array_equal(output, np.moveaxis(reference, tail_axis, 1))
    zero_state = golden.dynamic_mx_quant_golden(moved)
    assert not np.array_equal(tail[0], zero_state[0])


def test_sample_siblings_do_not_shadow_standard_library():
    result = subprocess.run(
        [sys.executable, "-c", "import random, tempfile; assert hasattr(random, 'Random')"],
        cwd=Path(mixed.__file__).parent,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_dtype", "bf16"),
        ("e5m2", True),
        ("ub_bytes", 200704),
        ("scale_alg", 0),
        ("max_low_bound_bits", 0x3F800000),
        ("layout", "wide32_non_tail"),
        ("npu_arch", "unsupported-test-arch"),
        ("device_type", "other-device"),
        ("device_index", 1),
        ("dsl_version", "other-wheel"),
        ("native_build", "other-native"),
        ("compiler_identity", "other-bisheng"),
        ("host_abi", 10),
        ("cache_format", "other-format"),
        ("cache_format_tag", "force-rebuild"),
        ("codegen_options", b"other-options"),
        ("rng_abi_version", 3),
        ("rng_abi_version", 4),
        ("fp32_eight_mapping", True),
        ("fp32_four_pair_schedule", True),
        ("bf16_crossdraw_prefix3", True),
    ],
)
def test_program_cache_isolates_complete_compile_identity(field, value):
    cache = mixed._ProgramCache(64)
    key = _cache_key()
    built = []

    def build():
        result = _fake_program()
        built.append(result)
        return result

    first = cache.get_or_build(key, build)
    assert cache.get_or_build(key, build) is first
    assert cache.get_or_build(replace(key, **{field: value}), build) is not first
    assert len(built) == 2
    with pytest.raises(FrozenInstanceError):
        key.ub_bytes = 1
    assert not {"shape", "blocks", "tile_rows", "quant_length", "width"} & key.__dataclass_fields__.keys()


def test_program_cache_coalesces_first_build_without_sleep(monkeypatch):
    waiting = _observe_cache_waiter(monkeypatch)
    cache = mixed._ProgramCache(64)
    entered, release = Event(), Event()
    calls = []

    def build():
        calls.append(1)
        entered.set()
        assert release.wait(10)
        return _fake_program()

    with ThreadPoolExecutor(2) as executor:
        owner = executor.submit(cache.get_or_build, _cache_key(), build)
        try:
            assert entered.wait(10)
            waiter = executor.submit(cache.get_or_build, _cache_key(), build)
            assert waiting.wait(10)
        finally:
            release.set()
        assert owner.result(10) is waiter.result(10)
    assert len(calls) == 1 and cache.info().currsize == 1


def test_program_cache_different_keys_build_outside_lock():
    cache = mixed._ProgramCache(64)
    rendezvous = Barrier(3, timeout=10)

    def build():
        rendezvous.wait()
        return _fake_program()

    with ThreadPoolExecutor(2) as executor:
        first = executor.submit(cache.get_or_build, _cache_key(), build)
        second = executor.submit(cache.get_or_build, _cache_key(e5m2=True), build)
        rendezvous.wait()
        assert first.result(10) is not second.result(10)
    assert cache.info().currsize == 2


def test_program_cache_clear_isolates_inflight_generations(monkeypatch):
    waiting = _observe_cache_waiter(monkeypatch)
    cache = mixed._ProgramCache(64)
    key = _cache_key()
    entered, release = Event(), Event()
    old, new = _fake_program("old.so"), _fake_program("new.so")

    def old_build():
        entered.set()
        assert release.wait(10)
        return old

    with ThreadPoolExecutor(2) as executor:
        owner = executor.submit(cache.get_or_build, key, old_build)
        try:
            assert entered.wait(10)
            waiter = executor.submit(cache.get_or_build, key, old_build)
            assert waiting.wait(10)
            generation = cache.info().generation
            cache.clear()
            assert cache.info().generation == generation + 1
            assert cache.get_or_build(key, lambda: new) is new
        finally:
            release.set()
        assert owner.result(10) is old and waiter.result(10) is old
    assert cache.get_or_build(key, lambda: pytest.fail("old build refilled cache")) is new
    assert not old.closed and cache.info().currsize == 1


def test_program_cache_failure_wakes_waiters_and_allows_retry(monkeypatch):
    waiting = _observe_cache_waiter(monkeypatch)
    cache = mixed._ProgramCache(64)
    entered, release = Event(), Event()
    failure = RuntimeError("controlled AOT failure")

    def build():
        entered.set()
        assert release.wait(10)
        raise failure

    with ThreadPoolExecutor(2) as executor:
        owner = executor.submit(cache.get_or_build, _cache_key(), build)
        try:
            assert entered.wait(10)
            waiter = executor.submit(cache.get_or_build, _cache_key(), build)
            assert waiting.wait(10)
        finally:
            release.set()
        for future in (owner, waiter):
            with pytest.raises(RuntimeError, match="controlled AOT failure") as caught:
                future.result(10)
            assert caught.value is failure
    assert cache.info().inflight == 0 and cache.info().currsize == 0
    assert cache.get_or_build(_cache_key(), _fake_program).so_path


def test_program_cache_lru_capacity_and_external_lifetime():
    cache = mixed._ProgramCache(2)
    first = cache.get_or_build(_cache_key(), _fake_program)
    cache.get_or_build(_cache_key(e5m2=True), _fake_program)
    assert cache.get_or_build(_cache_key(), _fake_program) is first
    cache.get_or_build(_cache_key(scale_alg=0), _fake_program)
    assert cache.info().currsize == 2
    assert cache.get_or_build(_cache_key(), _fake_program) is first
    cache.clear()
    assert not first.closed and first.so_path == "compiled.so"
    assert cache.info().currsize == 0


@pytest.mark.parametrize("action", ["clear", "evict"])
@pytest.mark.parametrize("external_owner", [False, True], ids=["cache-only", "external-owner"])
def test_program_cache_releases_last_reference_and_owned_resources(action, external_owner, tmp_path):
    cache = mixed._ProgramCache(1)
    directory = tmp_path / "owned-program"
    directory.mkdir()
    artifact = directory / "kernel.so"
    artifact.write_bytes(b"owned test artifact")

    class OwnedProgram:
        so_path = str(artifact)
        closed = False

    program = cache.get_or_build("first", OwnedProgram)
    reference = weakref.ref(program)
    cleanup = weakref.finalize(program, shutil.rmtree, directory)
    if not external_owner:
        del program
    if action == "clear":
        cache.clear()
    else:
        cache.get_or_build("replacement", _fake_program)
    if external_owner:
        gc.collect()
        assert reference() is program and not program.closed
        assert artifact.is_file() and cleanup.alive
        del program
    gc.collect()
    assert reference() is None and not cleanup.alive
    assert not directory.exists()


def test_program_cache_rebuilds_externally_closed_program():
    cache = mixed._ProgramCache(64)
    first = cache.get_or_build(_cache_key(), _fake_program)
    first.closed = True
    assert cache.get_or_build(_cache_key(), _fake_program) is not first


def test_program_cache_diagnostics_never_swallow_or_publish_stubs():
    cache = mixed._ProgramCache(64)
    key = _cache_key()
    normal = cache.get_or_build(key, _fake_program)
    calls = []

    def translate():
        calls.append("AOT")
        return _fake_program("")

    assert cache.get_or_build(key, translate, bypass=True).so_path == ""
    assert cache.get_or_build(key, translate, bypass=True).so_path == ""
    assert cache.get_or_build(key, _fake_program) is normal
    assert calls == ["AOT", "AOT"]
    cache.clear()
    assert cache.get_or_build(key, translate, bypass=True).so_path == ""
    assert cache.info().currsize == 0
    with pytest.raises(RuntimeError, match="so_path|stub"):
        cache.get_or_build(key, translate)
    assert cache.info().currsize == 0 and cache.info().inflight == 0
    assert cache.get_or_build(key, _fake_program).so_path


@pytest.mark.parametrize(
    "shape,axis,layout",
    [
        ((2, 64), -1, "tail"),
        ((2, 512, 2), 1, "non_tail_fused"),
        ((2, 576, 2), 1, "non_tail"),
        ((256, 8), 0, "wide8_non_tail"),
        ((256, 16), 0, "wide16_non_tail"),
        ((64, 32), 0, "wide32_non_tail"),
        ((64, 64), 0, "wide_non_tail"),
        ((64, 128), 0, "wide_non_tail"),
        ((2, 48, 3), 1, "transpose"),
        ((2, 3, 64), 2, "tail"),
    ],
)
def test_host_execution_plan_is_frozen_and_keeps_runtime_geometry(shape, axis, layout, monkeypatch):
    monkeypatch.setattr(mixed, "_resolve_block_dim", lambda x, blocks, **kwargs: 64)
    x = torch.empty(shape)
    plan = mixed._make_execution_plan(x, axis % x.ndim)
    assert isinstance(plan, mixed.ExecutionPlan) and plan.layout is mixed.Layout(layout)
    assert plan.quant_length == shape[axis] and plan.output_shape == shape
    expected_scale = list(shape)
    expected_scale[axis] = (shape[axis] + 63) // 64
    assert plan.scale_shape == (*expected_scale, 2)
    assert 1 <= plan.blocks <= 64
    assert plan.compile_layout is (mixed.Layout.TAIL if layout == "transpose" else mixed.Layout(layout))
    with pytest.raises(FrozenInstanceError):
        plan.blocks = 1


@pytest.mark.parametrize(
    "layout",
    [
        "tail",
        "non_tail",
        "non_tail_fused",
        "wide_non_tail",
        "wide32_non_tail",
        "wide8_non_tail",
        "wide16_non_tail",
    ],
)
def test_registry_uses_direct_aot_and_dynamic_specs(layout, monkeypatch):
    calls = []
    result = _fake_program()

    def compile_aot(entry, *specs):
        calls.append((entry, specs))
        return result

    monkeypatch.setattr(mixed.cannbotdsl, "compile", compile_aot)
    assert mixed._build_program(_cache_key(layout=mixed.Layout(layout))) is result
    assert len(calls) == 1
    _, specs = calls[0]
    tensors = [spec for spec in specs if isinstance(spec, mixed.TensorSpec)]
    # Only the fixed two-word RNG carrier is static; data geometry is dynamic.
    assert tensors[-1].shape == (2,) and tensors[-1].dtype == dtypes.int64
    assert all(isinstance(dim, mixed.Dim) for spec in tensors[:-1] for dim in spec.shape)
    assert all(dim.max is None for spec in tensors[:-1] for dim in spec.shape)
    registry = mixed._LAYOUT_REGISTRY
    assert set(registry) == set(mixed.Layout) - {mixed.Layout.TRANSPOSE}
    with pytest.raises(TypeError):
        registry[mixed.Layout.TAIL] = None
    assert registry[mixed.Layout.WIDE16].entry == registry[mixed.Layout.WIDE8].entry
    assert registry[mixed.Layout.WIDE16].wide_lanes == 16
    assert not hasattr(mixed, "_load_static_compiler")


@pytest.mark.parametrize(
    "setting,value",
    [
        ("CANNBOTDSL_AUTO_BUFFER_SYNC", "0"),
        ("CANNBOTDSL_BUFFER_READ_SHARING", "0"),
        ("CANNBOTDSL_DEBUG_ASC_SYNC", "all"),
        ("CANNBOTDSL_BARALL", "1"),
        ("CANNBOTDSL_FRONTEND_LINE_INFO", "1"),
        ("CANNBOTDSL_CACHE_FORMAT_TAG", "force-rebuild"),
    ],
)
def test_real_configuration_affects_program_identity(setting, value, monkeypatch):
    monkeypatch.delenv(setting, raising=False)
    first = _compile_context(monkeypatch)
    monkeypatch.setenv(setting, value)
    second = _compile_context(monkeypatch)
    assert first.key != second.key
    assert second.key.device_type == "npu" and second.key.device_index == 1


@pytest.mark.parametrize(
    "setting,value",
    [
        ("CANNBOTDSL_PIPE_STAGE", "translate"),
        ("CANNBOTDSL_PIPE_STAGE", "compile"),
        ("CANNBOTDSL_DUMP_MLIR", "1"),
        ("CANNBOTDSL_DUMP_ASCENDC", "1"),
        ("CANNBOTDSL_PRINT_MLIR", "1"),
        ("CANNBOTDSL_PRINT_ASCENDC", "1"),
        ("CANNBOTDSL_REUSE_ASCENDC", "1"),
        ("CANNBOTDSL_DUMP_FRONTEND", "source"),
    ],
)
def test_real_diagnostic_configuration_bypasses_outer_cache(setting, value, monkeypatch):
    monkeypatch.setenv(setting, value)
    assert _compile_context(monkeypatch).bypass_reason is not None


def test_dump_directory_alone_is_not_a_diagnostic_or_code_option(monkeypatch):
    first = _compile_context(monkeypatch)
    monkeypatch.setenv("CANNBOTDSL_DUMP_DIR", "/tmp/pr279-dump-identity")
    second = _compile_context(monkeypatch)
    assert first.key == second.key and second.bypass_reason is None


def test_both_official_cache_tiers_disabled_bypass_program_cache(monkeypatch):
    monkeypatch.setenv("CANNBOTDSL_CACHE_MEM_DISABLE", "yes")
    monkeypatch.setenv("CANNBOTDSL_CACHE_DISK_DISABLE", "on")
    assert _compile_context(monkeypatch).bypass_reason == "all_tiers_disabled"


def test_compile_context_rejects_unsupported_architecture_and_unresolved_device(monkeypatch):
    module = mixed
    _compile_context(monkeypatch)
    monkeypatch.setenv("CANNBOTDSL_NPU_ARCH", "dav-2201")
    arguments = ("fp32", False, 253952, 1, 0, mixed.Layout.TAIL)
    with pytest.raises(NotImplementedError, match="architecture"):
        module.capture_compile_context(*arguments, SimpleNamespace(type="npu", index=0))
    monkeypatch.setenv("CANNBOTDSL_NPU_ARCH", "dav-3510")
    with pytest.raises(ValueError, match="resolved"):
        module.capture_compile_context(*arguments, SimpleNamespace(type="npu", index=None))


def test_production_program_cache_wraps_aot_not_backend_compiler(monkeypatch):
    module = mixed
    monkeypatch.setenv("CANNBOTDSL_NPU_ARCH", "dav-3510")
    monkeypatch.setattr(module, "_build_identity", lambda: ("wheel", "native", "compiler", 9, "format"))
    backend_cache, aot_calls, backend_calls = {}, [], []

    def build(key):
        aot_calls.append(key)
        if key not in backend_cache:
            backend_calls.append(key)
            backend_cache[key] = "compiled.so"
        return _fake_program(backend_cache[key])

    monkeypatch.setattr(mixed, "_build_program", build)
    arguments = (dtypes.float32, False, 253952)
    options = dict(device=SimpleNamespace(type="npu", index=0))
    mixed.clear_caches()
    try:
        first = mixed._get_compiled_kernel(*arguments, **options)
        assert mixed._get_compiled_kernel(*arguments, **options) is first
        mixed.clear_caches()
        assert mixed._get_compiled_kernel(*arguments, **options) is not first
        assert not first.closed and len(aot_calls) == 2 and len(backend_calls) == 1
    finally:
        mixed.clear_caches()


@pytest.mark.parametrize("numel", [1, 3, 4, 5, 7, 8])
def test_sr_fields_use_high_rotate_low_low_rotate_high(numel, monkeypatch):
    """A wrong lane pairing must fail without sharing the DSL gather formula."""
    words = np.array([0x01234567, 0x89ABCDEF], dtype=np.uint32)
    expected = np.array(
        [0x0123, 0x6701, 0x4567, 0x2345, 0x89AB, 0xEF89, 0xCDEF, 0xABCD],
        dtype=np.uint32,
    )

    def fixed_words(count, *, quant_length, seed, block_offset):
        assert count == (numel + 3) // 4
        assert quant_length == 16
        assert seed == block_offset == 0
        return words[:count]

    monkeypatch.setattr(golden, "_philox_word_stream", fixed_words)
    np.testing.assert_array_equal(golden._random_bits(numel, quant_length=16), expected[:numel])


@pytest.mark.parametrize("shape,axis", LAYOUT_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
def test_rejects_unsupported_dtype_before_dispatch(shape, axis, dtype, monkeypatch):
    monkeypatch.setattr(torch.npu, "device", _unexpected_device_work)
    with pytest.raises(TypeError, match="dtype.*float32.*bfloat16"):
        mixed.dynamic_mx_quant_sr(torch.ones(shape, dtype=dtype), axis=axis)


@pytest.mark.parametrize("shape,axis", LAYOUT_CASES)
def test_rejects_cpu_input_before_dispatch(shape, axis, monkeypatch):
    monkeypatch.setattr(torch.npu, "device", _unexpected_device_work)
    with pytest.raises(ValueError, match="Ascend NPU"):
        mixed.dynamic_mx_quant_sr(torch.ones(shape), axis=axis)


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
        ({"max_low_bound": 1e100}, ValueError, "float32 range"),
        ({"scale_alg": 0, "max_low_bound": 1.0}, ValueError, "max_low_bound"),
    ],
)
def test_rejects_unsupported_attributes(kwargs, error, match):
    with pytest.raises(error, match=match):
        mixed.dynamic_mx_quant_sr(torch.ones((2, 64)), **kwargs)


@pytest.mark.parametrize(
    "value,axis,error,match",
    [
        (None, -1, TypeError, "input"),
        (torch.empty(()), -1, ValueError, "nonempty|dimension"),
        (torch.empty((0, 64)), -1, ValueError, "nonempty"),
        (torch.empty((2, 17)), -1, ValueError, "multiple of 16"),
        (torch.empty((2, 16)), -1, ValueError, "at least 32"),
        (torch.empty((2, 17, 3)), 1, ValueError, "multiple of 16"),
        (torch.empty((2, 64)).t(), 0, ValueError, "contiguous"),
    ],
)
def test_rejects_invalid_tensor_domain_before_device_work(value, axis, error, match, monkeypatch):
    monkeypatch.setattr(torch.npu, "device", _unexpected_device_work)
    with pytest.raises(error, match=match):
        mixed.dynamic_mx_quant_sr(value, axis=axis)


def test_compile_cache_includes_layout_algorithm_and_bound(monkeypatch):
    built = []

    def build(key):
        program = _fake_program()
        built.append((key, program))
        return program

    _compile_context(monkeypatch)
    monkeypatch.setattr(mixed, "_build_program", build)
    options = dict(device=SimpleNamespace(type="npu", index=0))
    mixed.clear_caches()
    try:
        args = (mixed.dtypes.float32, False, 253952)
        first = mixed._get_compiled_kernel(*args, 1, 0, layout=mixed.Layout.TAIL, **options)
        assert mixed._get_compiled_kernel(*args, 1, 0, layout=mixed.Layout.TAIL, **options) is first
        assert mixed._get_compiled_kernel(*args, 0, 0, layout=mixed.Layout.TAIL, **options) is not first
        assert mixed._get_compiled_kernel(*args, 1, 0x3F800000, layout=mixed.Layout.TAIL, **options) is not first
        assert mixed._get_compiled_kernel(*args, 1, 0, layout=mixed.Layout.WIDE32, **options) is not first
        assert len(built) == 4
        assert built[-1][0].layout is mixed.Layout.WIDE32
        assert mixed.cache_info().maxsize == 64
    finally:
        mixed.clear_caches()


@pytest.mark.parametrize("src", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dst_code", [23, 24])
@pytest.mark.parametrize("algorithm", [0, 1])
def test_finite_boundary_fixture_has_zero_and_nonzero_scales(src, dst_code, algorithm):
    cpu = finite_scale_boundary_input(src, dst_code, algorithm)
    assert torch.isfinite(cpu).all()
    values = cpu.float().numpy().astype(bfloat16 if src == torch.bfloat16 else np.float32)
    dst = "float8_e5m2" if dst_code == 23 else "float8_e4m3fn"
    _, scales = golden.dynamic_mx_quant_golden(values, dst, scale_alg=algorithm)
    # Literal E8M0 codes for zero, neighbors below/above underflow, and 1.0.
    last = 112 if dst_code == 23 else 119
    np.testing.assert_array_equal(scales[:, 0, 0], [0, 0, 0, 0, 1, 1, 2, last])
    np.testing.assert_array_equal(scales[:, 0, 1], np.zeros(8, dtype=np.uint8))


def test_golden_philox_known_zero_vector():
    # Published Philox4x32-10 zero counter / zero key vector, independent of DSL.
    expected = np.array([0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8], dtype=np.uint32)
    np.testing.assert_array_equal(golden._philox4x32_10(0, 0, 0, 0), expected)
    np.testing.assert_array_equal(golden._philox_word_stream(4, quant_length=16), expected)
    np.testing.assert_array_equal(
        golden._random_bits(8, quant_length=16),
        np.array([0x6627, 0xD566, 0xE8D5, 0x27E8, 0xE169, 0x8DE1, 0xC58D, 0x69C5], dtype=np.uint32),
    )


@pytest.mark.parametrize(
    "dst, half_ulp, carry_value, min_half, one_code",
    [
        ("float8_e4m3fn", 1.0625, 1.9375, 2.0**-10, 0x38),
        ("float8_e5m2", 1.125, 1.875, 2.0**-17, 0x3C),
    ],
)
def test_golden_stochastic_carry_and_subnormal(dst, half_ulp, carry_value, min_half, one_code):
    values = np.array(
        [half_ulp, half_ulp, -half_ulp, carry_value, min_half, min_half, -min_half, 2.0**-149], dtype=np.float32
    )
    random = np.array([0x7FFF, 0x8000, 0x8000, 0x8000, 0x7FFF, 0x8000, 0x8000, 0xFFFF], dtype=np.uint32)
    expected = np.array([one_code, one_code + 1, (one_code + 1) | 0x80, 0x40, 0, 1, 0x81, 0], dtype=np.uint8)
    np.testing.assert_array_equal(golden._sr_cast_fp32_to_fp8(values, random, dst), expected)


@pytest.mark.parametrize("dst,max_code", [("float8_e4m3fn", 0x7E), ("float8_e5m2", 0x7B)])
def test_golden_signed_zero_nan_inf_and_saturation(dst, max_code):
    bits = np.array(
        [0, 0x80000000, 0x7FC00000, 0xFFC00001, 0x7F800000, 0xFF800000, 0x7F7FFFFF, 0xFF7FFFFF], dtype=np.uint32
    )
    expected = np.array([0, 0x80, 0x7F, 0x7F, max_code, max_code | 0x80, max_code, max_code | 0x80], dtype=np.uint8)
    np.testing.assert_array_equal(golden._sr_cast_fp32_to_fp8(bits.view(np.float32), 0xFFFF, dst), expected)


@pytest.mark.parametrize("dst,max_value", [("float8_e4m3fn", 448.0), ("float8_e5m2", 57344.0)])
def test_golden_scale_ceil_and_special_bytes(dst, max_value):
    maxima = np.array(
        [
            0,
            max_value / 2,
            max_value,
            np.nextafter(np.float32(max_value), np.float32(np.inf)),
            2.0**-149,
            max_value * 2.0**-127,
            np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    scale, reciprocal = golden._mx_scale_alg1_kernel(maxima, dst)
    np.testing.assert_array_equal(scale, np.array([0, 126, 127, 128, 0, 0, 255, 255], dtype=np.uint8))
    np.testing.assert_array_equal(
        reciprocal, np.array([0, 0x4000, 0x3F80, 0x3F00, 0x7F00, 0x7F00, 0x7F81, 0x7F81], dtype=np.uint16)
    )


@pytest.mark.parametrize("k", [16, 48, 80, 272])
def test_golden_zero_block_and_scale_padding(k):
    x = np.zeros((3, k), dtype=np.float32)
    x[:, 1::2] = -0.0
    y, scale = golden.dynamic_mx_quant_golden(x)
    expected = np.zeros_like(x, dtype=np.uint8)
    expected[:, 1::2] = 0x80
    np.testing.assert_array_equal(y, expected)
    np.testing.assert_array_equal(scale, np.zeros((3, (k + 63) // 64, 2), dtype=np.uint8))


@pytest.mark.parametrize("dst,shift", [("float8_e4m3fn", 8), ("float8_e5m2", 15)])
def test_golden_ocp_exponent_and_underflow(dst, shift):
    bits = np.array(
        [0, 1, (shift << 23) | 0x7FFFFF, (shift + 1) << 23, 0x3F800000, 0x3FFFFFFF, 0x7F7FFFFF, 0x7F800000, 0x7FC00001],
        np.uint32,
    )
    code, recip = golden._mx_scale_alg0_kernel(bits.view(np.float32), dst)
    expected_code = [0, 0, 0, 1, 127 - shift, 127 - shift, 254 - shift, 255, 255]
    expected_recip = [0, 0, 0, 0x7E80, (127 + shift) << 7, (127 + shift) << 7, shift << 7, 0x7F81, 0x7F81]
    np.testing.assert_array_equal(code, np.array(expected_code, np.uint8))
    np.testing.assert_array_equal(recip, np.array(expected_recip, np.uint16))


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_golden_bound_clamps_nonzero_groups_but_preserves_zero_scale(dtype):
    # 1.0 with max_low_bound=Amax(FP8) has E8M0 scale 1.0 (code 127).
    bound = 448.0 if dtype == "float8_e4m3fn" else 57344.0
    x = np.concatenate((np.ones(32, dtype=np.float32), np.zeros(32, dtype=np.float32)))
    x[33::2] = -0.0
    y, scale = golden.dynamic_mx_quant_golden(x, dtype, scale_alg=1, max_low_bound=bound)
    np.testing.assert_array_equal(scale.reshape(-1), np.array([127, 0], dtype=np.uint8))
    np.testing.assert_array_equal(y[:32], np.full(32, 0x38 if dtype == "float8_e4m3fn" else 0x3C, dtype=np.uint8))
    np.testing.assert_array_equal(y[32::2], np.zeros(16, dtype=np.uint8))
    np.testing.assert_array_equal(y[33::2], np.full(16, 0x80, dtype=np.uint8))


def test_sr_public_signature_matches_torch_argument_names_and_kinds():
    # Catch missing/reordered attrs and accidentally exposing launch controls.
    sample = mixed
    assert sample.__all__ == ["dynamic_mx_quant_sr"]
    assert not hasattr(sample, "dynamic_mx_quant_bitreverse_simt")
    assert not hasattr(sample, "dynamic_mx_quant")
    parameters = inspect.signature(mixed.dynamic_mx_quant_sr).parameters
    assert list(parameters) == [
        "input",
        "axis",
        "round_mode",
        "dst_type",
        "block_size",
        "scale_alg",
        "dst_type_max",
        "max_low_bound",
    ]
    assert {name: value.default for name, value in parameters.items() if name != "input"} == {
        "axis": -1,
        "round_mode": "stochastic",
        "dst_type": 24,
        "block_size": 32,
        "scale_alg": 1,
        "dst_type_max": 0.0,
        "max_low_bound": 0.0,
    }
    assert parameters["input"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(p.kind == inspect.Parameter.KEYWORD_ONLY for name, p in parameters.items() if name != "input")


@pytest.mark.parametrize(
    "kwargs,error,attribute",
    [
        ({"axis": True}, TypeError, "axis"),
        ({"axis": 1.0}, TypeError, "axis"),
        ({"axis": -3}, ValueError, "axis"),
        ({"axis": 2}, ValueError, "axis"),
        ({"round_mode": None}, TypeError, "round_mode"),
        ({"round_mode": "rint"}, NotImplementedError, "round_mode"),
        ({"round_mode": "floor"}, NotImplementedError, "round_mode"),
        ({"dst_type": torch.float8_e4m3fn}, TypeError, "dst_type"),
        ({"dst_type": True}, TypeError, "dst_type"),
        ({"dst_type": 24.0}, TypeError, "dst_type"),
        ({"dst_type": 296}, NotImplementedError, "dst_type"),
        ({"dst_type": 35}, NotImplementedError, "dst_type"),
        ({"dst_type": 36}, NotImplementedError, "dst_type"),
        ({"block_size": True}, TypeError, "block_size"),
        ({"block_size": 32.0}, TypeError, "block_size"),
        ({"block_size": 0}, ValueError, "block_size"),
        ({"block_size": 16}, ValueError, "block_size"),
        ({"block_size": 2048}, ValueError, "block_size"),
        ({"block_size": 64}, NotImplementedError, "block_size"),
        ({"scale_alg": True}, TypeError, "scale_alg"),
        ({"scale_alg": 1.0}, TypeError, "scale_alg"),
        ({"scale_alg": 2}, NotImplementedError, "scale_alg"),
        ({"scale_alg": None}, NotImplementedError, "scale_alg"),
        ({"dst_type_max": "0"}, TypeError, "dst_type_max"),
        ({"dst_type_max": True}, TypeError, "dst_type_max"),
        ({"dst_type_max": float("nan")}, ValueError, "dst_type_max"),
        ({"dst_type_max": float("inf")}, ValueError, "dst_type_max"),
        ({"dst_type_max": 6.0}, NotImplementedError, "dst_type_max"),
    ],
)
def test_sr_rejects_unsupported_attributes_before_device_work(kwargs, error, attribute):
    # A CPU tensor makes validation observable without NPU work or mocks.
    # Silently ignoring an attribute would instead reach the NPU input check.
    with pytest.raises(error, match=attribute):
        mixed.dynamic_mx_quant_sr(torch.ones((2, 64)), **kwargs)


@pytest.mark.parametrize("kwargs", [{"block_dim": 1}, {"dst_dtype": torch.float8_e4m3fn}])
def test_sr_does_not_accept_old_public_keywords(kwargs):
    with pytest.raises(TypeError, match=next(iter(kwargs))):
        mixed.dynamic_mx_quant_sr(torch.ones((2, 64)), **kwargs)


def test_sr_attributes_are_keyword_only():
    with pytest.raises(TypeError):
        mixed.dynamic_mx_quant_sr(torch.ones((2, 64)), 24)


@pytest.mark.parametrize("input_value", [None, [1.0] * 64])
def test_sr_rejects_non_tensor_input(input_value):
    with pytest.raises(TypeError, match="input"):
        mixed.dynamic_mx_quant_sr(input_value)


@pytest.mark.parametrize("shape", [(), (0, 64), (2, 0)])
def test_sr_rejects_empty_or_scalar_before_axis_normalization(shape):
    with pytest.raises(ValueError, match="nonempty|dimension"):
        mixed.dynamic_mx_quant_sr(torch.empty(shape))


@pytest.mark.parametrize("boundary", (1024, 2048))
def test_cpu_golden_random_fields_across_batch_boundary(boundary):
    quant_length = boundary + 32
    fields = golden._random_bits(quant_length, quant_length=quant_length)
    expected = []
    for logical in range(boundary - 32, boundary + 32):
        group = logical // 16
        draw, subsequence = divmod(group, 16384)
        word = utils.scalar_philox(draw, subsequence=subsequence)[logical // 4 % 4]
        b3, b2, b1, b0 = int(word).to_bytes(4, "big")
        expected.append(((b3 << 8) | b2, (b0 << 8) | b3, (b1 << 8) | b0, (b2 << 8) | b1)[logical % 4])
    np.testing.assert_array_equal(fields[boundary - 32 : boundary + 32], np.asarray(expected, np.uint32))


@pytest.mark.parametrize("count,want", [(54, 54), (56, 56), (64, 64), (80, 80), (12, 12), (1, 1)])
def test_auto_budget_uses_effective_vector_count(monkeypatch, count, want):
    resolve = _resolver()
    x = _platform(monkeypatch, [count])
    assert resolve(x, None) == want


def test_auto_budget_refreshes_stream_quota(monkeypatch):
    resolve = _resolver()
    x = _platform(monkeypatch, [54, 56, 64, 80])
    assert resolve(x, None) == 54
    assert resolve(x, None) == 56
    assert resolve(x, None) == 64
    assert resolve(x, None) == 80


@pytest.mark.parametrize("count", [0, -1, True, 1.5, None, "64"])
def test_auto_budget_rejects_invalid_platform_count(monkeypatch, count):
    resolve = _resolver()
    x = _platform(monkeypatch, [count])
    with pytest.raises(RuntimeError, match="vector_core_num"):
        resolve(x, None)


def test_auto_budget_does_not_hide_query_failure(monkeypatch):
    resolve = _resolver()
    x = _platform(monkeypatch, [64])

    def unavailable(**kwargs):
        raise RuntimeError("platform query unavailable")

    monkeypatch.setattr(cannbotdsl, "get_platform_info", unavailable)
    with pytest.raises(RuntimeError, match="platform query unavailable"):
        resolve(x, None)


@pytest.mark.parametrize("blocks", [1, 54, 56, 64, 80])
def test_explicit_budget_does_not_query_platform(monkeypatch, blocks):
    resolve = _resolver()

    def unexpected(*args, **kwargs):
        raise AssertionError("explicit block_dim must bypass automatic query")

    monkeypatch.setattr(cannbotdsl, "get_platform_info", unexpected)
    monkeypatch.setattr(torch.npu, "current_stream", unexpected)
    assert resolve(SimpleNamespace(device=None), blocks) == blocks


@pytest.mark.parametrize(
    "shape,want_x,want_scale",
    [
        ((3, 64), (1, 192), (1, 6)),
        ((2, 3, 128), (1, 768), (1, 24)),
        ((65, 64), (1, 4160), (1, 130)),
        ((3, 32), (1, 96), (1, 6)),
        ((2, 3, 32), (1, 192), (1, 12)),
        ((65, 32), (1, 2080), (1, 130)),
        ((3, 48), (3, 48), (3, 2)),
        ((3, 80), (3, 80), (3, 4)),
        ((3, 16), (3, 16), (3, 2)),
    ],
)
def test_kernel_views_preserve_storage_order_and_padding(shape, want_x, want_scale):
    sample = mixed
    assert callable(getattr(sample, "_kernel_views", None)), "aligned-row carrier batching missing"
    size = torch.tensor(shape).prod().item()
    # Nonzero storage offset must survive the view operation.
    x = torch.arange(size + 7, dtype=torch.float32)[7:].reshape(shape)
    y = torch.arange(size, dtype=torch.int64).to(torch.uint8).reshape(shape)
    scales = torch.zeros((*shape[:-1], (shape[-1] + 63) // 64, 2), dtype=torch.uint8)
    plan = sample._make_execution_plan(x, x.ndim - 1, block_dim=1)
    views = sample._kernel_views(x, y, scales, plan)
    for view, original, want in zip(views, (x, y, scales), (want_x, want_x, want_scale)):
        assert tuple(view.shape) == want
        assert view.data_ptr() == original.data_ptr()
        assert view.storage_offset() == original.storage_offset()
        assert view.is_contiguous()
        torch.testing.assert_close(view.reshape(-1), original.reshape(-1))


@pytest.mark.parametrize(
    "dst,blocks,error,match",
    [
        ("unsupported", None, TypeError, "dst"),
        ("e4", 0, ValueError, "block"),
        ("e4", 2**32, ValueError, "block"),
        ("e4", True, ValueError, "block"),
    ],
)
def test_shared_launch_rejects_invalid_dtype_or_budget_before_device(dst, blocks, error, match):
    import torch

    from ._dynamic_mx_quant_sr_rng import launch

    dtype = torch.float8_e4m3fn if dst == "e4" else torch.float16
    with pytest.raises(error, match=match):
        launch(torch.ones((1, 64)), dtype, block_dim=blocks)


@pytest.mark.parametrize("blocks", [65, 80, 2**32 - 1])
def test_shared_launch_accepts_uint32_budget_before_device_validation(blocks):
    with pytest.raises(ValueError, match="Ascend NPU"):
        utils.launch(torch.ones((1, 64)), torch.float8_e4m3fn, block_dim=blocks)


@pytest.mark.parametrize("count,columns,want", [(32, 32, 1), (1024, 64, 1), (1088, 64, 2), (2048, 64, 2)])
def test_coalesced_aiv_cap_at_philox_batch_boundaries(monkeypatch, count, columns, want):
    x = _platform(monkeypatch, [64])
    x.shape = (count // columns, columns)
    x.numel = lambda: count
    assert mixed._resolve_block_dim(x, None) == want


@pytest.mark.parametrize("blocks", [0, 2**32, -1, 1.5, True])
def test_private_budget_rejected_before_device_setup(blocks, monkeypatch):
    # Tensor-domain validation is covered separately; isolate this Host guard.
    monkeypatch.setattr(mixed, "_validate_input", lambda *args: None)
    monkeypatch.setattr(torch.npu, "device", _unexpected_device_work)
    with pytest.raises((TypeError, ValueError), match="block_dim"):
        mixed._dynamic_mx_quant_sr_impl(torch.ones((2, 64)), block_dim=blocks)


@pytest.mark.parametrize("src", INPUT_DTYPES)
@pytest.mark.parametrize("fused", [False, True])
def test_native_tiling_reserves_simt_ub_and_validates_capacity(src, fused):
    dtype = dtypes.float32 if src == torch.float32 else dtypes.bfloat16
    for ub in (253952, 200704, 65536):
        tiling = mixed._make_tiling(dtype, ub, fused)
        assert tiling.tile_elements >= 1024 and tiling.tile_elements % 1024 == 0
        assert tiling.scale_capacity % 64 == 0
        assert tiling.buffer_bytes <= ub - 40 * 1024
        with pytest.raises(FrozenInstanceError):
            tiling.tile_elements = 1
    for ub in (0, 40 * 1024, 40 * 1024 + 1, True, 253952.0):
        with pytest.raises(ValueError, match="UB"):
            mixed._make_tiling(dtype, ub, fused)


@pytest.mark.parametrize(
    "src,shape,axis,want",
    [
        (torch.float32, (32, 4096, 16), 1, "wide16_non_tail"),
        (torch.float32, (64, 4096, 16), 1, "transpose"),
        (torch.bfloat16, (2, 4096, 16), 1, "wide16_non_tail"),
        (torch.bfloat16, (16, 8192, 16), 1, "transpose"),
        (torch.bfloat16, (32, 512, 8), 1, "wide8_non_tail"),
        (torch.bfloat16, (33, 512, 8), 1, "transpose"),
        (torch.float32, (2, 512, 2), 1, "non_tail_fused"),
        (torch.float32, (2, 576, 2), 1, "non_tail"),
    ],
)
def test_layout_policy_uses_current_crossover_not_archival_benchmark(src, shape, axis, want):
    plan = mixed._make_execution_plan(torch.empty(shape, dtype=src), axis, block_dim=64)
    assert plan.layout.value == want
    assert plan.blocks <= 64 and plan.compile_layout in mixed._LAYOUT_REGISTRY


@pytest.mark.parametrize("dst", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("family", ["exponent_edges", "bf16_all_bits", "random_bits"])
def test_vectorized_reference_preserves_independent_scalar_algorithm19(dst, family):
    values, fields = utils.q16_inputs(family)
    scalar = golden._sr_cast_fp32_to_fp8(values, fields, dst)
    vector = golden._sr_cast_fp32_to_fp8_vectorized(values, fields, dst)
    np.testing.assert_array_equal(vector, scalar)
    # Broadcast, empty, non-contiguous inputs and partial chunks are distinct
    # shape/stride contracts, not just one large flat array.
    for count in (0, 1, 17, 65):
        subset = values.reshape(-1)[: 2 * count : 2]
        np.testing.assert_array_equal(
            golden._sr_cast_fp32_to_fp8_vectorized(subset, 0xABCD8000, dst),
            golden._sr_cast_fp32_to_fp8(subset, 0xABCD8000, dst),
        )


def test_large_bf16_e4_inputs_always_have_four_panel_route_with_supported_aiv_budgets():
    # The removed single-panel flag started at 8 Mi elements. For up to 128
    # AIVs, that already supplies the four-panel minimum of 64 Ki per AIV.
    for blocks, algorithm, numel in itertools.product(range(1, 129), (0, 1), (8 * 1024 * 1024, 32 * 1024 * 1024)):
        assert mixed._select_bf16_four_panel_policy(
            numel, dtypes.bfloat16, torch.float8_e4m3fn, algorithm, 0, mixed.Layout.TAIL, 256, blocks
        )


def test_fp32_mapping_policy_and_reference():
    op = utils.sample()
    assert op.CompileKey.__dataclass_fields__["rng_abi_version"].default == 60
    assert op._select_fp32_eight_policy(8191 * 256, op.dtypes.float32, op.Layout.TAIL, 256) is False
    assert op._select_fp32_eight_policy(8192 * 256, op.dtypes.float32, op.Layout.TAIL, 256) is True
    assert op._select_fp32_eight_policy(8192 * 272, op.dtypes.float32, op.Layout.TAIL, 272) is False
    assert op._select_fp32_eight_policy(8192 * 256, op.dtypes.bfloat16, op.Layout.TAIL, 256) is False
    # The reference uses eight ordered adjacent-byte pairs from one Philox word.
    pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 0), (0, 3)]
    word = int(
        utils.golden._philox_blocks(np.array([0], dtype=np.uint64), quant_length=256, seed=0, block_offset=0).reshape(
            -1
        )[0]
    )
    b = [(word >> (8 * i)) & 255 for i in range(4)]
    expected = np.array([b[x] | b[y] << 8 for x, y in pairs], dtype=np.uint32)
    np.testing.assert_array_equal(
        utils.golden._random_bits_fp32_eight(8, quant_length=256, seed=0, block_offset=0), expected
    )


def test_alias_cache_and_scope():
    op = utils.sample()
    assert op.CompileKey.__dataclass_fields__["rng_abi_version"].default == 60
    key = _cache_key(input_dtype="fp32", fp32_eight_mapping=True)
    assert dataclasses.replace(key, output_rng_alias=True) != key
    with pytest.raises(ValueError, match="mapped FP32 tail"):
        op.capture_compile_context(
            "bf16", False, 253952, 0, 0, op.Layout.TAIL, torch.device("npu:0"), output_rng_alias=True
        )
    with pytest.raises(ValueError, match="FP32 tail"):
        op._make_tiling(op.dtypes.bfloat16, 253952, output_rng_alias=True)


@pytest.mark.parametrize(
    "shape,src,dst,algorithm,enabled",
    [
        pytest.param(
            (8192, 256),
            torch.float32,
            24,
            0,
            dict(fp32_eight_mapping=True, output_rng_alias=True),
            id="fp32-alias",
        ),
        pytest.param((8191, 256), torch.float32, 24, 0, {}, id="fp32-below-mapping-boundary"),
        pytest.param(
            (32768, 256),
            torch.bfloat16,
            24,
            0,
            dict(bf16_four_panel=True),
            id="bf16-four-panel-large",
        ),
        pytest.param(
            (32767, 256),
            torch.bfloat16,
            24,
            0,
            dict(bf16_four_panel=True),
            id="bf16-four-panel-nearby",
        ),
        pytest.param(
            (2048, 16384),
            torch.bfloat16,
            24,
            0,
            dict(bf16_compact_mode=36, bf16_compact_four_panel=True, bf16_tail_simd=True),
            id="bf16-compact-simd-tail",
        ),
        pytest.param(
            (16384, 16384),
            torch.bfloat16,
            24,
            0,
            dict(bf16_compact_mode=36, bf16_compact_four_panel=True),
            id="bf16-compact-without-simd-tail",
        ),
        pytest.param(
            (512, 7168),
            torch.bfloat16,
            24,
            1,
            dict(bf16_four_panel=True, bf16_full_tiles=True),
            id="bf16-e4-alg1-complete-tiles",
        ),
        pytest.param(
            (512, 7168),
            torch.bfloat16,
            24,
            0,
            dict(bf16_four_panel=True, bf16_full_tiles=True),
            id="bf16-e4-alg0-complete-tiles",
        ),
        pytest.param(
            (512, 7168),
            torch.bfloat16,
            23,
            0,
            dict(bf16_four_panel=True, bf16_full_tiles=True),
            id="bf16-e5-alg0-complete-tiles",
        ),
        pytest.param(
            (512, 7168),
            torch.bfloat16,
            23,
            1,
            dict(bf16_four_panel=True, bf16_full_tiles=True),
            id="bf16-e5-alg1-complete-tiles",
        ),
        pytest.param((1007, 4096), torch.bfloat16, 23, 0, {}, id="bf16-e5-below-four-panel-boundary"),
        pytest.param(
            (1008, 4096),
            torch.bfloat16,
            23,
            0,
            dict(bf16_four_panel=True),
            id="bf16-e5-four-panel",
        ),
    ],
)
def test_prepare_launch_forwards_selected_optimizations(monkeypatch, shape, src, dst, algorithm, enabled):
    # Exercise real planning and policy selection; replace only device setup and AOT builds.
    # Literal expectations also cover fallback sides, without reusing policy functions as an oracle.
    key = _compile_context(monkeypatch).key
    identity = mixed.RuntimeCompileIdentity(
        **{field.name: getattr(key, field.name) for field in dataclasses.fields(mixed.RuntimeCompileIdentity)}
    )
    monkeypatch.setattr(mixed, "_capture_runtime_identity", lambda device: (identity, None))
    monkeypatch.setattr(torch.npu, "device", lambda *args: nullcontext())
    monkeypatch.setattr(mixed, "get_mem_size", lambda name: 253952)
    monkeypatch.setattr(mixed, "_resolve_block_dim", lambda *args, **kwargs: 56)
    calls = []
    explicit, scalar = _fake_program(), _fake_program()
    framework = (_fake_program(), _fake_program())

    def capture(name, result):
        def build(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return build

    monkeypatch.setattr(mixed, "_get_compiled_kernel", capture("explicit", explicit))
    monkeypatch.setattr(mixed, "_get_compiled_tail_scalar_kernel", capture("scalar", scalar))
    monkeypatch.setattr(mixed, "_get_compiled_framework_programs", capture("framework", framework))
    x = torch.empty(shape, dtype=src, device="meta")
    dst_dtype = torch.float8_e5m2 if dst == 23 else torch.float8_e4m3fn
    prepared = mixed._prepare_launch(x, dst_dtype, 1, algorithm, 0)
    expected = dict.fromkeys(
        (
            "fp32_eight_mapping",
            "output_rng_alias",
            "bf16_compact_four_panel",
            "bf16_tail_simd",
            "bf16_four_panel",
            "bf16_full_tiles",
        ),
        False,
    )
    expected["bf16_compact_mode"] = 0
    expected.update(enabled)
    assert prepared.plan.layout is mixed.Layout.TAIL
    assert prepared.program is explicit and prepared.scalar_program is scalar
    assert prepared.framework_programs is framework
    assert [name for name, _, _ in calls] == ["explicit", "scalar", "framework"]
    dtype = dtypes.bfloat16 if src == torch.bfloat16 else dtypes.float32
    for name, args, options in calls:
        assert "bf16_half_q7_mapping" not in options
        assert args[:5] == (dtype, dst == 23, 253952, algorithm, 0), name
        assert {key: options[key] for key in expected} == expected, name
        assert {key: getattr(options["_context"].key, key) for key in expected} == expected, name
        if name != "scalar":
            assert options["layout"] is mixed.Layout.TAIL and options["device"] == x.device
        else:
            assert args[5] == x.device


@pytest.mark.parametrize("blocks", [28, 32, 56, 64])
def test_scoped_bf16_combo_preserves_dynamic_route_and_scope(blocks):
    op = utils.sample()
    select = op._select_bf16_compact_four_panel_policy
    args = [24 * 1024 * 1024, op.dtypes.bfloat16, torch.float8_e4m3fn, 0, 0, op.Layout.TAIL, 12288, blocks, 253952]
    parent = op._make_tiling(op.dtypes.bfloat16, args[-1], tail_axis=True)
    compact = op._make_tiling(op.dtypes.bfloat16, args[-1], tail_axis=True, bf16_compact_mode=36)
    preserves_route = not (
        args[0] > blocks * parent.tile_elements
        and args[0] % (blocks * parent.tile_elements) == 0
        and args[0] % (blocks * compact.tile_elements) != 0
    )
    assert select(*args) is preserves_route
    # Only legality and insufficient work/UB exclude this route; K is runtime.
    shorter_axis = args.copy()
    shorter_axis[6] = 8192
    assert select(*shorter_axis) is preserves_route
    for index, value in (
        (0, 24 * 1024 * 1024 - 256),
        (1, op.dtypes.float32),
        (2, torch.float8_e5m2),
        (3, 1),
        (4, 1),
        (5, op.Layout.TRANSPOSE),
        (6, 12289),
        (8, 65536),
    ):
        other = args.copy()
        other[index] = value
        assert select(*other) is False


def test_scoped_bf16_combo_has_independent_cache_key():
    key = _cache_key(input_dtype="bf16", bf16_compact_mode=36)
    assert dataclasses.replace(key, bf16_compact_four_panel=True) != key
    # This is an implementation revision of the existing Philox mapping.
    assert key.rng_abi_version == 60


def test_policy_uses_reported_block_budget():
    op = utils.sample()
    assert op.CompileKey.__dataclass_fields__["rng_abi_version"].default == 60
    assert "bf16_compact_mode" in op.CompileKey.__dataclass_fields__
    for blocks in (28, 56, 64):
        for rows, length in ((2048, 16384), (8192, 7168), (8192, 8192)):
            mode = op._select_bf16_compact_tile_policy(
                rows * length, op.dtypes.bfloat16, torch.float8_e4m3fn, op.Layout.TAIL, length, blocks, 253952
            )
            assert mode in (0, 36)
            assert (
                op._select_bf16_compact_tile_policy(
                    rows * length, op.dtypes.bfloat16, torch.float8_e5m2, op.Layout.TAIL, length, blocks, 253952
                )
                == 0
            )
    assert (
        op._select_bf16_compact_tile_policy(
            2048 * 16384, op.dtypes.bfloat16, torch.float8_e4m3fn, op.Layout.TAIL, 16384, 56, 253952
        )
        == 36
    )
    assert (
        op._select_bf16_compact_tile_policy(
            8192 * 7168, op.dtypes.bfloat16, torch.float8_e4m3fn, op.Layout.TAIL, 7168, 56, 253952
        )
        == 0
    )


@pytest.mark.parametrize("numel", [16, 32, 16 * (THREADS - 63), 16 * THREADS, 17749 * 144, 8192 * 16384, 2**64 - 16])
def test_linear_reservation_covers_padding(numel):
    # Includes at most63 unused groups generated by a 64-lane SIMD batch.
    last_draw = (numel // 16 + 62) // THREADS
    assert rng._rng_increment(numel) == 4 * (last_draw + 1)
    assert rng._rng_increment(numel) % 4 == 0
    assert last_draw < rng._rng_increment(numel) // 4


@pytest.mark.parametrize("seed", [0, 17, 2**32, 2**63 + 7, 2**64 - 1])
@pytest.mark.parametrize("offset", [0, 4, 4 * (2**32 - 1), 2**63, 2**64 - 8])
def test_linear_words_match_independent_cann_style_mapping(seed, offset):
    groups = [0, 1, THREADS - 1, THREADS, THREADS + 1, 2**32 - 1, 2**32, 2**60 - 1]
    expected = [scalar_philox(offset // 4 + group // THREADS, group % THREADS, seed) for group in groups]
    for columns in [16, 80, 128, 7168, 16 * (2**32 + 13)]:
        np.testing.assert_array_equal(
            golden._philox_blocks(groups, quant_length=columns, seed=seed, block_offset=offset),
            np.asarray(expected, np.uint32),
        )


@pytest.mark.parametrize("offset", [1, 2, 3, 2**64 - 1])
def test_linear_offset_requires_word_alignment(offset):
    with pytest.raises(ValueError):
        rng._validate_rng_args(0, offset)
    with pytest.raises(ValueError):
        golden._philox_blocks(0, quant_length=16, block_offset=offset)


def test_linear_calls_have_disjoint_counter_intervals():
    offset = 4 * (2**32 - 1)
    previous_end = offset // 4
    for count in [16, 17749 * 144, 8192 * 16384, 32]:
        increment = rng._rng_increment(count)
        last = offset // 4 + (count // 16 + 62) // THREADS
        assert offset // 4 >= previous_end
        assert last < (offset + increment) // 4
        previous_end = (offset + increment) // 4
        offset += increment


def test_linear_group_and_field_addresses_do_not_alias():
    groups = np.arange(THREADS * 3 + 7, dtype=np.uint64)
    addresses = {(int(g) // THREADS, int(g) % THREADS) for g in groups}
    assert len(addresses) == len(groups)
    # Distinct H/rotate-low/L/rotate-high slots; full16 fields are related.
    slots = {(word, field) for word in range(4) for field in range(4)}
    assert len(slots) == 16


def test_linear_field_transforms_preserve_uniform_16bit_probability():
    fields = np.arange(65536, dtype=np.uint32)
    for value, dst, upper_count in [
        (1.0625, "float8_e4m3fn", 32768),
        (1.125, "float8_e5m2", 32768),
        (2**-22, "float8_e5m2", 1024),
    ]:
        values = np.full(65536, value, np.float32)
        result = golden._sr_cast_fp32_to_fp8_vectorized(values, fields, dst)
        codes, counts = np.unique(result, return_counts=True)
        assert len(codes) == 2 and int(counts[-1]) == upper_count


def test_linear_philox_known_random123_zero_vector():
    assert scalar_philox(0, 0, 0) == [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]


def test_rotate8_fields_have_uniform16_marginals_and_disjoint_top8_bytes(monkeypatch):
    # Word byte positions independently expose every possible value of each field.
    x = np.arange(65536, dtype=np.uint32)
    for index, words in enumerate(
        (x << np.uint32(16), (x >> np.uint32(8)) | ((x & np.uint32(255)) << np.uint32(24)), x, x << np.uint32(8))
    ):
        monkeypatch.setattr(golden, "_philox_word_stream", lambda *args, words=words, **kwargs: words)
        actual = golden._random_bits(words.size * 4, quant_length=16).reshape(-1, 4)
        np.testing.assert_array_equal(actual[:, index], x)
    basis = np.left_shift(np.uint32(1), np.arange(32, dtype=np.uint32))
    monkeypatch.setattr(golden, "_philox_word_stream", lambda *args, **kwargs: basis)
    fields = golden._random_bits(128, quant_length=16).reshape(-1, 4) >> np.uint32(8)
    packed = sum(fields[:, j].astype(np.uint64) << np.uint64(8 * j) for j in range(4))
    np.testing.assert_array_equal(np.sort(packed), np.left_shift(np.uint64(1), np.arange(32, dtype=np.uint64)))


@pytest.mark.parametrize(
    "shape,axis,want",
    [
        ((511, 4096), 1, False),
        ((512, 4096), 1, True),
        ((513, 4096), 1, True),
        ((8192, 272), 1, False),
        ((683, 1024, 3), 1, True),
        ((512, 4096, 2), 1, False),
        ((512, 1024, 8), 1, False),
    ],
)
def test_reference_policy_boundaries(shape, axis, want):
    assert golden._uses_fp32_eight_mapping(shape, axis) == want


@pytest.mark.parametrize("captured,scalar_tail", [(False, False), (False, True), (True, False), (True, True)])
def test_public_launch_never_materializes_rng(monkeypatch, captured, scalar_tail):
    """Exercise the real host dispatch without a device or helper kernel."""
    op = mixed
    calls, reservations = [], []

    class Program:
        so_path = "already_compiled.so"
        closed = False

        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            calls.append((self.name, args))

    def forbidden(*args, **kwargs):
        raise AssertionError("public launch must not materialize RNG state")

    seed, offset = 0xFEDCBA9876543210, 4 * (2**32 - 1)
    carrier = (
        (True, False, torch.tensor([-7]), torch.tensor([offset]), 12) if captured else (False, False, seed, offset, 0)
    )

    def reserve(device, increment):
        reservations.append((device, increment))
        return carrier

    device = torch.device("npu", 0)
    arguments = (SimpleNamespace(device=device), "output", "scales", 512, 1)
    prepared = op._PreparedLaunch(
        SimpleNamespace(output_shape=(1, 512)),
        Program("explicit"),
        arguments,
        SimpleNamespace(),
        SimpleNamespace(),
        torch.float8_e4m3fn,
        Program("tail_scalar") if scalar_tail else None,
    )
    prepared = replace(prepared, framework_programs=(Program("eager"), Program("graph")))
    monkeypatch.setattr(op, "prepare_unpack", forbidden, raising=False)
    monkeypatch.setattr(op, "unpack_rng_state", forbidden, raising=False)
    monkeypatch.setattr(op, "_reserve_rng_state", forbidden, raising=False)
    monkeypatch.setattr(op._framework_rng, "_reserve_rng_carrier", reserve)
    monkeypatch.setattr(torch.npu, "device", lambda *args: nullcontext())
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: captured)
    prepared.launch()
    assert reservations == [(device, op._rng_increment(512))]
    assert len(calls) == 1
    name, values = calls[0]
    assert values[: len(arguments)] == arguments
    if captured:
        assert name == "graph"
        assert values[-3] is carrier[2] and values[-2] is carrier[3] and values[-1] == 12
    else:
        assert name == ("tail_scalar" if scalar_tail else "eager")
        assert values[-2:] == (op._tail_rng_signed64(seed), offset)


def test_returned_outputs_retain_framework_carrier_owners():
    op = mixed
    encoded = torch.empty((1, 512), dtype=torch.uint8)
    scales = torch.empty((1, 8, 2), dtype=torch.uint8)
    owners = (object(), (True, False, torch.tensor([17]), torch.tensor([4]), 8))
    encoded._sr_rng_owners = owners
    prepared = op._PreparedLaunch(
        SimpleNamespace(layout=op.Layout.TAIL),
        object(),
        (),
        encoded,
        scales,
        torch.float8_e4m3fn,
    )
    outputs = prepared.outputs()
    assert all(getattr(t, "_sr_rng_owners", None) is owners for t in outputs)


def test_public_carrier_secondary_capture_rejects_before_program(monkeypatch):
    op = mixed
    calls = []

    class Program:
        so_path = "compiled.so"
        closed = False

        def __call__(self, *args):
            calls.append(args)

    prepared = op._PreparedLaunch(
        SimpleNamespace(output_shape=(1, 512)),
        Program(),
        (SimpleNamespace(device=torch.device("npu", 0)),),
        SimpleNamespace(),
        SimpleNamespace(),
        torch.float8_e4m3fn,
        framework_programs=(Program(), Program()),
    )
    monkeypatch.setattr(torch.npu, "device", lambda *args: nullcontext())
    monkeypatch.setattr(op._framework_rng, "_reserve_rng_carrier", lambda *args: (True, True, object(), object(), 4))
    with pytest.raises(NotImplementedError, match="secondary-stream"):
        prepared.launch()
    assert not calls


@pytest.mark.parametrize("layout", [layout for layout in mixed.Layout if layout is not mixed.Layout.TRANSPOSE])
def test_framework_rng_program_cache_reuses_entries_and_clears(monkeypatch, layout):
    _compile_context(monkeypatch)
    for name, capacity in (
        ("_PROGRAM_CACHE", 64),
        ("_TAIL_SCALAR_PROGRAM_CACHE", 64),
        ("_FRAMEWORK_PROGRAM_CACHE", 128),
    ):
        monkeypatch.setattr(mixed, name, mixed._ProgramCache(capacity), raising=False)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    builds = []

    def compile_aot(entry, *specs):
        builds.append((entry, specs))
        return _fake_program()

    monkeypatch.setattr(mixed.cannbotdsl, "compile", compile_aot)
    arguments = (dtypes.float32, False, 253952, 1, 0)
    device = SimpleNamespace(type="npu", index=0)
    first = mixed._get_compiled_framework_programs(*arguments, layout=layout, device=device)
    second = mixed._get_compiled_framework_programs(*arguments, layout=layout, device=device)
    assert all(a is b for a, b in zip(first, second)) and len(builds) == 2
    eager_specs, graph_specs = builds[0][1], builds[1][1]
    assert eager_specs[-2:] == (dtypes.int64, dtypes.int64)
    assert all(isinstance(spec, mixed.TensorSpec) and spec.shape == (1,) for spec in graph_specs[-3:-1])
    assert graph_specs[-1] == dtypes.int64
    other = mixed._get_compiled_framework_programs(
        *arguments, layout=layout, device=SimpleNamespace(type="npu", index=1)
    )
    assert all(a is not b for a, b in zip(first, other)) and len(builds) == 4
    mixed.clear_caches()
    assert all(not program.closed for program in first)
    rebuilt = mixed._get_compiled_framework_programs(*arguments, layout=layout, device=device)
    assert all(a is not b for a, b in zip(first, rebuilt)) and len(builds) == 6


@pytest.mark.parametrize("e5,algorithm", [(False, 0), (False, 1), (True, 0), (True, 1)])
def test_full_tile_resource_budget_includes_reciprocals_and_dense_flags(e5, algorithm):
    op = mixed
    # 32K BF16 tiles: E4/alg0 has dense flags; all four variants store 2 KiB reciprocals.
    required_bytes = 246048 if not e5 and algorithm == 0 else 249856
    # Admission also requires the ordinary tiling to fit a complete 32K tile.
    route_minimum_bytes = 247808 if not e5 and algorithm == 0 else 249856
    elements = 56 * 32768
    plan = SimpleNamespace(layout=op.Layout.TAIL, quant_length=256, blocks=56, kernel_shape=(1, elements))
    target = torch.float8_e5m2 if e5 else torch.float8_e4m3fn
    for ub in (246047, 246048, 247807, 247808, 249855, 249856, 253952):
        admitted = op._select_bf16_full_tiles_policy(
            elements, op.dtypes.bfloat16, target, algorithm, 0, plan, ub, 0, True
        )
        assert admitted is (ub >= route_minimum_bytes)
        if admitted:
            tiling = op._make_tiling(op.dtypes.bfloat16, ub, bf16_full_tiles=True, e5m2=e5, scale_alg=algorithm)
            assert tiling.tile_elements == 32768
            assert tiling.buffer_bytes + op.SIMT_UB_RESERVE == required_bytes


@pytest.mark.parametrize("blocks", [54, 56, 64, 80])
@pytest.mark.parametrize("dst,algorithm", [(24, 0), (24, 1), (23, 0), (23, 1)])
def test_lower_wave_full_tiles_follow_aiv_work_and_reject_partial_waves(blocks, dst, algorithm):
    op = mixed
    target = torch.float8_e5m2 if dst == 23 else torch.float8_e4m3fn
    for waves in (1, 2, 3):
        elements = blocks * 32768 * waves
        plan = SimpleNamespace(layout=op.Layout.TAIL, quant_length=512, blocks=blocks, kernel_shape=(1, elements))
        args = (op.dtypes.bfloat16, target, algorithm, 0, plan, 253952, 0)
        assert op._select_bf16_lower_wave_four_panel_policy(elements, *args) is (waves <= 2)
        plan.kernel_shape = (1, elements + 256)
        assert not op._select_bf16_lower_wave_four_panel_policy(elements + 256, *args)


def test_bf16_four_panel_uses_work_per_aiv_and_legal_attributes():
    bf, tail = mixed.dtypes.bfloat16, mixed.Layout.TAIL
    e4, e5 = torch.float8_e4m3fn, torch.float8_e5m2
    select = mixed._select_bf16_four_panel_policy
    for blocks in (28, 32, 56, 64):
        for dst in (e4, e5):
            for alg in (0, 1):
                work_per_aiv = 72 * 1024 if dst == e5 and alg == 0 else 64 * 1024
                for length in (4096, 16384):
                    args = (bf, dst, alg, 0, tail, length, blocks)
                    assert not select(blocks * work_per_aiv - 256, *args)
                    assert select(blocks * work_per_aiv, *args)
                    assert select(blocks * (work_per_aiv + 1024), *args)

    work = 56 * 128 * 1024
    assert not select(work, mixed.dtypes.float32, e4, 0, 0, tail, 4096, 56)
    assert not select(work, bf, e5, 0, 1, tail, 4096, 56)
    assert not select(work, bf, e5, 0, 0, mixed.Layout.TRANSPOSE, 4096, 56)
    assert not select(work, bf, e5, 0, 0, tail, 4224, 56)


@pytest.mark.parametrize("blocks", [54, 56, 64, 80])
@pytest.mark.parametrize("waves", [1, 2, 3, 8, 32])
def test_full_tile_route_tracks_complete_aiv_waves(blocks, waves):
    op = mixed
    elements = blocks * 32768 * waves
    plan = SimpleNamespace(layout=op.Layout.TAIL, quant_length=256, blocks=blocks, kernel_shape=(1, elements))
    args = (elements, op.dtypes.bfloat16, torch.float8_e4m3fn, 0, 0, plan, 253952, 0, True)
    assert op._select_bf16_full_tiles_policy(*args)
    plan.kernel_shape = (1, elements + 256)
    assert not op._select_bf16_full_tiles_policy(elements + 256, *args[1:])


@pytest.mark.parametrize("axis_length", [256, 1536, 7168, 8192, 16384])
def test_full_tile_route_does_not_match_axis_length(axis_length):
    op = mixed
    elements = 56 * 32768 * 24
    plan = SimpleNamespace(layout=op.Layout.TAIL, quant_length=axis_length, blocks=56, kernel_shape=(1, elements))
    assert op._select_bf16_full_tiles_policy(
        elements, op.dtypes.bfloat16, torch.float8_e4m3fn, 0, 0, plan, 253952, 0, True
    )


@pytest.mark.parametrize(
    "override",
    [
        {"dtype": "fp32"},
        {"alg": 2},
        {"bound": 1},
        {"compact": 36},
        {"four": False},
        {"ub": 200000},
        {"axis_length": 64},
        {"non_tail": True},
    ],
)
def test_full_tile_route_preserves_fallback_categories(override):
    op = mixed
    values = dict(
        dtype="bf16", dst=24, alg=0, bound=0, compact=0, four=True, ub=253952, axis_length=7168, non_tail=False
    )
    values.update(override)
    elements = 56 * 32768 * 8
    plan = SimpleNamespace(
        layout=op.Layout.TRANSPOSE if values["non_tail"] else op.Layout.TAIL,
        quant_length=values["axis_length"],
        blocks=56,
        kernel_shape=(1, elements),
    )
    assert not op._select_bf16_full_tiles_policy(
        elements,
        op.dtypes.float32 if values["dtype"] == "fp32" else op.dtypes.bfloat16,
        torch.float8_e5m2 if values["dst"] == 23 else torch.float8_e4m3fn,
        values["alg"],
        values["bound"],
        plan,
        values["ub"],
        values["compact"],
        values["four"],
    )


def test_full_tile_category_is_in_compile_cache_key(monkeypatch):
    op = mixed
    identity = op.RuntimeCompileIdentity("dav-3510", "npu", 0, "test", "native", "compiler", 1, "format", "", b"{}")
    monkeypatch.setattr(op, "_capture_runtime_identity", lambda device: (identity, None))
    args = ("bf16", False, 253952, 0, 0, op.Layout.TAIL, torch.device("npu", 0))
    generic = op.capture_compile_context(*args, bf16_four_panel=True)
    complete = op.capture_compile_context(*args, bf16_four_panel=True, bf16_full_tiles=True)
    assert generic.key != complete.key
    assert not generic.key.bf16_full_tiles and complete.key.bf16_full_tiles


def test_bf16_compact_four_panel_uses_work_and_ub_capacity():
    bf, e4, e5, tail = mixed.dtypes.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, mixed.Layout.TAIL
    select = mixed._select_bf16_compact_four_panel_policy
    ub = 253952
    parent = mixed._make_tiling(bf, ub, tail_axis=True)
    compact = mixed._make_tiling(bf, ub, tail_axis=True, bf16_compact_mode=36)
    for blocks in (28, 32, 56, 64):
        work = max(24 * 1024 * 1024, blocks * 384 * 1024)
        no_uneven_tail = not (
            work > blocks * parent.tile_elements
            and work % (blocks * parent.tile_elements) == 0
            and work % (blocks * compact.tile_elements) != 0
        )
        expected = compact.tile_elements == 36 * 1024 and no_uneven_tail
        for length in (8192, 12288, 16384):
            assert select(work, bf, e4, 0, 0, tail, length, blocks, ub) is expected
            assert not select(work - 256, bf, e4, 0, 0, tail, length, blocks, ub)
        assert not select(work, bf, e5, 0, 0, tail, 12288, blocks, ub)
        assert not select(work, bf, e4, 1, 0, tail, 12288, blocks, ub)
        assert not select(work, bf, e4, 0, 1, tail, 12288, blocks, ub)
        assert not select(work, bf, e4, 0, 0, mixed.Layout.TRANSPOSE, 12288, blocks, ub)
        assert not select(work, bf, e4, 0, 0, tail, 12289, blocks, ub)


@pytest.mark.parametrize("count", [54, 56, 64, 80])
def test_effective_aiv_quota_is_not_hard_capped(count, quota):
    quota(count)
    assert mixed._resolve_block_dim(TensorGeometry((64, 80)), None) == count


@pytest.mark.parametrize("count", [54, 56, 64, 80])
def test_equal_direct_work_has_no_exact_axis_exception(count, quota):
    quota(count)
    a = mixed._make_execution_plan(TensorGeometry((17, 4096, 16)), 1)
    b = mixed._make_execution_plan(TensorGeometry((16, 4352, 16)), 1)
    assert a.layout is b.layout
    assert a.blocks == b.blocks


@pytest.mark.parametrize("count", [54, 56, 64, 80])
def test_serial_work_crossover_is_a_workload_boundary(count, quota):
    quota(count)
    seven = mixed._make_execution_plan(TensorGeometry((7 * count, 256, 16)), 1)
    eight = mixed._make_execution_plan(TensorGeometry((8 * count, 256, 16)), 1)
    assert seven.layout is mixed.Layout.WIDE16
    assert eight.layout is mixed.Layout.TRANSPOSE


def test_ub_fallback_retains_public_geometry(quota):
    quota(56)
    x = TensorGeometry((17, 4096, 16))
    direct = mixed._make_execution_plan(x, 1, ub_bytes=253952)
    fallback = mixed._make_execution_plan(x, 1, ub_bytes=65536)
    assert direct.layout is mixed.Layout.WIDE16
    assert fallback.layout is mixed.Layout.TRANSPOSE
    assert direct.output_shape == fallback.output_shape
    assert direct.scale_shape == fallback.scale_shape
    assert direct.blocks == fallback.blocks


@pytest.mark.parametrize("dst", ["float8_e4m3fn", "float8_e5m2"])
def test_width16_reference_mapping_does_not_follow_internal_layout(dst):
    for shape in ((2, 256, 16), (17, 4096, 16), (16, 4352, 16), (32, 4096, 16)):
        assert golden._uses_nibble_mapping(shape, 1, "bfloat16", dst) is True
    assert golden._uses_nibble_mapping((3, 272, 16), 1, "bfloat16", dst) is False


def test_wide16_random_cache_revision_is_category_specific(monkeypatch):
    identity = mixed.RuntimeCompileIdentity(
        "dav-3510", "npu", 0, "test", "native", "compiler", 1, "format", "", "".encode()
    )
    monkeypatch.setattr(mixed, "_capture_runtime_identity", lambda device: (identity, None))
    keys = {}
    for src, layout in (
        ("bf16", mixed.Layout.WIDE16),
        ("fp32", mixed.Layout.WIDE16),
        ("bf16", mixed.Layout.WIDE8),
        ("bf16", mixed.Layout.TAIL),
    ):
        keys[src, layout] = mixed.capture_compile_context(src, False, 253952, 0, 0, layout, TensorGeometry.device).key
    assert keys["bf16", mixed.Layout.WIDE16].bf16_wide16_rng_revision == 1
    assert all(key.bf16_wide16_rng_revision == 0 for kind, key in keys.items() if kind != ("bf16", mixed.Layout.WIDE16))
    assert all(key.rng_abi_version == 60 for key in keys.values())


@pytest.mark.parametrize("cores", [54, 56, 64, 80])
@pytest.mark.parametrize(
    "dst,algorithm,bound,eligible",
    [
        (24, 0, 0, True),
        (24, 1, 0, True),
        (24, 1, 1, False),
        (23, 0, 0, False),
        (23, 1, 0, True),
        (23, 1, 1, True),
    ],
)
def test_fp32_width16_transpose_work_boundary(cores, dst, algorithm, bound, eligible, quota):
    quota(cores)
    target = torch.float8_e5m2 if dst == 23 else torch.float8_e4m3fn
    for waves in (4, 5, 6):
        x = torch.empty((cores * waves, 256, 16))
        plan = mixed._make_execution_plan(x, 1, dst_dtype=target, scale_alg=algorithm, bound_bits=bound)
        want = mixed.Layout.TRANSPOSE if eligible and waves >= 5 else mixed.Layout.WIDE16
        assert plan.layout is want
        assert plan.output_shape == tuple(x.shape)
        assert plan.scale_shape == (cores * waves, 4, 16, 2)
    # At 2 Mi the transpose implementation changes random-field mapping.
    edge = torch.empty((512, 256, 16))
    assert (
        mixed._make_execution_plan(edge, 1, dst_dtype=target, scale_alg=algorithm, bound_bits=bound).layout
        is mixed.Layout.WIDE16
    )


@pytest.mark.parametrize("cores", [54, 56, 64, 80])
@pytest.mark.parametrize("axis_length", [256, 1280, 5120])
def test_fp32_four_pair_uses_tile_work_and_preserves_eight_field_route(cores, axis_length):
    op = mixed
    tile = op._make_tiling(op.dtypes.float32, 253952).tile_elements
    arguments = (op.dtypes.float32, op.Layout.TAIL, axis_length, cores, 253952)
    assert not op._select_fp32_four_pair_policy(cores * tile - axis_length, *arguments)
    assert op._select_fp32_four_pair_policy(cores * tile, *arguments)
    assert not op._select_fp32_four_pair_policy(op.FP32_EIGHT_MIN_ELEMENTS, *arguments)
    assert not op._select_fp32_four_pair_policy(
        cores * tile, op.dtypes.bfloat16, op.Layout.TAIL, axis_length, cores, 253952
    )
    assert not op._select_fp32_four_pair_policy(
        cores * tile, op.dtypes.float32, op.Layout.WIDE16, axis_length, cores, 253952
    )


@pytest.mark.parametrize("cores", [54, 56, 64, 80])
def test_bf16_crossdraw_prefix_preserves_geometry_and_algorithm_fallbacks(cores):
    op = mixed
    draw = op.FOUR_FIELD_DRAW_ELEMENTS
    tile = op._make_tiling(op.dtypes.bfloat16, 253952).tile_elements
    elements = (cores * tile + draw - 1) // draw * draw
    select = op._select_bf16_crossdraw_prefix3_policy
    plan = SimpleNamespace(compile_layout=op.Layout.TAIL, quant_length=64, blocks=cores)
    assert select(elements, op.dtypes.bfloat16, 0, plan, 253952)
    assert not select(elements - draw, op.dtypes.bfloat16, 0, plan, 253952)
    assert not select(elements + 64, op.dtypes.bfloat16, 0, plan, 253952)
    assert not select(elements, op.dtypes.bfloat16, 1, plan, 253952)
    assert not select(elements, op.dtypes.float32, 0, plan, 253952)
    plan.quant_length = 256
    assert not select(elements, op.dtypes.bfloat16, 0, plan, 253952)


@pytest.mark.parametrize("route", ["fp32-four-pair", "bf16-crossdraw"])
def test_prepare_launch_forwards_new_categories_to_all_rng_entries(monkeypatch, route):
    op = mixed
    key = _compile_context(monkeypatch).key
    identity = op.RuntimeCompileIdentity(
        **{field.name: getattr(key, field.name) for field in dataclasses.fields(op.RuntimeCompileIdentity)}
    )
    monkeypatch.setattr(op, "_capture_runtime_identity", lambda device: (identity, None))
    cores, ub = 4, 253952
    bf16 = route == "bf16-crossdraw"
    dtype = op.dtypes.bfloat16 if bf16 else op.dtypes.float32
    tile = op._make_tiling(dtype, ub).tile_elements
    elements = op.FOUR_FIELD_DRAW_ELEMENTS if bf16 else cores * tile
    columns = 64 if bf16 else 256
    x = torch.empty((elements // columns, columns), dtype=torch.bfloat16 if bf16 else torch.float32)
    monkeypatch.setattr(torch.npu, "device", lambda device: nullcontext())
    monkeypatch.setattr(op, "get_mem_size", lambda name: ub)
    monkeypatch.setattr(op, "_resolve_block_dim", lambda *args, **kwargs: cores)
    categories = {}

    def compile_entry(name):
        def capture(*args, **kwargs):
            categories[name] = kwargs
            return (object(), object()) if name == "framework" else object()

        return capture

    monkeypatch.setattr(op, "_get_compiled_kernel", compile_entry("explicit"))
    monkeypatch.setattr(op, "_get_compiled_tail_scalar_kernel", compile_entry("scalar"))
    monkeypatch.setattr(op, "_get_compiled_framework_programs", compile_entry("framework"))
    prepared = op._prepare_launch(x, torch.float8_e4m3fn, 1, 0, 0)
    assert prepared.plan.compile_layout is op.Layout.TAIL
    assert set(categories) == {"explicit", "scalar", "framework"}
    for options in categories.values():
        assert options["fp32_four_pair_schedule"] is (not bf16)
        assert options["bf16_crossdraw_prefix3"] is bf16
        assert options["_context"].key.fp32_four_pair_schedule is (not bf16)
        assert options["_context"].key.bf16_crossdraw_prefix3 is bf16


@pytest.mark.parametrize("implementation,device", [("torch", "npu"), ("torch", "cuda"), ("triton", "npu")])
def test_runtime_suite_rejects_other_provider_selections(implementation, device):
    from . import test_dynamic_mx_quant_sr_runtime as runtime

    with pytest.raises(pytest.skip.Exception):
        runtime.require_cannbotdsl_runtime.__wrapped__((implementation, device), (implementation, "npu.a5", device))


def test_benchmark_metadata_records_the_executed_compile_category(monkeypatch):
    from tests.performance.functions import test_dynamic_mx_quant_sr as bench

    key = _cache_key(fp32_eight_mapping=True, output_rng_alias=True)
    cache = mixed._ProgramCache(1)
    program = cache.get_or_build(key, _fake_program)
    monkeypatch.setattr(mixed, "_PROGRAM_CACHE", cache)
    prepared = SimpleNamespace(program=program, plan=SimpleNamespace(layout=mixed.Layout.TAIL, blocks=7))
    metadata = bench._execution_metadata(mixed, prepared)
    assert metadata["layout"] == "tail"
    assert metadata["blocks"] == 7
    assert metadata["compile_key"]["ub_bytes"] == 253952
    assert metadata["compile_key"]["dsl_version"] == "test-wheel"
    assert metadata["compile_key"]["fp32_eight_mapping"] is True
    assert metadata["compile_key"]["output_rng_alias"] is True
    # Reports must survive the JSON serializer used by the performance harness.
    import json

    assert json.loads(json.dumps(metadata)) == metadata
