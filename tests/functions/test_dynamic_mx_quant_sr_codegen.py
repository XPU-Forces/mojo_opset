"""SR codegen regressions adapted to the Mojo provider and capability gates."""

import pytest

pytest.importorskip("cannbotdsl", reason="SR provider tests require CANNBot-DSL 0.8")
import gc
import re
import weakref

from pathlib import Path
from types import SimpleNamespace

import cannbotdsl
import torch

from cannbotdsl import dtypes
from cannbotdsl import get_mem_size

from mojo_opset.kernels.npu_a5_cannbotdsl import dynamic_mx_quant_sr as mixed

from . import _dynamic_mx_quant_sr_rng as utils
from ._dynamic_mx_quant_sr_rng import DTYPE_PAIRS
from ._dynamic_mx_quant_sr_rng import _cache_key
from ._dynamic_mx_quant_sr_rng import _fake_program
from ._dynamic_mx_quant_sr_rng import _unexpected_device_work

pytestmark = [pytest.mark.api("functions.dynamic_mx_quant_sr"), pytest.mark.ascendc_toolchain]

pytestmark.append(pytest.mark.reference)


@pytest.fixture(autouse=True)
def codegen_npu_interface(monkeypatch):
    from types import SimpleNamespace

    import torch

    if not hasattr(torch, "npu"):
        monkeypatch.setattr(
            torch,
            "npu",
            SimpleNamespace(is_current_stream_capturing=lambda: False, current_device=lambda: 0),
            raising=False,
        )


@pytest.mark.ascendc_toolchain
def test_eight_field_rng_translates_without_quantization_resources(monkeypatch, tmp_path):
    monkeypatch.setenv("CANNBOTDSL_PIPE_STAGE", "translate")
    monkeypatch.setenv("CANNBOTDSL_DUMP_ASCENDC", "1")
    monkeypatch.setenv("CANNBOTDSL_DUMP_DIR", str(tmp_path))
    program = cannbotdsl.compile(
        utils.run_eight_field_words,
        cannbotdsl.TensorSpec((1, 512), dtypes.int32),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
    )
    assert not program.so_path
    source = next(tmp_path.glob("*.asc")).read_text()
    # DSL 0.8 stores the slot as bytes and binds int32 views at each use.
    arrays = re.findall(r"__ubuf__ uint8_t \w+\[(\d+)\];", source)
    assert [int(size) for size in arrays] == [512 * 4]
    assert "__ubuf__ int32_t*" in source
    assert "__umulhi(" in source


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize(
    "probe",
    [
        "runtime-words",
        "counter-words",
        "words-aos",
        "words-soa",
        "paired-words",
        "multi-aos",
        "multi-soa",
        "explicit-e4",
        "explicit-e5",
        "q16-e4",
        "q16-e5",
        "scale-e4",
        "scale-e5",
        "short-fp32",
        "short-bf16",
        "short-words",
        "batched-fields",
    ],
)
def test_device_precision_probes_translate_without_npu(probe, monkeypatch, tmp_path):
    """Catch stale device helper interfaces even on a host without an NPU."""
    monkeypatch.setenv("CANNBOTDSL_PIPE_STAGE", "translate")
    monkeypatch.setenv("CANNBOTDSL_DUMP_ASCENDC", "1")
    monkeypatch.setenv("CANNBOTDSL_DUMP_DIR", str(tmp_path))
    spec = cannbotdsl.TensorSpec
    i64 = dtypes.int64
    words = spec((1, 512), dtypes.int32)
    fields = [spec((1, 64), dtype) for dtype in (dtypes.float32, dtypes.uint32, dtypes.uint8)]
    entries = {
        "runtime-words": (
            utils.run_runtime_state_words,
            [spec((2,), i64), spec((4, 128), dtypes.uint32), i64, i64, i64],
        ),
        "counter-words": (utils.run_counter_words, [spec((4, 64), dtypes.uint32)] * 2),
        "words-aos": (utils.run_words_aos, [spec((1, 256), dtypes.int32), i64, i64]),
        "words-soa": (utils.run_words_soa, [spec((1, 256), dtypes.int32), i64, i64]),
        "paired-words": (utils.run_paired_words, [words, i64, i64]),
        "multi-aos": (utils.MultiBatchWordsLaunch(True, 768).run, [spec((1, 768), dtypes.int32), i64, i64, i64]),
        "multi-soa": (utils.MultiBatchWordsLaunch(False, 768).run, [spec((1, 768), dtypes.int32), i64, i64, i64]),
        "explicit-e4": (utils.ExplicitSRLaunch("float8_e4m3fn").run, fields),
        "explicit-e5": (utils.ExplicitSRLaunch("float8_e5m2").run, fields),
        "q16-e4": (utils.Q16Launch("float8_e4m3fn").run, fields),
        "q16-e5": (utils.Q16Launch("float8_e5m2").run, fields),
        "scale-e4": (utils.IntegerScaleLaunch(991054117).run, [spec((1, 64), dtypes.uint32)] * 3),
        "scale-e5": (utils.IntegerScaleLaunch(932333861).run, [spec((1, 64), dtypes.uint32)] * 3),
        "short-fp32": (
            utils.ShortRowsLaunch(dtypes.float32, False, 5, 48, 6).run,
            [spec((1, 240), dtypes.float32), spec((1, 240), dtypes.uint8), spec((1, 10), dtypes.uint8), i64],
        ),
        "short-bf16": (
            utils.ShortRowsLaunch(dtypes.bfloat16, True, 5, 48, 6).run,
            [spec((1, 240), dtypes.bfloat16), spec((1, 240), dtypes.uint8), spec((1, 10), dtypes.uint8), i64],
        ),
        "short-words": (
            utils.ShortRowsWordsLaunch(dtypes.float32, False, 48, 48, 6).run,
            [
                spec((1, 2304), dtypes.float32),
                spec((1, 2304), dtypes.uint8),
                spec((1, 96), dtypes.uint8),
                spec((1, 768), dtypes.int32),
                i64,
            ],
        ),
        "batched-fields": (
            utils.run_batched_fields,
            [spec((4, 64), dtypes.uint32), spec((16, 64), dtypes.uint32), i64, i64, i64, i64],
        ),
    }
    entry, specs = entries[probe]
    program = cannbotdsl.compile(entry, *specs)
    assert not program.so_path
    sources = list(tmp_path.glob("*.asc"))
    assert len(sources) == 1
    source = sources[0].read_text()
    assert "__vector__ __global__ __aicore__ void " in source


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize("action", ["clear", "evict"])
def test_program_cache_releases_real_aot_artifact_after_last_owner(monkeypatch, action):
    # Disable both SDK cache tiers so the real Program owns its temporary binary.
    # Persistent SDK cache artifacts must survive clearing this operator's cache.
    monkeypatch.setenv("CANNBOTDSL_CACHE_MEM_DISABLE", "1")
    monkeypatch.setenv("CANNBOTDSL_CACHE_DISK_DISABLE", "1")
    cache = mixed._ProgramCache(1)
    program = cache.get_or_build("first", lambda: mixed._build_program(_cache_key()))
    reference = weakref.ref(program)
    artifact = Path(program.so_path)
    directory = artifact.parent
    assert artifact.is_file()
    if action == "clear":
        cache.clear()
    else:
        cache.get_or_build("replacement", _fake_program)
    assert reference() is program and not program.closed and artifact.is_file()
    del program
    gc.collect()
    assert reference() is None
    assert not artifact.exists() and not directory.exists()


@pytest.mark.ascendc_toolchain
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
@pytest.mark.parametrize("dtype", [dtypes.float32, dtypes.bfloat16])
@pytest.mark.parametrize("e5m2", [False, True])
@pytest.mark.parametrize("algorithm", [0, 1])
def test_actual_translated_artifact_uses_native_static_ub_and_abi5_fields(
    layout, dtype, e5m2, algorithm, monkeypatch, tmp_path
):
    monkeypatch.setenv("CANNBOTDSL_PIPE_STAGE", "translate")
    monkeypatch.setenv("CANNBOTDSL_DUMP_ASCENDC", "1")
    monkeypatch.setenv("CANNBOTDSL_DUMP_DIR", str(tmp_path))
    ub = get_mem_size("ub")
    stub = mixed._get_compiled_kernel(
        dtype, e5m2, ub, algorithm, layout=mixed.Layout(layout), device=SimpleNamespace(type="npu", index=0)
    )
    assert not stub.so_path
    sources = list(tmp_path.glob("*.asc"))
    assert len(sources) == 1
    source = sources[0].read_text()
    tiling = mixed._make_tiling(
        dtype,
        ub,
        layout == "non_tail_fused",
        tail_axis=layout == "tail",
    )
    arrays = re.findall(r"__ubuf__ uint8_t (ub_storage\d+)\[(\d+)\];", source)
    # ABI5 leaves the historical non-tail budget conservative. The fused owner
    # retains its reserve; the other non-tail entries omit the unused field.
    live_bytes = tiling.buffer_bytes - (0 if layout in ("tail", "non_tail_fused") else tiling.tile_elements)
    assert live_bytes <= tiling.buffer_bytes <= ub - 40 * 1024
    if arrays:
        assert len(arrays) == 1 and int(arrays[0][1]) == live_bytes
    else:
        # Pure SIMD wide entries use fixed UB offsets instead of a SIMT-visible
        # storage array. Check every channel slot and buffer offset explicitly.
        assert layout.startswith("wide")
        pointers = re.findall(
            r"__ubuf__ \w+\* (buf\d+) = \(__ubuf__ \w+\*\)\(uintptr_t\)\(uint64_t\)\((\d+)\);", source
        )
        tile = tiling.tile_elements
        item = 4 if dtype == dtypes.float32 else 2
        sizes = [tile * item, tile * item, tile, tile, tile // 16, tile // 16, 4 * tiling.scale_capacity, tile]
        expected_offsets, end = [], 0
        for size in sizes:
            expected_offsets.append(end)
            end += size
        assert len(pointers) == len(sizes) and len({name for name, _ in pointers}) == len(sizes)
        assert [int(offset) for _, offset in pointers] == expected_offsets
        assert end == live_bytes
    launches = re.findall(r"(\w+)<<<([^,\n]+),\s*([^,\n]+),\s*([^\n]+?)>>>", source)
    assert len(launches) == 1 and launches[0][2].strip() == "0"
    assert ("__simt_vf__" in source) == (layout in ("tail", "non_tail", "non_tail_fused"))
    assert not re.search(r"\bbrev\b", source), "ABI5 must not execute the obsolete bit reversal"
    assert source.count("__vector__ __global__ __aicore__ void ") == 1


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize("scalar_entry", [False, True], ids=["device-state", "scalar-state"])
@pytest.mark.parametrize(
    "src,e5,algorithm,options",
    [
        (dtypes.float32, True, 0, dict(fp32_eight_mapping=True, output_rng_alias=True)),
        (dtypes.bfloat16, False, 0, dict(bf16_compact_mode=36, bf16_compact_four_panel=True)),
        (dtypes.bfloat16, False, 0, dict(bf16_compact_mode=36, bf16_compact_four_panel=True, bf16_tail_simd=True)),
        (dtypes.bfloat16, False, 1, dict(bf16_compact_four_panel=True, bf16_four_panel=True)),
        (dtypes.bfloat16, True, 0, dict(bf16_four_panel=True)),
        (dtypes.float32, False, 0, dict(fp32_four_pair_schedule=True)),
        (dtypes.float32, True, 1, dict(fp32_four_pair_schedule=True)),
        (dtypes.bfloat16, False, 0, dict(bf16_crossdraw_prefix3=True)),
        (dtypes.bfloat16, False, 0, dict(bf16_four_panel=True)),
        (dtypes.bfloat16, False, 0, dict(bf16_four_panel=True, bf16_full_tiles=True)),
        (dtypes.bfloat16, False, 1, dict(bf16_four_panel=True, bf16_full_tiles=True)),
        (dtypes.bfloat16, True, 0, dict(bf16_four_panel=True, bf16_full_tiles=True)),
        (dtypes.bfloat16, True, 1, dict(bf16_four_panel=True, bf16_full_tiles=True)),
    ],
    ids=[
        "fp32-output-alias",
        "bf16-compact",
        "bf16-simd-tail",
        "bf16-e4-panels",
        "bf16-e5-panels",
        "fp32-four-pair-e4",
        "fp32-four-pair-e5",
        "bf16-crossdraw",
        "bf16-batched-e4",
        "bf16-full-e4-alg0",
        "bf16-full-e4-alg1",
        "bf16-full-e5-alg0",
        "bf16-full-e5-alg1",
    ],
)
def test_optimized_stages_translate_with_both_rng_entries(
    src, e5, algorithm, options, scalar_entry, monkeypatch, tmp_path
):
    monkeypatch.setenv("CANNBOTDSL_PIPE_STAGE", "translate")
    monkeypatch.setenv("CANNBOTDSL_DUMP_ASCENDC", "1")
    monkeypatch.setenv("CANNBOTDSL_DUMP_DIR", str(tmp_path))
    ub = get_mem_size("ub")
    device = SimpleNamespace(type="npu", index=0)
    if scalar_entry:
        monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
        program = mixed._get_compiled_tail_scalar_kernel(src, e5, ub, algorithm, 0, device, **options)
    else:
        program = mixed._get_compiled_kernel(src, e5, ub, algorithm, device=device, **options)
    assert not program.so_path
    sources = list(tmp_path.glob("*.asc"))
    assert len(sources) == 1
    source = sources[0].read_text()
    arrays = re.findall(r"__ubuf__ uint8_t ub_storage\d+\[(\d+)\];", source)
    assert len(arrays) == 1
    tiling = mixed._make_tiling(
        src,
        ub,
        tail_axis=True,
        output_rng_alias=options.get("output_rng_alias", False),
        bf16_compact_mode=options.get("bf16_compact_mode", 0),
        bf16_full_tiles=options.get("bf16_full_tiles", False),
        e5m2=e5,
        scale_alg=algorithm,
    )
    # DSL 0.8 retains declared scale scratch in the shared UB allocation.
    assert int(arrays[0]) == tiling.buffer_bytes <= ub - mixed.SIMT_UB_RESERVE
    assert source.count("__vector__ __global__ __aicore__ void ") == 1


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize("src,dst", DTYPE_PAIRS)
def test_real_dynamic_codegen_without_npu_query(src, dst, monkeypatch):
    monkeypatch.setattr(torch.npu, "current_device", _unexpected_device_work)
    dtype = dtypes.bfloat16 if src == torch.bfloat16 else dtypes.float32
    program = mixed._get_compiled_kernel(
        dtype, dst == torch.float8_e5m2, get_mem_size("ub"), device=SimpleNamespace(type="npu", index=0)
    )
    assert Path(program.so_path).is_file() and not program.closed


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize("layout", [layout for layout in mixed.Layout if layout is not mixed.Layout.TRANSPOSE])
def test_framework_rng_layout_entries_translate(layout, monkeypatch, tmp_path):
    monkeypatch.setenv("CANNBOTDSL_PIPE_STAGE", "translate")
    monkeypatch.setenv("CANNBOTDSL_DUMP_ASCENDC", "1")
    monkeypatch.setenv("CANNBOTDSL_DUMP_DIR", str(tmp_path))
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    programs = mixed._get_compiled_framework_programs(
        dtypes.bfloat16,
        False,
        253952,
        1,
        0,
        layout=layout,
        device=SimpleNamespace(type="npu", index=0),
    )
    assert len(programs) == 2 and all(not program.so_path for program in programs)
    sources = list(tmp_path.glob("*.asc"))
    assert len(sources) == 2
    assert all("__vector__ __global__ __aicore__ void " in source.read_text() for source in sources)


@pytest.mark.ascendc_toolchain
@pytest.mark.parametrize("layout", ["non_tail", "non_tail_fused"])
@pytest.mark.parametrize("dtype", [dtypes.float32, dtypes.bfloat16])
def test_stride2_inputs_use_two_nddma_copies(layout, dtype, monkeypatch, tmp_path):
    monkeypatch.setenv("CANNBOTDSL_PIPE_STAGE", "translate")
    monkeypatch.setenv("CANNBOTDSL_DUMP_ASCENDC", "1")
    monkeypatch.setenv("CANNBOTDSL_DUMP_DIR", str(tmp_path))
    program = mixed._get_compiled_kernel(
        dtype, False, get_mem_size("ub"), 0, layout=mixed.Layout(layout), device=SimpleNamespace(type="npu", index=0)
    )
    assert not program.so_path
    sources = list(tmp_path.glob("*.asc"))
    assert len(sources) == 1
    assert sources[0].read_text().count("asc_set_ndim_loop0_stride((uint64_t)1, (uint64_t)2)") == 2
