"""Dynamic MXFP8 SR: SIMD numerical work, full Philox10 with NPU linear virtual streams.

One public operator and seven dynamic layout specializations. Standard AOT
owns binary management; this sample keeps bounded, generation-safe Program
references. Compilation requires stable process configuration.
"""

import math
import os
import struct

from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from functools import partial
from threading import Lock
from types import MappingProxyType
from typing import NamedTuple

import cannbotdsl
import torch

from cannbotdsl import Buffer
from cannbotdsl import Dim
from cannbotdsl import TensorSpec
from cannbotdsl import dtypes
from cannbotdsl import get_mem_size
from cannbotdsl.core.compiler import cache as dsl_cache
from cannbotdsl.core.compiler import cache_key
from cannbotdsl.core.compiler.abi import HOST_ABI_VERSION
from cannbotdsl.core.utils.env import env
from cannbotdsl.lang import const_expr
from cannbotdsl.lang import host
from cannbotdsl.lang import jit
from cannbotdsl.lang import kernel
from cannbotdsl.lang import vf
from cannbotdsl.ops import reg
from cannbotdsl.ops import scalar
from cannbotdsl.ops import simt
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.arch import get_block_num
from cannbotdsl.ops.memcpy import copy_options
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import MemLoc
from cannbotdsl.tensor import Tensor
from cannbotdsl.tensor import make_tiler
from cannbotdsl.tensor import reinterpret
from cannbotdsl.tensor import tile_slice

from . import _rng_state as _framework_rng
from ._rng_state import _rng_increment
from ._rng_state import _validate_rng_geometry


@jit
def _copy_to_gm(destination, source):
    """Bound the source domain before a partial UB-to-GM DMA.

    DSL 0.8 copies the source View's extent, not the intersection of source
    and destination. Keep the selected slot's pitch while clipping its view.
    """
    rows = destination.shape[0]
    if rows > source.shape[0]:
        rows = source.shape[0]
    columns = destination.shape[1]
    if columns > source.shape[1]:
        columns = source.shape[1]
    mem_copy(destination, reinterpret(source, shape=(rows, columns), stride=source.stride))


class Layout(str, Enum):
    TAIL = "tail"
    NON_TAIL = "non_tail"
    NON_TAIL_FUSED = "non_tail_fused"
    WIDE = "wide_non_tail"
    WIDE32 = "wide32_non_tail"
    WIDE8 = "wide8_non_tail"
    WIDE16 = "wide16_non_tail"
    TRANSPOSE = "transpose"


@dataclass(frozen=True)
class ExecutionPlan:
    layout: Layout
    axis: int
    quant_length: int
    width: int
    logical_rows: int
    output_shape: tuple[int, ...]
    scale_shape: tuple[int, ...]
    storage_output_shape: tuple[int, ...]
    storage_scale_shape: tuple[int, ...]
    kernel_shape: tuple[int, int]
    kernel_scale_shape: tuple[int, int]
    blocks: int
    base_task_rows: int

    @property
    def compile_layout(self):
        return Layout.TAIL if self.layout is Layout.TRANSPOSE else self.layout


BF16_TILE_LIMIT_ELEMENTS = 32 * 1024
# Non-tail layouts keep smaller BF16 tiles to fit their additional staging.
BF16_NON_TAIL_TILE_LIMIT_ELEMENTS = 16 * 1024
# Eight-field gathers load 256 bytes at a 128-byte stride; pad the last load.
BF16_RANDOM_GATHER_PADDING_BYTES = 128
SUPPORTED_ARCHITECTURES = frozenset({"dav-3510"})

# Performance calibration for dav-3510, measured in input elements. These
# amortization thresholds never relax layout or Philox correctness constraints.
BF16_COMPACT_MIN_ELEMENTS = 24 * 1024 * 1024
# Compact tiling without the four-panel route amortizes at 32 Mi elements on dav-3510.
BF16_COMPACT_TILE_MIN_ELEMENTS = 32 * 1024 * 1024
# Compact-mode sizes are in units of 1024 input elements.
BF16_COMPACT_TILE_KI_ELEMENTS = 36
BF16_SINGLE_INPUT_TILE_KI_ELEMENTS = 48
BF16_COMPACT_MIN_ELEMENTS_PER_AIV = 384 * 1024
BF16_FOUR_PANEL_MIN_ELEMENTS_PER_AIV = 64 * 1024
BF16_E5_FLOOR_MIN_ELEMENTS_PER_AIV = 72 * 1024
# Complete waves below the ordinary per-AIV threshold use at most two 32K tiles per AIV.
BF16_FULL_TILE_MAX_LOWER_WAVES = 2
BF16_WIDE16_TASK_ELEMENTS = 4096
# FP32 dav-3510 calibration: amortize the four-task transpose chain from five
# serial direct waves; total input must stay below the mapping's 2 Mi edge.
FP32_WIDE16_TASK_ELEMENTS = 4096
FP32_WIDE16_TRANSPOSE_MIN_WAVES = 5
# A four-field draw spans 16 elements per virtual stream. The cross-draw
# route admits complete draws and scale-pair-aligned rows; its per-AIV
# minimum work is a separate performance selection below.
FOUR_FIELD_DRAW_ELEMENTS = _framework_rng.ELEMENTS_PER_PHILOX_BLOCK * _framework_rng.RNG_VIRTUAL_STREAMS
BF16_CROSSDRAW_ROW_ALIGNMENT_ELEMENTS = 64
BF16_WIDE16_MAX_WAVES = 7
# Relative cost calibration: one direct wave costs 5 Mi units; the three
# transpose launches cost 14 Mi units plus 15 units per input element.
BF16_WIDE16_WAVE_COST = 5 * 1024 * 1024
BF16_TRANSPOSE_LAUNCH_COST = 14 * 1024 * 1024
BF16_TRANSPOSE_ELEMENT_COST = 15
PHILOX_DRAW_ELEMENTS = 16384 * 32
# Positive BF16 bits are ordered. The half-Q7 path is exact from exponent 118
# (2**-9) upward; each stored exponent step occupies seven mantissa bits.
BF16_HALF_Q7_MIN_NORMALIZED_BITS = 118 << 7


@dataclass(frozen=True)
class CompileKey:
    input_dtype: str
    e5m2: bool
    ub_bytes: int
    scale_alg: int
    max_low_bound_bits: int
    layout: Layout
    npu_arch: str
    device_type: str
    device_index: int
    dsl_version: str
    native_build: str
    compiler_identity: str
    host_abi: int
    cache_format: str
    cache_format_tag: str
    codegen_options: bytes
    rng_abi_version: int = 60
    fp32_eight_mapping: bool = False
    output_rng_alias: bool = False
    bf16_compact_mode: int = 0
    bf16_compact_four_panel: bool = False
    bf16_compact_rng_revision: int = 0
    bf16_tail_simd: bool = False
    bf16_four_panel: bool = False
    bf16_wide16_rng_revision: int = 0
    bf16_full_tiles: bool = False
    fp32_four_pair_schedule: bool = False
    bf16_crossdraw_prefix3: bool = False


@dataclass(frozen=True)
class RuntimeCompileIdentity:
    npu_arch: str
    device_type: str
    device_index: int
    dsl_version: str
    native_build: str
    compiler_identity: str
    host_abi: int
    cache_format: str
    cache_format_tag: str
    codegen_options: bytes


@dataclass(frozen=True)
class CompileContext:
    key: CompileKey
    bypass_reason: str | None


def _require_cache_identity_api():
    """Isolate private SDK cache identity hooks; never probe at module import."""
    required = (
        "_runtime_identity",
        "_toolchain_version",
        "_FORMAT_VERSION_TAG",
        "_KEY_IGNORED_ENV",
        "_encode_env_snapshot",
    )
    missing = [name for name in required if not hasattr(cache_key, name)]
    if missing:
        raise RuntimeError(
            f"CANNBotDSL {cannbotdsl.__version__} lacks SR cache identity capabilities: " + ", ".join(missing)
        )


def _build_identity():
    """Version-specific SDK hooks keep cache entries distinct across builds."""
    _require_cache_identity_api()
    return (
        cannbotdsl.__version__,
        cache_key._runtime_identity(),
        cache_key._toolchain_version(),
        HOST_ABI_VERSION,
        cache_key._FORMAT_VERSION_TAG,
    )


def _diagnostic_reason(config):
    reason = dsl_cache.bypass_reason(config, has_debug_plans=False)
    return reason or ("frontend_dumps" if config.frontend_dumps else None)


def _capture_runtime_identity(device):
    """One configuration/diagnostics path for quantization and state helpers."""
    config = env()
    arch = config.npu_arch
    if arch not in SUPPORTED_ARCHITECTURES:
        raise NotImplementedError(f"dynamic_mx_quant_sr does not support architecture {arch!r}")
    if device.type not in ("npu", "privateuseone") or device.index is None:
        raise ValueError("compile device must have a resolved NPU index")
    version, native, compiler, abi, format_version = _build_identity()
    options = {name: value for name, value in vars(config).items() if name not in cache_key._KEY_IGNORED_ENV}
    identity = RuntimeCompileIdentity(
        arch,
        "npu",
        device.index,
        version,
        native,
        compiler,
        abi,
        format_version,
        os.environ.get("CANNBOTDSL_CACHE_FORMAT_TAG", ""),
        cache_key._encode_env_snapshot(options),
    )
    return (identity, _diagnostic_reason(config))


def capture_compile_context(
    input_dtype,
    e5m2,
    ub_bytes,
    scale_alg,
    max_low_bound_bits,
    layout,
    device,
    *,
    fp32_eight_mapping=False,
    output_rng_alias=False,
    bf16_compact_mode=0,
    bf16_compact_four_panel=False,
    bf16_tail_simd=False,
    bf16_four_panel=False,
    bf16_full_tiles=False,
    fp32_four_pair_schedule=False,
    bf16_crossdraw_prefix3=False,
):
    """Capture DSL configuration and explicit shape-derived mapping category."""
    if bf16_crossdraw_prefix3 and (input_dtype != "bf16" or Layout(layout) is not Layout.TAIL):
        raise ValueError("cross-draw prefix3 requires BF16 tail")
    if fp32_four_pair_schedule and (input_dtype != "fp32" or fp32_eight_mapping or Layout(layout) is not Layout.TAIL):
        raise ValueError("four-field paired schedule requires FP32 four-field tail")
    if output_rng_alias and (input_dtype != "fp32" or not fp32_eight_mapping or Layout(layout) is not Layout.TAIL):
        raise ValueError("upper alias requires mapped FP32 tail")
    identity, reason = _capture_runtime_identity(device)
    key = CompileKey(
        input_dtype,
        e5m2,
        ub_bytes,
        scale_alg,
        max_low_bound_bits,
        Layout(layout),
        **asdict(identity),
        fp32_eight_mapping=fp32_eight_mapping,
        output_rng_alias=output_rng_alias,
        bf16_compact_mode=bf16_compact_mode,
        bf16_compact_four_panel=bf16_compact_four_panel,
        bf16_compact_rng_revision=90 if bf16_tail_simd else 0,
        bf16_tail_simd=bf16_tail_simd,
        bf16_four_panel=bf16_four_panel,
        bf16_full_tiles=bf16_full_tiles,
        bf16_wide16_rng_revision=1 if input_dtype == "bf16" and Layout(layout) is Layout.WIDE16 else 0,
        fp32_four_pair_schedule=fp32_four_pair_schedule,
        bf16_crossdraw_prefix3=bf16_crossdraw_prefix3,
    )
    return CompileContext(key, reason)


class CacheInfo(NamedTuple):
    hits: int
    misses: int
    maxsize: int
    currsize: int
    inflight: int
    generation: int


class _ProgramCache:
    """Coalesce first Program builds per (generation,key), with bounded LRU."""

    def __init__(self, capacity=64):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Program cache capacity must be a positive integer")
        self._capacity = capacity
        self._lock = Lock()
        self._programs = OrderedDict()
        self._inflight = {}
        self._generation = 0
        self._hits = self._misses = 0

    def get_or_build(self, key, builder, *, bypass=False):
        if bypass:
            return builder()
        with self._lock:
            program = self._programs.get(key)
            if program is not None and (not program.closed):
                self._programs.move_to_end(key)
                self._hits += 1
                return program
            if program is not None:
                del self._programs[key]
            self._misses += 1
            generation = self._generation
            token = (generation, key)
            pending = self._inflight.get(token)
            owner = pending is None
            if owner:
                pending = self._inflight[token] = Future()
        if not owner:
            return pending.result()
        try:
            program = builder()
            if not program.so_path or program.closed:
                raise RuntimeError("production AOT returned a closed Program or empty so_path stub")
        except BaseException as error:
            with self._lock:
                if self._inflight.get(token) is pending:
                    del self._inflight[token]
            pending.set_exception(error)
            raise
        evicted = None
        with self._lock:
            if self._inflight.get(token) is pending:
                del self._inflight[token]
            if generation == self._generation:
                self._programs[key] = program
                self._programs.move_to_end(key)
                if len(self._programs) > self._capacity:
                    _, evicted = self._programs.popitem(last=False)
        del evicted
        pending.set_result(program)
        return program

    def clear(self):
        with self._lock:
            released = list(self._programs.values())
            self._generation += 1
            self._programs.clear()
            self._hits = self._misses = 0
        del released

    def info(self):
        with self._lock:
            return CacheInfo(
                self._hits, self._misses, self._capacity, len(self._programs), len(self._inflight), self._generation
            )


PHILOX_ROUNDS = 10
UINT32_MASK = (1 << 32) - 1
# Random123 Philox4x32-10: two multiplication constants and two Weyl key increments.
# Hex spellings match the algorithm specification; arithmetic wraps modulo 2**32.
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85


def _split_uint64(value):
    """Clear the unsigned low word before exact signed division.

    UInt32 -> Int64 widens with zero extension. Clearing that low word
    leaves a multiple of 2**32 in signed range, so trunc division is exact
    for every 64-bit seed/offset bit pattern, including negative int64.
    """
    low = dtypes.uint32(value)
    cleared = dtypes.int64(value) - dtypes.int64(low)
    return (low, dtypes.uint32(cleared // 4294967296))


def _pair_add(low0, high0, low1, high1, mask):
    carry, low = reg.vaddco(low0, low1, mask=mask)
    _, high = reg.vaddc(high0, high1, carry, mask=mask)
    return (low, high)


def _linear_counter_coordinates(row_low, row_high, col_low, col_high, axis_groups, block_offset, mask):
    """Logical group=row*axis_groups+column; fixed virtual-stream mapping."""
    axis_low, axis_high = _split_uint64(axis_groups)
    group_low, group_high = reg.vmull(row_low, reg.vdups(axis_low, dtypes.uint32, mask=mask), dtypes.uint32, mask=mask)
    group_high = reg.vadd(group_high, reg.vmuls(row_low, axis_high, mask=mask), mask=mask)
    group_high = reg.vadd(group_high, reg.vmuls(row_high, axis_low, mask=mask), mask=mask)
    group_low, group_high = _pair_add(group_low, group_high, col_low, col_high, mask)
    return _linear_counter_groups(group_low, group_high, block_offset, mask)


def _linear_counter_groups(group_low, group_high, word_offset, mask):
    offset_low, offset_high = _split_uint64(word_offset)
    base_low = dtypes.uint32(offset_low // 4 + offset_high % 4 * 1073741824)
    base_high = dtypes.uint32(offset_high // 4)
    draw_low = reg.vbitwise_or(reg.vshr(group_low, 14, mask=mask), reg.vshl(group_high, 18, mask=mask), mask=mask)
    draw_high = reg.vshr(group_high, 14, mask=mask)
    c0, c1 = _pair_add(
        reg.vdups(base_low, dtypes.uint32, mask=mask),
        reg.vdups(base_high, dtypes.uint32, mask=mask),
        draw_low,
        draw_high,
        mask,
    )
    return (
        reg.vdups(0, dtypes.uint32, mask=mask),
        reg.vbitwise_and(group_low, reg.vdups(16383, dtypes.uint32, mask=mask), mask=mask),
        c0,
        c1,
    )


@jit
def _linear_counter_contiguous(row_base, col_base, ids, axis_groups, reciprocal, block_offset, mask):
    group = dtypes.uint64(row_base) * dtypes.uint64(axis_groups) + dtypes.uint64(col_base)
    low, high = _split_uint64(group)
    group_low, group_high = _pair_add(
        reg.vdups(low, dtypes.uint32, mask=mask),
        reg.vdups(high, dtypes.uint32, mask=mask),
        ids,
        reg.vdups(0, dtypes.uint32, mask=mask),
        mask,
    )
    return _linear_counter_groups(group_low, group_high, block_offset, mask)


def _linear_counter_rows(row_base, ids, col_group, axis_groups, block_offset, mask):
    """Wide layouts visit consecutive original logical rows at one column."""
    row_low_bits, row_high_bits = _split_uint64(row_base)
    col_low_bits, col_high_bits = _split_uint64(col_group)
    row_low, row_high = _pair_add(
        reg.vdups(row_low_bits, dtypes.uint32, mask=mask),
        reg.vdups(row_high_bits, dtypes.uint32, mask=mask),
        ids,
        reg.vdups(0, dtypes.uint32, mask=mask),
        mask,
    )
    return _linear_counter_coordinates(
        row_low,
        row_high,
        reg.vdups(col_low_bits, dtypes.uint32, mask=mask),
        reg.vdups(col_high_bits, dtypes.uint32, mask=mask),
        axis_groups,
        block_offset,
        mask,
    )


def _rng_geometry(start, quant_length):
    return _rng_group_geometry(dtypes.int64(start) // 16, quant_length)


@jit
def _rng_group_geometry(group, quant_length):
    """Nonnegative scalar division stays outside all SIMD VF loops."""
    axis_groups = dtypes.int64(quant_length) // 16
    group = dtypes.int64(group)
    row_base, col_base = (group // axis_groups, group % axis_groups)
    reciprocal = dtypes.uint32(0)
    if axis_groups > 1 and axis_groups < 64:
        reciprocal = dtypes.uint32(4294967296 // axis_groups)
    return (axis_groups, reciprocal, row_base, col_base)


def _advance_group(row_base, col_base, step_q, step_r, axis_groups):
    total = col_base + step_r
    carry = dtypes.int64(total >= axis_groups)
    return (row_base + step_q + carry, total - carry * axis_groups)


def _cursor_coordinates(row_low, row_high, col_low, col_high):
    """Rebuild temporary coordinates from explicitly zero-extended words.

    gc8c8581 narrows scalar VF loop-carried values to u32, even after an
    int64 cast. Carry four u32 words through loops instead; these temporary
    int64 coordinates retain the validated nonnegative geometry domain.
    """
    row = dtypes.int64(dtypes.uint64(dtypes.uint32(row_low)) + dtypes.uint64(dtypes.uint32(row_high)) * 4294967296)
    col = dtypes.int64(dtypes.uint64(dtypes.uint32(col_low)) + dtypes.uint64(dtypes.uint32(col_high)) * 4294967296)
    return (row, col)


def _advance_cursor_words(row, col, step_q, step_r, axis_groups):
    row, col = _advance_group(row, col, step_q, step_r, axis_groups)
    row_low, row_high = _split_uint64(row)
    col_low, col_high = _split_uint64(col)
    return (row_low, row_high, col_low, col_high)


def _philox_counter10(c0, c1, c2, c3, key0, key1, mask):
    """Full ten-round Philox4x32 for runtime counter and 64-bit seed key."""
    m0 = reg.vdups(PHILOX_M0, dtypes.uint32, mask=mask)
    m1 = reg.vdups(PHILOX_M1, dtypes.uint32, mask=mask)
    for step in range(PHILOX_ROUNDS):
        lo0, hi0 = reg.vmull(c0, m0, dtypes.uint32, mask=mask)
        lo1, hi1 = reg.vmull(c2, m1, dtypes.uint32, mask=mask)
        next0 = reg.vbitwise_xor(hi1, c1, mask=mask)
        next2 = reg.vbitwise_xor(hi0, c3, mask=mask)
        k0 = reg.vdups(dtypes.uint32(key0) + (step * PHILOX_W0 & UINT32_MASK), dtypes.uint32, mask=mask)
        k1 = reg.vdups(dtypes.uint32(key1) + (step * PHILOX_W1 & UINT32_MASK), dtypes.uint32, mask=mask)
        c0 = reg.vbitwise_xor(next0, k0, mask=mask)
        c2 = reg.vbitwise_xor(next2, k1, mask=mask)
        c1 = lo1
        c3 = lo0
    return (c0, c1, c2, c3)


def _store_random_words(scratch, offset, w0, w1, w2, w3, mask):
    """Store one Philox result in the existing four-plane SoA order."""
    reg.vstore(scratch, offset, reg.vreinterpret(w0, dtypes.int32), mask)
    reg.vstore(scratch, offset + 64, reg.vreinterpret(w1, dtypes.int32), mask)
    reg.vstore(scratch, offset + 128, reg.vreinterpret(w2, dtypes.int32), mask)
    reg.vstore(scratch, offset + 192, reg.vreinterpret(w3, dtypes.int32), mask)


def _generate_random_words_keyed(
    scratch, row_base, col_base, axis_groups, reciprocal, offset, key0, key1, block_offset, *, aos=False
):
    mask = reg.full_mask()
    c0, c1, c2, c3 = _linear_counter_contiguous(
        row_base, col_base, reg.varange(0, dtypes.uint32), axis_groups, reciprocal, block_offset, mask
    )
    w0, w1, w2, w3 = _philox_counter10(c0, c1, c2, c3, key0, key1, mask)
    if aos:
        _store_random_words_aos(scratch, offset, w0, w1, w2, w3, mask)
    else:
        _store_random_words(scratch, offset, w0, w1, w2, w3, mask)


def _store_random_words_aos(scratch, offset, w0, w1, w2, w3, mask):
    a0, a1 = reg.vinterleave(w0, w2)
    b0, b1 = reg.vinterleave(w1, w3)
    reg.vstore_interleave(scratch, offset, a0, b0)
    reg.vstore_interleave(scratch, offset + 128, a1, b1)


def _rotate32_words8(word, mask):
    """ABI5: byte rotate; fields H, rot.low16, L, rot.high16."""
    return reg.vbitwise_or(reg.vshl(word, 8, mask=mask), reg.vshr(word, 24, mask=mask), mask=mask)


def _random_fields_from_words(ids, scratch, segment, mask):
    slot = reg.vbitwise_and(reg.vshr(ids, 2, mask=mask), reg.vdups(3, dtypes.uint32, mask=mask), mask=mask)
    index = reg.vadd(reg.vshl(slot, 6, mask=mask), reg.vshr(ids, 4, mask=mask), mask=mask)
    index = reg.vadds(index, segment // 16 * 256 + segment % 16 * 4, mask=mask)
    word = reg.vreinterpret(reg.vgather(scratch, index, mask=mask), dtypes.uint32)
    reverse = _rotate32_words8(word, mask)
    lane = reg.vbitwise_and(ids, reg.vdups(3, dtypes.uint32, mask=mask), mask=mask)
    even = reg.veqs(reg.vbitwise_and(lane, reg.vdups(1, dtypes.uint32, mask=mask), mask=mask), 0, mask=mask)
    selected = reg.vselect(word, reverse, cond_mask=even)
    high_half = reg.mask_or(reg.veqs(lane, 0, mask=mask), reg.veqs(lane, 3, mask=mask), exec_mask=mask)
    return reg.vselect(
        reg.vshr(selected, 16, mask=mask),
        reg.vbitwise_and(selected, reg.vdups(65535, dtypes.uint32, mask=mask), mask=mask),
        cond_mask=high_half,
    )


def _simt_umulhi(lhs, rhs):
    """Unsigned mulhi adapter for the DSL scalar carrier ABI.

    The public mulhi infers signedness from the signless i32 carrier in this
    SDK, rejecting a uint32 SIMTScalar. Philox requires an unsigned high word;
    keep the ui32 IR operation here until the public API preserves its dtype.
    """
    required = ("_create_simt_cast", "_wrap_simt_scalar")
    missing = [name for name in required if not callable(getattr(simt, name, None))]
    if missing:
        raise RuntimeError(
            f"CANNBotDSL {cannbotdsl.__version__} lacks unsigned Philox capabilities: " + ", ".join(missing)
        )
    from cannbotdsl._mlir import ir
    from cannbotdsl.core.ir_transport import unwrap_operand

    unsigned = ir.IntegerType.get_unsigned(32)
    a = ir.Operation.create(
        "ascsimt.bitcast", operands=[unwrap_operand(dtypes.uint32(lhs))], results=[unsigned]
    ).results[0]
    if isinstance(rhs, int):
        b = simt._create_simt_cast(
            unsigned, unwrap_operand(dtypes.int32(rhs - 4294967296)), simt.RoundingMode.RZ, False
        )
    else:
        b = ir.Operation.create(
            "ascsimt.bitcast", operands=[unwrap_operand(dtypes.uint32(rhs))], results=[unsigned]
        ).results[0]
    result = ir.Operation.create(
        "ascsimt.math_binary", operands=[a, b], results=[unsigned], attributes={"function": ir.StringAttr.get("mulhi")}
    ).results[0]
    return simt._wrap_simt_scalar(result, storage_type=unsigned)


def _nibble_fields32(self, ids, scratch, part, mask):
    index = reg.vadds(reg.vshr(ids, 3, mask=mask), part * 32, mask=mask)
    word = reg.vreinterpret(reg.vgather(scratch, index, mask=mask), dtypes.uint32)
    field = reg.vbitwise_and(ids, reg.vdups(7, dtypes.uint32, mask=mask), mask=mask)
    if const_expr(self.mantissa == 3):
        value = reg.vbitwise_or(reg.vshl(word, 12, mask=mask), reg.vshr(word, 20, mask=mask), mask=mask)
        first = reg.vbitwise_or(reg.vshr(value, 4, mask=mask), reg.vshl(value, 28, mask=mask), mask=mask)
    else:
        value = reg.vbitwise_or(reg.vshl(word, 11, mask=mask), reg.vshr(word, 21, mask=mask), mask=mask)
        first = reg.vbitwise_or(reg.vshr(value, 3, mask=mask), reg.vshl(value, 29, mask=mask), mask=mask)
    value = reg.vselect(
        first,
        value,
        cond_mask=reg.vnes(reg.vbitwise_and(field, reg.vdups(1, dtypes.uint32, mask=mask), mask=mask), 0, mask=mask),
    )
    r8 = reg.vbitwise_or(reg.vshr(value, 8, mask=mask), reg.vshl(value, 24, mask=mask), mask=mask)
    value = reg.vselect(
        r8,
        value,
        cond_mask=reg.vnes(reg.vbitwise_and(field, reg.vdups(2, dtypes.uint32, mask=mask), mask=mask), 0, mask=mask),
    )
    r16 = reg.vbitwise_or(reg.vshr(value, 16, mask=mask), reg.vshl(value, 16, mask=mask), mask=mask)
    value = reg.vselect(
        r16,
        value,
        cond_mask=reg.vnes(reg.vbitwise_and(field, reg.vdups(4, dtypes.uint32, mask=mask), mask=mask), 0, mask=mask),
    )
    return reg.vbitwise_and(value, reg.vdups(65535, dtypes.uint32, mask=mask), mask=mask)


@jit
def _quantize_four_field_tile(self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset):
    segments = (count + 63) // 64
    batches = (count + 1023) // 1024
    prepare_random_words(random, start, batches, quant_length, seed, block_offset)
    with vf(mode="simd"):
        full = reg.full_mask()
        ids = reg.varange(0, dtypes.uint32)
        lower = reg.vlts(ids, 32, mask=full)
        upper = reg.vges(ids, 32, mask=full)
        for segment in range(segments):
            offset = segment * 64
            active = reg.update_mask(count - offset, elem_bits=32)[0]
            if const_expr(self.bf16):
                loaded = reg.vload_unpack(x, offset, unpack_mode=reg.UnpackMode.B16_TO_B32)
                values = reg.vcast(loaded, dtypes.float32, mask=full)
            else:
                values = reg.vload(x, offset)
            values = reg.vselect(values, reg.vdups(0.0, dtypes.float32, mask=full), cond_mask=active)
            absolute = reg.vbitwise_and(
                reg.vreinterpret(values, dtypes.uint32), reg.vdups(2147483647, dtypes.uint32, mask=full), mask=full
            )
            max0 = reg.vdup(reg.vreduce_max(absolute, mask=lower), mask=full)
            max1 = reg.vdup(reg.vreduce_max(absolute, mask=upper), mask=full)
            maxima = reg.vselect(max0, max1, cond_mask=lower)
            block_scales, reciprocal = compute_scale(self, maxima, full)
            normalized = reg.vmul(values, reciprocal, mask=full)
            random16 = _random_fields_from_words(ids, random.buffer, segment, full)
            codes = _sr_fp32_to_fp8_q16(normalized, random16, self.mantissa, self.bias, self.max_code, full)
            reg.vstore_pack(y, offset, codes, active, pack_mode=reg.PackMode.B32_TO_B8)
            reg.vstore_first(self.scale32, segment * 2, block_scales)
            upper_scale = reg.vgather_reg(block_scales, reg.vdups(32, dtypes.uint32, mask=full))
            reg.vstore_first(self.scale32, segment * 2 + 1, upper_scale)
        store_scale(self, scales, count, scale_factor, segments, ids, full)


@jit
def _quantize_small_nibble_tail(
    self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
):
    segments = (count + 63) // 64
    batches = (count + 1023) // 1024
    _prepare_eight_field_aos_linear(random, start, batches, seed, block_offset)
    with vf(mode="simd"):
        full = reg.full_mask()
        ids = reg.varange(0, dtypes.uint32)
        lower = reg.vlts(ids, 32, mask=full)
        upper = reg.vges(ids, 32, mask=full)
        for segment in range(segments):
            offset = segment * 64
            active = reg.update_mask(count - offset, elem_bits=32)[0]
            if const_expr(self.bf16):
                loaded = reg.vload_unpack(x, offset, unpack_mode=reg.UnpackMode.B16_TO_B32)
                values = reg.vcast(loaded, dtypes.float32, mask=full)
            else:
                values = reg.vload(x, offset)
            values = reg.vselect(values, reg.vdups(0.0, dtypes.float32, mask=full), cond_mask=active)
            absolute = reg.vbitwise_and(
                reg.vreinterpret(values, dtypes.uint32), reg.vdups(2147483647, dtypes.uint32, mask=full), mask=full
            )
            max0 = reg.vdup(reg.vreduce_max(absolute, mask=lower), mask=full)
            max1 = reg.vdup(reg.vreduce_max(absolute, mask=upper), mask=full)
            maxima = reg.vselect(max0, max1, cond_mask=lower)
            block_scales, reciprocal = compute_scale(self, maxima, full)
            normalized = reg.vmul(values, reciprocal, mask=full)
            if const_expr(self.bf16):
                rounded = reg.vcast(normalized, dtypes.bfloat16, mask=full, rounding=reg.RoundingMode.RN)
                normalized = reg.vcast(rounded, dtypes.float32, mask=full)
            random16 = _nibble_fields32(
                self, reg.vadds(ids, segment % 4 * 64, mask=full), random.buffer, segment // 4, full
            )
            codes = _sr_fp32_to_fp8_q16(normalized, random16, self.mantissa, self.bias, self.max_code, full)
            reg.vstore_pack(y, offset, codes, active, pack_mode=reg.PackMode.B32_TO_B8)
            reg.vstore_first(self.scale32, segment * 2, block_scales)
            upper_scale = reg.vgather_reg(block_scales, reg.vdups(32, dtypes.uint32, mask=full))
            reg.vstore_first(self.scale32, segment * 2 + 1, upper_scale)
        store_scale(self, scales, count, scale_factor, segments, ids, full)


@jit
def _prepare_eight_field_aos_linear(random, start, batches, seed, block_offset):
    key0, key1 = _split_uint64(seed)
    group_low, group_high = _split_uint64(dtypes.uint64(start) // 32)
    offset_low, offset_high = _split_uint64(block_offset)
    base_low = dtypes.uint32(offset_low // 4 + offset_high % 4 * 1073741824)
    base_high = dtypes.uint32(offset_high // 4)
    linear_group_count = dtypes.int64(batches) * 32
    first_group = dtypes.uint64(start) // 32
    draw = first_group // 16384
    last_draw = (first_group + dtypes.uint64(linear_group_count) - 1) // 16384
    if draw == last_draw:
        draw_low, draw_high = _split_uint64(draw)
        counter_low = dtypes.uint32(base_low) + dtypes.uint32(draw_low)
        counter_high = (
            dtypes.uint32(base_high) + dtypes.uint32(draw_high) + dtypes.uint32(counter_low < dtypes.uint32(base_low))
        )
        first_low, first_high = _split_uint64(dtypes.uint64(counter_low) * dtypes.uint64(PHILOX_M1))
        uniform2 = dtypes.uint32(counter_high) ^ dtypes.uint32(key1)
        second_low, second_high = _split_uint64(dtypes.uint64(uniform2) * dtypes.uint64(PHILOX_M1))
        next_key0 = dtypes.uint32(key0) + dtypes.uint32(PHILOX_W0)
        next_key1 = dtypes.uint32(key1) + dtypes.uint32(PHILOX_W1)
        uniform_second0 = dtypes.uint32(second_high) ^ dtypes.uint32(first_low) ^ dtypes.uint32(next_key0)
        third_low, third_high = _split_uint64(dtypes.uint64(uniform_second0) * dtypes.uint64(PHILOX_M0))
        uniform_initial = dtypes.uint32(first_high) ^ dtypes.uint32(key0)
        uniform_x0 = dtypes.uint32(second_low) ^ dtypes.uint32(key0) + dtypes.uint32(2 * PHILOX_W0 & UINT32_MASK)
        uniform_x2 = dtypes.uint32(third_high) ^ dtypes.uint32(key1) + dtypes.uint32(2 * PHILOX_W1 & UINT32_MASK)
        with vf(mode="simt", thread=256):
            for group in range(simt.thread_idx()[0], linear_group_count, 256):
                subsequence = (dtypes.uint32(group_low) + dtypes.uint32(group)) % 16384
                first_c0 = subsequence ^ dtypes.uint32(uniform_initial)
                variable_low = dtypes.uint32(first_c0) * dtypes.uint32(PHILOX_M0)
                variable_high = _simt_umulhi(first_c0, PHILOX_M0)
                second_c2 = variable_high ^ dtypes.uint32(next_key1)
                p1lo = dtypes.uint32(second_c2) * dtypes.uint32(PHILOX_M1)
                p1hi = _simt_umulhi(second_c2, PHILOX_M1)
                c0 = p1hi ^ dtypes.uint32(uniform_x0)
                c1 = p1lo
                c2 = variable_low ^ dtypes.uint32(uniform_x2)
                c3 = dtypes.uint32(third_low)
                for step in range(3, 10):
                    p0lo = dtypes.uint32(c0) * dtypes.uint32(PHILOX_M0)
                    p0hi = _simt_umulhi(c0, PHILOX_M0)
                    p1lo = dtypes.uint32(c2) * dtypes.uint32(PHILOX_M1)
                    p1hi = _simt_umulhi(c2, PHILOX_M1)
                    k0 = dtypes.uint32(key0) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W0)
                    k1 = dtypes.uint32(key1) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W1)
                    c0, c1, c2, c3 = (p1hi ^ c1 ^ k0, p1lo, p0hi ^ c3 ^ k1, p0lo)
                random.buffer[0, group * 4] = dtypes.int32(c0)
                random.buffer[0, group * 4 + 1] = dtypes.int32(c1)
                random.buffer[0, group * 4 + 2] = dtypes.int32(c2)
                random.buffer[0, group * 4 + 3] = dtypes.int32(c3)
    else:
        with vf(mode="simt", thread=256):
            for group in range(simt.thread_idx()[0], linear_group_count, 256):
                gl = dtypes.uint32(group_low) + dtypes.uint32(group)
                gh = dtypes.uint32(group_high) + dtypes.uint32(gl < dtypes.uint32(group_low))
                draw_low = dtypes.uint32(gl // 16384) + dtypes.uint32(gh % 16384) * dtypes.uint32(262144)
                draw_high = dtypes.uint32(gh // 16384)
                local_counter_low = dtypes.uint32(base_low) + dtypes.uint32(draw_low)
                local_counter_high = (
                    dtypes.uint32(base_high)
                    + dtypes.uint32(draw_high)
                    + dtypes.uint32(local_counter_low < dtypes.uint32(base_low))
                )
                c0 = dtypes.uint32(0)
                c1 = dtypes.uint32(gl % 16384)
                c2 = local_counter_low
                c3 = local_counter_high
                for step in range(10):
                    p0lo = dtypes.uint32(c0) * dtypes.uint32(PHILOX_M0)
                    p0hi = _simt_umulhi(c0, PHILOX_M0)
                    p1lo = dtypes.uint32(c2) * dtypes.uint32(PHILOX_M1)
                    p1hi = _simt_umulhi(c2, PHILOX_M1)
                    k0 = dtypes.uint32(key0) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W0)
                    k1 = dtypes.uint32(key1) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W1)
                    c0, c1, c2, c3 = (p1hi ^ c1 ^ k0, p1lo, p0hi ^ c3 ^ k1, p0lo)
                random.buffer[0, group * 4] = dtypes.int32(c0)
                random.buffer[0, group * 4 + 1] = dtypes.int32(c1)
                random.buffer[0, group * 4 + 2] = dtypes.int32(c2)
                random.buffer[0, group * 4 + 3] = dtypes.int32(c3)


@jit
def _emit_nibble_clean_part(self, random, x, y, first, stop):
    sign_source = reinterpret(x, dtypes.uint8, shape=(1, self.tile_elements * 2))
    with vf(mode="simd"):
        full16 = reg.full_mask(elem_bits=16)
        full8 = reg.full_mask(elem_bits=8)
        byte_mask16 = reg.vdups(255, dtypes.uint16, mask=full16)
        sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
        random_mask = reg.vdups((1 << 7 - self.mantissa) - 1, dtypes.uint16, mask=full16)
        random_bytes = reinterpret(random.buffer, dtypes.uint8, shape=(1, random.capacity_bytes))
        for part in range(first, stop):
            offset = part * 256
            av, bv = reg.vload_deinterleave(x, offset)
            raw = reg.vload(random_bytes, part * 128)
            expanded = reg.vunpack(raw, dtypes.uint16)
            ra = reg.vbitwise_and(expanded, random_mask, mask=full16)
            reverse = reg.vshr(expanded, 1 + self.mantissa, mask=full16)
            ca = reg.vshr(
                reg.vadd(reg.vreinterpret(av, dtypes.uint16), ra, mask=full16), 7 - self.mantissa, mask=full16
            )
            cb = reg.vshr(
                reg.vadd(reg.vreinterpret(bv, dtypes.uint16), reverse, mask=full16), 7 - self.mantissa, mask=full16
            )
            paired = reg.vbitwise_or(
                reg.vbitwise_and(ca, byte_mask16, mask=full16), reg.vshl(cb, 8, mask=full16), mask=full16
            )
            codes8 = reg.vreinterpret_lanes(paired, dtypes.uint8)
            codes8 = reg.vadds(codes8, -(127 - self.bias << self.mantissa) & 255, mask=full8)
            codes8 = reg.vmins(codes8, self.max_code, mask=full8)
            unused_low, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
            codes8 = reg.vbitwise_or(codes8, reg.vbitwise_and(signbytes, sign_mask8, mask=full8), mask=full8)
            reg.vstore(y, offset, codes8, full8)
        # Complete encoded byte stores before subsequent VFs reuse UB or DMA consumes this output slot.
        reg.vmem_bar("vst_vld")


@jit
def _sr_bf16_middle_q12_direct(self, value, random12, absolute_mask, multiplier, mask):
    absolute = reg.vbitwise_and(reg.vreinterpret(value, dtypes.uint16), absolute_mask, mask=mask)
    small = reg.vlts(absolute, (128 - self.bias) * 128, mask=mask)
    normal = reg.vshr(
        reg.vadd(absolute, reg.vshr(random12, 5 + self.mantissa, mask=mask), mask=mask), 7 - self.mantissa, mask=mask
    )
    scaled = reg.vmul(reg.vreinterpret(absolute, dtypes.bfloat16), multiplier, mask=small)
    half = reg.vcast(scaled, dtypes.float16, mask=mask, rounding=reg.RoundingMode.RZ)
    q12 = reg.vreinterpret(reg.vcast(half, dtypes.int16, mask=mask, rounding=reg.RoundingMode.RZ), dtypes.uint16)
    small_code = reg.vshr(reg.vadd(q12, random12, mask=mask), 12, mask=mask)
    small_code = reg.vadds(small_code, 127 - self.bias << self.mantissa & 65535, mask=mask)
    return reg.vselect(small_code, normal, cond_mask=small)


@jit
def _emit_nibble_middle_part(self, random, x, y, first, stop):
    sign_source = reinterpret(x, dtypes.uint8, shape=(1, self.tile_elements * 2))
    with vf(mode="simd"):
        full16 = reg.full_mask(elem_bits=16)
        full8 = reg.full_mask(elem_bits=8)
        byte_mask16 = reg.vdups(255, dtypes.uint16, mask=full16)
        sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
        ids_byte = reg.varange(0, dtypes.uint8)
        pairs = reg.vshr(ids_byte, 1, mask=full8)
        base = reg.vshl(reg.vshr(pairs, 2, mask=full8), 2, mask=full8)
        byte = reg.vbitwise_and(ids_byte, reg.vdups(1, dtypes.uint8, mask=full8), mask=full8)
        position = reg.vbitwise_and(
            reg.vadd(reg.vadds(pairs, 3, mask=full8), byte, mask=full8),
            reg.vdups(3, dtypes.uint8, mask=full8),
            mask=full8,
        )
        pair_index = reg.vbitwise_or(base, position, mask=full8)
        field_mask = reg.vdups(4095, dtypes.uint16, mask=full16)
        random_bytes = reinterpret(random.buffer, dtypes.uint8, shape=(1, random.capacity_bytes))
        absolute_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
        multiplier = reg.vdups(2.0 ** (self.bias + self.mantissa + 11), dtypes.bfloat16, mask=full16)
        for part in range(first, stop):
            offset = part * 256
            av, bv = reg.vload_deinterleave(x, offset)
            raw = reg.vload(random_bytes, part * 128)
            window = reg.vreinterpret_lanes(reg.vgather_reg(raw, pair_index), dtypes.uint16)
            if const_expr(self.mantissa == 3):
                ra = reg.vbitwise_and(window, field_mask, mask=full16)
            else:
                ra = reg.vbitwise_and(reg.vshr(window, 1, mask=full16), field_mask, mask=full16)
            reverse = reg.vshr(window, 4, mask=full16)
            ca = _sr_bf16_middle_q12_direct(self, av, ra, absolute_mask, multiplier, full16)
            cb = _sr_bf16_middle_q12_direct(self, bv, reverse, absolute_mask, multiplier, full16)
            paired = reg.vbitwise_or(
                reg.vbitwise_and(ca, byte_mask16, mask=full16), reg.vshl(cb, 8, mask=full16), mask=full16
            )
            codes8 = reg.vreinterpret_lanes(paired, dtypes.uint8)
            codes8 = reg.vadds(codes8, -(127 - self.bias << self.mantissa) & 255, mask=full8)
            codes8 = reg.vmins(codes8, self.max_code, mask=full8)
            unused_low, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
            codes8 = reg.vbitwise_or(codes8, reg.vbitwise_and(signbytes, sign_mask8, mask=full8), mask=full8)
            reg.vstore(y, offset, codes8, full8)
        # Complete encoded byte stores before subsequent VFs reuse UB or DMA consumes this output slot.
        reg.vmem_bar("vst_vld")


@jit
def _emit_nibble_general_part(self, random, x, y, first, stop, packed_scales=None):
    if const_expr(self.bf16_four_panel or self.bf16_compact_four_panel):
        sign_source = reinterpret(x, dtypes.uint8, shape=(1, self.tile_elements * 2))
        scale16 = reinterpret(self.scale32, dtypes.uint16, shape=(1, self.tile_elements // 16))
        with vf(mode="simd"):
            full = reg.full_mask()
            full16 = reg.full_mask(elem_bits=16)
            full8 = reg.full_mask(elem_bits=8)
            eight16 = reg.update_mask(8, elem_bits=16)[0]
            ids8 = reg.varange(0, dtypes.uint8)
            sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
            index8 = reg.vshr(ids8, 5, mask=full8)
            pack_group = reg.vbitwise_and(ids8, reg.vdups(3, dtypes.uint8, mask=full8), mask=full8)
            pack_index = reg.vbitwise_or(reg.vshl(pack_group, 6, mask=full8), reg.vshr(ids8, 2, mask=full8), mask=full8)
            random_bytes = reinterpret(random.buffer, dtypes.uint8, shape=(1, random.capacity_bytes))
            byte_ids = reg.varange(0, dtypes.uint8)
            word_base = reg.vshl(reg.vshr(byte_ids, 3, mask=full8), 2, mask=full8)
            lane_parity = reg.vbitwise_and(
                reg.vshr(byte_ids, 2, mask=full8), reg.vdups(1, dtypes.uint8, mask=full8), mask=full8
            )
            start_byte = reg.vshl(
                reg.vbitwise_xor(lane_parity, reg.vdups(1, dtypes.uint8, mask=full8), mask=full8), 1, mask=full8
            )
            byte_position = reg.vbitwise_and(byte_ids, reg.vdups(3, dtypes.uint8, mask=full8), mask=full8)
            position_a = reg.vbitwise_and(
                reg.vadd(start_byte, byte_position, mask=full8), reg.vdups(3, dtypes.uint8, mask=full8), mask=full8
            )
            index_a = reg.vbitwise_or(word_base, position_a, mask=full8)
            position_b = reg.vbitwise_and(
                reg.vadds(position_a, 1, mask=full8), reg.vdups(3, dtypes.uint8, mask=full8), mask=full8
            )
            index_b = reg.vbitwise_or(word_base, position_b, mask=full8)
            field_mask = reg.vdups(65535, dtypes.uint32, mask=full)
            if const_expr(self.bf16_full_tiles and self.scale_alg == 0 and self.mantissa == 3):
                packed_scale_batch = reg.vload(packed_scales, first // 32 * 256)
            for part in range(first, stop):
                offset = part * 256
                av, bv = reg.vload_deinterleave(x, offset)
                absolute_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
                av = reg.vreinterpret(
                    reg.vbitwise_and(reg.vreinterpret(av, dtypes.uint16), absolute_mask, mask=full16), dtypes.bfloat16
                )
                bv = reg.vreinterpret(
                    reg.vbitwise_and(reg.vreinterpret(bv, dtypes.uint16), absolute_mask, mask=full16), dtypes.bfloat16
                )
                v0 = reg.vcast(av, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v2 = reg.vcast(av, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                v1 = reg.vcast(bv, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v3 = reg.vcast(bv, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                raw_fields = reg.vload(random_bytes, part * 128)
                rotated_a = reg.vreinterpret_lanes(reg.vgather_reg(raw_fields, index_a), dtypes.uint32)
                rotated_b = reg.vreinterpret_lanes(reg.vgather_reg(raw_fields, index_b), dtypes.uint32)
                r0 = reg.vbitwise_and(reg.vshr(rotated_a, 7 - self.mantissa, mask=full), field_mask, mask=full)
                r1 = reg.vbitwise_and(rotated_b, field_mask, mask=full)
                r2 = reg.vbitwise_and(reg.vshr(rotated_b, 7 - self.mantissa, mask=full), field_mask, mask=full)
                r3 = reg.vshr(rotated_a, 16, mask=full)
                c0 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v0, dtypes.uint32), r0, self.mantissa, self.bias, full
                )
                c1 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v1, dtypes.uint32), r1, self.mantissa, self.bias, full
                )
                c2 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v2, dtypes.uint32), r2, self.mantissa, self.bias, full
                )
                c3 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v3, dtypes.uint32), r3, self.mantissa, self.bias, full
                )
                unused01, code01 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(c0, dtypes.uint16), reg.vreinterpret_lanes(c1, dtypes.uint16)
                )
                unused23, code23 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(c2, dtypes.uint16), reg.vreinterpret_lanes(c3, dtypes.uint16)
                )
                code_bytes, unused_bytes = reg.vdeinterleave(
                    reg.vreinterpret_lanes(code01, dtypes.uint8), reg.vreinterpret_lanes(code23, dtypes.uint8)
                )
                codes8 = reg.vgather_reg(code_bytes, pack_index)
                magnitude8 = reg.vadds(codes8, -(128 - self.bias << self.mantissa) & 255, mask=full8)
                codes8 = reg.vmins(magnitude8, self.max_code, mask=reg.vnes(magnitude8, 255, mask=full8))
                unused_low, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
                codes8 = reg.vbitwise_or(codes8, reg.vbitwise_and(signbytes, sign_mask8, mask=full8), mask=full8)
                if const_expr(self.bf16_full_tiles and self.scale_alg == 0 and self.mantissa == 3):
                    scale_indices = reg.vadds(index8, (part % 32) * 8, mask=full8)
                    scale_codes = reg.vgather_reg(packed_scale_batch, scale_indices)
                else:
                    ids = reg.vadds(reg.varange(0, dtypes.uint16), part * 16, mask=full16)
                    scales16 = reg.vgather(scale16, ids, mask=eight16)
                    scales8 = reg.vpack(scales16, dtypes.uint8, part="lower")
                    scale_codes = reg.vgather_reg(scales8, index8)
                invalid = reg.veqs(scale_codes, 255, mask=full8)
                codes8 = reg.vdups(127, dtypes.uint8, mask=invalid, mode="merging", merge=codes8)
                reg.vstore(y, offset, codes8, full8)
            # Finish encoded byte stores before the next VF or DMA reuses this output slot.
            reg.vmem_bar("vst_vld")
    else:
        sign_source = reinterpret(x, dtypes.uint8, shape=(1, self.tile_elements * 2))
        scale16 = reinterpret(self.scale32, dtypes.uint16, shape=(1, self.tile_elements // 16))
        with vf(mode="simd"):
            full = reg.full_mask()
            full16 = reg.full_mask(elem_bits=16)
            full8 = reg.full_mask(elem_bits=8)
            eight16 = reg.update_mask(8, elem_bits=16)[0]
            ids8 = reg.varange(0, dtypes.uint8)
            sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
            index8 = reg.vshr(ids8, 5, mask=full8)
            pack_group = reg.vbitwise_and(ids8, reg.vdups(3, dtypes.uint8, mask=full8), mask=full8)
            pack_index = reg.vbitwise_or(reg.vshl(pack_group, 6, mask=full8), reg.vshr(ids8, 2, mask=full8), mask=full8)
            for part in range(first, stop):
                offset = part * 256
                av, bv = reg.vload_deinterleave(x, offset)
                absolute_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
                av = reg.vreinterpret(
                    reg.vbitwise_and(reg.vreinterpret(av, dtypes.uint16), absolute_mask, mask=full16), dtypes.bfloat16
                )
                bv = reg.vreinterpret(
                    reg.vbitwise_and(reg.vreinterpret(bv, dtypes.uint16), absolute_mask, mask=full16), dtypes.bfloat16
                )
                v0 = reg.vcast(av, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v2 = reg.vcast(av, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                v1 = reg.vcast(bv, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v3 = reg.vcast(bv, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                r0 = _nibble_fields32(
                    self,
                    reg.vadds(reg.vshl(reg.varange(0, dtypes.uint32), 2, mask=full), 0, mask=full),
                    random.buffer,
                    part,
                    full,
                )
                r1 = _nibble_fields32(
                    self,
                    reg.vadds(reg.vshl(reg.varange(0, dtypes.uint32), 2, mask=full), 1, mask=full),
                    random.buffer,
                    part,
                    full,
                )
                r2 = _nibble_fields32(
                    self,
                    reg.vadds(reg.vshl(reg.varange(0, dtypes.uint32), 2, mask=full), 2, mask=full),
                    random.buffer,
                    part,
                    full,
                )
                r3 = _nibble_fields32(
                    self,
                    reg.vadds(reg.vshl(reg.varange(0, dtypes.uint32), 2, mask=full), 3, mask=full),
                    random.buffer,
                    part,
                    full,
                )
                c0 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v0, dtypes.uint32), r0, self.mantissa, self.bias, full
                )
                c1 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v1, dtypes.uint32), r1, self.mantissa, self.bias, full
                )
                c2 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v2, dtypes.uint32), r2, self.mantissa, self.bias, full
                )
                c3 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v3, dtypes.uint32), r3, self.mantissa, self.bias, full
                )
                unused01, code01 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(c0, dtypes.uint16), reg.vreinterpret_lanes(c1, dtypes.uint16)
                )
                unused23, code23 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(c2, dtypes.uint16), reg.vreinterpret_lanes(c3, dtypes.uint16)
                )
                code_bytes, unused_bytes = reg.vdeinterleave(
                    reg.vreinterpret_lanes(code01, dtypes.uint8), reg.vreinterpret_lanes(code23, dtypes.uint8)
                )
                codes8 = reg.vgather_reg(code_bytes, pack_index)
                magnitude8 = reg.vadds(codes8, -(128 - self.bias << self.mantissa) & 255, mask=full8)
                codes8 = reg.vmins(magnitude8, self.max_code, mask=reg.vnes(magnitude8, 255, mask=full8))
                unused_low, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
                codes8 = reg.vbitwise_or(codes8, reg.vbitwise_and(signbytes, sign_mask8, mask=full8), mask=full8)
                ids = reg.vadds(reg.varange(0, dtypes.uint16), part * 16, mask=full16)
                scales16 = reg.vgather(scale16, ids, mask=eight16)
                scales8 = reg.vpack(scales16, dtypes.uint8, part="lower")
                invalid = reg.veqs(reg.vgather_reg(scales8, index8), 255, mask=full8)
                codes8 = reg.vdups(127, dtypes.uint8, mask=invalid, mode="merging", merge=codes8)
                reg.vstore(y, offset, codes8, full8)
            # Finish encoded byte stores before the next VF or DMA reuses this output slot.
            reg.vmem_bar("vst_vld")


@jit
def _quantize_bf16_nibble_partial_256(
    self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
):
    parts = count // 256
    batches = (count + 1023) // 1024
    scale16 = reinterpret(self.scale32, dtypes.uint16, shape=(1, self.tile_elements // 16))
    _prepare_eight_field_aos_linear(random, start, batches, seed, block_offset)
    with vf(mode="simd"):
        full = reg.full_mask()
        full16 = reg.full_mask(elem_bits=16)
        full8 = reg.full_mask(elem_bits=8)
        eight16 = reg.update_mask(8, elem_bits=16)[0]
        abs_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
        index16 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
        pack_ids8 = reg.varange(0, dtypes.uint8)
        scale_byte_slot = reg.vbitwise_and(pack_ids8, reg.vdups(1, dtypes.uint8, mask=full8), mask=full8)
        scale_byte_mask = reg.mask_and(
            reg.vlts(pack_ids8, 16, mask=full8),
            reg.vlts(scale_byte_slot, dtypes.uint8(scale_factor // 2), mask=full8),
            exec_mask=full8,
        )
        scale_cursor = reg.vstore_unalign_begin(scales)
        group_min = reg.vdups(65535, dtypes.uint16, mask=full16)
        group_invalid = reg.vdups(0, dtypes.uint16, mask=full16)
        for part in range(parts):
            offset = part * 256
            a, b = reg.vload_deinterleave(x, offset)
            aa = reg.vbitwise_and(reg.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
            ab = reg.vbitwise_and(reg.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
            maxima16 = reg.vreduce_max_datablock(reg.vmax(aa, ab, mask=full16), mask=full16)
            if const_expr(self.scale_alg == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                exponent16 = reg.vshr(maxima16, 7, mask=full16)
                invalid16 = reg.vges(exponent16, 255, mask=full16)
                code16 = reg.vadds(reg.vmaxs(exponent16, scale_shift, mask=full16), -scale_shift & 65535, mask=full16)
                code16 = reg.vdups(255, dtypes.uint16, mask=invalid16, mode="merging", merge=code16)
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16),
                    7,
                    mask=reg.vnes(code16, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                active16 = reg.vgts(maxima16, scale_shift << 7 | 96, mask=full16)
                code16 = reg.vshr(reg.vadds(maxima16, 31 - (scale_shift << 7) & 65535, mask=full16), 7, mask=active16)
                code16 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16, 32640, mask=full16), mode="merging", merge=code16
                )
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16), 7, mask=full16
                )
            else:
                maxima = reg.vshl(reg.vunpack(maxima16, dtypes.uint32), 16, mask=full)
                block_scales, reciprocal = compute_scale(self, maxima, full, False)
                bits16 = reg.vshr(reg.vreinterpret(reciprocal, dtypes.uint32), 16, mask=full)
                r16 = reg.vpack(bits16, dtypes.uint16, part="lower")
                code16 = reg.vpack(block_scales, dtypes.uint16, part="lower")
            group_invalid = reg.vmax(group_invalid, code16, mask=eight16)
            broadcast = reg.vreinterpret(reg.vgather_reg(r16, index16), dtypes.bfloat16)
            av = reg.vmul(a, broadcast, mask=full16)
            bv = reg.vmul(b, broadcast, mask=full16)
            reg.vstore_interleave(x, offset, av, bv)
            aa_norm = reg.vbitwise_and(reg.vreinterpret(av, dtypes.uint16), abs_mask, mask=full16)
            ab_norm = reg.vbitwise_and(reg.vreinterpret(bv, dtypes.uint16), abs_mask, mask=full16)
            panel_min = reg.vmin(aa_norm, ab_norm, mask=full16)
            group_min = reg.vmin(group_min, panel_min, mask=full16)
            reg.vstore(scale16, part * 16, code16, eight16)
            scale_bytes = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(code16, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes, scale_cursor)
        minimum = reg.vreduce_min(group_min, mask=full16)
        maximum_scale = reg.vreduce_max(group_invalid, mask=eight16)
        minimum = reg.vdups(
            0, dtypes.uint16, mask=reg.veqs(maximum_scale, 255, mask=full16), mode="merging", merge=minimum
        )
        reg.vsqueeze_and_storeunalign_finalize(scales, 0, scale_cursor)
        reg.vstore_first(self.norm_flags, 0, minimum)
        # Publish normalized x, scales and minima before scalar classification and encoding reloads.
        reg.vmem_bar("vst_vld")
    flag_min = dtypes.int64(self.norm_flags[0, 0])
    if flag_min >= (128 - self.bias) * 128:
        _emit_nibble_clean_part(self, random, x, y, 0, parts)
    elif flag_min >= (123 - self.bias - self.mantissa) * 128:
        _emit_nibble_middle_part(self, random, x, y, 0, parts)
    else:
        _emit_nibble_general_part(self, random, x, y, 0, parts)


@jit
def _quantize_bf16_nibble_tile_256(
    self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
):
    if const_expr(self.bf16_compact_four_panel or self.bf16_four_panel):
        _quantize_bf16_four_panel(
            self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
        )
    else:
        _quantize_bf16_single_panel(
            self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
        )


def _fp32_eight_fields_from_aos(ids, scratch, segment, mask):
    full8 = reg.full_mask(elem_bits=8)
    raw = reg.vreinterpret_lanes(reg.vload(scratch, segment * 8), dtypes.uint8)
    byte_ids = reg.varange(0, dtypes.uint8)
    elements = reg.vshr(byte_ids, 2, mask=full8)
    base = reg.vshl(reg.vshr(elements, 3, mask=full8), 2, mask=full8)
    field = reg.vbitwise_and(elements, reg.vdups(7, dtypes.uint8, mask=full8), mask=full8)
    side = reg.vbitwise_and(
        reg.vbitwise_xor(field, byte_ids, mask=full8), reg.vdups(1, dtypes.uint8, mask=full8), mask=full8
    )
    position = reg.vbitwise_and(
        reg.vadd(reg.vshr(field, 1, mask=full8), side, mask=full8), reg.vdups(3, dtypes.uint8, mask=full8), mask=full8
    )
    index = reg.vbitwise_or(base, position, mask=full8)
    selected = reg.vreinterpret_lanes(reg.vgather_reg(raw, index), dtypes.uint32)
    return reg.vbitwise_and(selected, reg.vdups(65535, dtypes.uint32, mask=mask), mask=mask)


@jit
def _quantize_fp32_eight_small(
    self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
):
    segments = (count + 63) // 64
    batches = (count + 1023) // 1024
    _prepare_eight_field_aos_linear(random, start, batches, seed, block_offset)
    with vf(mode="simd"):
        full = reg.full_mask()
        ids = reg.varange(0, dtypes.uint32)
        lower = reg.vlts(ids, 32, mask=full)
        upper = reg.vges(ids, 32, mask=full)
        for segment in range(segments):
            offset = segment * 64
            active = reg.update_mask(count - offset, elem_bits=32)[0]
            if const_expr(self.bf16):
                loaded = reg.vload_unpack(x, offset, unpack_mode=reg.UnpackMode.B16_TO_B32)
                values = reg.vcast(loaded, dtypes.float32, mask=full)
            else:
                values = reg.vload(x, offset)
            values = reg.vselect(values, reg.vdups(0.0, dtypes.float32, mask=full), cond_mask=active)
            absolute = reg.vbitwise_and(
                reg.vreinterpret(values, dtypes.uint32), reg.vdups(2147483647, dtypes.uint32, mask=full), mask=full
            )
            max0 = reg.vdup(reg.vreduce_max(absolute, mask=lower), mask=full)
            max1 = reg.vdup(reg.vreduce_max(absolute, mask=upper), mask=full)
            maxima = reg.vselect(max0, max1, cond_mask=lower)
            block_scales, reciprocal = compute_scale(self, maxima, full)
            normalized = reg.vmul(values, reciprocal, mask=full)
            if const_expr(self.bf16):
                rounded = reg.vcast(normalized, dtypes.bfloat16, mask=full, rounding=reg.RoundingMode.RN)
                normalized = reg.vcast(rounded, dtypes.float32, mask=full)
            random16 = _fp32_eight_fields_from_aos(ids, random.buffer, segment, full)
            codes = _sr_fp32_to_fp8_q16(normalized, random16, self.mantissa, self.bias, self.max_code, full)
            reg.vstore_pack(y, offset, codes, active, pack_mode=reg.PackMode.B32_TO_B8)
            reg.vstore_first(self.scale32, segment * 2, block_scales)
            upper_scale = reg.vgather_reg(block_scales, reg.vdups(32, dtypes.uint32, mask=full))
            reg.vstore_first(self.scale32, segment * 2 + 1, upper_scale)
        store_scale(self, scales, count, scale_factor, segments, ids, full)


@jit
def prepare_random_words_aos_linear(random, start, batches, seed, block_offset):
    key0, key1 = _split_uint64(seed)
    group_low, group_high = _split_uint64(dtypes.uint64(start) // 16)
    offset_low, offset_high = _split_uint64(block_offset)
    base_low = dtypes.uint32(offset_low // 4 + offset_high % 4 * 1073741824)
    base_high = dtypes.uint32(offset_high // 4)
    linear_group_count = dtypes.int64(batches) * 64
    first_group = dtypes.uint64(start) // 16
    draw = first_group // 16384
    last_draw = (first_group + dtypes.uint64(linear_group_count) - 1) // 16384
    if draw == last_draw:
        draw_low, draw_high = _split_uint64(draw)
        counter_low = dtypes.uint32(base_low) + dtypes.uint32(draw_low)
        counter_high = (
            dtypes.uint32(base_high) + dtypes.uint32(draw_high) + dtypes.uint32(counter_low < dtypes.uint32(base_low))
        )
        first_low, first_high = _split_uint64(dtypes.uint64(counter_low) * dtypes.uint64(PHILOX_M1))
        uniform2 = dtypes.uint32(counter_high) ^ dtypes.uint32(key1)
        second_low, second_high = _split_uint64(dtypes.uint64(uniform2) * dtypes.uint64(PHILOX_M1))
        next_key0 = dtypes.uint32(key0) + dtypes.uint32(PHILOX_W0)
        next_key1 = dtypes.uint32(key1) + dtypes.uint32(PHILOX_W1)
        uniform_second0 = dtypes.uint32(second_high) ^ dtypes.uint32(first_low) ^ dtypes.uint32(next_key0)
        third_low, third_high = _split_uint64(dtypes.uint64(uniform_second0) * dtypes.uint64(PHILOX_M0))
        uniform_initial = dtypes.uint32(first_high) ^ dtypes.uint32(key0)
        uniform_x0 = dtypes.uint32(second_low) ^ dtypes.uint32(key0) + dtypes.uint32(2 * PHILOX_W0 & UINT32_MASK)
        uniform_x2 = dtypes.uint32(third_high) ^ dtypes.uint32(key1) + dtypes.uint32(2 * PHILOX_W1 & UINT32_MASK)
        with vf(mode="simt", thread=256):
            for group in range(simt.thread_idx()[0], linear_group_count, 256):
                subsequence = (dtypes.uint32(group_low) + dtypes.uint32(group)) % 16384
                first_c0 = subsequence ^ dtypes.uint32(uniform_initial)
                variable_low = dtypes.uint32(first_c0) * dtypes.uint32(PHILOX_M0)
                variable_high = _simt_umulhi(first_c0, PHILOX_M0)
                second_c2 = variable_high ^ dtypes.uint32(next_key1)
                p1lo = dtypes.uint32(second_c2) * dtypes.uint32(PHILOX_M1)
                p1hi = _simt_umulhi(second_c2, PHILOX_M1)
                c0 = p1hi ^ dtypes.uint32(uniform_x0)
                c1 = p1lo
                c2 = variable_low ^ dtypes.uint32(uniform_x2)
                c3 = dtypes.uint32(third_low)
                for step in range(3, 10):
                    p0lo = dtypes.uint32(c0) * dtypes.uint32(PHILOX_M0)
                    p0hi = _simt_umulhi(c0, PHILOX_M0)
                    p1lo = dtypes.uint32(c2) * dtypes.uint32(PHILOX_M1)
                    p1hi = _simt_umulhi(c2, PHILOX_M1)
                    k0 = dtypes.uint32(key0) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W0)
                    k1 = dtypes.uint32(key1) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W1)
                    c0, c1, c2, c3 = (p1hi ^ c1 ^ k0, p1lo, p0hi ^ c3 ^ k1, p0lo)
                random.buffer[0, group * 4] = dtypes.int32(c0)
                random.buffer[0, group * 4 + 1] = dtypes.int32(c1)
                random.buffer[0, group * 4 + 2] = dtypes.int32(c2)
                random.buffer[0, group * 4 + 3] = dtypes.int32(c3)
    else:
        with vf(mode="simt", thread=256):
            for group in range(simt.thread_idx()[0], linear_group_count, 256):
                gl = dtypes.uint32(group_low) + dtypes.uint32(group)
                gh = dtypes.uint32(group_high) + dtypes.uint32(gl < dtypes.uint32(group_low))
                draw_low = dtypes.uint32(gl // 16384) + dtypes.uint32(gh % 16384) * dtypes.uint32(262144)
                draw_high = dtypes.uint32(gh // 16384)
                local_counter_low = dtypes.uint32(base_low) + dtypes.uint32(draw_low)
                local_counter_high = (
                    dtypes.uint32(base_high)
                    + dtypes.uint32(draw_high)
                    + dtypes.uint32(local_counter_low < dtypes.uint32(base_low))
                )
                c0 = dtypes.uint32(0)
                c1 = dtypes.uint32(gl % 16384)
                c2 = local_counter_low
                c3 = local_counter_high
                for step in range(10):
                    p0lo = dtypes.uint32(c0) * dtypes.uint32(PHILOX_M0)
                    p0hi = _simt_umulhi(c0, PHILOX_M0)
                    p1lo = dtypes.uint32(c2) * dtypes.uint32(PHILOX_M1)
                    p1hi = _simt_umulhi(c2, PHILOX_M1)
                    k0 = dtypes.uint32(key0) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W0)
                    k1 = dtypes.uint32(key1) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W1)
                    c0, c1, c2, c3 = (p1hi ^ c1 ^ k0, p1lo, p0hi ^ c3 ^ k1, p0lo)
                random.buffer[0, group * 4] = dtypes.int32(c0)
                random.buffer[0, group * 4 + 1] = dtypes.int32(c1)
                random.buffer[0, group * 4 + 2] = dtypes.int32(c2)
                random.buffer[0, group * 4 + 3] = dtypes.int32(c3)


@jit
def prepare_random_words_aos(self, random, start, batches, quant_length, seed, block_offset):
    if const_expr(self.fp32_eight_tail):
        if quant_length % PACKED_K_ALIGNMENT_ELEMENTS == 0:
            _prepare_eight_field_aos_linear(random, start, batches, seed, block_offset)
        else:
            prepare_random_words_aos_linear(random, start, batches, seed, block_offset)
    else:
        if const_expr(self.bf16_crossdraw_prefix3):
            _prepare_fp4_crossdraw_aos_linear(random, start, batches, seed, block_offset)
        else:
            prepare_random_words_aos_linear(random, start, batches, seed, block_offset)


@jit
def prepare_random_words(random, start, batches, quant_length, seed, block_offset):
    key0, key1 = _split_uint64(seed)
    axis_groups, reciprocal, row_base, col_base = _rng_geometry(start, quant_length)
    row_low, row_high = _split_uint64(row_base)
    col_low, col_high = _split_uint64(col_base)
    step_q = dtypes.int64(64) // axis_groups
    step_r = dtypes.int64(64) % axis_groups
    with vf(mode="simd"):
        rl, rh, cl, ch = (row_low, row_high, col_low, col_high)
        for batch in range(batches):
            row_cursor, col_cursor = _cursor_coordinates(rl, rh, cl, ch)
            _generate_random_words_keyed(
                random.buffer,
                row_cursor,
                col_cursor,
                axis_groups,
                reciprocal,
                batch * 256,
                key0,
                key1,
                block_offset,
            )
            rl, rh, cl, ch = _advance_cursor_words(row_cursor, col_cursor, step_q, step_r, axis_groups)
        # Publish Philox scratch stores before SR VFs gather random words.
        reg.vmem_bar("vst_vld")


def _sr_fp32_to_fp8_q16(value, r16, mantissa, bias, max_code, mask):
    """Encode base/discarded16 together; retain Algorithm 19's exact carry."""
    bits = reg.vreinterpret(value, dtypes.uint32)
    sign = reg.vbitwise_and(reg.vshr(bits, 24, mask=mask), reg.vdups(128, dtypes.uint32, mask=mask), mask=mask)
    absolute = reg.vbitwise_and(bits, reg.vdups(2147483647, dtypes.uint32, mask=mask), mask=mask)
    threshold = 128 - bias << 23
    normal_q = reg.vsub(
        reg.vshr(absolute, 7 - mantissa, mask=mask),
        reg.vdups(127 - bias << mantissa + 16, dtypes.uint32, mask=mask),
        mask=mask,
    )
    absolute_value = reg.vreinterpret(absolute, dtypes.float32)
    scaled = reg.vmuls(absolute_value, float(2 ** (bias + mantissa + 15)), mask=mask)
    sub_q = reg.vreinterpret(reg.vcast(scaled, dtypes.int32, mask=mask, rounding=reg.RoundingMode.RZ), dtypes.uint32)
    q = reg.vselect(normal_q, sub_q, cond_mask=reg.vges(absolute, threshold, mask=mask))
    code = reg.vshr(
        reg.vadd(q, reg.vbitwise_and(r16, reg.vdups(65535, dtypes.uint32, mask=mask), mask=mask), mask=mask),
        16,
        mask=mask,
    )
    code = reg.vbitwise_or(reg.vmins(code, max_code, mask=mask), sign, mask=mask)
    return reg.vselect(
        reg.vdups(127, dtypes.uint32, mask=mask), code, cond_mask=reg.vgts(absolute, 2139095040, mask=mask)
    )


def _scale_alg0(amax_bits, inv_max_bits, mask, canonical_invalid=True):
    """ASC ComputeScaleOcp: floor exponent; zero reciprocal when scale is zero."""
    shift = 15 if inv_max_bits == 932333861 else 8
    exponent = reg.vshr(amax_bits, 23, mask=mask)
    invalid = reg.vges(exponent, 255, mask=mask)
    code = reg.vadds(reg.vmaxs(exponent, shift, mask=mask), -shift & UINT32_MASK, mask=mask)
    code = reg.vdups(255, dtypes.uint32, mask=invalid, mode="merging", merge=code)
    reciprocal = reg.vshl(
        reg.vsub(reg.vdups(254, dtypes.uint32, mask=mask), code, mask=mask), 23, mask=reg.vnes(code, 0, mask=mask)
    )
    if canonical_invalid:
        reciprocal = reg.vdups(2139160576, dtypes.uint32, mask=invalid, mode="merging", merge=reciprocal)
    return (code, reg.vreinterpret(reciprocal, dtypes.float32))


def _scale_alg1(amax_bits, inv_max_bits, mask, canonical_invalid=True):
    """Return E8M0 code and the power-of-two normalization multiplier."""
    return _scale_alg1_integer(amax_bits, inv_max_bits, mask, canonical_invalid)


def _scale_alg1_integer(amax_bits, inv_max_bits, mask, canonical_invalid=True):
    """Fixed-format scale ceil, including the rounded half-subnormal boundary.

    Both reciprocal-max constants have the same significand. Above the
    exponent-zero boundary, product ceil advances iff the input significand
    exceeds 1.75. At that boundary, RN keeps the next FP32 value at exactly
    half the smallest normal product as well (hence 0x600001, not 0x600000).
    """
    shift = 15 if inv_max_bits == 932333861 else 8
    invalid = reg.vges(amax_bits, 2139095040, mask=mask)
    code = reg.vshr(
        reg.vadds(amax_bits, 2097151 - (shift << 23) & UINT32_MASK, mask=mask),
        23,
        mask=reg.vgts(amax_bits, shift << 23 | 6291457, mask=mask),
    )
    code = reg.vdups(255, dtypes.uint32, mask=invalid, mode="merging", merge=code)
    recip = reg.vshl(reg.vsub(reg.vdups(254, dtypes.uint32, mask=mask), code, mask=mask), 23, mask=mask)
    if canonical_invalid:
        recip = reg.vdups(2139160576, dtypes.uint32, mask=invalid, mode="merging", merge=recip)
    return (code, reg.vreinterpret(recip, dtypes.float32))


def _sr_magnitude_q16_unbiased_sum(absolute, r16, mantissa, bias, mask):
    value = reg.vreinterpret(absolute, dtypes.float32)
    magic = 2.0 ** (1 - bias) - 2.0 ** (-bias - 23)
    mapped = reg.vadd(value, reg.vmaxs(value, magic, mask=mask), mask=mask)
    q = reg.vshr(reg.vreinterpret(mapped, dtypes.uint32), 7 - mantissa, mask=mask)
    return reg.vadd(q, r16, mask=mask)


def compute_scale(self, maxima, mask, canonical_invalid=True):
    """Shared scale rule; packed SR handles invalid output bytes itself."""
    if const_expr(self.scale_alg == 1 and self.max_low_bound_bits != 0):
        original_zero = reg.veqs(maxima, 0, mask=mask)
        maxima = reg.vmaxs(maxima, self.max_low_bound_bits, mask=mask)
        maxima = reg.vselect(reg.vdups(0, dtypes.uint32, mask=mask), maxima, cond_mask=original_zero)
    if const_expr(self.scale_alg == 0):
        return _scale_alg0(maxima, self.inv_max_bits, mask, canonical_invalid)
    return _scale_alg1(maxima, self.inv_max_bits, mask, canonical_invalid)


@jit
def store_scale(self, scales, count, scale_factor, segments, ids, full):
    # Complete scale32 stores before gather.
    reg.vmem_bar("vst_vld")
    scale_count = segments * 2
    if scale_factor == 4:
        scale_count = count // 32 * 2
    for offset in range(0, scale_count, 64):
        scale_mask = reg.update_mask(scale_count - offset, elem_bits=32)[0]
        scale_ids = reg.vadds(ids, offset, mask=full)
        if scale_factor == 4:
            scale_codes = reg.vgather(self.scale32, reg.vshr(scale_ids, 1, mask=full), mask=scale_mask)
            even = reg.veqs(reg.vbitwise_and(ids, reg.vdups(1, dtypes.uint32, mask=full), mask=full), 0, mask=full)
            scale_codes = reg.vselect(scale_codes, reg.vdups(0, dtypes.uint32, mask=full), cond_mask=even)
            reg.vstore_pack(scales, offset, scale_codes, scale_mask, pack_mode=reg.PackMode.B32_TO_B8)
        else:
            scale_codes = reg.vgather(self.scale32, scale_ids, mask=scale_mask)
            reg.vstore_pack(scales, offset, scale_codes, scale_mask, pack_mode=reg.PackMode.B32_TO_B8)
    # Complete scale byte stores before the next UB consumer/DMA release.
    reg.vmem_bar("vst_vld")


@jit
def quantize_tile(self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset):
    if const_expr(self.fp32_eight_tail):
        if quant_length % PACKED_K_ALIGNMENT_ELEMENTS == 0:
            _quantize_fp32_eight_small(
                self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
            )
        else:
            _quantize_four_field_tile(
                self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
            )
    elif const_expr(self.bf16_guard_tail):
        if quant_length % PACKED_K_ALIGNMENT_ELEMENTS == 0:
            _quantize_small_nibble_tail(
                self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
            )
        else:
            _quantize_four_field_tile(
                self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
            )
    else:
        _quantize_four_field_tile(
            self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
        )


@jit
def _emit_bf16_clean_normal_part(self, random, x, y, first, stop):
    sign_source = reinterpret(x, dtypes.uint8, shape=(1, self.tile_elements * 2))
    with vf(mode="simd"):
        full = reg.full_mask()
        full16 = reg.full_mask(elem_bits=16)
        full8 = reg.full_mask(elem_bits=8)
        byte_mask16 = reg.vdups(255, dtypes.uint16, mask=full16)
        sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
        swap16 = reg.vbitwise_xor(reg.varange(0, dtypes.uint16), reg.vdups(1, dtypes.uint16, mask=full16), mask=full16)
        for part in range(first, stop):
            offset = part * 256
            av, bv = reg.vload_deinterleave(x, offset)
            word = reg.vreinterpret_lanes(reg.vload(random.buffer, part * 64), dtypes.uint16)
            reverse = reg.vreinterpret_lanes(
                _rotate32_words8(reg.vreinterpret_lanes(word, dtypes.uint32), full), dtypes.uint16
            )
            ra = reg.vgather_reg(word, swap16)
            ca = reg.vshr(
                reg.vadd(
                    reg.vreinterpret(av, dtypes.uint16), reg.vshr(ra, 9 + self.mantissa, mask=full16), mask=full16
                ),
                7 - self.mantissa,
                mask=full16,
            )
            cb = reg.vshr(
                reg.vadd(
                    reg.vreinterpret(bv, dtypes.uint16), reg.vshr(reverse, 9 + self.mantissa, mask=full16), mask=full16
                ),
                7 - self.mantissa,
                mask=full16,
            )
            paired = reg.vbitwise_or(
                reg.vbitwise_and(ca, byte_mask16, mask=full16), reg.vshl(cb, 8, mask=full16), mask=full16
            )
            codes8 = reg.vreinterpret_lanes(paired, dtypes.uint8)
            codes8 = reg.vadds(codes8, -(127 - self.bias << self.mantissa) & 255, mask=full8)
            codes8 = reg.vmins(codes8, self.max_code, mask=full8)
            unused_low, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
            codes8 = reg.vbitwise_or(codes8, reg.vbitwise_and(signbytes, sign_mask8, mask=full8), mask=full8)
            reg.vstore(y, offset, codes8, full8)
        # Complete encoded byte stores before subsequent VFs reuse UB or DMA consumes this output slot.
        reg.vmem_bar("vst_vld")


def _sr_bf16_middle_q12(self, value, random16, absolute_mask, multiplier, mask):
    absolute = reg.vbitwise_and(reg.vreinterpret(value, dtypes.uint16), absolute_mask, mask=mask)
    small = reg.vlts(absolute, (128 - self.bias) * 128, mask=mask)
    normal = reg.vshr(
        reg.vadd(absolute, reg.vshr(random16, 9 + self.mantissa, mask=mask), mask=mask), 7 - self.mantissa, mask=mask
    )
    scaled = reg.vmul(reg.vreinterpret(absolute, dtypes.bfloat16), multiplier, mask=small)
    half = reg.vcast(scaled, dtypes.float16, mask=mask, rounding=reg.RoundingMode.RZ)
    q12 = reg.vreinterpret(reg.vcast(half, dtypes.int16, mask=mask, rounding=reg.RoundingMode.RZ), dtypes.uint16)
    small_code = reg.vshr(reg.vadd(q12, reg.vshr(random16, 4, mask=mask), mask=mask), 12, mask=mask)
    small_code = reg.vadds(small_code, 127 - self.bias << self.mantissa & 65535, mask=mask)
    return reg.vselect(small_code, normal, cond_mask=small)


@jit
def _emit_bf16_middle_q12_part(self, random, x, y, first, stop):
    sign_source = reinterpret(x, dtypes.uint8, shape=(1, self.tile_elements * 2))
    with vf(mode="simd"):
        full = reg.full_mask()
        full16 = reg.full_mask(elem_bits=16)
        full8 = reg.full_mask(elem_bits=8)
        byte_mask16 = reg.vdups(255, dtypes.uint16, mask=full16)
        sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
        swap16 = reg.vbitwise_xor(reg.varange(0, dtypes.uint16), reg.vdups(1, dtypes.uint16, mask=full16), mask=full16)
        absolute_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
        multiplier = reg.vdups(2.0 ** (self.bias + self.mantissa + 11), dtypes.bfloat16, mask=full16)
        for part in range(first, stop):
            offset = part * 256
            av, bv = reg.vload_deinterleave(x, offset)
            word = reg.vreinterpret_lanes(reg.vload(random.buffer, part * 64), dtypes.uint16)
            reverse = reg.vreinterpret_lanes(
                _rotate32_words8(reg.vreinterpret_lanes(word, dtypes.uint32), full), dtypes.uint16
            )
            ra = reg.vgather_reg(word, swap16)
            ca = _sr_bf16_middle_q12(self, av, ra, absolute_mask, multiplier, full16)
            cb = _sr_bf16_middle_q12(self, bv, reverse, absolute_mask, multiplier, full16)
            paired = reg.vbitwise_or(
                reg.vbitwise_and(ca, byte_mask16, mask=full16), reg.vshl(cb, 8, mask=full16), mask=full16
            )
            codes8 = reg.vreinterpret_lanes(paired, dtypes.uint8)
            codes8 = reg.vadds(codes8, -(127 - self.bias << self.mantissa) & 255, mask=full8)
            codes8 = reg.vmins(codes8, self.max_code, mask=full8)
            unused_low, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
            codes8 = reg.vbitwise_or(codes8, reg.vbitwise_and(signbytes, sign_mask8, mask=full8), mask=full8)
            reg.vstore(y, offset, codes8, full8)
        # Complete encoded byte stores before subsequent VFs reuse UB or DMA consumes this output slot.
        reg.vmem_bar("vst_vld")


@jit
def _emit_bf16_general_part(self, random, x, y, first, stop):
    sign_source = reinterpret(x, dtypes.uint8, shape=(1, self.tile_elements * 2))
    scale16 = reinterpret(self.scale32, dtypes.uint16, shape=(1, self.tile_elements // 16))
    with vf(mode="simd"):
        full = reg.full_mask()
        full16 = reg.full_mask(elem_bits=16)
        full8 = reg.full_mask(elem_bits=8)
        eight16 = reg.update_mask(8, elem_bits=16)[0]
        ids8 = reg.varange(0, dtypes.uint8)
        sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
        index8 = reg.vshr(ids8, 5, mask=full8)
        pack_group = reg.vbitwise_and(ids8, reg.vdups(3, dtypes.uint8, mask=full8), mask=full8)
        pack_index = reg.vbitwise_or(reg.vshl(pack_group, 6, mask=full8), reg.vshr(ids8, 2, mask=full8), mask=full8)
        random_mask = reg.vdups(65535, dtypes.uint32, mask=full)
        for part in range(first, stop):
            offset = part * 256
            av, bv = reg.vload_deinterleave(x, offset)
            absolute_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
            av = reg.vreinterpret(
                reg.vbitwise_and(reg.vreinterpret(av, dtypes.uint16), absolute_mask, mask=full16), dtypes.bfloat16
            )
            bv = reg.vreinterpret(
                reg.vbitwise_and(reg.vreinterpret(bv, dtypes.uint16), absolute_mask, mask=full16), dtypes.bfloat16
            )
            v0 = reg.vcast(av, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
            v2 = reg.vcast(av, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
            v1 = reg.vcast(bv, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
            v3 = reg.vcast(bv, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
            word = reg.vreinterpret(reg.vload(random.buffer, part * 64), dtypes.uint32)
            reverse = _rotate32_words8(word, full)
            r0 = reg.vshr(word, 16, mask=full)
            r1 = reg.vbitwise_and(reverse, random_mask, mask=full)
            r2 = reg.vbitwise_and(word, random_mask, mask=full)
            r3 = reg.vshr(reverse, 16, mask=full)
            c0 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v0, dtypes.uint32), r0, self.mantissa, self.bias, full)
            c1 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v1, dtypes.uint32), r1, self.mantissa, self.bias, full)
            c2 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v2, dtypes.uint32), r2, self.mantissa, self.bias, full)
            c3 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v3, dtypes.uint32), r3, self.mantissa, self.bias, full)
            unused01, code01 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c0, dtypes.uint16), reg.vreinterpret_lanes(c1, dtypes.uint16)
            )
            unused23, code23 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c2, dtypes.uint16), reg.vreinterpret_lanes(c3, dtypes.uint16)
            )
            code_bytes, unused_bytes = reg.vdeinterleave(
                reg.vreinterpret_lanes(code01, dtypes.uint8), reg.vreinterpret_lanes(code23, dtypes.uint8)
            )
            codes8 = reg.vgather_reg(code_bytes, pack_index)
            magnitude8 = reg.vadds(codes8, -(128 - self.bias << self.mantissa) & 255, mask=full8)
            codes8 = reg.vmins(magnitude8, self.max_code, mask=reg.vnes(magnitude8, 255, mask=full8))
            unused_low, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
            codes8 = reg.vbitwise_or(codes8, reg.vbitwise_and(signbytes, sign_mask8, mask=full8), mask=full8)
            ids = reg.vadds(reg.varange(0, dtypes.uint16), part * 16, mask=full16)
            scales16 = reg.vgather(scale16, ids, mask=eight16)
            scales8 = reg.vpack(scales16, dtypes.uint8, part="lower")
            invalid = reg.veqs(reg.vgather_reg(scales8, index8), 255, mask=full8)
            codes8 = reg.vdups(127, dtypes.uint8, mask=invalid, mode="merging", merge=codes8)
            reg.vstore(y, offset, codes8, full8)
        # Complete encoded byte stores before subsequent VFs reuse UB or DMA consumes this output slot.
        reg.vmem_bar("vst_vld")


@jit
def _quantize_bf16_four_field_panels(
    self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
):
    parts = count // 256
    batches = (count + 1023) // 1024
    scale16 = reinterpret(self.scale32, dtypes.uint16, shape=(1, self.tile_elements // 16))
    prepare_random_words_aos(self, random, start, batches, quant_length, seed, block_offset)
    with vf(mode="simd"):
        full = reg.full_mask()
        full16 = reg.full_mask(elem_bits=16)
        full8 = reg.full_mask(elem_bits=8)
        eight16 = reg.update_mask(8, elem_bits=16)[0]
        abs_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
        index16 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
        pack_ids8 = reg.varange(0, dtypes.uint8)
        scale_byte_slot = reg.vbitwise_and(pack_ids8, reg.vdups(1, dtypes.uint8, mask=full8), mask=full8)
        scale_byte_mask = reg.mask_and(
            reg.vlts(pack_ids8, 16, mask=full8),
            reg.vlts(scale_byte_slot, dtypes.uint8(scale_factor // 2), mask=full8),
            exec_mask=full8,
        )
        scale_cursor = reg.vstore_unalign_begin(scales)
        one16 = reg.update_mask(1, elem_bits=16)[0]
        tile_minimum = reg.vdups(65535, dtypes.uint16, mask=one16)
        for first4 in range(0, parts, 4):
            group_min = reg.vdups(65535, dtypes.uint16, mask=full16)
            group_invalid = reg.vdups(0, dtypes.uint16, mask=full16)
            part = first4 + 0
            offset = part * 256
            a, b = reg.vload_deinterleave(x, offset)
            aa = reg.vbitwise_and(reg.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
            ab = reg.vbitwise_and(reg.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
            maxima16 = reg.vreduce_max_datablock(reg.vmax(aa, ab, mask=full16), mask=full16)
            if const_expr(self.scale_alg == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                exponent16 = reg.vshr(maxima16, 7, mask=full16)
                invalid16 = reg.vges(exponent16, 255, mask=full16)
                code16 = reg.vadds(reg.vmaxs(exponent16, scale_shift, mask=full16), -scale_shift & 65535, mask=full16)
                code16 = reg.vdups(255, dtypes.uint16, mask=invalid16, mode="merging", merge=code16)
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16),
                    7,
                    mask=reg.vnes(code16, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                active16 = reg.vgts(maxima16, scale_shift << 7 | 96, mask=full16)
                code16 = reg.vshr(reg.vadds(maxima16, 31 - (scale_shift << 7) & 65535, mask=full16), 7, mask=active16)
                code16 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16, 32640, mask=full16), mode="merging", merge=code16
                )
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16), 7, mask=full16
                )
            else:
                maxima = reg.vshl(reg.vunpack(maxima16, dtypes.uint32), 16, mask=full)
                block_scales, reciprocal = compute_scale(self, maxima, full, False)
                bits16 = reg.vshr(reg.vreinterpret(reciprocal, dtypes.uint32), 16, mask=full)
                r16 = reg.vpack(bits16, dtypes.uint16, part="lower")
                code16 = reg.vpack(block_scales, dtypes.uint16, part="lower")
            group_invalid = reg.vmax(group_invalid, code16, mask=eight16)
            broadcast = reg.vreinterpret(reg.vgather_reg(r16, index16), dtypes.bfloat16)
            av = reg.vmul(a, broadcast, mask=full16)
            bv = reg.vmul(b, broadcast, mask=full16)
            reg.vstore_interleave(x, offset, av, bv)
            aa_norm = reg.vbitwise_and(reg.vreinterpret(av, dtypes.uint16), abs_mask, mask=full16)
            ab_norm = reg.vbitwise_and(reg.vreinterpret(bv, dtypes.uint16), abs_mask, mask=full16)
            panel_min = reg.vmin(aa_norm, ab_norm, mask=full16)
            group_min = reg.vmin(group_min, panel_min, mask=full16)
            reg.vstore(scale16, part * 16, code16, eight16)
            scale_bytes = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(code16, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes, scale_cursor)
            part = first4 + 1
            offset = part * 256
            a, b = reg.vload_deinterleave(x, offset)
            aa = reg.vbitwise_and(reg.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
            ab = reg.vbitwise_and(reg.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
            maxima16 = reg.vreduce_max_datablock(reg.vmax(aa, ab, mask=full16), mask=full16)
            if const_expr(self.scale_alg == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                exponent16 = reg.vshr(maxima16, 7, mask=full16)
                invalid16 = reg.vges(exponent16, 255, mask=full16)
                code16 = reg.vadds(reg.vmaxs(exponent16, scale_shift, mask=full16), -scale_shift & 65535, mask=full16)
                code16 = reg.vdups(255, dtypes.uint16, mask=invalid16, mode="merging", merge=code16)
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16),
                    7,
                    mask=reg.vnes(code16, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                active16 = reg.vgts(maxima16, scale_shift << 7 | 96, mask=full16)
                code16 = reg.vshr(reg.vadds(maxima16, 31 - (scale_shift << 7) & 65535, mask=full16), 7, mask=active16)
                code16 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16, 32640, mask=full16), mode="merging", merge=code16
                )
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16), 7, mask=full16
                )
            else:
                maxima = reg.vshl(reg.vunpack(maxima16, dtypes.uint32), 16, mask=full)
                block_scales, reciprocal = compute_scale(self, maxima, full, False)
                bits16 = reg.vshr(reg.vreinterpret(reciprocal, dtypes.uint32), 16, mask=full)
                r16 = reg.vpack(bits16, dtypes.uint16, part="lower")
                code16 = reg.vpack(block_scales, dtypes.uint16, part="lower")
            group_invalid = reg.vmax(group_invalid, code16, mask=eight16)
            broadcast = reg.vreinterpret(reg.vgather_reg(r16, index16), dtypes.bfloat16)
            av = reg.vmul(a, broadcast, mask=full16)
            bv = reg.vmul(b, broadcast, mask=full16)
            reg.vstore_interleave(x, offset, av, bv)
            aa_norm = reg.vbitwise_and(reg.vreinterpret(av, dtypes.uint16), abs_mask, mask=full16)
            ab_norm = reg.vbitwise_and(reg.vreinterpret(bv, dtypes.uint16), abs_mask, mask=full16)
            panel_min = reg.vmin(aa_norm, ab_norm, mask=full16)
            group_min = reg.vmin(group_min, panel_min, mask=full16)
            reg.vstore(scale16, part * 16, code16, eight16)
            scale_bytes = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(code16, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes, scale_cursor)
            part = first4 + 2
            offset = part * 256
            a, b = reg.vload_deinterleave(x, offset)
            aa = reg.vbitwise_and(reg.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
            ab = reg.vbitwise_and(reg.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
            maxima16 = reg.vreduce_max_datablock(reg.vmax(aa, ab, mask=full16), mask=full16)
            if const_expr(self.scale_alg == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                exponent16 = reg.vshr(maxima16, 7, mask=full16)
                invalid16 = reg.vges(exponent16, 255, mask=full16)
                code16 = reg.vadds(reg.vmaxs(exponent16, scale_shift, mask=full16), -scale_shift & 65535, mask=full16)
                code16 = reg.vdups(255, dtypes.uint16, mask=invalid16, mode="merging", merge=code16)
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16),
                    7,
                    mask=reg.vnes(code16, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                active16 = reg.vgts(maxima16, scale_shift << 7 | 96, mask=full16)
                code16 = reg.vshr(reg.vadds(maxima16, 31 - (scale_shift << 7) & 65535, mask=full16), 7, mask=active16)
                code16 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16, 32640, mask=full16), mode="merging", merge=code16
                )
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16), 7, mask=full16
                )
            else:
                maxima = reg.vshl(reg.vunpack(maxima16, dtypes.uint32), 16, mask=full)
                block_scales, reciprocal = compute_scale(self, maxima, full, False)
                bits16 = reg.vshr(reg.vreinterpret(reciprocal, dtypes.uint32), 16, mask=full)
                r16 = reg.vpack(bits16, dtypes.uint16, part="lower")
                code16 = reg.vpack(block_scales, dtypes.uint16, part="lower")
            group_invalid = reg.vmax(group_invalid, code16, mask=eight16)
            broadcast = reg.vreinterpret(reg.vgather_reg(r16, index16), dtypes.bfloat16)
            av = reg.vmul(a, broadcast, mask=full16)
            bv = reg.vmul(b, broadcast, mask=full16)
            reg.vstore_interleave(x, offset, av, bv)
            aa_norm = reg.vbitwise_and(reg.vreinterpret(av, dtypes.uint16), abs_mask, mask=full16)
            ab_norm = reg.vbitwise_and(reg.vreinterpret(bv, dtypes.uint16), abs_mask, mask=full16)
            panel_min = reg.vmin(aa_norm, ab_norm, mask=full16)
            group_min = reg.vmin(group_min, panel_min, mask=full16)
            reg.vstore(scale16, part * 16, code16, eight16)
            scale_bytes = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(code16, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes, scale_cursor)
            part = first4 + 3
            offset = part * 256
            a, b = reg.vload_deinterleave(x, offset)
            aa = reg.vbitwise_and(reg.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
            ab = reg.vbitwise_and(reg.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
            maxima16 = reg.vreduce_max_datablock(reg.vmax(aa, ab, mask=full16), mask=full16)
            if const_expr(self.scale_alg == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                exponent16 = reg.vshr(maxima16, 7, mask=full16)
                invalid16 = reg.vges(exponent16, 255, mask=full16)
                code16 = reg.vadds(reg.vmaxs(exponent16, scale_shift, mask=full16), -scale_shift & 65535, mask=full16)
                code16 = reg.vdups(255, dtypes.uint16, mask=invalid16, mode="merging", merge=code16)
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16),
                    7,
                    mask=reg.vnes(code16, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                active16 = reg.vgts(maxima16, scale_shift << 7 | 96, mask=full16)
                code16 = reg.vshr(reg.vadds(maxima16, 31 - (scale_shift << 7) & 65535, mask=full16), 7, mask=active16)
                code16 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16, 32640, mask=full16), mode="merging", merge=code16
                )
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16), 7, mask=full16
                )
            else:
                maxima = reg.vshl(reg.vunpack(maxima16, dtypes.uint32), 16, mask=full)
                block_scales, reciprocal = compute_scale(self, maxima, full, False)
                bits16 = reg.vshr(reg.vreinterpret(reciprocal, dtypes.uint32), 16, mask=full)
                r16 = reg.vpack(bits16, dtypes.uint16, part="lower")
                code16 = reg.vpack(block_scales, dtypes.uint16, part="lower")
            group_invalid = reg.vmax(group_invalid, code16, mask=eight16)
            broadcast = reg.vreinterpret(reg.vgather_reg(r16, index16), dtypes.bfloat16)
            av = reg.vmul(a, broadcast, mask=full16)
            bv = reg.vmul(b, broadcast, mask=full16)
            reg.vstore_interleave(x, offset, av, bv)
            aa_norm = reg.vbitwise_and(reg.vreinterpret(av, dtypes.uint16), abs_mask, mask=full16)
            ab_norm = reg.vbitwise_and(reg.vreinterpret(bv, dtypes.uint16), abs_mask, mask=full16)
            panel_min = reg.vmin(aa_norm, ab_norm, mask=full16)
            group_min = reg.vmin(group_min, panel_min, mask=full16)
            reg.vstore(scale16, part * 16, code16, eight16)
            scale_bytes = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(code16, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes, scale_cursor)
            minimum = reg.vreduce_min(group_min, mask=full16)
            maximum_scale = reg.vreduce_max(group_invalid, mask=eight16)
            minimum = reg.vdups(
                0, dtypes.uint16, mask=reg.veqs(maximum_scale, 255, mask=full16), mode="merging", merge=minimum
            )
            reg.vstore_first(self.norm_flags, first4 * self.norm_flag_stride, minimum)
            tile_minimum = reg.vmin(tile_minimum, minimum, mask=one16)
        reg.vsqueeze_and_storeunalign_finalize(scales, 0, scale_cursor)
        reg.vstore_first(self.norm_flags, self.norm_tile_flag, tile_minimum)
        # Publish normalized x, scales and minima before scalar classification and encoding reloads.
        reg.vmem_bar("vst_vld")
    if dtypes.int64(self.norm_flags[0, self.norm_tile_flag]) >= (128 - self.bias) * 128:
        _emit_bf16_clean_normal_part(self, random, x, y, 0, parts)
    else:
        for first in range(0, parts, 16):
            stop = first + 16
            if stop > parts:
                stop = parts
            flag_min = dtypes.int64(self.norm_flags[0, first * self.norm_flag_stride])
            for cursor in range(first + 4, stop, 4):
                flag = dtypes.int64(self.norm_flags[0, cursor * self.norm_flag_stride])
                if flag < flag_min:
                    flag_min = flag
            if flag_min >= (128 - self.bias) * 128:
                _emit_bf16_clean_normal_part(self, random, x, y, first, stop)
            elif flag_min >= (123 - self.bias - self.mantissa) * 128:
                _emit_bf16_middle_q12_part(self, random, x, y, first, stop)
            else:
                _emit_bf16_general_part(self, random, x, y, first, stop)


@jit
def quantize_tile_256(self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset):
    if const_expr(self.bf16_guard_tail):
        if quant_length % PACKED_K_ALIGNMENT_ELEMENTS == 0:
            if count >= 1024 and count % 1024 == 0:
                _quantize_bf16_nibble_tile_256(
                    self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
                )
            else:
                _quantize_bf16_nibble_partial_256(
                    self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
                )
        else:
            _dispatch_panel_quantization(
                self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
            )
    else:
        _dispatch_panel_quantization(
            self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
        )


@jit
def _quantize_fp32_eight_compact_256(
    self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
):
    sign_source = reinterpret(x, dtypes.uint8 if self.bf16 else dtypes.uint16, shape=(1, self.tile_elements * 2))
    parts = count // 256
    batches = (count + 1023) // 1024
    _prepare_eight_field_aos_linear(random, start, batches, seed, block_offset)
    with vf(mode="simd"):
        full = reg.full_mask()
        full16 = reg.full_mask(elem_bits=16)
        full8 = reg.full_mask(elem_bits=8)
        ids = reg.varange(0, dtypes.uint32)
        scale_index = reg.vshr(ids, 3, mask=full)
        sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
        pack_ids8 = reg.varange(0, dtypes.uint8)
        pack_group8 = reg.vbitwise_and(pack_ids8, reg.vdups(3, dtypes.uint8, mask=full8), mask=full8)
        pack_index8 = reg.vbitwise_or(
            reg.vshl(pack_group8, 6, mask=full8), reg.vshr(pack_ids8, 2, mask=full8), mask=full8
        )
        random_mask32 = reg.vdups(65535, dtypes.uint32, mask=full)
        scale_byte_slot = reg.vbitwise_and(pack_ids8, reg.vdups(3, dtypes.uint8, mask=full8), mask=full8)
        scale_byte_mask = reg.mask_and(
            reg.vlts(pack_ids8, 32, mask=full8),
            reg.vlts(scale_byte_slot, dtypes.uint8(scale_factor // 2), mask=full8),
            exec_mask=full8,
        )
        scale_cursor = reg.vstore_unalign_begin(scales)
        pair_ids = reg.varange(0, dtypes.uint8)
        pair_base = reg.vshl(reg.vshr(pair_ids, 3, mask=full8), 2, mask=full8)
        pair_position = reg.vbitwise_and(
            reg.vshr(reg.vadds(pair_ids, 1, mask=full8), 1, mask=full8),
            reg.vdups(3, dtypes.uint8, mask=full8),
            mask=full8,
        )
        pair_index02 = reg.vbitwise_or(pair_base, pair_position, mask=full8)
        pair_index13 = reg.vgather_reg(
            pair_index02, reg.vbitwise_xor(pair_ids, reg.vdups(1, dtypes.uint8, mask=full8), mask=full8)
        )
        for part in range(parts):
            offset = part * 256
            if const_expr(self.bf16):
                a, b = reg.vload_deinterleave(x, offset)
                abs_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
                aa = reg.vbitwise_and(reg.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
                ab = reg.vbitwise_and(reg.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
                maxima16 = reg.vreduce_max_datablock(reg.vmax(aa, ab, mask=full16), mask=full16)
                maxima = reg.vshl(reg.vunpack(maxima16, dtypes.uint32), 16, mask=full)
                unused_signbytes, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
                signs = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes, sign_mask8, mask=full8), dtypes.uint32)
            else:
                a, b = reg.vload_deinterleave(x, offset)
                c, d = reg.vload_deinterleave(x, offset + 128)
                v0, v2 = reg.vdeinterleave(a, c)
                v1, v3 = reg.vdeinterleave(b, d)
                a0 = reg.vreinterpret(reg.vabs(v0, mask=full), dtypes.uint32)
                a1 = reg.vreinterpret(reg.vabs(v1, mask=full), dtypes.uint32)
                a2 = reg.vreinterpret(reg.vabs(v2, mask=full), dtypes.uint32)
                a3 = reg.vreinterpret(reg.vabs(v3, mask=full), dtypes.uint32)
                pair0 = reg.vmax(a0, a1, mask=full)
                pair1 = reg.vmax(a2, a3, mask=full)
                maxima = reg.vreduce_max_datablock(reg.vmax(pair0, pair1, mask=full), mask=full)
                unused_sign01, sign01 = reg.vload_deinterleave(sign_source, offset * 2)
                unused_sign23, sign23 = reg.vload_deinterleave(sign_source, offset * 2 + 256)
                unused_signbytes, signbytes = reg.vdeinterleave(
                    reg.vreinterpret_lanes(sign01, dtypes.uint8), reg.vreinterpret_lanes(sign23, dtypes.uint8)
                )
                signs = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes, sign_mask8, mask=full8), dtypes.uint32)
            block_scales, reciprocal = compute_scale(self, maxima, full, False)
            if const_expr(self.bf16):
                bits16 = reg.vshr(reg.vreinterpret(reciprocal, dtypes.uint32), 16, mask=full)
                r16 = reg.vpack(bits16, dtypes.uint16, part="lower")
                index16 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
                broadcast = reg.vreinterpret(reg.vgather_reg(r16, index16), dtypes.bfloat16)
                a = reg.vmul(reg.vreinterpret(aa, dtypes.bfloat16), broadcast, mask=full16)
                b = reg.vmul(reg.vreinterpret(ab, dtypes.bfloat16), broadcast, mask=full16)
                v0 = reg.vcast(a, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v2 = reg.vcast(a, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                v1 = reg.vcast(b, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v3 = reg.vcast(b, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                invalid_reciprocal = reg.vreinterpret_lanes(broadcast, dtypes.uint32)
            else:
                reciprocal = reg.vgather_reg(reciprocal, scale_index)
                v0 = reg.vmul(reg.vreinterpret(a0, dtypes.float32), reciprocal, mask=full)
                v1 = reg.vmul(reg.vreinterpret(a1, dtypes.float32), reciprocal, mask=full)
                v2 = reg.vmul(reg.vreinterpret(a2, dtypes.float32), reciprocal, mask=full)
                v3 = reg.vmul(reg.vreinterpret(a3, dtypes.float32), reciprocal, mask=full)
                invalid_reciprocal = reciprocal
            raw = reg.vreinterpret(reg.vload(random.buffer, part * 32), dtypes.uint32)
            raw_bytes = reg.vreinterpret_lanes(raw, dtypes.uint8)
            word02 = reg.vreinterpret_lanes(reg.vgather_reg(raw_bytes, pair_index02), dtypes.uint32)
            word13 = reg.vreinterpret_lanes(reg.vgather_reg(raw_bytes, pair_index13), dtypes.uint32)
            r0 = reg.vbitwise_and(word02, random_mask32, mask=full)
            r1 = reg.vbitwise_and(word13, random_mask32, mask=full)
            r2 = reg.vshr(word02, 16, mask=full)
            r3 = reg.vshr(word13, 16, mask=full)
            c0 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v0, dtypes.uint32), r0, self.mantissa, self.bias, full)
            c1 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v1, dtypes.uint32), r1, self.mantissa, self.bias, full)
            c2 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v2, dtypes.uint32), r2, self.mantissa, self.bias, full)
            c3 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v3, dtypes.uint32), r3, self.mantissa, self.bias, full)
            unused_code01, code01 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c0, dtypes.uint16), reg.vreinterpret_lanes(c1, dtypes.uint16)
            )
            unused_code23, code23 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c2, dtypes.uint16), reg.vreinterpret_lanes(c3, dtypes.uint16)
            )
            code_bytes, unused_code_bytes = reg.vdeinterleave(
                reg.vreinterpret_lanes(code01, dtypes.uint8), reg.vreinterpret_lanes(code23, dtypes.uint8)
            )
            packed = reg.vreinterpret_lanes(reg.vgather_reg(code_bytes, pack_index8), dtypes.uint32)
            magnitude8 = reg.vadds(
                reg.vreinterpret_lanes(packed, dtypes.uint8), -(128 - self.bias << self.mantissa) & 255, mask=full8
            )
            saturated = reg.vmins(magnitude8, self.max_code, mask=reg.vnes(magnitude8, 255, mask=full8))
            packed = reg.vbitwise_or(reg.vreinterpret_lanes(saturated, dtypes.uint32), signs, mask=full)
            invalid = reg.veqs(
                reg.vreinterpret(invalid_reciprocal, dtypes.uint32), self.invalid_packed_reciprocal, mask=full
            )
            packed = reg.vdups(2139062143, dtypes.uint32, mask=invalid, mode="merging", merge=packed)
            reg.vstore(y, offset, reg.vreinterpret_lanes(packed, dtypes.uint8), full8)
            scale_bytes = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(block_scales, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes, scale_cursor)
        reg.vsqueeze_and_storeunalign_finalize(scales, 0, scale_cursor)
        # Complete encoded byte stores before subsequent VFs reuse UB or DMA consumes this output slot.
        reg.vmem_bar("vst_vld")


@jit
def _dispatch_panel_quantization(
    self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
):
    if const_expr(self.fp32_four_pair_schedule):
        _quantize_fp32_four_pair_256(
            self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
        )
    elif const_expr(self.fp32_eight_tail):
        if quant_length % PACKED_K_ALIGNMENT_ELEMENTS == 0:
            _quantize_fp32_eight_compact_256(
                self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
            )
        else:
            _quantize_tile_256_baseline(
                self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
            )
    elif const_expr(self.bf16_guard_tail):
        if count >= 1024 and count % 1024 == 0:
            _quantize_bf16_four_field_panels(
                self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
            )
        else:
            _quantize_tile_256_baseline(
                self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
            )
    else:
        _quantize_tile_256_baseline(
            self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
        )


@jit
def _quantize_tile_256_baseline(
    self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
):
    sign_source = reinterpret(x, dtypes.uint8 if self.bf16 else dtypes.uint16, shape=(1, self.tile_elements * 2))
    parts = count // 256
    batches = (count + 1023) // 1024
    prepare_random_words_aos(self, random, start, batches, quant_length, seed, block_offset)
    with vf(mode="simd"):
        full = reg.full_mask()
        full16 = reg.full_mask(elem_bits=16)
        full8 = reg.full_mask(elem_bits=8)
        ids = reg.varange(0, dtypes.uint32)
        scale_index = reg.vshr(ids, 3, mask=full)
        sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
        pack_ids8 = reg.varange(0, dtypes.uint8)
        pack_group8 = reg.vbitwise_and(pack_ids8, reg.vdups(3, dtypes.uint8, mask=full8), mask=full8)
        pack_index8 = reg.vbitwise_or(
            reg.vshl(pack_group8, 6, mask=full8), reg.vshr(pack_ids8, 2, mask=full8), mask=full8
        )
        random_mask32 = reg.vdups(65535, dtypes.uint32, mask=full)
        scale_byte_slot = reg.vbitwise_and(pack_ids8, reg.vdups(3, dtypes.uint8, mask=full8), mask=full8)
        scale_byte_mask = reg.mask_and(
            reg.vlts(pack_ids8, 32, mask=full8),
            reg.vlts(scale_byte_slot, dtypes.uint8(scale_factor // 2), mask=full8),
            exec_mask=full8,
        )
        scale_cursor = reg.vstore_unalign_begin(scales)
        for part in range(parts):
            offset = part * 256
            if const_expr(self.bf16):
                a, b = reg.vload_deinterleave(x, offset)
                abs_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
                aa = reg.vbitwise_and(reg.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
                ab = reg.vbitwise_and(reg.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
                maxima16 = reg.vreduce_max_datablock(reg.vmax(aa, ab, mask=full16), mask=full16)
                maxima = reg.vshl(reg.vunpack(maxima16, dtypes.uint32), 16, mask=full)
                unused_signbytes, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
                signs = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes, sign_mask8, mask=full8), dtypes.uint32)
            else:
                a, b = reg.vload_deinterleave(x, offset)
                c, d = reg.vload_deinterleave(x, offset + 128)
                v0, v2 = reg.vdeinterleave(a, c)
                v1, v3 = reg.vdeinterleave(b, d)
                a0 = reg.vreinterpret(reg.vabs(v0, mask=full), dtypes.uint32)
                a1 = reg.vreinterpret(reg.vabs(v1, mask=full), dtypes.uint32)
                a2 = reg.vreinterpret(reg.vabs(v2, mask=full), dtypes.uint32)
                a3 = reg.vreinterpret(reg.vabs(v3, mask=full), dtypes.uint32)
                pair0 = reg.vmax(a0, a1, mask=full)
                pair1 = reg.vmax(a2, a3, mask=full)
                maxima = reg.vreduce_max_datablock(reg.vmax(pair0, pair1, mask=full), mask=full)
                unused_sign01, sign01 = reg.vload_deinterleave(sign_source, offset * 2)
                unused_sign23, sign23 = reg.vload_deinterleave(sign_source, offset * 2 + 256)
                unused_signbytes, signbytes = reg.vdeinterleave(
                    reg.vreinterpret_lanes(sign01, dtypes.uint8), reg.vreinterpret_lanes(sign23, dtypes.uint8)
                )
                signs = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes, sign_mask8, mask=full8), dtypes.uint32)
            block_scales, reciprocal = compute_scale(self, maxima, full, False)
            if const_expr(self.bf16):
                bits16 = reg.vshr(reg.vreinterpret(reciprocal, dtypes.uint32), 16, mask=full)
                r16 = reg.vpack(bits16, dtypes.uint16, part="lower")
                index16 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
                broadcast = reg.vreinterpret(reg.vgather_reg(r16, index16), dtypes.bfloat16)
                a = reg.vmul(reg.vreinterpret(aa, dtypes.bfloat16), broadcast, mask=full16)
                b = reg.vmul(reg.vreinterpret(ab, dtypes.bfloat16), broadcast, mask=full16)
                v0 = reg.vcast(a, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v2 = reg.vcast(a, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                v1 = reg.vcast(b, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v3 = reg.vcast(b, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                invalid_reciprocal = reg.vreinterpret_lanes(broadcast, dtypes.uint32)
            else:
                reciprocal = reg.vgather_reg(reciprocal, scale_index)
                v0 = reg.vmul(reg.vreinterpret(a0, dtypes.float32), reciprocal, mask=full)
                v1 = reg.vmul(reg.vreinterpret(a1, dtypes.float32), reciprocal, mask=full)
                v2 = reg.vmul(reg.vreinterpret(a2, dtypes.float32), reciprocal, mask=full)
                v3 = reg.vmul(reg.vreinterpret(a3, dtypes.float32), reciprocal, mask=full)
                invalid_reciprocal = reciprocal
            word = reg.vreinterpret(reg.vload(random.buffer, part * 64), dtypes.uint32)
            reversed_word = _rotate32_words8(word, full)
            r0 = reg.vshr(word, 16, mask=full)
            r1 = reg.vbitwise_and(reversed_word, random_mask32, mask=full)
            r2 = reg.vbitwise_and(word, random_mask32, mask=full)
            r3 = reg.vshr(reversed_word, 16, mask=full)
            c0 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v0, dtypes.uint32), r0, self.mantissa, self.bias, full)
            c1 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v1, dtypes.uint32), r1, self.mantissa, self.bias, full)
            c2 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v2, dtypes.uint32), r2, self.mantissa, self.bias, full)
            c3 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v3, dtypes.uint32), r3, self.mantissa, self.bias, full)
            unused_code01, code01 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c0, dtypes.uint16), reg.vreinterpret_lanes(c1, dtypes.uint16)
            )
            unused_code23, code23 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c2, dtypes.uint16), reg.vreinterpret_lanes(c3, dtypes.uint16)
            )
            code_bytes, unused_code_bytes = reg.vdeinterleave(
                reg.vreinterpret_lanes(code01, dtypes.uint8), reg.vreinterpret_lanes(code23, dtypes.uint8)
            )
            packed = reg.vreinterpret_lanes(reg.vgather_reg(code_bytes, pack_index8), dtypes.uint32)
            magnitude8 = reg.vadds(
                reg.vreinterpret_lanes(packed, dtypes.uint8), -(128 - self.bias << self.mantissa) & 255, mask=full8
            )
            saturated = reg.vmins(magnitude8, self.max_code, mask=reg.vnes(magnitude8, 255, mask=full8))
            packed = reg.vbitwise_or(reg.vreinterpret_lanes(saturated, dtypes.uint32), signs, mask=full)
            invalid = reg.veqs(
                reg.vreinterpret(invalid_reciprocal, dtypes.uint32), self.invalid_packed_reciprocal, mask=full
            )
            packed = reg.vdups(2139062143, dtypes.uint32, mask=invalid, mode="merging", merge=packed)
            reg.vstore(y, offset, reg.vreinterpret_lanes(packed, dtypes.uint8), full8)
            scale_bytes = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(block_scales, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes, scale_cursor)
        reg.vsqueeze_and_storeunalign_finalize(scales, 0, scale_cursor)
        # Complete encoded byte stores before subsequent VFs reuse UB or DMA consumes this output slot.
        reg.vmem_bar("vst_vld")


@jit
def quantize_short_rows(self, random, x, y, scales, start, rows, columns, pitch_shift, seed, block_offset):
    """Batch compact GM rows, padding only the SIMD reduction geometry."""
    pitch = 64
    if pitch_shift == 7:
        pitch = 128
    parts = (rows * pitch + 255) // 256
    scale_count = rows * (pitch // 32)
    pitch_bits = dtypes.uint32(pitch_shift)
    column_mask = dtypes.uint32(pitch // 4 - 1)
    columns4 = dtypes.uint32(columns // 4)
    row_count = dtypes.uint32(rows)
    packed_y = reinterpret(y, dtypes.uint32, shape=(1, self.tile_elements // 4))
    if const_expr(self.bf16):
        packed_x = reinterpret(x, dtypes.uint32, shape=(1, self.tile_elements // 2))
    batches = (rows * columns + 1023) // 1024
    key0, key1 = _split_uint64(seed)
    axis_groups, reciprocal, row_base, col_base = _rng_geometry(start, columns)
    row_low, row_high = _split_uint64(row_base)
    col_low, col_high = _split_uint64(col_base)
    step_q, step_r = (64 // axis_groups, 64 % axis_groups)
    with vf(mode="simd"):
        rl, rh, cl, ch = (row_low, row_high, col_low, col_high)
        for batch in range(batches):
            batch_row, batch_col = _cursor_coordinates(rl, rh, cl, ch)
            _generate_random_words_keyed(
                random.buffer,
                batch_row,
                batch_col,
                axis_groups,
                reciprocal,
                batch * 256,
                key0,
                key1,
                block_offset,
                aos=True,
            )
            rl, rh, cl, ch = _advance_cursor_words(batch_row, batch_col, step_q, step_r, axis_groups)
        # Publish Philox scratch before random-word gathers in the following stage.
        reg.vmem_bar("vst_vld")
    with vf(mode="simd"):
        full = reg.full_mask()
        ids = reg.varange(0, dtypes.uint32)
        eight = reg.update_mask(8, elem_bits=32)[0]
        scale_index = reg.vshr(ids, 3, mask=full)
        zero = reg.vdups(0, dtypes.uint32, mask=full)
        wide_pitch = reg.veqs(reg.vdups(pitch_bits, dtypes.uint32, mask=full), 7, mask=full)
        for part in range(parts):
            virtual4 = reg.vadds(ids, part * 64, mask=full)
            row = reg.vselect(reg.vshr(virtual4, 5, mask=full), reg.vshr(virtual4, 4, mask=full), cond_mask=wide_pitch)
            column4 = reg.vbitwise_and(virtual4, reg.vdups(column_mask, dtypes.uint32, mask=full), mask=full)
            logical4 = reg.vadd(reg.vmuls(row, columns4, mask=full), column4, mask=full)
            valid = reg.mask_and(
                reg.vlts(row, row_count, mask=full), reg.vlts(column4, columns4, mask=full), exec_mask=full
            )
            if const_expr(self.bf16):
                index = reg.vshl(logical4, 1, mask=full)
                a = reg.vgather(packed_x, index, mask=valid)
                b = reg.vgather(packed_x, reg.vadds(index, 1, mask=full), mask=valid)
                a = reg.vselect(a, zero, cond_mask=valid)
                b = reg.vselect(b, zero, cond_mask=valid)
                v0 = reg.vreinterpret(reg.vshl(a, 16, mask=full), dtypes.float32)
                v1 = reg.vreinterpret(
                    reg.vbitwise_and(a, reg.vdups(4294901760, dtypes.uint32, mask=full), mask=full), dtypes.float32
                )
                v2 = reg.vreinterpret(reg.vshl(b, 16, mask=full), dtypes.float32)
                v3 = reg.vreinterpret(
                    reg.vbitwise_and(b, reg.vdups(4294901760, dtypes.uint32, mask=full), mask=full), dtypes.float32
                )
            else:
                index = reg.vshl(logical4, 2, mask=full)
                v0 = reg.vgather(x, index, mask=valid)
                v1 = reg.vgather(x, reg.vadds(index, 1, mask=full), mask=valid)
                v2 = reg.vgather(x, reg.vadds(index, 2, mask=full), mask=valid)
                v3 = reg.vgather(x, reg.vadds(index, 3, mask=full), mask=valid)
                zeros = reg.vdups(0.0, dtypes.float32, mask=full)
                v0 = reg.vselect(v0, zeros, cond_mask=valid)
                v1 = reg.vselect(v1, zeros, cond_mask=valid)
                v2 = reg.vselect(v2, zeros, cond_mask=valid)
                v3 = reg.vselect(v3, zeros, cond_mask=valid)
            a0 = reg.vbitwise_and(
                reg.vreinterpret(v0, dtypes.uint32), reg.vdups(2147483647, dtypes.uint32, mask=full), mask=full
            )
            a1 = reg.vbitwise_and(
                reg.vreinterpret(v1, dtypes.uint32), reg.vdups(2147483647, dtypes.uint32, mask=full), mask=full
            )
            a2 = reg.vbitwise_and(
                reg.vreinterpret(v2, dtypes.uint32), reg.vdups(2147483647, dtypes.uint32, mask=full), mask=full
            )
            a3 = reg.vbitwise_and(
                reg.vreinterpret(v3, dtypes.uint32), reg.vdups(2147483647, dtypes.uint32, mask=full), mask=full
            )
            maxima = reg.vreduce_max_datablock(
                reg.vmax(reg.vmax(a0, a1, mask=full), reg.vmax(a2, a3, mask=full), mask=full), mask=full
            )
            block_scales, reciprocal = compute_scale(self, maxima, full)
            reciprocal = reg.vgather_reg(reciprocal, scale_index)
            v0 = reg.vmul(v0, reciprocal, mask=full)
            v1 = reg.vmul(v1, reciprocal, mask=full)
            v2 = reg.vmul(v2, reciprocal, mask=full)
            v3 = reg.vmul(v3, reciprocal, mask=full)
            word = reg.vreinterpret(reg.vgather(random.buffer, logical4, mask=valid), dtypes.uint32)
            reversed_word = _rotate32_words8(word, valid)
            rab = reg.vshr(word, 16, mask=full)
            ref = reg.vbitwise_and(word, reg.vdups(65535, dtypes.uint32, mask=full), mask=full)
            c0 = _sr_fp32_to_fp8_q16(v0, rab, self.mantissa, self.bias, self.max_code, full)
            c1 = _sr_fp32_to_fp8_q16(
                v1,
                reg.vbitwise_and(reversed_word, reg.vdups(65535, dtypes.uint32, mask=full), mask=full),
                self.mantissa,
                self.bias,
                self.max_code,
                full,
            )
            c2 = _sr_fp32_to_fp8_q16(v2, ref, self.mantissa, self.bias, self.max_code, full)
            c3 = _sr_fp32_to_fp8_q16(
                v3, reg.vshr(reversed_word, 16, mask=full), self.mantissa, self.bias, self.max_code, full
            )
            lo = reg.vbitwise_or(c0, reg.vshl(c1, 8, mask=full), mask=full)
            hi = reg.vbitwise_or(reg.vshl(c2, 16, mask=full), reg.vshl(c3, 24, mask=full), mask=full)
            reg.vscatter(packed_y, reg.vbitwise_or(lo, hi, mask=full), logical4, mask=valid)
            reg.vstore(self.scale32, part * 8, block_scales, eight)
        # Publish scale32 stores before the packing loop reloads and truncates scale codes.
        reg.vmem_bar("vst_vld")
        for offset in range(0, scale_count, 64):
            scale_codes = reg.vload(self.scale32, offset)
            scale_mask = reg.update_mask(scale_count - offset, elem_bits=32)[0]
            reg.vstore_pack(scales, offset, scale_codes, scale_mask, pack_mode=reg.PackMode.B32_TO_B8)
        # Finish encoded output and packed scale-byte stores before DMA reads these slots.
        reg.vmem_bar("vst_vld")


@jit
def quantize_wide64(
    self, random, x, y, scales, row, group, quant_length, width, d_offset, tile_rows, seed, block_offset
):
    """Quantize one S tile by 64 physical D lanes without GM transposes."""
    row_base = dtypes.int64(row) * dtypes.int64(width) + dtypes.int64(d_offset)
    col_group = dtypes.int64(group) * dtypes.int64(tile_rows) // 16
    axis_groups = dtypes.int64(quant_length) // 16
    key0, key1 = _split_uint64(seed)
    scale_pairs = reinterpret(scales, dtypes.uint16, shape=(1, self.tile_elements // 32))
    with vf(mode="simd"):
        full = reg.full_mask()
        lane = reg.varange(0, dtypes.uint32)
        for counter in range(tile_rows // 16):
            c0, c1, c2, c3 = _linear_counter_rows(row_base, lane, col_group + counter, axis_groups, block_offset, full)
            w0, w1, w2, w3 = _philox_counter10(c0, c1, c2, c3, key0, key1, full)
            reg.vstore(random.buffer, counter * 256, reg.vreinterpret(w0, dtypes.int32), full)
            reg.vstore(random.buffer, counter * 256 + 64, reg.vreinterpret(w1, dtypes.int32), full)
            reg.vstore(random.buffer, counter * 256 + 128, reg.vreinterpret(w2, dtypes.int32), full)
            reg.vstore(random.buffer, counter * 256 + 192, reg.vreinterpret(w3, dtypes.int32), full)
        # Publish Philox scratch before random-word gathers in the following stage.
        reg.vmem_bar("vst_vld")
    with vf(mode="simd"):
        full = reg.full_mask()
        for half in range(tile_rows // 32):
            maxima = reg.vdups(0, dtypes.uint32, mask=full)
            for inner in range(32):
                offset = (half * 32 + inner) * width + d_offset
                if const_expr(self.bf16):
                    loaded = reg.vload_unpack(x, offset, unpack_mode=reg.UnpackMode.B16_TO_B32)
                    values = reg.vcast(loaded, dtypes.float32, mask=full)
                else:
                    values = reg.vload(x, offset)
                absolute = reg.vbitwise_and(
                    reg.vreinterpret(values, dtypes.uint32), reg.vdups(2147483647, dtypes.uint32, mask=full), mask=full
                )
                maxima = reg.vmax(maxima, absolute, mask=full)
            scale_codes, reciprocal = compute_scale(self, maxima, full)
            reg.vstore(self.scale32, half * 64, scale_codes, full)
            for word in range(8):
                slot = half * 512 + word * 64
                original = reg.vreinterpret(reg.vload(random.buffer, slot), dtypes.uint32)
                reversed_word = _rotate32_words8(original, full)
                first = half * 32 + word * 4
                store_wide_row(
                    self, x, y, first * width + d_offset, reciprocal, reg.vshr(original, 16, mask=full), full
                )
                store_wide_row(
                    self,
                    x,
                    y,
                    (first + 1) * width + d_offset,
                    reciprocal,
                    reg.vbitwise_and(reversed_word, reg.vdups(65535, dtypes.uint32, mask=full), mask=full),
                    full,
                )
                store_wide_row(
                    self,
                    x,
                    y,
                    (first + 2) * width + d_offset,
                    reciprocal,
                    reg.vbitwise_and(original, reg.vdups(65535, dtypes.uint32, mask=full), mask=full),
                    full,
                )
                store_wide_row(
                    self, x, y, (first + 3) * width + d_offset, reciprocal, reg.vshr(reversed_word, 16, mask=full), full
                )
        # Publish scale32 stores before the pair-packing loop reloads both scale rows.
        reg.vmem_bar("vst_vld")
        pair_mask = reg.update_mask(64, elem_bits=16)[0]
        for pair in range(tile_rows // 64):
            first_scale = reg.vload(self.scale32, pair * 128)
            second_scale = reg.vload(self.scale32, pair * 128 + 64)
            code_pairs = reg.vbitwise_or(first_scale, reg.vshl(second_scale, 8, mask=full), mask=full)
            packed = reg.vpack(code_pairs, dtypes.uint16, part="lower")
            reg.vstore(scale_pairs, pair * width + d_offset, packed, pair_mask)
        # Finish encoded output and packed scale-byte stores before DMA reads these slots.
        reg.vmem_bar("vst_vld")


@jit
def quantize_wide32(
    self, random, x, y, scales, row, group, quant_length, width, d_offset, tile_rows, seed, block_offset
):
    """Quantize one S tile by 32 physical D lanes without GM transposes."""
    lanes = 32
    row_base = dtypes.int64(row) * dtypes.int64(width) + dtypes.int64(d_offset)
    col_group = dtypes.int64(group) * dtypes.int64(tile_rows) // 16
    axis_groups = dtypes.int64(quant_length) // 16
    key0, key1 = _split_uint64(seed)
    scale_pairs = reinterpret(scales, dtypes.uint16, shape=(1, self.tile_elements // 32))
    with vf(mode="simd"):
        full = reg.update_mask(lanes, elem_bits=32)[0]
        lane = reg.varange(0, dtypes.uint32)
        for counter in range(tile_rows // 16):
            c0, c1, c2, c3 = _linear_counter_rows(row_base, lane, col_group + counter, axis_groups, block_offset, full)
            w0, w1, w2, w3 = _philox_counter10(c0, c1, c2, c3, key0, key1, full)
            reg.vstore(random.buffer, counter * 4 * lanes, reg.vreinterpret(w0, dtypes.int32), full)
            reg.vstore(random.buffer, counter * 4 * lanes + lanes, reg.vreinterpret(w1, dtypes.int32), full)
            reg.vstore(random.buffer, counter * 4 * lanes + 2 * lanes, reg.vreinterpret(w2, dtypes.int32), full)
            reg.vstore(random.buffer, counter * 4 * lanes + 3 * lanes, reg.vreinterpret(w3, dtypes.int32), full)
        # Publish Philox scratch before random-word gathers in the following stage.
        reg.vmem_bar("vst_vld")
    with vf(mode="simd"):
        full = reg.update_mask(lanes, elem_bits=32)[0]
        for half in range(tile_rows // 32):
            if const_expr(not self.bf16):
                full64 = reg.full_mask()
                lane64 = reg.varange(0, dtypes.uint32)
                maxima64 = reg.vdups(0, dtypes.uint32, mask=full64)
                for inner in range(16):
                    offset = (half * 32 + inner * 2) * width + d_offset
                    values64 = reg.vload(x, offset)
                    absolute64 = reg.vbitwise_and(
                        reg.vreinterpret(values64, dtypes.uint32),
                        reg.vdups(2147483647, dtypes.uint32, mask=full64),
                        mask=full64,
                    )
                    maxima64 = reg.vmax(maxima64, absolute64, mask=full64)
                swap32 = reg.vbitwise_xor(lane64, reg.vdups(32, dtypes.uint32, mask=full64), mask=full64)
                maxima = reg.vmax(maxima64, reg.vgather_reg(maxima64, swap32), mask=full64)
            else:
                full64 = reg.full_mask()
                lane64 = reg.varange(0, dtypes.uint32)
                maxima64 = reg.vdups(0, dtypes.uint32, mask=full64)
                for inner in range(16):
                    offset = (half * 32 + inner * 2) * width + d_offset
                    loaded64 = reg.vload_unpack(x, offset, unpack_mode=reg.UnpackMode.B16_TO_B32)
                    values64 = reg.vcast(loaded64, dtypes.float32, mask=full64)
                    absolute64 = reg.vbitwise_and(
                        reg.vreinterpret(values64, dtypes.uint32),
                        reg.vdups(2147483647, dtypes.uint32, mask=full64),
                        mask=full64,
                    )
                    maxima64 = reg.vmax(maxima64, absolute64, mask=full64)
                swap32 = reg.vbitwise_xor(lane64, reg.vdups(32, dtypes.uint32, mask=full64), mask=full64)
                maxima = reg.vmax(maxima64, reg.vgather_reg(maxima64, swap32), mask=full64)
            scale_codes, reciprocal = compute_scale(self, maxima, full)
            reg.vstore(self.scale32, half * lanes, scale_codes, full)
            for word in range(8):
                slot = half * 8 * lanes + word * lanes
                original = reg.vreinterpret(reg.vload(random.buffer, slot), dtypes.uint32)
                reversed_word = _rotate32_words8(original, full)
                first = half * 32 + word * 4
                store_wide_row(
                    self, x, y, first * width + d_offset, reciprocal, reg.vshr(original, 16, mask=full), full
                )
                store_wide_row(
                    self,
                    x,
                    y,
                    (first + 1) * width + d_offset,
                    reciprocal,
                    reg.vbitwise_and(reversed_word, reg.vdups(65535, dtypes.uint32, mask=full), mask=full),
                    full,
                )
                store_wide_row(
                    self,
                    x,
                    y,
                    (first + 2) * width + d_offset,
                    reciprocal,
                    reg.vbitwise_and(original, reg.vdups(65535, dtypes.uint32, mask=full), mask=full),
                    full,
                )
                store_wide_row(
                    self, x, y, (first + 3) * width + d_offset, reciprocal, reg.vshr(reversed_word, 16, mask=full), full
                )
        # Publish scale32 stores before the pair-packing loop reloads both scale rows.
        reg.vmem_bar("vst_vld")
        pair_mask = reg.update_mask(lanes, elem_bits=16)[0]
        for pair in range(tile_rows // 64):
            first_scale = reg.vload(self.scale32, pair * 2 * lanes)
            second_scale = reg.vload(self.scale32, pair * 2 * lanes + lanes)
            code_pairs = reg.vbitwise_or(first_scale, reg.vshl(second_scale, 8, mask=full), mask=full)
            packed = reg.vpack(code_pairs, dtypes.uint16, part="lower")
            reg.vstore(scale_pairs, pair * width + d_offset, packed, pair_mask)
        # Finish encoded output and packed scale-byte stores before DMA reads these slots.
        reg.vmem_bar("vst_vld")


@jit
def load_wide8_bf16_aligned(self, x, offset, mask):
    aligned = offset // 16 * 16
    loaded = reg.vload_unpack(x, aligned, unpack_mode=reg.UnpackMode.B16_TO_B32)
    sixteen = reg.update_mask(16, elem_bits=32)[0]
    values = reg.vcast(loaded, dtypes.float32, mask=sixteen)
    ids = reg.varange(0, dtypes.uint32)
    indices = reg.vadd(ids, reg.vdups(offset % 16, dtypes.uint32, mask=mask), mask=mask)
    return reg.vgather_reg(values, indices)


@jit
def quantize_wide8(
    self, random, x, y, scales, row, group, quant_length, width, d_offset, tile_rows, seed, block_offset
):
    """Quantize one S tile by compile-time physical D lanes without GM transposes."""
    lanes = self.wide_lanes
    row_base = dtypes.int64(row) * dtypes.int64(width) + dtypes.int64(d_offset)
    col_group = dtypes.int64(group) * dtypes.int64(tile_rows) // 16
    axis_groups = dtypes.int64(quant_length) // 16
    random_group_width = 16
    if const_expr(self.bf16 and lanes == 16):
        random_group_width = 32
        col_group = dtypes.int64(group) * dtypes.int64(tile_rows) // 32
        axis_groups = dtypes.int64(quant_length) // 32
    key0, key1 = _split_uint64(seed)
    with vf(mode="simd"):
        if const_expr(lanes == 8):
            full = reg.full_mask()
            ids = reg.varange(0, dtypes.uint32)
            channel = reg.vbitwise_and(ids, reg.vdups(7, dtypes.uint32, mask=full), mask=full)
            counter_lane = reg.vshr(ids, 3, mask=full)
            row_low_bits, row_high_bits = _split_uint64(row_base)
            for base in range(0, tile_rows // random_group_width, 8):
                counter = reg.vadds(counter_lane, base, mask=full)
                col_low_bits, col_high_bits = _split_uint64(col_group + base)
                row_low, row_high = _pair_add(
                    reg.vdups(row_low_bits, dtypes.uint32, mask=full),
                    reg.vdups(row_high_bits, dtypes.uint32, mask=full),
                    channel,
                    reg.vdups(0, dtypes.uint32, mask=full),
                    full,
                )
                col_low, col_high = _pair_add(
                    reg.vdups(col_low_bits, dtypes.uint32, mask=full),
                    reg.vdups(col_high_bits, dtypes.uint32, mask=full),
                    counter_lane,
                    reg.vdups(0, dtypes.uint32, mask=full),
                    full,
                )
                c0, c1, c2, c3 = _linear_counter_coordinates(
                    row_low, row_high, col_low, col_high, axis_groups, block_offset, full
                )
                w0, w1, w2, w3 = _philox_counter10(c0, c1, c2, c3, key0, key1, full)
                slot = reg.vadd(reg.vshl(counter, 5, mask=full), channel, mask=full)
                reg.vscatter(random.buffer, reg.vreinterpret(w0, dtypes.int32), slot, mask=full)
                reg.vscatter(
                    random.buffer, reg.vreinterpret(w1, dtypes.int32), reg.vadds(slot, 8, mask=full), mask=full
                )
                reg.vscatter(
                    random.buffer, reg.vreinterpret(w2, dtypes.int32), reg.vadds(slot, 16, mask=full), mask=full
                )
                reg.vscatter(
                    random.buffer, reg.vreinterpret(w3, dtypes.int32), reg.vadds(slot, 24, mask=full), mask=full
                )
            # Publish Philox words before the following SIMD VF gathers them.
            reg.vmem_bar("vst_vld")
        else:
            full = reg.update_mask(lanes, elem_bits=32)[0]
            lane = reg.varange(0, dtypes.uint32)
            for counter in range(tile_rows // random_group_width):
                c0, c1, c2, c3 = _linear_counter_rows(
                    row_base, lane, col_group + counter, axis_groups, block_offset, full
                )
                w0, w1, w2, w3 = _philox_counter10(c0, c1, c2, c3, key0, key1, full)
                reg.vstore(random.buffer, counter * 4 * lanes, reg.vreinterpret(w0, dtypes.int32), full)
                reg.vstore(random.buffer, counter * 4 * lanes + lanes, reg.vreinterpret(w1, dtypes.int32), full)
                reg.vstore(random.buffer, counter * 4 * lanes + 2 * lanes, reg.vreinterpret(w2, dtypes.int32), full)
                reg.vstore(random.buffer, counter * 4 * lanes + 3 * lanes, reg.vreinterpret(w3, dtypes.int32), full)
            # Publish Philox words before the following SIMD VF gathers them.
            reg.vmem_bar("vst_vld")
    with vf(mode="simd"):
        full = reg.update_mask(lanes, elem_bits=32)[0]
        y_cursor = reg.vstore_unalign_begin(y)
        for half in range(tile_rows // 32):
            if const_expr(lanes == 16):
                full64 = reg.full_mask()
                lane64 = reg.varange(0, dtypes.uint32)
                maxima64 = reg.vdups(0, dtypes.uint32, mask=full64)
                for inner in range(8):
                    offset = (half * 32 + inner * 4) * width + d_offset
                    if const_expr(self.bf16):
                        loaded64 = reg.vload_unpack(x, offset, unpack_mode=reg.UnpackMode.B16_TO_B32)
                        values64 = reg.vcast(loaded64, dtypes.float32, mask=full64)
                    else:
                        values64 = reg.vload(x, offset)
                    absolute64 = reg.vbitwise_and(
                        reg.vreinterpret(values64, dtypes.uint32),
                        reg.vdups(2147483647, dtypes.uint32, mask=full64),
                        mask=full64,
                    )
                    maxima64 = reg.vmax(maxima64, absolute64, mask=full64)
                swap16 = reg.vbitwise_xor(lane64, reg.vdups(16, dtypes.uint32, mask=full64), mask=full64)
                maxima64 = reg.vmax(maxima64, reg.vgather_reg(maxima64, swap16), mask=full64)
                swap32 = reg.vbitwise_xor(lane64, reg.vdups(32, dtypes.uint32, mask=full64), mask=full64)
                maxima = reg.vmax(maxima64, reg.vgather_reg(maxima64, swap32), mask=full64)
            elif const_expr(lanes == 8):
                full64 = reg.full_mask()
                lane64 = reg.varange(0, dtypes.uint32)
                maxima64 = reg.vdups(0, dtypes.uint32, mask=full64)
                for inner in range(4):
                    offset = (half * 32 + inner * 8) * width + d_offset
                    if const_expr(self.bf16):
                        loaded64 = reg.vload_unpack(x, offset, unpack_mode=reg.UnpackMode.B16_TO_B32)
                        values64 = reg.vcast(loaded64, dtypes.float32, mask=full64)
                    else:
                        values64 = reg.vload(x, offset)
                    absolute64 = reg.vbitwise_and(
                        reg.vreinterpret(values64, dtypes.uint32),
                        reg.vdups(2147483647, dtypes.uint32, mask=full64),
                        mask=full64,
                    )
                    maxima64 = reg.vmax(maxima64, absolute64, mask=full64)
                for bit in (8, 16, 32):
                    swapped = reg.vbitwise_xor(lane64, reg.vdups(bit, dtypes.uint32, mask=full64), mask=full64)
                    maxima64 = reg.vmax(maxima64, reg.vgather_reg(maxima64, swapped), mask=full64)
                maxima = maxima64
            else:
                maxima = reg.vdups(0, dtypes.uint32, mask=full)
                for inner in range(32):
                    offset = (half * 32 + inner) * width + d_offset
                    if const_expr(self.bf16):
                        loaded = reg.vload_unpack(x, offset, unpack_mode=reg.UnpackMode.B16_TO_B32)
                        values = reg.vcast(loaded, dtypes.float32, mask=full)
                    else:
                        values = reg.vload(x, offset)
                    absolute = reg.vbitwise_and(
                        reg.vreinterpret(values, dtypes.uint32),
                        reg.vdups(2147483647, dtypes.uint32, mask=full),
                        mask=full,
                    )
                    maxima = reg.vmax(maxima, absolute, mask=full)
            scale_codes, reciprocal = compute_scale(self, maxima, full)
            reg.vstore(self.scale32, half * lanes, scale_codes, full)
            if const_expr(self.bf16 and lanes == 16):
                for word in range(4):
                    slot = half * 4 * lanes + word * lanes
                    original = reg.vreinterpret(reg.vload(random.buffer, slot), dtypes.uint32)
                    first = half * 32 + word * 8
                    if const_expr(self.mantissa == 3):
                        r0 = reg.vbitwise_or(
                            reg.vshr(original, 20, mask=full), reg.vshl(original, 12, mask=full), mask=full
                        )
                        r2 = reg.vbitwise_or(
                            reg.vshr(original, 28, mask=full), reg.vshl(original, 4, mask=full), mask=full
                        )
                        r4 = reg.vbitwise_or(
                            reg.vshr(original, 4, mask=full), reg.vshl(original, 28, mask=full), mask=full
                        )
                        r6 = reg.vbitwise_or(
                            reg.vshr(original, 12, mask=full), reg.vshl(original, 20, mask=full), mask=full
                        )
                    else:
                        r0 = reg.vbitwise_or(
                            reg.vshr(original, 21, mask=full), reg.vshl(original, 11, mask=full), mask=full
                        )
                        r2 = reg.vbitwise_or(
                            reg.vshr(original, 29, mask=full), reg.vshl(original, 3, mask=full), mask=full
                        )
                        r4 = reg.vbitwise_or(
                            reg.vshr(original, 5, mask=full), reg.vshl(original, 27, mask=full), mask=full
                        )
                        r6 = reg.vbitwise_or(
                            reg.vshr(original, 13, mask=full), reg.vshl(original, 19, mask=full), mask=full
                        )
                    r1 = reg.vbitwise_or(reg.vshr(original, 24, mask=full), reg.vshl(original, 8, mask=full), mask=full)
                    r5 = reg.vbitwise_or(reg.vshr(original, 8, mask=full), reg.vshl(original, 24, mask=full), mask=full)
                    r7 = reg.vbitwise_or(
                        reg.vshr(original, 16, mask=full), reg.vshl(original, 16, mask=full), mask=full
                    )
                    r3 = original
                    store_wide8_row_unaligned(
                        self,
                        x,
                        y,
                        (first + 0) * width + d_offset,
                        reciprocal,
                        reg.vbitwise_and(r0, reg.vdups(65535, dtypes.uint32, mask=full), mask=full),
                        full,
                        y_cursor,
                    )
                    store_wide8_row_unaligned(
                        self,
                        x,
                        y,
                        (first + 1) * width + d_offset,
                        reciprocal,
                        reg.vbitwise_and(r1, reg.vdups(65535, dtypes.uint32, mask=full), mask=full),
                        full,
                        y_cursor,
                    )
                    store_wide8_row_unaligned(
                        self,
                        x,
                        y,
                        (first + 2) * width + d_offset,
                        reciprocal,
                        reg.vbitwise_and(r2, reg.vdups(65535, dtypes.uint32, mask=full), mask=full),
                        full,
                        y_cursor,
                    )
                    store_wide8_row_unaligned(
                        self,
                        x,
                        y,
                        (first + 3) * width + d_offset,
                        reciprocal,
                        reg.vbitwise_and(r3, reg.vdups(65535, dtypes.uint32, mask=full), mask=full),
                        full,
                        y_cursor,
                    )
                    store_wide8_row_unaligned(
                        self,
                        x,
                        y,
                        (first + 4) * width + d_offset,
                        reciprocal,
                        reg.vbitwise_and(r4, reg.vdups(65535, dtypes.uint32, mask=full), mask=full),
                        full,
                        y_cursor,
                    )
                    store_wide8_row_unaligned(
                        self,
                        x,
                        y,
                        (first + 5) * width + d_offset,
                        reciprocal,
                        reg.vbitwise_and(r5, reg.vdups(65535, dtypes.uint32, mask=full), mask=full),
                        full,
                        y_cursor,
                    )
                    store_wide8_row_unaligned(
                        self,
                        x,
                        y,
                        (first + 6) * width + d_offset,
                        reciprocal,
                        reg.vbitwise_and(r6, reg.vdups(65535, dtypes.uint32, mask=full), mask=full),
                        full,
                        y_cursor,
                    )
                    store_wide8_row_unaligned(
                        self,
                        x,
                        y,
                        (first + 7) * width + d_offset,
                        reciprocal,
                        reg.vbitwise_and(r7, reg.vdups(65535, dtypes.uint32, mask=full), mask=full),
                        full,
                        y_cursor,
                    )
            elif const_expr(lanes == 8):
                full64 = reg.full_mask()
                lane64 = reg.varange(0, dtypes.uint32)
                row8 = reg.vshr(lane64, 3, mask=full64)
                channel = reg.vbitwise_and(lane64, reg.vdups(7, dtypes.uint32, mask=full64), mask=full64)
                row4 = reg.vbitwise_and(row8, reg.vdups(3, dtypes.uint32, mask=full64), mask=full64)
                reciprocal64 = reg.vgather_reg(reciprocal, channel)
                for inner in range(4):
                    offset = (half * 32 + inner * 8) * width + d_offset
                    if const_expr(self.bf16):
                        loaded64 = reg.vload_unpack(x, offset, unpack_mode=reg.UnpackMode.B16_TO_B32)
                        values64 = reg.vcast(loaded64, dtypes.float32, mask=full64)
                    else:
                        values64 = reg.vload(x, offset)
                    normalized64 = reg.vmul(values64, reciprocal64, mask=full64)
                    if const_expr(self.bf16):
                        rounded64 = reg.vcast(normalized64, dtypes.bfloat16, mask=full64, rounding=reg.RoundingMode.RN)
                        normalized64 = reg.vcast(rounded64, dtypes.float32, mask=full64)
                    word_index = reg.vadd(
                        reg.vshl(reg.vshr(row8, 2, mask=full64), 3, mask=full64), channel, mask=full64
                    )
                    word_index = reg.vadds(word_index, half * 8 * lanes + inner * 2 * lanes, mask=full64)
                    word64 = reg.vreinterpret(reg.vgather(random.buffer, word_index, mask=full64), dtypes.uint32)
                    rotated64 = _rotate32_words8(word64, full64)
                    r16 = reg.vdups(0, dtypes.uint32, mask=full64)
                    r16 = reg.vselect(reg.vshr(word64, 16, mask=full64), r16, cond_mask=reg.veqs(row4, 0, mask=full64))
                    r16 = reg.vselect(
                        reg.vbitwise_and(rotated64, reg.vdups(65535, dtypes.uint32, mask=full64), mask=full64),
                        r16,
                        cond_mask=reg.veqs(row4, 1, mask=full64),
                    )
                    r16 = reg.vselect(
                        reg.vbitwise_and(word64, reg.vdups(65535, dtypes.uint32, mask=full64), mask=full64),
                        r16,
                        cond_mask=reg.veqs(row4, 2, mask=full64),
                    )
                    r16 = reg.vselect(
                        reg.vshr(rotated64, 16, mask=full64), r16, cond_mask=reg.veqs(row4, 3, mask=full64)
                    )
                    codes = _sr_fp32_to_fp8_q16(normalized64, r16, self.mantissa, self.bias, self.max_code, full64)
                    reg.vstore_pack(y, offset, codes, full64, pack_mode=reg.PackMode.B32_TO_B8)
            else:
                for word in range(8):
                    slot = half * 8 * lanes + word * lanes
                    original = reg.vreinterpret(reg.vload(random.buffer, slot), dtypes.uint32)
                    reversed_word = _rotate32_words8(original, full)
                    first = half * 32 + word * 4
                    store_wide8_row_unaligned(
                        self,
                        x,
                        y,
                        first * width + d_offset,
                        reciprocal,
                        reg.vshr(original, 16, mask=full),
                        full,
                        y_cursor,
                    )
                    store_wide8_row_unaligned(
                        self,
                        x,
                        y,
                        (first + 1) * width + d_offset,
                        reciprocal,
                        reg.vbitwise_and(reversed_word, reg.vdups(65535, dtypes.uint32, mask=full), mask=full),
                        full,
                        y_cursor,
                    )
                    store_wide8_row_unaligned(
                        self,
                        x,
                        y,
                        (first + 2) * width + d_offset,
                        reciprocal,
                        reg.vbitwise_and(original, reg.vdups(65535, dtypes.uint32, mask=full), mask=full),
                        full,
                        y_cursor,
                    )
                    store_wide8_row_unaligned(
                        self,
                        x,
                        y,
                        (first + 3) * width + d_offset,
                        reciprocal,
                        reg.vshr(reversed_word, 16, mask=full),
                        full,
                        y_cursor,
                    )
        if const_expr(lanes != 8):
            reg.vsqueeze_and_storeunalign_finalize(y, 0, y_cursor)
        # Publish scale32 stores before the scale gather below.
        reg.vmem_bar("vst_vld")
        pair_mask = reg.update_mask(lanes * 2, elem_bits=8)[0]
        scale_cursor = reg.vstore_unalign_begin(scales)
        for pair in range(tile_rows // 64):
            first_scale = reg.vload(self.scale32, pair * 2 * lanes)
            second_scale = reg.vload(self.scale32, pair * 2 * lanes + lanes)
            code_pairs = reg.vbitwise_or(first_scale, reg.vshl(second_scale, 8, mask=full), mask=full)
            packed = reg.vpack(code_pairs, dtypes.uint16, part="lower")
            packed_bytes = reg.vreinterpret_lanes(packed, dtypes.uint8)
            squeezed = reg.vsqueeze_and_storeunalign_init(packed_bytes, mask=pair_mask)
            reg.vsqueeze_and_storeunalign(scales, 0, squeezed, scale_cursor)
        reg.vsqueeze_and_storeunalign_finalize(scales, 0, scale_cursor)
        # Finish output/scale stores before the owner DMA reads these slots.
        reg.vmem_bar("vst_vld")


@jit
def store_wide_row(self, x, y, offset, reciprocal, random16, mask):
    if const_expr(self.bf16):
        loaded = reg.vload_unpack(x, offset, unpack_mode=reg.UnpackMode.B16_TO_B32)
        value = reg.vcast(loaded, dtypes.float32, mask=mask)
    else:
        value = reg.vload(x, offset)
    normalized = reg.vmul(value, reciprocal, mask=mask)
    if const_expr(self.bf16):
        rounded = reg.vcast(normalized, dtypes.bfloat16, mask=mask, rounding=reg.RoundingMode.RN)
        normalized = reg.vcast(rounded, dtypes.float32, mask=mask)
    codes = _sr_fp32_to_fp8_q16(normalized, random16, self.mantissa, self.bias, self.max_code, mask)
    reg.vstore_pack(y, offset, codes, mask, pack_mode=reg.PackMode.B32_TO_B8)


def store_wide8_row_unaligned(self, x, y, offset, reciprocal, random16, mask, y_cursor):
    if const_expr(self.bf16):
        if const_expr(self.wide_lanes == 8):
            value = load_wide8_bf16_aligned(self, x, offset, mask)
        else:
            loaded = reg.vload_unpack(x, offset, unpack_mode=reg.UnpackMode.B16_TO_B32)
            value = reg.vcast(loaded, dtypes.float32, mask=mask)
    else:
        value = reg.vload(x, offset)
    normalized = reg.vmul(value, reciprocal, mask=mask)
    if const_expr(self.bf16):
        rounded = reg.vcast(normalized, dtypes.bfloat16, mask=mask, rounding=reg.RoundingMode.RN)
        normalized = reg.vcast(rounded, dtypes.float32, mask=mask)
    codes = _sr_fp32_to_fp8_q16(normalized, random16, self.mantissa, self.bias, self.max_code, mask)
    packed16 = reg.vpack(codes, dtypes.uint16, part="lower")
    packed8 = reg.vpack(packed16, dtypes.uint8, part="lower")
    byte_mask = reg.update_mask(self.wide_lanes, elem_bits=8)[0]
    squeezed = reg.vsqueeze_and_storeunalign_init(packed8, mask=byte_mask)
    reg.vsqueeze_and_storeunalign(y, 0, squeezed, y_cursor)


@jit
def run_short_rows(self, x, y, scales, quant_length, seed, block_offset):
    columns = x.shape[1]
    rows = x.shape[0]
    pitch_shift = 6
    max_rows = self.quantization.tile_elements // 64
    if columns > 64:
        pitch_shift = 7
        max_rows = self.quantization.tile_elements // 128
    batch_rows = rows // get_block_num()
    if batch_rows < 1:
        batch_rows = 1
    if batch_rows > max_rows:
        batch_rows = max_rows
    tiles = (rows + batch_rows - 1) // batch_rows
    scale_columns = scales.shape[1]
    flat_x = x.view(1, rows * columns)
    flat_y = y.view(1, rows * columns)
    flat_scales = scales.view(1, rows * scale_columns)
    for task in range(get_block_idx(), tiles, get_block_num()):
        count = rows - task * batch_rows
        if count > batch_rows:
            count = batch_rows
        input_slot = self.input.next_write()
        mem_copy(
            tile_slice(input_slot, make_tiler((1, self.quantization.tile_elements), alignment=(1, 16)), (0, 0)),
            tile_slice(flat_x, make_tiler((1, batch_rows * columns), alignment=(1, 16)), (0, task)),
        )
        input_tile = tile_slice(
            self.input.next_read(), make_tiler((1, self.quantization.tile_elements), alignment=(1, 16)), (0, 0)
        )
        output_tile = self.output.next_write()
        scale_tile = self.scale.next_write()
        quantize_short_rows(
            self.quantization,
            self.random.words,
            input_tile,
            output_tile,
            scale_tile,
            task * batch_rows * columns,
            count,
            columns,
            pitch_shift,
            seed,
            block_offset,
        )
        output_copy = self.output.next_read()
        _copy_to_gm(
            tile_slice(flat_y, make_tiler((1, batch_rows * columns), alignment=(1, 16)), (0, task)), output_copy
        )
        scale_copy = self.scale.next_read()
        _copy_to_gm(
            tile_slice(flat_scales, make_tiler((1, batch_rows * scale_columns), alignment=(1, 2)), (0, task)),
            scale_copy,
        )


@jit
def run_tail(self, x, y, scales, quant_length, seed, block_offset):
    columns = x.shape[1]
    pairs = (columns + 63) // 64
    scale_factor = 2
    if scales.shape[1] == columns // 16:
        scale_factor = 4
    pairs_per_core = (x.shape[0] * pairs + get_block_num() - 1) // get_block_num()
    tiles_per_core = (pairs_per_core + self.max_pairs - 1) // self.max_pairs
    pair_batch = (pairs_per_core + tiles_per_core - 1) // tiles_per_core
    if pair_batch >= 4:
        pair_batch = (pair_batch + 3) // 4 * 4
    if const_expr(self.quantization.bf16_guard_tail):
        if pairs * x.shape[0] >= get_block_num() * 64:
            pair_batch = (pair_batch + 15) // 16 * 16
    tile_columns = pair_batch * 64
    column_tiles = (pairs + pair_batch - 1) // pair_batch
    tasks = x.shape[0] * column_tiles
    for task in range(get_block_idx(), tasks, get_block_num()):
        row = task // column_tiles
        tile = task % column_tiles
        count = columns - tile * tile_columns
        if count > tile_columns:
            count = tile_columns
        input_slot = self.input.next_write()
        mem_copy(
            tile_slice(input_slot, make_tiler((1, self.quantization.tile_elements), alignment=(1, 64)), (0, 0)),
            tile_slice(x, make_tiler((1, tile_columns), alignment=(1, 64)), (row, tile)),
        )
        input_tile = tile_slice(
            self.input.next_read(), make_tiler((1, self.quantization.tile_elements), alignment=(1, 64)), (0, 0)
        )
        output_tile = self.output.next_write()
        scale_tile = self.scale.next_write()
        if count >= 256 and count % 256 == 0:
            quantize_tile_256(
                self.quantization,
                self.random.words,
                input_tile,
                output_tile,
                scale_tile,
                row * columns + tile * tile_columns,
                count,
                scale_factor,
                quant_length,
                seed,
                block_offset,
            )
        else:
            quantize_tile(
                self.quantization,
                self.random.words,
                input_tile,
                output_tile,
                scale_tile,
                row * columns + tile * tile_columns,
                count,
                scale_factor,
                quant_length,
                seed,
                block_offset,
            )
        output_copy = self.output.next_read()
        _copy_to_gm(tile_slice(y, make_tiler((1, tile_columns), alignment=(1, 64)), (row, tile)), output_copy)
        scale_copy = self.scale.next_read()
        _copy_to_gm(
            tile_slice(scales, make_tiler((1, pair_batch * scale_factor), alignment=(1, 2)), (row, tile)), scale_copy
        )


@jit
def run_coalesced(self, x, y, scales, quant_length, seed, block_offset):
    columns = x.shape[1]
    pairs = columns // 64
    core_pairs = (pairs + get_block_num() - 1) // get_block_num()
    if core_pairs >= 4:
        core_pairs = (core_pairs + 3) // 4 * 4
    if const_expr(self.quantization.bf16_guard_tail):
        if pairs >= get_block_num() * 64:
            core_pairs = (core_pairs + 15) // 16 * 16
    core_columns = core_pairs * 64
    core_start = get_block_idx() * core_columns
    scale_factor = 2
    if scales.shape[1] == columns // 16:
        scale_factor = 4
    if core_start < columns:
        core_x = tile_slice(x, make_tiler((1, core_columns), alignment=(1, 64)), (0, get_block_idx()))
        core_y = tile_slice(y, make_tiler((1, core_columns), alignment=(1, 64)), (0, get_block_idx()))
        core_s = tile_slice(scales, make_tiler((1, core_pairs * scale_factor), alignment=(1, 2)), (0, get_block_idx()))
        core_count = columns - core_start
        if core_count > core_columns:
            core_count = core_columns
        tiles = (core_count + self.quantization.tile_elements - 1) // self.quantization.tile_elements
        for tile in range(tiles):
            count = core_count - tile * self.quantization.tile_elements
            if count > self.quantization.tile_elements:
                count = self.quantization.tile_elements
            input_slot = self.input.next_write()
            mem_copy(
                tile_slice(input_slot, make_tiler((1, self.quantization.tile_elements), alignment=(1, 64)), (0, 0)),
                tile_slice(core_x, make_tiler((1, self.quantization.tile_elements), alignment=(1, 64)), (0, tile)),
            )
            input_tile = tile_slice(
                self.input.next_read(), make_tiler((1, self.quantization.tile_elements), alignment=(1, 64)), (0, 0)
            )
            output_tile = self.output.next_write()
            scale_tile = self.scale.next_write()
            start = core_start + tile * self.quantization.tile_elements
            if count >= 256 and count % 256 == 0:
                quantize_tile_256(
                    self.quantization,
                    self.random.words,
                    input_tile,
                    output_tile,
                    scale_tile,
                    start,
                    count,
                    scale_factor,
                    quant_length,
                    seed,
                    block_offset,
                )
            else:
                quantize_tile(
                    self.quantization,
                    self.random.words,
                    input_tile,
                    output_tile,
                    scale_tile,
                    start,
                    count,
                    scale_factor,
                    quant_length,
                    seed,
                    block_offset,
                )
            output_copy = self.output.next_read()
            _copy_to_gm(
                tile_slice(core_y, make_tiler((1, self.quantization.tile_elements), alignment=(1, 64)), (0, tile)),
                output_copy,
            )
            scale_copy = self.scale.next_read()
            _copy_to_gm(
                tile_slice(core_s, make_tiler((1, self.max_pairs * scale_factor), alignment=(1, 2)), (0, tile)),
                scale_copy,
            )


@jit
def _dispatch_standard_tail(self, x, y, scales, quant_length, seed, block_offset):
    """One geometry dispatcher for GM and eager scalar RNG carriers."""
    if (
        x.shape[0] == 1
        and x.shape[1] % 64 == 0
        and (x.shape[1] > get_block_num() * self.quantization.tile_elements)
        and (x.shape[1] % (get_block_num() * self.quantization.tile_elements) != 0)
    ):
        run_coalesced(self, x, y, scales, quant_length, seed, block_offset)
    elif x.shape[0] > 1 and x.shape[1] < 128:
        run_short_rows(self, x, y, scales, quant_length, seed, block_offset)
    else:
        run_tail(self, x, y, scales, quant_length, seed, block_offset)


@jit
def _stage_run_wide_non_tail(
    self, x: Tensor, y: Tensor, scales: Tensor, quant_length: int, width: int, seed, block_offset
):
    base_tasks = x.shape[0] * (quant_length // 64)
    tile_rows = 64
    if base_tasks >= 256 and quant_length % PACKED_K_ALIGNMENT_ELEMENTS == 0 and (width == 64):
        tile_rows = 256
    elif base_tasks >= 256 and quant_length % 128 == 0:
        tile_rows = 128
    groups = quant_length // tile_rows
    tasks = x.shape[0] * groups
    for task in range(get_block_idx(), tasks, get_block_num()):
        row = task // groups
        group = task % groups
        tile = tile_rows * width
        mem_copy(self.input.next_write(), tile_slice(x, make_tiler((1, tile), alignment=(1, 64)), (row, group)))
        x_tile = self.input.next_read()
        y_tile = self.output.next_write()
        scale_tile = self.scale.next_write()
        for d_chunk in range(width // 64):
            quantize_wide64(
                self.quantization,
                self.random.words,
                x_tile,
                y_tile,
                scale_tile,
                row,
                group,
                quant_length,
                width,
                d_chunk * 64,
                tile_rows,
                seed,
                block_offset,
            )
        _copy_to_gm(tile_slice(y, make_tiler((1, tile), alignment=(1, 64)), (row, group)), self.output.next_read())
        _copy_to_gm(
            tile_slice(scales, make_tiler((1, tile_rows * width // 32), alignment=(1, 128)), (row, group)),
            self.scale.next_read(),
        )


@jit
def _stage_run_wide32_non_tail(
    self, x: Tensor, y: Tensor, scales: Tensor, quant_length: int, width: int, seed, block_offset
):
    base_tasks = x.shape[0] * (quant_length // 64)
    tile_rows = 64
    if base_tasks >= 256 and quant_length % 512 == 0 and (width == 32):
        tile_rows = 512
    elif base_tasks >= 256 and quant_length % PACKED_K_ALIGNMENT_ELEMENTS == 0 and (width == 32):
        tile_rows = 256
    elif base_tasks >= 256 and quant_length % 128 == 0:
        tile_rows = 128
    groups = quant_length // tile_rows
    tasks = x.shape[0] * groups
    for task in range(get_block_idx(), tasks, get_block_num()):
        row = task // groups
        group = task % groups
        tile = tile_rows * width
        mem_copy(self.input.next_write(), tile_slice(x, make_tiler((1, tile), alignment=(1, 64)), (row, group)))
        x_tile = self.input.next_read()
        y_tile = self.output.next_write()
        scale_tile = self.scale.next_write()
        quantize_wide32(
            self.quantization,
            self.random.words,
            x_tile,
            y_tile,
            scale_tile,
            row,
            group,
            quant_length,
            width,
            0,
            tile_rows,
            seed,
            block_offset,
        )
        _copy_to_gm(tile_slice(y, make_tiler((1, tile), alignment=(1, 64)), (row, group)), self.output.next_read())
        _copy_to_gm(
            tile_slice(scales, make_tiler((1, tile_rows * width // 32), alignment=(1, 64)), (row, group)),
            self.scale.next_read(),
        )


@jit
def _stage_run_wide8_non_tail(
    self, x: Tensor, y: Tensor, scales: Tensor, quant_length: int, width: int, seed, block_offset
):
    tile_rows = 256
    groups = quant_length // tile_rows
    tasks = x.shape[0] * groups
    for task in range(get_block_idx(), tasks, get_block_num()):
        row = task // groups
        group = task % groups
        tile = tile_rows * width
        mem_copy(self.input.next_write(), tile_slice(x, make_tiler((1, tile), alignment=(1, 64)), (row, group)))
        x_tile = self.input.next_read()
        y_tile = self.output.next_write()
        scale_tile = self.scale.next_write()
        quantize_wide8(
            self.quantization,
            self.random.words,
            x_tile,
            y_tile,
            scale_tile,
            row,
            group,
            quant_length,
            width,
            0,
            tile_rows,
            seed,
            block_offset,
        )
        _copy_to_gm(tile_slice(y, make_tiler((1, tile), alignment=(1, 64)), (row, group)), self.output.next_read())
        _copy_to_gm(
            tile_slice(scales, make_tiler((1, tile_rows * width // 32), alignment=(1, 64)), (row, group)),
            self.scale.next_read(),
        )


@jit
def _stage_run_non_tail(self, x0: Tensor, x1: Tensor, y: Tensor, scales: Tensor, quant_length: int, seed, block_offset):
    """Quantize two stride-2 source channels without an input transpose.

    The output carriers remain in moved-axis order. One Torch transpose per
    output restores the public contiguous layout after this kernel.
    """
    columns = x0.shape[1]
    pairs = columns // 64
    logical_rows = x0.shape[0] * 2
    pairs_per_core = (logical_rows * pairs + get_block_num() - 1) // get_block_num()
    tiles_per_core = (pairs_per_core + self.max_pairs - 1) // self.max_pairs
    pair_batch = (pairs_per_core + tiles_per_core - 1) // tiles_per_core
    if pair_batch >= 4:
        pair_batch = (pair_batch + 3) // 4 * 4
    tile_columns = pair_batch * 64
    column_tiles = (pairs + pair_batch - 1) // pair_batch
    tasks = logical_rows * column_tiles
    for task in range(get_block_idx(), tasks, get_block_num()):
        logical_row = task // column_tiles
        row = logical_row // 2
        channel = logical_row % 2
        tile = task % column_tiles
        count = columns - tile * tile_columns
        if count > tile_columns:
            count = tile_columns
        if channel == 0:
            mem_copy(
                self.input.next_write(),
                tile_slice(x0, make_tiler((1, tile_columns), alignment=(1, 64)), (row, tile)),
                copy_options(engine="nddma"),
            )
        else:
            mem_copy(
                self.input.next_write(),
                tile_slice(x1, make_tiler((1, tile_columns), alignment=(1, 64)), (row, tile)),
                copy_options(engine="nddma"),
            )
        input_tile = self.input.next_read()
        output_tile = self.output.next_write()
        scale_tile = self.scale.next_write()
        start = logical_row * columns + tile * tile_columns
        if count >= 256 and count % 256 == 0:
            quantize_tile_256(
                self.quantization,
                self.random.words,
                input_tile,
                output_tile,
                scale_tile,
                start,
                count,
                2,
                quant_length,
                seed,
                block_offset,
            )
        else:
            quantize_tile(
                self.quantization,
                self.random.words,
                input_tile,
                output_tile,
                scale_tile,
                start,
                count,
                2,
                quant_length,
                seed,
                block_offset,
            )
        _copy_to_gm(
            tile_slice(y, make_tiler((1, tile_columns), alignment=(1, 64)), (logical_row, tile)),
            self.output.next_read(),
        )
        _copy_to_gm(
            tile_slice(scales, make_tiler((1, pair_batch * 2), alignment=(1, 2)), (logical_row, tile)),
            self.scale.next_read(),
        )


@jit
def _stage_run_non_tail_fused(
    self, x0: Tensor, x1: Tensor, y: Tensor, scales: Tensor, quant_length: int, seed, block_offset
):
    """Quantize both stride-2 channels and interleave in UB before GM stores."""
    columns = x0.shape[1]
    pairs = columns // 64
    rows = x0.shape[0]
    pairs_per_core = (rows * pairs + get_block_num() - 1) // get_block_num()
    tiles_per_core = (pairs_per_core + self.max_pairs - 1) // self.max_pairs
    pair_batch = (pairs_per_core + tiles_per_core - 1) // tiles_per_core
    pair_batch = (pair_batch + 3) // 4 * 4
    tile_columns = pair_batch * 64
    column_tiles = (pairs + pair_batch - 1) // pair_batch
    tasks = rows * column_tiles
    for task in range(get_block_idx(), tasks, get_block_num()):
        row = task // column_tiles
        tile = task % column_tiles
        count = columns - tile * tile_columns
        if count > tile_columns:
            count = tile_columns
        mem_copy(
            self.input.next_write(),
            tile_slice(x0, make_tiler((1, tile_columns), alignment=(1, 64)), (row, tile)),
            copy_options(engine="nddma"),
        )
        x_tile = self.input.next_read()
        y_tile = self.output.next_write()
        s_tile = self.scale.next_write()
        quantize_tile_256(
            self.quantization,
            self.random.words,
            x_tile,
            y_tile,
            s_tile,
            row * 2 * columns + tile * tile_columns,
            count,
            2,
            quant_length,
            seed,
            block_offset,
        )
        mem_copy(self.saved_output, self.output.next_read())
        mem_copy(self.saved_scale, self.scale.next_read())
        mem_copy(
            self.input.next_write(),
            tile_slice(x1, make_tiler((1, tile_columns), alignment=(1, 64)), (row, tile)),
            copy_options(engine="nddma"),
        )
        x_tile = self.input.next_read()
        y_tile = self.output.next_write()
        s_tile = self.scale.next_write()
        quantize_tile_256(
            self.quantization,
            self.random.words,
            x_tile,
            y_tile,
            s_tile,
            (row * 2 + 1) * columns + tile * tile_columns,
            count,
            2,
            quant_length,
            seed,
            block_offset,
        )
        second_y = self.output.next_read()
        second_s = self.scale.next_read()
        first_pairs = reinterpret(self.saved_scale, dtypes.uint16, shape=(1, self.quantization.tile_elements // 32))
        second_pairs = reinterpret(second_s, dtypes.uint16, shape=(1, self.quantization.tile_elements // 32))
        with vf(mode="simd"):
            for offset in range(0, count, 256):
                a = reg.vload(self.saved_output, offset)
                b = reg.vload(second_y, offset)
                reg.vstore_interleave(self.interleaved_output, offset * 2, a, b)
            for offset in range(0, count // 64, 128):
                a = reg.vload(first_pairs, offset)
                b = reg.vload(second_pairs, offset)
                reg.vstore_interleave(self.interleaved_scale, offset * 2, a, b)
        _copy_to_gm(
            tile_slice(y, make_tiler((1, tile_columns * 2), alignment=(1, 512)), (row, tile)), self.interleaved_output
        )
        scale_bytes = reinterpret(self.interleaved_scale, dtypes.uint8, shape=(1, self.quantization.tile_elements // 8))
        _copy_to_gm(tile_slice(scales, make_tiler((1, pair_batch * 4), alignment=(1, 16)), (row, tile)), scale_bytes)


SIMT_UB_RESERVE = 40 * 1024


@jit
def _read_rng_state(rng_state):
    if const_expr(isinstance(rng_state, tuple)):
        if const_expr(isinstance(rng_state[0], Tensor)):
            seed = dtypes.uint64(scalar.load_bypass(rng_state[0].ptr(0)))
            block_offset = dtypes.uint64(scalar.load_bypass(rng_state[1].ptr(0))) + dtypes.uint64(rng_state[2])
        else:
            seed = dtypes.uint64(rng_state[0])
            block_offset = dtypes.uint64(rng_state[1])
    else:
        seed = dtypes.uint64(scalar.load_bypass(rng_state.ptr(0)))
        block_offset = dtypes.uint64(scalar.load_bypass(rng_state.ptr(1)))
    return (seed, block_offset)


@dataclass(frozen=True)
class _RandomWords:
    """Narrow Philox-to-SR handoff: UB storage and its byte capacity."""

    buffer: object
    capacity_bytes: int


class _RandomStage:
    """Own Philox storage; numerical encoders receive only its words view."""

    def __init__(self, tile, *, compact_mode=0, output_alias=False, guard_tail=False):
        capacity = (
            tile // 2 + BF16_RANDOM_GATHER_PADDING_BYTES if compact_mode == BF16_COMPACT_TILE_KI_ELEMENTS else tile
        )
        self.words = None
        if not output_alias:
            self.words = _RandomWords(Buffer(MemLoc.UB, (1, capacity // 4), dtypes.int32), capacity)
        if not guard_tail:
            # Preserve the conservative non-tail UB budget used by tiling.
            self.compatibility_reserve = Buffer(MemLoc.UB, (1, tile // 4), dtypes.int32)


class _ScaleSrStage:
    """Own scale/SR parameters, scale scratch and normalization flags.

    Random-word views and input/output slots arrive as explicit arguments.
    This stage owns neither Philox storage nor DMA channels.
    """

    def __init__(
        self,
        input_dtype,
        e5m2,
        tile,
        scale_alg=1,
        max_low_bound_bits=0,
        fused_non_tail=False,
        wide_lanes=8,
        guard_tail=False,
        fp32_eight_mapping=False,
        bf16_compact_mode=0,
        bf16_compact_four_panel=False,
        bf16_tail_simd=False,
        bf16_four_panel=False,
        fp32_four_pair_schedule=False,
        bf16_crossdraw_prefix3=False,
        bf16_full_tiles=False,
    ):
        self.wide_lanes = wide_lanes
        self.scale_alg = scale_alg
        self.max_low_bound_bits = max_low_bound_bits
        self.bf16 = input_dtype == dtypes.bfloat16
        self.bf16_guard_tail = self.bf16 and guard_tail and (not fused_non_tail)
        self.bf16_compact_mode = bf16_compact_mode if self.bf16_guard_tail else 0
        dense_full_flags = bf16_full_tiles and scale_alg == 0 and not e5m2
        self.norm_flag_stride = 1 if dense_full_flags else (4 if self.bf16_compact_mode else 16)
        self.norm_tile_flag = tile // 256 if dense_full_flags else (tile // 1024 * 16 if self.bf16_compact_mode else 16)
        self.fp32_eight_tail = fp32_eight_mapping and (not self.bf16) and guard_tail and (not fused_non_tail)
        self.invalid_packed_reciprocal = 4286644096 if self.bf16 else 4286578688
        self.mantissa = 2 if e5m2 else 3
        self.bias = 15 if e5m2 else 7
        self.max_code = 123 if e5m2 else 126
        self.inv_max_bits = 932333861 if e5m2 else 991054117
        self.tile_elements = tile
        self.bf16_compact_four_panel = bf16_compact_four_panel
        self.bf16_tail_simd = bf16_tail_simd
        self.bf16_four_panel = bf16_four_panel
        self.fp32_four_pair_schedule = fp32_four_pair_schedule
        self.bf16_crossdraw_prefix3 = bf16_crossdraw_prefix3
        self.bf16_full_tiles = bf16_full_tiles

    def _allocate_scale(self, scale_capacity):
        self.scale32 = Buffer(MemLoc.UB, (1, scale_capacity), dtypes.uint32)

    def _allocate_normalization(self, tile):
        if self.bf16_guard_tail:
            flag_slots, reciprocal_slots = _bf16_normalization_slots(
                tile,
                compact_mode=self.bf16_compact_mode,
                full_tiles=self.bf16_full_tiles,
                e5m2=self.mantissa == 2,
                scale_alg=self.scale_alg,
            )
            self.norm_flags = Buffer(MemLoc.UB, (1, flag_slots), dtypes.uint16)
            if reciprocal_slots:
                self.recip_scale = Buffer(MemLoc.UB, (1, reciprocal_slots), dtypes.uint16)


@kernel
class _DynamicMxQuantSrKernel:
    def __init__(
        self,
        input_dtype,
        e5m2,
        tile,
        scale_capacity,
        scale_alg=1,
        max_low_bound_bits=0,
        fused_non_tail=False,
        wide_lanes=8,
        guard_tail=False,
        fp32_eight_mapping=False,
        output_rng_alias=False,
        bf16_compact_mode=0,
        bf16_compact_four_panel=False,
        bf16_tail_simd=False,
        bf16_four_panel=False,
        bf16_full_tiles=False,
        fp32_four_pair_schedule=False,
        bf16_crossdraw_prefix3=False,
    ):
        self.bf16_full_tiles = bf16_full_tiles
        self.quantization = _ScaleSrStage(
            input_dtype=input_dtype,
            e5m2=e5m2,
            tile=tile,
            scale_alg=scale_alg,
            max_low_bound_bits=max_low_bound_bits,
            fused_non_tail=fused_non_tail,
            wide_lanes=wide_lanes,
            guard_tail=guard_tail,
            fp32_eight_mapping=fp32_eight_mapping,
            bf16_compact_mode=bf16_compact_mode,
            bf16_compact_four_panel=bf16_compact_four_panel,
            bf16_tail_simd=bf16_tail_simd,
            bf16_four_panel=bf16_four_panel,
            fp32_four_pair_schedule=fp32_four_pair_schedule,
            bf16_crossdraw_prefix3=bf16_crossdraw_prefix3,
            bf16_full_tiles=bf16_full_tiles,
        )
        quantization = self.quantization
        self.max_pairs = tile // 64
        self.output_rng_alias = output_rng_alias and quantization.fp32_eight_tail
        input_tile = tile
        input_depth = 1 if quantization.bf16_compact_mode == BF16_SINGLE_INPUT_TILE_KI_ELEMENTS else 2
        self.input = RingBuffer(MemLoc.UB, (1, input_tile), input_dtype, depth=input_depth)
        output_tile = tile
        output_depth = 2 if self.output_rng_alias else 1 if guard_tail and (not fused_non_tail) else 2
        self.output = RingBuffer(MemLoc.UB, (1, output_tile), dtypes.uint8, depth=output_depth)
        self.scale = RingBuffer(MemLoc.UB, (1, output_tile // 16), dtypes.uint8, depth=output_depth)
        quantization._allocate_scale(scale_capacity)
        self.random = _RandomStage(
            tile, compact_mode=quantization.bf16_compact_mode, output_alias=self.output_rng_alias, guard_tail=guard_tail
        )
        quantization._allocate_normalization(tile)
        if fused_non_tail:
            self.saved_output = Buffer(MemLoc.UB, (1, tile), dtypes.uint8)
            self.saved_scale = Buffer(MemLoc.UB, (1, tile // 16), dtypes.uint8)
            self.interleaved_output = Buffer(MemLoc.UB, (1, 2 * tile), dtypes.uint8)
            self.interleaved_scale = Buffer(MemLoc.UB, (1, tile // 16), dtypes.uint16)

    def __call__(self, x: Tensor, y: Tensor, scales: Tensor, quant_length: int, rng_state: Tensor):
        seed, block_offset = _read_rng_state(rng_state)
        dispatch_tail(self, x, y, scales, quant_length, seed, block_offset)

    def run_tail_scalar_rng(
        self, x: Tensor, y: Tensor, scales: Tensor, quant_length: int, seed_bits, block_offset_bits
    ):
        """Eager scalar RNG entry; graph capture uses the device-state entry."""
        seed = dtypes.uint64(seed_bits)
        block_offset = dtypes.uint64(block_offset_bits)
        dispatch_tail(self, x, y, scales, quant_length, seed, block_offset)

    def run_wide_non_tail(self, x, y, scales, quant_length, width, rng_state):
        seed, block_offset = _read_rng_state(rng_state)
        _stage_run_wide_non_tail(self, x, y, scales, quant_length, width, seed, block_offset)

    def run_wide32_non_tail(self, x, y, scales, quant_length, width, rng_state):
        seed, block_offset = _read_rng_state(rng_state)
        _stage_run_wide32_non_tail(self, x, y, scales, quant_length, width, seed, block_offset)

    def run_wide8_non_tail(self, x, y, scales, quant_length, width, rng_state):
        seed, block_offset = _read_rng_state(rng_state)
        _stage_run_wide8_non_tail(self, x, y, scales, quant_length, width, seed, block_offset)

    def run_non_tail(self, x0, x1, y, scales, quant_length, rng_state):
        seed, block_offset = _read_rng_state(rng_state)
        _stage_run_non_tail(self, x0, x1, y, scales, quant_length, seed, block_offset)

    def run_non_tail_fused(self, x0, x1, y, scales, quant_length, rng_state):
        seed, block_offset = _read_rng_state(rng_state)
        _stage_run_non_tail_fused(self, x0, x1, y, scales, quant_length, seed, block_offset)


__all__ = ["dynamic_mx_quant_sr"]
STRIDE2_MIN_QUANT_LENGTH = 512
STRIDE2_ALIGNMENT = 64
FUSED_STRIDE2_ALIGNMENT = 256
SMALL_WIDE_ALIGNMENT = 256
WIDE_ALIGNMENT = 64
SMALL_WIDE_SUFFIX_WIDTHS = frozenset((8, 16))
WIDE_SUFFIX_WIDTHS = frozenset((32, 64, 128))
# Element alignment required by packed indexing, independent of performance.
PACKED_K_ALIGNMENT_ELEMENTS = 256
PHILOX_BATCH_ELEMENTS = 1024
BF16_WIDE8_MAX_ELEMENTS = 128 * 1024
WIDE16_MAX_ELEMENTS = 2 * 1024 * 1024


@dataclass(frozen=True)
class QuantizationTiling:
    tile_elements: int
    scale_capacity: int
    buffer_bytes: int
    ub_bytes: int


def _bf16_normalization_slots(tile, *, compact_mode=0, full_tiles=False, e5m2=False, scale_alg=1):
    """Count uint16 slots shared by UB allocation, tiling and route admission."""
    flags = (
        tile // 256 + 16
        if full_tiles and scale_alg == 0 and not e5m2
        else ((tile + 1023) // 1024 + 1) * 16
        if compact_mode
        else max(tile // 256, 1) * 16
    )
    reciprocals = tile // 32 if full_tiles and scale_alg in (0, 1) else 0
    return flags, reciprocals


def _bf16_guard_buffer_bytes(tile):
    return 2 * _bf16_normalization_slots(tile)[0]


def _make_tiling(
    input_dtype,
    ub_bytes,
    fused_non_tail=False,
    tail_axis=True,
    *,
    output_rng_alias=False,
    bf16_compact_mode=0,
    bf16_full_tiles=False,
    e5m2=False,
    scale_alg=1,
):
    if output_rng_alias and (input_dtype != dtypes.float32 or not tail_axis or fused_non_tail):
        raise ValueError("upper alias requires FP32 tail")
    if input_dtype not in (dtypes.float32, dtypes.bfloat16):
        raise TypeError("tiling requires float32 or bfloat16")
    if type(ub_bytes) is not int or ub_bytes <= SIMT_UB_RESERVE:
        raise ValueError("UB capacity must exceed the 40 KiB SIMT reserve")
    item = 4 if input_dtype == dtypes.float32 else 2
    budget = ub_bytes - SIMT_UB_RESERVE
    guard_inplace = input_dtype == dtypes.bfloat16 and tail_axis and (not fused_non_tail)
    input_depth = 1 if bf16_compact_mode == BF16_SINGLE_INPUT_TILE_KI_ELEMENTS and guard_inplace else 2
    tail_single = tail_axis and (not fused_non_tail)
    tile = (
        budget // 5 // 1024 * 1024
        if bf16_compact_mode == BF16_COMPACT_TILE_KI_ELEMENTS and guard_inplace
        else budget // (input_depth * item + (2 if tail_single else 4)) // 1024 * 1024
    )
    if input_dtype == dtypes.bfloat16:
        tile = min(
            tile,
            bf16_compact_mode * 1024
            if bf16_compact_mode and guard_inplace
            else BF16_TILE_LIMIT_ELEMENTS
            if guard_inplace
            else BF16_NON_TAIL_TILE_LIMIT_ELEMENTS,
        )
    while tile >= 1024:
        scales = (tile // 32 + 63) // 64 * 64
        output_depth = 2 if output_rng_alias else 1 if tail_single else 2
        used = (
            input_depth * tile * item
            + output_depth * tile
            + output_depth * (tile // 16)
            + 4 * scales
            + (0 if output_rng_alias else 1 if tail_single else 2) * tile
        )
        if bf16_compact_mode == BF16_COMPACT_TILE_KI_ELEMENTS and guard_inplace:
            used += tile // 2 + BF16_RANDOM_GATHER_PADDING_BYTES - tile
        if input_dtype == dtypes.bfloat16 and tail_axis and (not fused_non_tail):
            flags, reciprocals = _bf16_normalization_slots(
                tile, compact_mode=bf16_compact_mode, full_tiles=bf16_full_tiles, e5m2=e5m2, scale_alg=scale_alg
            )
            used += 2 * (flags + reciprocals)
        if fused_non_tail:
            used += tile + tile // 16 + 2 * tile + tile // 8
        if used <= budget:
            return QuantizationTiling(tile, scales, used, ub_bytes)
        tile -= 1024
    raise ValueError("UB remaining after the SIMT reserve cannot fit one tile")


@jit
def _new_kernel(config, tail_axis=False):
    return _DynamicMxQuantSrKernel(
        config.input_dtype,
        config.e5m2,
        config.tiling.tile_elements,
        config.tiling.scale_capacity,
        config.scale_alg,
        config.max_low_bound_bits,
        config.fused,
        config.wide_lanes,
        tail_axis,
        config.fp32_eight_mapping,
        config.output_rng_alias,
        config.bf16_compact_mode,
        config.bf16_compact_four_panel,
        config.bf16_tail_simd,
        config.bf16_four_panel,
        config.bf16_full_tiles,
        config.fp32_four_pair_schedule,
        config.bf16_crossdraw_prefix3,
    )


class _Launch:
    def __init__(
        self,
        input_dtype,
        e5m2,
        tiling,
        scale_alg,
        max_low_bound_bits,
        *,
        fused=False,
        wide_lanes=8,
        fp32_eight_mapping=False,
        output_rng_alias=False,
        bf16_compact_mode=0,
        bf16_compact_four_panel=False,
        bf16_tail_simd=False,
        bf16_four_panel=False,
        bf16_full_tiles=False,
        fp32_four_pair_schedule=False,
        bf16_crossdraw_prefix3=False,
    ):
        self.input_dtype, self.e5m2, self.tiling = (input_dtype, e5m2, tiling)
        self.scale_alg, self.max_low_bound_bits = (scale_alg, max_low_bound_bits)
        self.fused, self.wide_lanes = (fused, wide_lanes)
        self.fp32_eight_mapping = fp32_eight_mapping
        self.output_rng_alias = output_rng_alias
        self.bf16_compact_mode = bf16_compact_mode
        self.bf16_compact_four_panel = bf16_compact_four_panel
        self.bf16_tail_simd = bf16_tail_simd
        self.bf16_four_panel = bf16_four_panel
        self.bf16_full_tiles = bf16_full_tiles
        self.fp32_four_pair_schedule = fp32_four_pair_schedule
        self.bf16_crossdraw_prefix3 = bf16_crossdraw_prefix3

    @host
    def run(self, x, y, scales, quant_length: int, blocks: int, rng_state):
        op = _new_kernel(self, tail_axis=True)
        op[blocks](x, y, scales, quant_length, rng_state)

    @host
    def run_tail_scalar_rng(self, x, y, scales, quant_length: int, blocks: int, seed_bits, block_offset_bits):
        op = _new_kernel(self, tail_axis=True)
        op[blocks].run_tail_scalar_rng(x, y, scales, quant_length, seed_bits, block_offset_bits)

    @host
    def run_non_tail(self, x0, x1, y, scales, quant_length: int, blocks: int, rng_state):
        op = _new_kernel(self)
        op[blocks].run_non_tail(x0, x1, y, scales, quant_length, rng_state)

    @host
    def run_non_tail_fused(self, x0, x1, y, scales, quant_length: int, blocks: int, rng_state):
        op = _new_kernel(self)
        op[blocks].run_non_tail_fused(x0, x1, y, scales, quant_length, rng_state)

    @host
    def run_wide_non_tail(self, x, y, scales, quant_length: int, width: int, blocks: int, rng_state):
        op = _new_kernel(self)
        op[blocks].run_wide_non_tail(x, y, scales, quant_length, width, rng_state)

    @host
    def run_wide32_non_tail(self, x, y, scales, quant_length: int, width: int, blocks: int, rng_state):
        op = _new_kernel(self)
        op[blocks].run_wide32_non_tail(x, y, scales, quant_length, width, rng_state)

    @host
    def run_wide8_non_tail(self, x, y, scales, quant_length: int, width: int, blocks: int, rng_state):
        op = _new_kernel(self)
        op[blocks].run_wide8_non_tail(x, y, scales, quant_length, width, rng_state)

    @host
    def run_graph_rng(self, x, y, scales, quant_length: int, blocks: int, seed_tensor, offset_tensor, intragraph):
        op = _new_kernel(self, tail_axis=True)
        op[blocks](x, y, scales, quant_length, (seed_tensor, offset_tensor, intragraph))

    @host
    def run_non_tail_eager_rng(self, x0, x1, y, scales, quant_length: int, blocks: int, seed_bits, offset_bits):
        op = _new_kernel(self)
        op[blocks].run_non_tail(x0, x1, y, scales, quant_length, (seed_bits, offset_bits))

    @host
    def run_non_tail_graph_rng(
        self, x0, x1, y, scales, quant_length: int, blocks: int, seed_tensor, offset_tensor, intragraph
    ):
        op = _new_kernel(self)
        op[blocks].run_non_tail(x0, x1, y, scales, quant_length, (seed_tensor, offset_tensor, intragraph))

    @host
    def run_non_tail_fused_eager_rng(self, x0, x1, y, scales, quant_length: int, blocks: int, seed_bits, offset_bits):
        op = _new_kernel(self)
        op[blocks].run_non_tail_fused(x0, x1, y, scales, quant_length, (seed_bits, offset_bits))

    @host
    def run_non_tail_fused_graph_rng(
        self, x0, x1, y, scales, quant_length: int, blocks: int, seed_tensor, offset_tensor, intragraph
    ):
        op = _new_kernel(self)
        op[blocks].run_non_tail_fused(x0, x1, y, scales, quant_length, (seed_tensor, offset_tensor, intragraph))

    @host
    def run_wide_non_tail_eager_rng(
        self, x, y, scales, quant_length: int, width: int, blocks: int, seed_bits, offset_bits
    ):
        op = _new_kernel(self)
        op[blocks].run_wide_non_tail(x, y, scales, quant_length, width, (seed_bits, offset_bits))

    @host
    def run_wide_non_tail_graph_rng(
        self, x, y, scales, quant_length: int, width: int, blocks: int, seed_tensor, offset_tensor, intragraph
    ):
        op = _new_kernel(self)
        op[blocks].run_wide_non_tail(x, y, scales, quant_length, width, (seed_tensor, offset_tensor, intragraph))

    @host
    def run_wide32_non_tail_eager_rng(
        self, x, y, scales, quant_length: int, width: int, blocks: int, seed_bits, offset_bits
    ):
        op = _new_kernel(self)
        op[blocks].run_wide32_non_tail(x, y, scales, quant_length, width, (seed_bits, offset_bits))

    @host
    def run_wide32_non_tail_graph_rng(
        self, x, y, scales, quant_length: int, width: int, blocks: int, seed_tensor, offset_tensor, intragraph
    ):
        op = _new_kernel(self)
        op[blocks].run_wide32_non_tail(x, y, scales, quant_length, width, (seed_tensor, offset_tensor, intragraph))

    @host
    def run_wide8_non_tail_eager_rng(
        self, x, y, scales, quant_length: int, width: int, blocks: int, seed_bits, offset_bits
    ):
        op = _new_kernel(self)
        op[blocks].run_wide8_non_tail(x, y, scales, quant_length, width, (seed_bits, offset_bits))

    @host
    def run_wide8_non_tail_graph_rng(
        self, x, y, scales, quant_length: int, width: int, blocks: int, seed_tensor, offset_tensor, intragraph
    ):
        op = _new_kernel(self)
        op[blocks].run_wide8_non_tail(x, y, scales, quant_length, width, (seed_tensor, offset_tensor, intragraph))


def _tail_specs(dtype):
    rows = Dim("rows", min=1)
    columns = Dim("columns", min=16 if dtype == dtypes.bfloat16 else 32, multiple_of=16)
    scales = Dim("scale_columns", min=2, multiple_of=2)
    return (
        TensorSpec((rows, columns), dtype),
        TensorSpec((rows, columns), dtypes.uint8),
        TensorSpec((rows, scales), dtypes.uint8),
        dtypes.int64,
        dtypes.int64,
        TensorSpec((2,), dtypes.int64),
    )


def _tail_scalar_rng_specs(dtype):
    return (*_tail_specs(dtype)[:5], dtypes.int64, dtypes.int64)


def _tail_rng_signed64(value):
    """Preserve every unsigned seed/block-offset bit in an Int64 carrier."""
    if type(value) is not int or not 0 <= value < 1 << 64:
        raise ValueError("scalar RNG value must be a Python int in [0, 2**64)")
    return value if value < 1 << 63 else value - (1 << 64)


def _stride2_specs(dtype, *, fused):
    rows = Dim("rows", min=1)
    alignment = FUSED_STRIDE2_ALIGNMENT if fused else STRIDE2_ALIGNMENT
    columns = Dim("columns", min=alignment, multiple_of=alignment)
    row_stride = Dim("source_row_stride", min=alignment * 2, multiple_of=alignment * 2)
    source = TensorSpec((rows, columns), dtype, stride=(row_stride, 2))
    if fused:
        out_rows = rows
        out_columns = Dim("packed_columns", min=512, multiple_of=512)
        scale_columns = Dim("packed_scale_columns", min=16, multiple_of=16)
    else:
        out_rows = Dim("logical_rows", min=2, multiple_of=2)
        out_columns = columns
        scale_columns = Dim("scale_columns", min=2, multiple_of=2)
    return (
        source,
        source,
        TensorSpec((out_rows, out_columns), dtypes.uint8),
        TensorSpec((out_rows, scale_columns), dtypes.uint8),
        dtypes.int64,
        dtypes.int64,
        TensorSpec((2,), dtypes.int64),
    )


def _wide_specs(dtype, *, column_alignment):
    rows = Dim("rows", min=1)
    columns = Dim("physical_columns", min=column_alignment, multiple_of=column_alignment)
    scale_alignment = column_alignment // 32
    scales = Dim("physical_scale_columns", min=scale_alignment, multiple_of=scale_alignment)
    return (
        TensorSpec((rows, columns), dtype),
        TensorSpec((rows, columns), dtypes.uint8),
        TensorSpec((rows, scales), dtypes.uint8),
        dtypes.int64,
        dtypes.int64,
        dtypes.int64,
        TensorSpec((2,), dtypes.int64),
    )


@dataclass(frozen=True)
class _LayoutSpec:
    entry: str
    spec_factory: object
    wide_lanes: int = 8
    fused: bool = False


_LAYOUT_REGISTRY = MappingProxyType(
    {
        Layout.TAIL: _LayoutSpec("run", _tail_specs),
        Layout.NON_TAIL: _LayoutSpec("run_non_tail", partial(_stride2_specs, fused=False)),
        Layout.NON_TAIL_FUSED: _LayoutSpec("run_non_tail_fused", partial(_stride2_specs, fused=True), fused=True),
        Layout.WIDE: _LayoutSpec("run_wide_non_tail", partial(_wide_specs, column_alignment=4096)),
        Layout.WIDE32: _LayoutSpec("run_wide32_non_tail", partial(_wide_specs, column_alignment=2048)),
        Layout.WIDE8: _LayoutSpec("run_wide8_non_tail", partial(_wide_specs, column_alignment=2048)),
        Layout.WIDE16: _LayoutSpec("run_wide8_non_tail", partial(_wide_specs, column_alignment=4096), wide_lanes=16),
    }
)


def _make_launch(key):
    """Construct resources once from the immutable specialization key."""
    spec = _LAYOUT_REGISTRY[key.layout]
    dtype = dtypes.bfloat16 if key.input_dtype == "bf16" else dtypes.float32
    tiling = _make_tiling(
        dtype,
        key.ub_bytes,
        spec.fused,
        tail_axis=key.layout is Layout.TAIL,
        output_rng_alias=key.output_rng_alias,
        bf16_compact_mode=key.bf16_compact_mode,
        bf16_full_tiles=key.bf16_full_tiles,
        e5m2=key.e5m2,
        scale_alg=key.scale_alg,
    )
    launch = _Launch(
        dtype,
        key.e5m2,
        tiling,
        key.scale_alg,
        key.max_low_bound_bits,
        fused=spec.fused,
        wide_lanes=spec.wide_lanes,
        fp32_eight_mapping=key.fp32_eight_mapping,
        output_rng_alias=key.output_rng_alias,
        bf16_compact_mode=key.bf16_compact_mode,
        bf16_compact_four_panel=key.bf16_compact_four_panel,
        bf16_tail_simd=key.bf16_tail_simd,
        bf16_four_panel=key.bf16_four_panel,
        bf16_full_tiles=key.bf16_full_tiles,
        fp32_four_pair_schedule=key.fp32_four_pair_schedule,
        bf16_crossdraw_prefix3=key.bf16_crossdraw_prefix3,
    )
    return launch


def _build_program(key):
    """Exactly one standard AOT call; DSL manages its own binary cache."""
    spec = _LAYOUT_REGISTRY[key.layout]
    dtype = dtypes.bfloat16 if key.input_dtype == "bf16" else dtypes.float32
    launch = _make_launch(key)
    return cannbotdsl.compile(getattr(launch, spec.entry), *spec.spec_factory(dtype))


_PROGRAM_CACHE = _ProgramCache(64)
_TAIL_SCALAR_PROGRAM_CACHE = _ProgramCache(64)
_FRAMEWORK_PROGRAM_CACHE = _ProgramCache(128)


def clear_caches():
    """Release this operator's references only; external Programs stay live."""
    _PROGRAM_CACHE.clear()
    _TAIL_SCALAR_PROGRAM_CACHE.clear()
    _FRAMEWORK_PROGRAM_CACHE.clear()


def cache_info():
    return _PROGRAM_CACHE.info()


def _get_compiled_kernel(
    input_dtype,
    e5m2,
    ub_bytes,
    scale_alg=1,
    max_low_bound_bits=0,
    *,
    layout=Layout.TAIL,
    device=None,
    fp32_eight_mapping=False,
    output_rng_alias=False,
    bf16_compact_mode=0,
    bf16_compact_four_panel=False,
    bf16_tail_simd=False,
    bf16_four_panel=False,
    bf16_full_tiles=False,
    fp32_four_pair_schedule=False,
    bf16_crossdraw_prefix3=False,
    _context=None,
):
    if input_dtype not in (dtypes.float32, dtypes.bfloat16):
        raise TypeError("compile input dtype must be float32 or bfloat16")
    if device is None or device.index is None:
        device = torch.device("npu", torch.npu.current_device())
    context = _context or capture_compile_context(
        "bf16" if input_dtype == dtypes.bfloat16 else "fp32",
        e5m2,
        ub_bytes,
        scale_alg,
        max_low_bound_bits,
        layout,
        device,
        fp32_eight_mapping=fp32_eight_mapping,
        output_rng_alias=output_rng_alias,
        bf16_compact_mode=bf16_compact_mode,
        bf16_compact_four_panel=bf16_compact_four_panel,
        bf16_tail_simd=bf16_tail_simd,
        bf16_four_panel=bf16_four_panel,
        bf16_full_tiles=bf16_full_tiles,
        fp32_four_pair_schedule=fp32_four_pair_schedule,
        bf16_crossdraw_prefix3=bf16_crossdraw_prefix3,
    )
    return _PROGRAM_CACHE.get_or_build(
        context.key, lambda: _build_program(context.key), bypass=context.bypass_reason is not None
    )


def _get_compiled_tail_scalar_kernel(
    input_dtype,
    e5m2,
    ub_bytes,
    scale_alg,
    bound_bits,
    device,
    *,
    fp32_eight_mapping=False,
    output_rng_alias=False,
    bf16_compact_mode=0,
    bf16_compact_four_panel=False,
    bf16_tail_simd=False,
    bf16_four_panel=False,
    bf16_full_tiles=False,
    fp32_four_pair_schedule=False,
    bf16_crossdraw_prefix3=False,
    _context=None,
):
    context = _context or capture_compile_context(
        "bf16" if input_dtype == dtypes.bfloat16 else "fp32",
        e5m2,
        ub_bytes,
        scale_alg,
        bound_bits,
        Layout.TAIL,
        device,
        fp32_eight_mapping=fp32_eight_mapping,
        output_rng_alias=output_rng_alias,
        bf16_compact_mode=bf16_compact_mode,
        bf16_compact_four_panel=bf16_compact_four_panel,
        bf16_tail_simd=bf16_tail_simd,
        bf16_four_panel=bf16_four_panel,
        bf16_full_tiles=bf16_full_tiles,
        fp32_four_pair_schedule=fp32_four_pair_schedule,
        bf16_crossdraw_prefix3=bf16_crossdraw_prefix3,
    )

    def build():
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("warm up scalar RNG entry before graph capture")
        launch = _make_launch(context.key)
        return cannbotdsl.compile(launch.run_tail_scalar_rng, *_tail_scalar_rng_specs(input_dtype))

    return _TAIL_SCALAR_PROGRAM_CACHE.get_or_build(context.key, build, bypass=context.bypass_reason is not None)


def _get_compiled_framework_programs(
    input_dtype, e5m2, ub_bytes, scale_alg, bound_bits, *, layout, device, _context=None, **options
):
    """Flat AOT carriers; values and addresses never enter the cache key."""
    context = _context or capture_compile_context(
        "bf16" if input_dtype == dtypes.bfloat16 else "fp32",
        e5m2,
        ub_bytes,
        scale_alg,
        bound_bits,
        layout,
        device,
        **options,
    )
    spec = _LAYOUT_REGISTRY[layout]
    programs = []
    for captured in (False, True):
        if not captured and layout is Layout.TAIL:
            programs.append(
                _get_compiled_tail_scalar_kernel(
                    input_dtype,
                    e5m2,
                    ub_bytes,
                    scale_alg,
                    bound_bits,
                    device,
                    _context=context,
                    **options,
                )
            )
            continue

        def build(captured=captured):
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("warm up framework RNG entries before graph capture")
            launch = _make_launch(context.key)
            entry = spec.entry + ("_graph_rng" if captured else "_eager_rng")
            rng_specs = (
                (TensorSpec((1,), dtypes.int64), TensorSpec((1,), dtypes.int64), dtypes.int64)
                if captured
                else (dtypes.int64, dtypes.int64)
            )
            return cannbotdsl.compile(getattr(launch, entry), *spec.spec_factory(input_dtype)[:-1], *rng_specs)

        programs.append(
            _FRAMEWORK_PROGRAM_CACHE.get_or_build(
                (context.key, captured),
                build,
                bypass=context.bypass_reason is not None,
            )
        )
    return tuple(programs)


def _resolve_block_dim(x, block_dim, *, quant_length=None):
    """Host-only AIV budget; refresh stream quota without specializing code."""
    if block_dim is not None:
        return block_dim
    stream = torch.npu.current_stream(x.device)
    info = cannbotdsl.get_platform_info(stream=stream)
    count = getattr(info, "vector_core_num", None)
    if type(count) is not int or count <= 0:
        raise RuntimeError(f"NPU {x.device} reports invalid vector_core_num={count!r}")
    budget = count
    columns = x.shape[-1] if quant_length is None else quant_length
    if columns % 64 == 0 or columns == 32:
        budget = min(budget, (x.numel() + PHILOX_BATCH_ELEMENTS - 1) // PHILOX_BATCH_ELEMENTS)
    return budget


def _validate_input(x, axis=-1):
    """Apply the same tensor contract before every layout dispatch."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if x.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("input must have dtype float32 or bfloat16")
    if x.ndim < 1 or x.numel() == 0:
        raise ValueError("input must be nonempty and have at least one dimension")
    length = x.shape[axis]
    if length % 16 or (x.dtype == torch.float32 and length < 32):
        raise ValueError("quantization dimension must be a multiple of 16 (at least 32 for float32)")
    if not x.is_contiguous():
        raise ValueError("input must be contiguous")
    if x.device.type not in ("npu", "privateuseone"):
        raise ValueError("input must be on an Ascend NPU")


def _encode_bound(value):
    """Normalize to FP32 once; cache positive and negative zero together."""
    try:
        packed = struct.pack("<f", float(value))
    except OverflowError as error:
        raise ValueError("max_low_bound exceeds float32 range") from error
    normalized = struct.unpack("<f", packed)[0]
    if not math.isfinite(normalized):
        raise ValueError("max_low_bound exceeds float32 range")
    return 0 if normalized == 0 else struct.unpack("<I", packed)[0]


def _validate_attributes(input, axis, round_mode, dst_type, block_size, scale_alg, dst_type_max, max_low_bound):
    """Validate the public attributes before any device setup or allocation."""
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if input.ndim < 1 or input.numel() == 0:
        raise ValueError("input must be nonempty and have at least one dimension")
    if type(axis) is not int:
        raise TypeError("axis must be an integer")
    if not -input.ndim <= axis < input.ndim:
        raise ValueError("axis is out of range for input dimensions")
    axis = axis % input.ndim
    if not isinstance(round_mode, str):
        raise TypeError("round_mode must be a string")
    if round_mode != "stochastic":
        raise NotImplementedError("dynamic_mx_quant_sr requires round_mode='stochastic'")
    if type(dst_type) is not int:
        raise TypeError("dst_type must be a Torch integer dtype code: 23 or 24")
    if dst_type not in (23, 24):
        raise NotImplementedError("dst_type must be 23 (E5M2) or 24 (E4M3FN)")
    if type(block_size) is not int:
        raise TypeError("block_size must be an integer")
    if not 0 < block_size <= 1024 or block_size % 32:
        raise ValueError("block_size must be a positive multiple of 32, at most 1024")
    if block_size != 32:
        raise NotImplementedError("dynamic_mx_quant_sr requires block_size=32")
    if scale_alg is not None and type(scale_alg) is not int:
        raise TypeError("scale_alg must be an integer or None")
    if scale_alg not in (0, 1):
        raise NotImplementedError("dynamic_mx_quant_sr requires scale_alg=0 or 1")
    if type(dst_type_max) not in (float, int):
        raise TypeError("dst_type_max must be a float or integer")
    if isinstance(dst_type_max, float) and (not math.isfinite(dst_type_max)):
        raise ValueError("dst_type_max must be finite")
    if dst_type_max != 0:
        raise NotImplementedError("dynamic_mx_quant_sr requires dst_type_max=0.0")
    if type(max_low_bound) not in (float, int) or not math.isfinite(max_low_bound) or max_low_bound < 0:
        raise ValueError("max_low_bound must be a finite non-negative number")
    if scale_alg != 1 and max_low_bound != 0:
        raise ValueError("max_low_bound must be 0 when scale_alg != 1")
    bound_bits = _encode_bound(max_low_bound)
    dst_dtype = torch.float8_e5m2 if dst_type == 23 else torch.float8_e4m3fn
    return (axis, dst_dtype, bound_bits)


def _supports_stride2(x, axis):
    return (
        x.ndim >= 2
        and 0 <= axis < x.ndim - 1
        and (math.prod(x.shape[axis + 1 :]) == 2)
        and (x.shape[axis] >= STRIDE2_MIN_QUANT_LENGTH)
        and (x.shape[axis] % STRIDE2_ALIGNMENT == 0)
    )


def _supports_small_wide(quant_length, width):
    return (
        width in SMALL_WIDE_SUFFIX_WIDTHS
        and quant_length >= SMALL_WIDE_ALIGNMENT
        and (quant_length % SMALL_WIDE_ALIGNMENT == 0)
    )


def _prefer_bf16_wide16(numel, aiv_budget, ub_bytes):
    """Select legal direct VFs by per-AIV work and avoided transfer cost.

    The direct VF serializes fixed4096-element tasks on each AIV. It avoids
    three transpose kernels whose launch and volume costs amortize over N.
    Coefficients are relative cost units calibrated on the supported device;
    neither K nor an exact input shape participates in performance choice.
    """
    if _make_tiling(dtypes.bfloat16, ub_bytes, tail_axis=False).tile_elements < BF16_WIDE16_TASK_ELEMENTS:
        return False
    tasks = numel // BF16_WIDE16_TASK_ELEMENTS
    blocks = min(aiv_budget, tasks)
    waves = (tasks + blocks - 1) // blocks
    # At eight serialized tasks the accepted same-shape A/A screen favored transpose;
    # the three copies amortize better than another direct VF wave. This workload
    # bound scales with queried AIV quota, independent of an exact axis/shape.
    return (
        waves <= BF16_WIDE16_MAX_WAVES
        and BF16_WIDE16_WAVE_COST * waves <= BF16_TRANSPOSE_LAUNCH_COST + BF16_TRANSPOSE_ELEMENT_COST * numel
    )


def _prefer_fp32_wide16_transpose(numel, aiv_budget, dst_dtype, scale_alg, bound_bits):
    """Avoid serial WIDE16 waves where the existing four-field transpose wins.

    At exactly 2 Mi elements the transpose path selects a different random
    field map, so this same-map routing rule stops strictly before that edge.
    """
    if (dst_dtype == torch.float8_e4m3fn and bound_bits == 0) or (dst_dtype == torch.float8_e5m2 and scale_alg == 1):
        tasks = numel // FP32_WIDE16_TASK_ELEMENTS
        blocks = min(aiv_budget, tasks)
        return (tasks + blocks - 1) // blocks >= FP32_WIDE16_TRANSPOSE_MIN_WAVES
    return False


def _select_layout(x, axis, *, aiv_budget=None, ub_bytes=None, dst_dtype=None, scale_alg=None, bound_bits=0):
    """Keep layout legality separate from workload/transfer cost selection."""
    if axis == x.ndim - 1:
        return Layout.TAIL
    if _supports_stride2(x, axis):
        return Layout.NON_TAIL_FUSED if x.shape[axis] % FUSED_STRIDE2_ALIGNMENT == 0 else Layout.NON_TAIL
    quant_length, width = (x.shape[axis], math.prod(x.shape[axis + 1 :]))
    elements = x.numel()
    if _supports_small_wide(quant_length, width):
        if width == 8 and (x.dtype == torch.float32 or elements <= BF16_WIDE8_MAX_ELEMENTS):
            return Layout.WIDE8
        if width == 16:
            if x.dtype == torch.float32 and elements <= WIDE16_MAX_ELEMENTS:
                if elements < WIDE16_MAX_ELEMENTS and dst_dtype is not None:
                    if aiv_budget is None:
                        aiv_budget = _resolve_block_dim(x, None, quant_length=quant_length)
                    if _prefer_fp32_wide16_transpose(elements, aiv_budget, dst_dtype, scale_alg, bound_bits):
                        return Layout.TRANSPOSE
                return Layout.WIDE16
            if x.dtype == torch.bfloat16:
                if aiv_budget is None:
                    aiv_budget = _resolve_block_dim(x, None, quant_length=quant_length)
                if ub_bytes is None:
                    ub_bytes = get_mem_size("ub")
                if _prefer_bf16_wide16(elements, aiv_budget, ub_bytes):
                    return Layout.WIDE16
    if width in WIDE_SUFFIX_WIDTHS and quant_length >= WIDE_ALIGNMENT and (quant_length % WIDE_ALIGNMENT == 0):
        return Layout.WIDE32 if width == 32 else Layout.WIDE
    return Layout.TRANSPOSE


def _make_execution_plan(x, axis, *, block_dim=None, ub_bytes=None, dst_dtype=None, scale_alg=None, bound_bits=0):
    shape = tuple(x.shape)
    quant_length = shape[axis]
    budget = _resolve_block_dim(x, block_dim, quant_length=quant_length)
    layout = _select_layout(
        x,
        axis,
        aiv_budget=budget,
        ub_bytes=ub_bytes,
        dst_dtype=dst_dtype,
        scale_alg=scale_alg,
        bound_bits=bound_bits,
    )
    pairs = (quant_length + 63) // 64
    scale_shape = (*shape[:axis], pairs, *shape[axis + 1 :], 2)
    width = math.prod(shape[axis + 1 :])
    base_task_rows = 1
    if layout in (Layout.TAIL, Layout.TRANSPOSE):
        logical_rows = x.numel() // quant_length
        moved_shape = (*shape[:axis], *shape[axis + 1 :], quant_length)
        storage_output_shape = shape if layout is Layout.TAIL else moved_shape
        storage_scale_shape = (*storage_output_shape[:-1], pairs, 2)
        coalesced = quant_length % 64 == 0 or quant_length == 32
        kernel_shape = (1, x.numel()) if coalesced else (logical_rows, quant_length)
        kernel_scale_shape = (1, logical_rows * pairs * 2) if coalesced else (logical_rows, pairs * 2)
        if layout is Layout.TRANSPOSE and coalesced:
            budget = min(budget, (x.numel() + PHILOX_BATCH_ELEMENTS - 1) // PHILOX_BATCH_ELEMENTS)
        blocks = min(budget, logical_rows * pairs)
    elif layout in (Layout.NON_TAIL, Layout.NON_TAIL_FUSED):
        rows = math.prod(shape[:axis])
        fused = layout is Layout.NON_TAIL_FUSED
        logical_rows = rows if fused else rows * 2
        storage_output_shape = shape if fused else (rows, 2, quant_length)
        storage_scale_shape = scale_shape if fused else (rows, 2, pairs, 2)
        kernel_shape = (logical_rows, quant_length * (2 if fused else 1))
        kernel_scale_shape = (logical_rows, pairs * (4 if fused else 2))
        blocks = min(budget, (x.numel() + PHILOX_BATCH_ELEMENTS - 1) // PHILOX_BATCH_ELEMENTS, logical_rows * pairs)
    else:
        logical_rows = x.numel() // (quant_length * width)
        storage_output_shape, storage_scale_shape = (shape, scale_shape)
        kernel_shape = (logical_rows, quant_length * width)
        kernel_scale_shape = (logical_rows, pairs * width * 2)
        base_task_rows = 256 if layout in (Layout.WIDE8, Layout.WIDE16) else 64
        blocks = min(budget, logical_rows * (quant_length // base_task_rows))
    return ExecutionPlan(
        layout,
        axis,
        quant_length,
        width,
        logical_rows,
        shape,
        scale_shape,
        storage_output_shape,
        storage_scale_shape,
        kernel_shape,
        kernel_scale_shape,
        blocks,
        base_task_rows,
    )


def _kernel_views(x, encoded, scales, plan):
    return (x.view(plan.kernel_shape), encoded.view(plan.kernel_shape), scales.view(plan.kernel_scale_shape))


@dataclass(frozen=True)
class _PreparedLaunch:
    plan: ExecutionPlan
    program: object
    arguments: tuple
    encoded: torch.Tensor
    scales: torch.Tensor
    dst_dtype: torch.dtype
    scalar_program: object = None
    framework_programs: tuple = ()

    def launch(self, *, rng_state=None):
        device = self.arguments[0].device
        if not self.program.so_path or self.program.closed:
            raise RuntimeError("SR launch requires a live executable AOT Program")
        if rng_state is not None:
            if not isinstance(rng_state, torch.Tensor):
                raise TypeError("rng_state must be an NPU int64[2] tensor")
            if (
                rng_state.device != device
                or rng_state.dtype != torch.int64
                or rng_state.shape != (2,)
                or (not rng_state.is_contiguous())
            ):
                raise ValueError("rng_state must be contiguous int64[2] on the input device")
        with torch.npu.device(device):
            if rng_state is not None:
                self.program(*self.arguments, rng_state)
                return
            if len(self.framework_programs) != 2:
                raise RuntimeError("framework RNG entries must be prepared before launch")
            if self.scalar_program is not None and (not self.scalar_program.so_path or self.scalar_program.closed):
                raise RuntimeError("scalar RNG launch requires a live executable Program")
            for program in self.framework_programs:
                if not program.so_path or program.closed:
                    raise RuntimeError("framework RNG launch requires live executable Programs")
            state = _framework_rng._reserve_rng_carrier(device, _rng_increment(math.prod(self.plan.output_shape)))
            captured, secondary, seed, offset, intragraph = state
            if secondary:
                raise NotImplementedError("secondary-stream RNG capture requires separate validation")
            if captured:
                program = self.framework_programs[1]
                program(*self.arguments, seed, offset, _tail_rng_signed64(intragraph))
                # Actual Tensor arguments are stream-recorded by Program; keep
                # owning framework handles with the output storage as well.
                self.encoded._sr_rng_owners = (program, state)
                self.scales._sr_rng_owners = (program, state)
            else:
                program = self.scalar_program or self.framework_programs[0]
                if not program.so_path or program.closed:
                    raise RuntimeError("scalar RNG launch requires a live executable Program")
                program(*self.arguments, _tail_rng_signed64(seed), _tail_rng_signed64(offset))

    def outputs(self):
        y, scales = (self.encoded.view(self.dst_dtype), self.scales)
        if self.plan.layout is Layout.NON_TAIL:
            y = y.movedim(1, 2).contiguous().reshape(self.plan.output_shape)
            scales = scales.movedim(1, 2).contiguous().reshape(self.plan.scale_shape)
        elif self.plan.layout is Layout.TRANSPOSE:
            y = y.movedim(-1, self.plan.axis).contiguous()
            scales = scales.movedim(-2, self.plan.axis).contiguous()
        scales = scales.view(torch.float8_e8m0fnu)
        owners = getattr(self.encoded, "_sr_rng_owners", None)
        if owners is not None:
            # Dtype views do not retain the source Tensor's Python attributes.
            # Keep carrier/program handles on the actual public return values.
            y._sr_rng_owners = owners
            scales._sr_rng_owners = owners
        return (y, scales)


FP32_EIGHT_MIN_ELEMENTS = 2 * 1024 * 1024


def _select_fp32_eight_policy(numel, dtype, compile_layout, quant_length):
    if dtype != dtypes.float32 or compile_layout is not Layout.TAIL or quant_length % PACKED_K_ALIGNMENT_ELEMENTS:
        return False
    return numel >= FP32_EIGHT_MIN_ELEMENTS


def _select_bf16_crossdraw_prefix3_policy(numel, dtype, scale_alg, plan, ub_bytes):
    """Select complete four-field draws with scale-pair-aligned BF16 rows."""
    if dtype != dtypes.bfloat16 or scale_alg != 0 or plan.compile_layout is not Layout.TAIL:
        return False
    if (
        plan.quant_length % BF16_CROSSDRAW_ROW_ALIGNMENT_ELEMENTS
        or plan.quant_length % PACKED_K_ALIGNMENT_ELEMENTS == 0
        or numel % FOUR_FIELD_DRAW_ELEMENTS
    ):
        return False
    tile = _make_tiling(dtype, ub_bytes, tail_axis=True).tile_elements
    # Amortize the prefix3 schedule with at least one full tile per AIV.
    return numel >= plan.blocks * tile


def _prepare_launch(x, dst_dtype, axis, scale_alg, bound_bits, *, block_dim=None):
    """Host setup shared with benchmarks; launch arguments retain runtime sizes."""
    with torch.npu.device(x.device):
        ub_bytes = get_mem_size("ub")
        plan = _make_execution_plan(
            x,
            axis,
            block_dim=block_dim,
            ub_bytes=ub_bytes,
            dst_dtype=dst_dtype,
            scale_alg=scale_alg,
            bound_bits=bound_bits,
        )
        _validate_rng_geometry(x.numel(), plan.quant_length)
        encoded = torch.empty(plan.storage_output_shape, dtype=torch.uint8, device=x.device)
        scales = torch.empty(plan.storage_scale_shape, dtype=torch.uint8, device=x.device)
        source = x.movedim(axis, -1).contiguous() if plan.layout is Layout.TRANSPOSE else x
        dtype = dtypes.bfloat16 if x.dtype == torch.bfloat16 else dtypes.float32
        fp32_eight_mapping = _select_fp32_eight_policy(x.numel(), dtype, plan.compile_layout, plan.quant_length)
        compact_mode = _select_bf16_compact_tile_policy(
            x.numel(), dtype, dst_dtype, plan.layout, plan.quant_length, plan.blocks, ub_bytes
        )
        bf16_compact_four_panel = _select_bf16_compact_four_panel_policy(
            x.numel(), dtype, dst_dtype, scale_alg, bound_bits, plan.layout, plan.quant_length, plan.blocks, ub_bytes
        )
        bf16_four_panel = not bf16_compact_four_panel and (
            _select_bf16_four_panel_policy(
                x.numel(), dtype, dst_dtype, scale_alg, bound_bits, plan.layout, plan.quant_length, plan.blocks
            )
            or _select_bf16_lower_wave_four_panel_policy(
                x.numel(), dtype, dst_dtype, scale_alg, bound_bits, plan, ub_bytes, compact_mode
            )
        )
        draw_elements = PHILOX_DRAW_ELEMENTS
        bf16_tail_simd = (
            bf16_compact_four_panel and plan.blocks * draw_elements <= x.numel() < plan.blocks * 4 * draw_elements
        )
        if bf16_compact_four_panel:
            compact_mode = BF16_COMPACT_TILE_KI_ELEMENTS
        bf16_full_tiles = _select_bf16_full_tiles_policy(
            x.numel(), dtype, dst_dtype, scale_alg, bound_bits, plan, ub_bytes, compact_mode, bf16_four_panel
        )
        fp32_four_pair_schedule = _select_fp32_four_pair_policy(
            x.numel(), dtype, plan.compile_layout, plan.quant_length, plan.blocks, ub_bytes
        )
        bf16_crossdraw_prefix3 = _select_bf16_crossdraw_prefix3_policy(x.numel(), dtype, scale_alg, plan, ub_bytes)
        # One immutable snapshot per preparation; subsequent calls still rediscover identity.
        context = capture_compile_context(
            "bf16" if dtype == dtypes.bfloat16 else "fp32",
            dst_dtype == torch.float8_e5m2,
            ub_bytes,
            scale_alg,
            bound_bits,
            plan.compile_layout,
            x.device,
            fp32_eight_mapping=fp32_eight_mapping,
            output_rng_alias=fp32_eight_mapping,
            bf16_compact_mode=compact_mode,
            bf16_compact_four_panel=bf16_compact_four_panel,
            bf16_tail_simd=bf16_tail_simd,
            bf16_four_panel=bf16_four_panel,
            bf16_full_tiles=bf16_full_tiles,
            fp32_four_pair_schedule=fp32_four_pair_schedule,
            bf16_crossdraw_prefix3=bf16_crossdraw_prefix3,
        )
        program = _get_compiled_kernel(
            dtype,
            dst_dtype == torch.float8_e5m2,
            ub_bytes,
            scale_alg,
            bound_bits,
            layout=plan.compile_layout,
            device=x.device,
            _context=context,
            fp32_eight_mapping=fp32_eight_mapping,
            bf16_compact_mode=compact_mode,
            output_rng_alias=fp32_eight_mapping,
            bf16_compact_four_panel=bf16_compact_four_panel,
            bf16_tail_simd=bf16_tail_simd,
            bf16_four_panel=bf16_four_panel,
            bf16_full_tiles=bf16_full_tiles,
            fp32_four_pair_schedule=fp32_four_pair_schedule,
            bf16_crossdraw_prefix3=bf16_crossdraw_prefix3,
        )
        scalar_program = None
        if plan.compile_layout is Layout.TAIL:
            scalar_program = _get_compiled_tail_scalar_kernel(
                dtype,
                dst_dtype == torch.float8_e5m2,
                ub_bytes,
                scale_alg,
                bound_bits,
                x.device,
                _context=context,
                fp32_eight_mapping=fp32_eight_mapping,
                bf16_compact_mode=compact_mode,
                output_rng_alias=fp32_eight_mapping,
                bf16_compact_four_panel=bf16_compact_four_panel,
                bf16_tail_simd=bf16_tail_simd,
                bf16_four_panel=bf16_four_panel,
                bf16_full_tiles=bf16_full_tiles,
                fp32_four_pair_schedule=fp32_four_pair_schedule,
                bf16_crossdraw_prefix3=bf16_crossdraw_prefix3,
            )
        framework_programs = _get_compiled_framework_programs(
            dtype,
            dst_dtype == torch.float8_e5m2,
            ub_bytes,
            scale_alg,
            bound_bits,
            layout=plan.compile_layout,
            device=x.device,
            _context=context,
            fp32_eight_mapping=fp32_eight_mapping,
            bf16_compact_mode=compact_mode,
            output_rng_alias=fp32_eight_mapping,
            bf16_compact_four_panel=bf16_compact_four_panel,
            bf16_tail_simd=bf16_tail_simd,
            bf16_four_panel=bf16_four_panel,
            bf16_full_tiles=bf16_full_tiles,
            fp32_four_pair_schedule=fp32_four_pair_schedule,
            bf16_crossdraw_prefix3=bf16_crossdraw_prefix3,
        )
        if plan.layout in (Layout.TAIL, Layout.TRANSPOSE):
            arguments = (*_kernel_views(source, encoded, scales, plan), plan.quant_length, plan.blocks)
        elif plan.layout in (Layout.NON_TAIL, Layout.NON_TAIL_FUSED):
            source = x.reshape(-1, plan.quant_length, 2)
            arguments = (
                source[:, :, 0],
                source[:, :, 1],
                encoded.view(plan.kernel_shape),
                scales.view(plan.kernel_scale_shape),
                plan.quant_length,
                plan.blocks,
            )
        else:
            arguments = (*_kernel_views(source, encoded, scales, plan), plan.quant_length, plan.width, plan.blocks)
    return _PreparedLaunch(plan, program, arguments, encoded, scales, dst_dtype, scalar_program, framework_programs)


def _dynamic_mx_quant_sr_impl(
    x, dst_dtype=torch.float8_e4m3fn, *, block_dim=None, scale_alg=1, max_low_bound=0.0, rng_state=None
):
    """Private tail-axis launch budget for tuning; no second public API."""
    _validate_input(x)
    if dst_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise TypeError("dst_dtype must be float8_e4m3fn or float8_e5m2")
    if block_dim is not None and (type(block_dim) is not int or not 1 <= block_dim <= 0xFFFFFFFF):
        raise ValueError("block_dim must be None or a positive uint32 integer")
    prepared = _prepare_launch(x, dst_dtype, x.ndim - 1, scale_alg, _encode_bound(max_low_bound), block_dim=block_dim)
    prepared.launch(rng_state=rng_state)
    return prepared.outputs()


def dynamic_mx_quant_sr(
    input,
    *,
    axis=-1,
    round_mode="stochastic",
    dst_type=24,
    block_size=32,
    scale_alg=1,
    dst_type_max=0.0,
    max_low_bound=0.0,
):
    """Dynamic MXFP8 stochastic quantization, Torch-compatible argument names.

    FP32/BF16 -> E4M3FN (24)/E5M2 (23); any axis; block_size=32;
    scale_alg=0 (floor exponent) or 1 (ceil rounded product). max_low_bound
    extends the Torch interface for scale_alg=1. Returns input-shaped FP8 y
    and E8M0 scales: replace axis with ceil(K/64), then append dimension 2.

    Each call reserves from the input device's default framework generator.
    State restore replays the sequence; logical mapping is scheduling-independent.
    Generic paths use four fields per word; optimized BF16 and large FP32
    tails use eight correlated fields. See README for mapping and Graph usage.
    The input must be contiguous/nonempty and on a supported dav-3510 NPU;
    axis length is a multiple of 16, at least 32 for FP32.
    """
    axis, dst_dtype, bound_bits = _validate_attributes(
        input, axis, round_mode, dst_type, block_size, scale_alg, dst_type_max, max_low_bound
    )
    _validate_input(input, axis)
    prepared = _prepare_launch(input, dst_dtype, axis, scale_alg, bound_bits)
    prepared.launch()
    return prepared.outputs()


@jit
def _quantize_bf16_single_panel(
    self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
):
    parts = count // 256
    batches = (count + 1023) // 1024
    scale16 = reinterpret(self.scale32, dtypes.uint16, shape=(1, self.tile_elements // 16))
    _prepare_eight_field_aos_linear(random, start, batches, seed, block_offset)
    with vf(mode="simd"):
        full = reg.full_mask()
        full16 = reg.full_mask(elem_bits=16)
        full8 = reg.full_mask(elem_bits=8)
        eight16 = reg.update_mask(8, elem_bits=16)[0]
        abs_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
        index16 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
        pack_ids8 = reg.varange(0, dtypes.uint8)
        scale_byte_slot = reg.vbitwise_and(pack_ids8, reg.vdups(1, dtypes.uint8, mask=full8), mask=full8)
        scale_byte_mask = reg.mask_and(
            reg.vlts(pack_ids8, 16, mask=full8),
            reg.vlts(scale_byte_slot, dtypes.uint8(scale_factor // 2), mask=full8),
            exec_mask=full8,
        )
        scale_cursor = reg.vstore_unalign_begin(scales)
        one16 = reg.update_mask(1, elem_bits=16)[0]
        tile_minimum = reg.vdups(65535, dtypes.uint16, mask=one16)
        for first4 in range(0, parts, 4):
            group_min = reg.vdups(65535, dtypes.uint16, mask=full16)
            group_invalid = reg.vdups(0, dtypes.uint16, mask=full16)
            part = first4 + 0
            offset = part * 256
            a, b = reg.vload_deinterleave(x, offset)
            aa = reg.vbitwise_and(reg.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
            ab = reg.vbitwise_and(reg.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
            maxima16 = reg.vreduce_max_datablock(reg.vmax(aa, ab, mask=full16), mask=full16)
            if const_expr(self.scale_alg == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                exponent16 = reg.vshr(maxima16, 7, mask=full16)
                invalid16 = reg.vges(exponent16, 255, mask=full16)
                code16 = reg.vadds(reg.vmaxs(exponent16, scale_shift, mask=full16), -scale_shift & 65535, mask=full16)
                code16 = reg.vdups(255, dtypes.uint16, mask=invalid16, mode="merging", merge=code16)
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16),
                    7,
                    mask=reg.vnes(code16, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                active16 = reg.vgts(maxima16, scale_shift << 7 | 96, mask=full16)
                code16 = reg.vshr(reg.vadds(maxima16, 31 - (scale_shift << 7) & 65535, mask=full16), 7, mask=active16)
                code16 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16, 32640, mask=full16), mode="merging", merge=code16
                )
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16), 7, mask=full16
                )
            else:
                maxima = reg.vshl(reg.vunpack(maxima16, dtypes.uint32), 16, mask=full)
                block_scales, reciprocal = compute_scale(self, maxima, full, False)
                bits16 = reg.vshr(reg.vreinterpret(reciprocal, dtypes.uint32), 16, mask=full)
                r16 = reg.vpack(bits16, dtypes.uint16, part="lower")
                code16 = reg.vpack(block_scales, dtypes.uint16, part="lower")
            group_invalid = reg.vmax(group_invalid, code16, mask=eight16)
            broadcast = reg.vreinterpret(reg.vgather_reg(r16, index16), dtypes.bfloat16)
            av = reg.vmul(a, broadcast, mask=full16)
            bv = reg.vmul(b, broadcast, mask=full16)
            reg.vstore_interleave(x, offset, av, bv)
            aa_norm = reg.vbitwise_and(reg.vreinterpret(av, dtypes.uint16), abs_mask, mask=full16)
            ab_norm = reg.vbitwise_and(reg.vreinterpret(bv, dtypes.uint16), abs_mask, mask=full16)
            panel_min = reg.vmin(aa_norm, ab_norm, mask=full16)
            group_min = reg.vmin(group_min, panel_min, mask=full16)
            reg.vstore(scale16, part * 16, code16, eight16)
            scale_bytes = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(code16, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes, scale_cursor)
            part = first4 + 1
            offset = part * 256
            a, b = reg.vload_deinterleave(x, offset)
            aa = reg.vbitwise_and(reg.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
            ab = reg.vbitwise_and(reg.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
            maxima16 = reg.vreduce_max_datablock(reg.vmax(aa, ab, mask=full16), mask=full16)
            if const_expr(self.scale_alg == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                exponent16 = reg.vshr(maxima16, 7, mask=full16)
                invalid16 = reg.vges(exponent16, 255, mask=full16)
                code16 = reg.vadds(reg.vmaxs(exponent16, scale_shift, mask=full16), -scale_shift & 65535, mask=full16)
                code16 = reg.vdups(255, dtypes.uint16, mask=invalid16, mode="merging", merge=code16)
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16),
                    7,
                    mask=reg.vnes(code16, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                active16 = reg.vgts(maxima16, scale_shift << 7 | 96, mask=full16)
                code16 = reg.vshr(reg.vadds(maxima16, 31 - (scale_shift << 7) & 65535, mask=full16), 7, mask=active16)
                code16 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16, 32640, mask=full16), mode="merging", merge=code16
                )
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16), 7, mask=full16
                )
            else:
                maxima = reg.vshl(reg.vunpack(maxima16, dtypes.uint32), 16, mask=full)
                block_scales, reciprocal = compute_scale(self, maxima, full, False)
                bits16 = reg.vshr(reg.vreinterpret(reciprocal, dtypes.uint32), 16, mask=full)
                r16 = reg.vpack(bits16, dtypes.uint16, part="lower")
                code16 = reg.vpack(block_scales, dtypes.uint16, part="lower")
            group_invalid = reg.vmax(group_invalid, code16, mask=eight16)
            broadcast = reg.vreinterpret(reg.vgather_reg(r16, index16), dtypes.bfloat16)
            av = reg.vmul(a, broadcast, mask=full16)
            bv = reg.vmul(b, broadcast, mask=full16)
            reg.vstore_interleave(x, offset, av, bv)
            aa_norm = reg.vbitwise_and(reg.vreinterpret(av, dtypes.uint16), abs_mask, mask=full16)
            ab_norm = reg.vbitwise_and(reg.vreinterpret(bv, dtypes.uint16), abs_mask, mask=full16)
            panel_min = reg.vmin(aa_norm, ab_norm, mask=full16)
            group_min = reg.vmin(group_min, panel_min, mask=full16)
            reg.vstore(scale16, part * 16, code16, eight16)
            scale_bytes = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(code16, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes, scale_cursor)
            part = first4 + 2
            offset = part * 256
            a, b = reg.vload_deinterleave(x, offset)
            aa = reg.vbitwise_and(reg.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
            ab = reg.vbitwise_and(reg.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
            maxima16 = reg.vreduce_max_datablock(reg.vmax(aa, ab, mask=full16), mask=full16)
            if const_expr(self.scale_alg == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                exponent16 = reg.vshr(maxima16, 7, mask=full16)
                invalid16 = reg.vges(exponent16, 255, mask=full16)
                code16 = reg.vadds(reg.vmaxs(exponent16, scale_shift, mask=full16), -scale_shift & 65535, mask=full16)
                code16 = reg.vdups(255, dtypes.uint16, mask=invalid16, mode="merging", merge=code16)
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16),
                    7,
                    mask=reg.vnes(code16, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                active16 = reg.vgts(maxima16, scale_shift << 7 | 96, mask=full16)
                code16 = reg.vshr(reg.vadds(maxima16, 31 - (scale_shift << 7) & 65535, mask=full16), 7, mask=active16)
                code16 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16, 32640, mask=full16), mode="merging", merge=code16
                )
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16), 7, mask=full16
                )
            else:
                maxima = reg.vshl(reg.vunpack(maxima16, dtypes.uint32), 16, mask=full)
                block_scales, reciprocal = compute_scale(self, maxima, full, False)
                bits16 = reg.vshr(reg.vreinterpret(reciprocal, dtypes.uint32), 16, mask=full)
                r16 = reg.vpack(bits16, dtypes.uint16, part="lower")
                code16 = reg.vpack(block_scales, dtypes.uint16, part="lower")
            group_invalid = reg.vmax(group_invalid, code16, mask=eight16)
            broadcast = reg.vreinterpret(reg.vgather_reg(r16, index16), dtypes.bfloat16)
            av = reg.vmul(a, broadcast, mask=full16)
            bv = reg.vmul(b, broadcast, mask=full16)
            reg.vstore_interleave(x, offset, av, bv)
            aa_norm = reg.vbitwise_and(reg.vreinterpret(av, dtypes.uint16), abs_mask, mask=full16)
            ab_norm = reg.vbitwise_and(reg.vreinterpret(bv, dtypes.uint16), abs_mask, mask=full16)
            panel_min = reg.vmin(aa_norm, ab_norm, mask=full16)
            group_min = reg.vmin(group_min, panel_min, mask=full16)
            reg.vstore(scale16, part * 16, code16, eight16)
            scale_bytes = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(code16, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes, scale_cursor)
            part = first4 + 3
            offset = part * 256
            a, b = reg.vload_deinterleave(x, offset)
            aa = reg.vbitwise_and(reg.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
            ab = reg.vbitwise_and(reg.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
            maxima16 = reg.vreduce_max_datablock(reg.vmax(aa, ab, mask=full16), mask=full16)
            if const_expr(self.scale_alg == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                exponent16 = reg.vshr(maxima16, 7, mask=full16)
                invalid16 = reg.vges(exponent16, 255, mask=full16)
                code16 = reg.vadds(reg.vmaxs(exponent16, scale_shift, mask=full16), -scale_shift & 65535, mask=full16)
                code16 = reg.vdups(255, dtypes.uint16, mask=invalid16, mode="merging", merge=code16)
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16),
                    7,
                    mask=reg.vnes(code16, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                active16 = reg.vgts(maxima16, scale_shift << 7 | 96, mask=full16)
                code16 = reg.vshr(reg.vadds(maxima16, 31 - (scale_shift << 7) & 65535, mask=full16), 7, mask=active16)
                code16 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16, 32640, mask=full16), mode="merging", merge=code16
                )
                r16 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16, mask=full16), 7, mask=full16
                )
            else:
                maxima = reg.vshl(reg.vunpack(maxima16, dtypes.uint32), 16, mask=full)
                block_scales, reciprocal = compute_scale(self, maxima, full, False)
                bits16 = reg.vshr(reg.vreinterpret(reciprocal, dtypes.uint32), 16, mask=full)
                r16 = reg.vpack(bits16, dtypes.uint16, part="lower")
                code16 = reg.vpack(block_scales, dtypes.uint16, part="lower")
            group_invalid = reg.vmax(group_invalid, code16, mask=eight16)
            broadcast = reg.vreinterpret(reg.vgather_reg(r16, index16), dtypes.bfloat16)
            av = reg.vmul(a, broadcast, mask=full16)
            bv = reg.vmul(b, broadcast, mask=full16)
            reg.vstore_interleave(x, offset, av, bv)
            aa_norm = reg.vbitwise_and(reg.vreinterpret(av, dtypes.uint16), abs_mask, mask=full16)
            ab_norm = reg.vbitwise_and(reg.vreinterpret(bv, dtypes.uint16), abs_mask, mask=full16)
            panel_min = reg.vmin(aa_norm, ab_norm, mask=full16)
            group_min = reg.vmin(group_min, panel_min, mask=full16)
            reg.vstore(scale16, part * 16, code16, eight16)
            scale_bytes = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(code16, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes, scale_cursor)
            minimum = reg.vreduce_min(group_min, mask=full16)
            maximum_scale = reg.vreduce_max(group_invalid, mask=eight16)
            minimum = reg.vdups(
                0, dtypes.uint16, mask=reg.veqs(maximum_scale, 255, mask=full16), mode="merging", merge=minimum
            )
            reg.vstore_first(self.norm_flags, first4 * self.norm_flag_stride, minimum)
            tile_minimum = reg.vmin(tile_minimum, minimum, mask=one16)
        reg.vsqueeze_and_storeunalign_finalize(scales, 0, scale_cursor)
        reg.vstore_first(self.norm_flags, self.norm_tile_flag, tile_minimum)
        # Publish normalized x, scales and minima before scalar classification and encoding reloads.
        reg.vmem_bar("vst_vld")
    if dtypes.int64(self.norm_flags[0, self.norm_tile_flag]) >= (128 - self.bias) * 128:
        _emit_nibble_clean_part(self, random, x, y, 0, parts)
    else:
        for first in range(0, parts, 16):
            stop = first + 16
            if stop > parts:
                stop = parts
            flag_min = dtypes.int64(self.norm_flags[0, first * self.norm_flag_stride])
            for cursor in range(first + 4, stop, 4):
                flag = dtypes.int64(self.norm_flags[0, cursor * self.norm_flag_stride])
                if flag < flag_min:
                    flag_min = flag
            if flag_min >= (128 - self.bias) * 128:
                _emit_nibble_clean_part(self, random, x, y, first, stop)
            elif flag_min >= (123 - self.bias - self.mantissa) * 128:
                _emit_nibble_middle_part(self, random, x, y, first, stop)
            else:
                _emit_nibble_general_part(self, random, x, y, first, stop)


@jit
def _emit_nibble_half_q7_part(self, random, x, y, first, stop):
    sign_source = reinterpret(x, dtypes.uint8, shape=(1, self.tile_elements * 2))
    with vf(mode="simd"):
        full16 = reg.full_mask(elem_bits=16)
        full8 = reg.full_mask(elem_bits=8)
        byte_mask16 = reg.vdups(255, dtypes.uint16, mask=full16)
        sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
        ids_byte = reg.varange(0, dtypes.uint8)
        pairs = reg.vshr(ids_byte, 1, mask=full8)
        base = reg.vshl(reg.vshr(pairs, 2, mask=full8), 2, mask=full8)
        byte = reg.vbitwise_and(ids_byte, reg.vdups(1, dtypes.uint8, mask=full8), mask=full8)
        position = reg.vbitwise_and(
            reg.vadd(reg.vadds(pairs, 3, mask=full8), byte, mask=full8),
            reg.vdups(3, dtypes.uint8, mask=full8),
            mask=full8,
        )
        pair_index = reg.vbitwise_or(base, position, mask=full8)
        field_mask = reg.vdups(4095, dtypes.uint16, mask=full16)
        random_bytes = reinterpret(random.buffer, dtypes.uint8, shape=(1, random.capacity_bytes))
        absolute_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
        for part in range(first, stop):
            offset = part * 256
            av, bv = reg.vload_deinterleave(x, offset)
            raw = reg.vload(random_bytes, part * 128)
            window = reg.vreinterpret_lanes(reg.vgather_reg(raw, pair_index), dtypes.uint16)
            if const_expr(self.mantissa == 3):
                ra = reg.vbitwise_and(window, field_mask, mask=full16)
            else:
                ra = reg.vbitwise_and(reg.vshr(window, 1, mask=full16), field_mask, mask=full16)
            reverse = reg.vshr(window, 4, mask=full16)
            ca = _sr_bf16_half_q7(av, ra, absolute_mask, full16)
            cb = _sr_bf16_half_q7(bv, reverse, absolute_mask, full16)
            paired = reg.vbitwise_or(
                reg.vbitwise_and(ca, byte_mask16, mask=full16), reg.vshl(cb, 8, mask=full16), mask=full16
            )
            codes8 = reg.vreinterpret_lanes(paired, dtypes.uint8)
            codes8 = reg.vadds(codes8, -(16 - self.bias << self.mantissa) & 255, mask=full8)
            codes8 = reg.vmins(codes8, self.max_code, mask=full8)
            unused_low, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
            codes8 = reg.vbitwise_or(codes8, reg.vbitwise_and(signbytes, sign_mask8, mask=full8), mask=full8)
            reg.vstore(y, offset, codes8, full8)
        # Complete encoded byte stores before subsequent VFs reuse UB or DMA consumes this output slot.
        reg.vmem_bar("vst_vld")


def _sr_bf16_half_q7(value, random12, absolute_mask, mask):
    absolute = reg.vbitwise_and(reg.vreinterpret(value, dtypes.uint16), absolute_mask, mask=mask)
    half = reg.vcast(
        reg.vreinterpret(absolute, dtypes.bfloat16), dtypes.float16, mask=mask, rounding=reg.RoundingMode.RZ
    )
    mapped = reg.vadd(half, reg.vmaxs(half, 2.0 ** (-6), mask=mask), mask=mask)
    return reg.vshr(
        reg.vadd(reg.vreinterpret(mapped, dtypes.uint16), reg.vshr(random12, 5, mask=mask), mask=mask), 7, mask=mask
    )


def _sr_bf16_half_q7_random7(value, random7, absolute_mask, mask):
    absolute = reg.vbitwise_and(reg.vreinterpret(value, dtypes.uint16), absolute_mask, mask=mask)
    half = reg.vcast(
        reg.vreinterpret(absolute, dtypes.bfloat16), dtypes.float16, mask=mask, rounding=reg.RoundingMode.RZ
    )
    mapped = reg.vadd(half, reg.vmaxs(half, 2.0 ** (-6), mask=mask), mask=mask)
    return reg.vshr(reg.vadd(reg.vreinterpret(mapped, dtypes.uint16), random7, mask=mask), 7, mask=mask)


@jit
def _produce_fp32_random_in_output(scratch, start, batches, seed, block_offset):
    key0, key1 = _split_uint64(seed)
    group_low, group_high = _split_uint64(dtypes.uint64(start) // 32)
    offset_low, offset_high = _split_uint64(block_offset)
    base_low = dtypes.uint32(offset_low // 4 + offset_high % 4 * 1073741824)
    base_high = dtypes.uint32(offset_high // 4)
    linear_group_count = dtypes.int64(batches) * 32
    first_group = dtypes.uint64(start) // 32
    draw = first_group // 16384
    last_draw = (first_group + dtypes.uint64(linear_group_count) - 1) // 16384
    if draw == last_draw:
        draw_low, draw_high = _split_uint64(draw)
        counter_low = dtypes.uint32(base_low) + dtypes.uint32(draw_low)
        counter_high = (
            dtypes.uint32(base_high) + dtypes.uint32(draw_high) + dtypes.uint32(counter_low < dtypes.uint32(base_low))
        )
        first_low, first_high = _split_uint64(dtypes.uint64(counter_low) * dtypes.uint64(PHILOX_M1))
        uniform2 = dtypes.uint32(counter_high) ^ dtypes.uint32(key1)
        second_low, second_high = _split_uint64(dtypes.uint64(uniform2) * dtypes.uint64(PHILOX_M1))
        next_key0 = dtypes.uint32(key0) + dtypes.uint32(PHILOX_W0)
        next_key1 = dtypes.uint32(key1) + dtypes.uint32(PHILOX_W1)
        uniform_second0 = dtypes.uint32(second_high) ^ dtypes.uint32(first_low) ^ dtypes.uint32(next_key0)
        third_low, third_high = _split_uint64(dtypes.uint64(uniform_second0) * dtypes.uint64(PHILOX_M0))
        uniform_initial = dtypes.uint32(first_high) ^ dtypes.uint32(key0)
        uniform_x0 = dtypes.uint32(second_low) ^ dtypes.uint32(key0) + dtypes.uint32(2 * PHILOX_W0 & UINT32_MASK)
        uniform_x2 = dtypes.uint32(third_high) ^ dtypes.uint32(key1) + dtypes.uint32(2 * PHILOX_W1 & UINT32_MASK)
        if linear_group_count >= 128 and linear_group_count % 128 == 0:
            with vf(mode="simd"):
                full = reg.full_mask()
                ids = reg.varange(0, dtypes.uint32)
                m0 = reg.vdups(PHILOX_M0, dtypes.uint32, mask=full)
                m1 = reg.vdups(PHILOX_M1, dtypes.uint32, mask=full)
                uinitial = reg.vdups(uniform_initial, dtypes.uint32, mask=full)
                uk1 = reg.vdups(next_key1, dtypes.uint32, mask=full)
                ux0 = reg.vdups(uniform_x0, dtypes.uint32, mask=full)
                ux2 = reg.vdups(uniform_x2, dtypes.uint32, mask=full)
                third = reg.vdups(third_low, dtypes.uint32, mask=full)
                for base in range(0, linear_group_count, 64):
                    sub0 = reg.vbitwise_and(
                        reg.vadds(ids, dtypes.uint32(group_low) + dtypes.uint32(base) + dtypes.uint32(0), mask=full),
                        reg.vdups(16383, dtypes.uint32, mask=full),
                        mask=full,
                    )
                    first0 = reg.vbitwise_xor(sub0, uinitial, mask=full)
                    varlo0, varhi0 = reg.vmull(first0, m0, dtypes.uint32, mask=full)
                    second0 = reg.vbitwise_xor(varhi0, uk1, mask=full)
                    p1lo0, p1hi0 = reg.vmull(second0, m1, dtypes.uint32, mask=full)
                    s0_0 = reg.vbitwise_xor(p1hi0, ux0, mask=full)
                    s0_1 = p1lo0
                    s0_2 = reg.vbitwise_xor(varlo0, ux2, mask=full)
                    s0_3 = third
                    k0 = reg.vdups(
                        dtypes.uint32(key0) + dtypes.uint32((3 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    k1 = reg.vdups(
                        dtypes.uint32(key1) + dtypes.uint32((3 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
                    lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
                    s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
                    s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
                    s0_1 = lo1_0
                    s0_3 = lo0_0
                    k0 = reg.vdups(
                        dtypes.uint32(key0) + dtypes.uint32((4 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    k1 = reg.vdups(
                        dtypes.uint32(key1) + dtypes.uint32((4 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
                    lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
                    s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
                    s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
                    s0_1 = lo1_0
                    s0_3 = lo0_0
                    k0 = reg.vdups(
                        dtypes.uint32(key0) + dtypes.uint32((5 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    k1 = reg.vdups(
                        dtypes.uint32(key1) + dtypes.uint32((5 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
                    lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
                    s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
                    s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
                    s0_1 = lo1_0
                    s0_3 = lo0_0
                    k0 = reg.vdups(
                        dtypes.uint32(key0) + dtypes.uint32((6 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    k1 = reg.vdups(
                        dtypes.uint32(key1) + dtypes.uint32((6 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
                    lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
                    s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
                    s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
                    s0_1 = lo1_0
                    s0_3 = lo0_0
                    k0 = reg.vdups(
                        dtypes.uint32(key0) + dtypes.uint32((7 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    k1 = reg.vdups(
                        dtypes.uint32(key1) + dtypes.uint32((7 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
                    lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
                    s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
                    s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
                    s0_1 = lo1_0
                    s0_3 = lo0_0
                    k0 = reg.vdups(
                        dtypes.uint32(key0) + dtypes.uint32((8 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    k1 = reg.vdups(
                        dtypes.uint32(key1) + dtypes.uint32((8 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
                    lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
                    s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
                    s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
                    s0_1 = lo1_0
                    s0_3 = lo0_0
                    k0 = reg.vdups(
                        dtypes.uint32(key0) + dtypes.uint32((9 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    k1 = reg.vdups(
                        dtypes.uint32(key1) + dtypes.uint32((9 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
                    lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
                    s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
                    s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
                    s0_1 = lo1_0
                    s0_3 = lo0_0
                    _store_random_words_aos(scratch, (base + 0) * 4, s0_0, s0_1, s0_2, s0_3, full)
                # Publish Philox scratch stores before SR VFs gather random words.
                reg.vmem_bar("vst_vld")
        else:
            with vf(mode="simt", thread=256):
                for group in range(simt.thread_idx()[0], linear_group_count, 256):
                    subsequence = (dtypes.uint32(group_low) + dtypes.uint32(group)) % 16384
                    first_c0 = subsequence ^ dtypes.uint32(uniform_initial)
                    variable_low = dtypes.uint32(first_c0) * dtypes.uint32(PHILOX_M0)
                    variable_high = _simt_umulhi(first_c0, PHILOX_M0)
                    second_c2 = variable_high ^ dtypes.uint32(next_key1)
                    p1lo = dtypes.uint32(second_c2) * dtypes.uint32(PHILOX_M1)
                    p1hi = _simt_umulhi(second_c2, PHILOX_M1)
                    c0 = p1hi ^ dtypes.uint32(uniform_x0)
                    c1 = p1lo
                    c2 = variable_low ^ dtypes.uint32(uniform_x2)
                    c3 = dtypes.uint32(third_low)
                    for step in range(3, 10):
                        p0lo = dtypes.uint32(c0) * dtypes.uint32(PHILOX_M0)
                        p0hi = _simt_umulhi(c0, PHILOX_M0)
                        p1lo = dtypes.uint32(c2) * dtypes.uint32(PHILOX_M1)
                        p1hi = _simt_umulhi(c2, PHILOX_M1)
                        k0 = dtypes.uint32(key0) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W0)
                        k1 = dtypes.uint32(key1) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W1)
                        c0, c1, c2, c3 = (p1hi ^ c1 ^ k0, p1lo, p0hi ^ c3 ^ k1, p0lo)
                    scratch[0, group * 4] = dtypes.int32(c0)
                    scratch[0, group * 4 + 1] = dtypes.int32(c1)
                    scratch[0, group * 4 + 2] = dtypes.int32(c2)
                    scratch[0, group * 4 + 3] = dtypes.int32(c3)
    else:
        with vf(mode="simt", thread=256):
            for group in range(simt.thread_idx()[0], linear_group_count, 256):
                gl = dtypes.uint32(group_low) + dtypes.uint32(group)
                gh = dtypes.uint32(group_high) + dtypes.uint32(gl < dtypes.uint32(group_low))
                draw_low = dtypes.uint32(gl // 16384) + dtypes.uint32(gh % 16384) * dtypes.uint32(262144)
                draw_high = dtypes.uint32(gh // 16384)
                local_counter_low = dtypes.uint32(base_low) + dtypes.uint32(draw_low)
                local_counter_high = (
                    dtypes.uint32(base_high)
                    + dtypes.uint32(draw_high)
                    + dtypes.uint32(local_counter_low < dtypes.uint32(base_low))
                )
                c0 = dtypes.uint32(0)
                c1 = dtypes.uint32(gl % 16384)
                c2 = local_counter_low
                c3 = local_counter_high
                for step in range(10):
                    p0lo = dtypes.uint32(c0) * dtypes.uint32(PHILOX_M0)
                    p0hi = _simt_umulhi(c0, PHILOX_M0)
                    p1lo = dtypes.uint32(c2) * dtypes.uint32(PHILOX_M1)
                    p1hi = _simt_umulhi(c2, PHILOX_M1)
                    k0 = dtypes.uint32(key0) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W0)
                    k1 = dtypes.uint32(key1) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W1)
                    c0, c1, c2, c3 = (p1hi ^ c1 ^ k0, p1lo, p0hi ^ c3 ^ k1, p0lo)
                scratch[0, group * 4] = dtypes.int32(c0)
                scratch[0, group * 4 + 1] = dtypes.int32(c1)
                scratch[0, group * 4 + 2] = dtypes.int32(c2)
                scratch[0, group * 4 + 3] = dtypes.int32(c3)


@jit
def _quantize_fp32_output_alias(self, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset):
    sign_source = reinterpret(x, dtypes.uint8 if self.bf16 else dtypes.uint16, shape=(1, self.tile_elements * 2))
    parts = count // 256
    batches = (count + 1023) // 1024
    scratch = reinterpret(
        y, dtypes.int32, shape=(1, self.tile_elements // 8 + 32), offset=self.tile_elements // 2 - 128
    )
    _produce_fp32_random_in_output(scratch, start, batches, seed, block_offset)
    with vf(mode="simd"):
        full = reg.full_mask()
        full16 = reg.full_mask(elem_bits=16)
        full8 = reg.full_mask(elem_bits=8)
        ids = reg.varange(0, dtypes.uint32)
        scale_index = reg.vshr(ids, 3, mask=full)
        sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
        pack_ids8 = reg.varange(0, dtypes.uint8)
        pack_group8 = reg.vbitwise_and(pack_ids8, reg.vdups(3, dtypes.uint8, mask=full8), mask=full8)
        pack_index8 = reg.vbitwise_or(
            reg.vshl(pack_group8, 6, mask=full8), reg.vshr(pack_ids8, 2, mask=full8), mask=full8
        )
        random_mask32 = reg.vdups(65535, dtypes.uint32, mask=full)
        scale_byte_slot = reg.vbitwise_and(pack_ids8, reg.vdups(3, dtypes.uint8, mask=full8), mask=full8)
        scale_byte_mask = reg.mask_and(
            reg.vlts(pack_ids8, 32, mask=full8),
            reg.vlts(scale_byte_slot, dtypes.uint8(scale_factor // 2), mask=full8),
            exec_mask=full8,
        )
        scale_cursor = reg.vstore_unalign_begin(scales)
        pair_ids = reg.varange(0, dtypes.uint8)
        pair_base = reg.vshl(reg.vshr(pair_ids, 3, mask=full8), 2, mask=full8)
        pair_position = reg.vbitwise_and(
            reg.vshr(reg.vadds(pair_ids, 1, mask=full8), 1, mask=full8),
            reg.vdups(3, dtypes.uint8, mask=full8),
            mask=full8,
        )
        pair_index02 = reg.vbitwise_or(pair_base, pair_position, mask=full8)
        pair_index13 = reg.vgather_reg(
            pair_index02, reg.vbitwise_xor(pair_ids, reg.vdups(1, dtypes.uint8, mask=full8), mask=full8)
        )
        for pair in range(parts // 2):
            part_p0 = pair * 2
            part_p1 = part_p0 + 1
            offset_p0 = part_p0 * 256
            if const_expr(self.bf16):
                a_p0, b_p0 = reg.vload_deinterleave(x, offset_p0)
                abs_mask_p0 = reg.vdups(32767, dtypes.uint16, mask=full16)
                aa_p0 = reg.vbitwise_and(reg.vreinterpret(a_p0, dtypes.uint16), abs_mask_p0, mask=full16)
                ab_p0 = reg.vbitwise_and(reg.vreinterpret(b_p0, dtypes.uint16), abs_mask_p0, mask=full16)
                maxima16_p0 = reg.vreduce_max_datablock(reg.vmax(aa_p0, ab_p0, mask=full16), mask=full16)
                maxima_p0 = reg.vshl(reg.vunpack(maxima16_p0, dtypes.uint32), 16, mask=full)
                unused_signbytes_p0, signbytes_p0 = reg.vload_deinterleave(sign_source, offset_p0 * 2)
                signs_p0 = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes_p0, sign_mask8, mask=full8), dtypes.uint32)
            else:
                a_p0, b_p0 = reg.vload_deinterleave(x, offset_p0)
                c_p0, d_p0 = reg.vload_deinterleave(x, offset_p0 + 128)
                v0_p0, v2_p0 = reg.vdeinterleave(a_p0, c_p0)
                v1_p0, v3_p0 = reg.vdeinterleave(b_p0, d_p0)
                a0_p0 = reg.vreinterpret(reg.vabs(v0_p0, mask=full), dtypes.uint32)
                a1_p0 = reg.vreinterpret(reg.vabs(v1_p0, mask=full), dtypes.uint32)
                a2_p0 = reg.vreinterpret(reg.vabs(v2_p0, mask=full), dtypes.uint32)
                a3_p0 = reg.vreinterpret(reg.vabs(v3_p0, mask=full), dtypes.uint32)
                pair0_p0 = reg.vmax(a0_p0, a1_p0, mask=full)
                pair1_p0 = reg.vmax(a2_p0, a3_p0, mask=full)
                maxima_p0 = reg.vreduce_max_datablock(reg.vmax(pair0_p0, pair1_p0, mask=full), mask=full)
                unused_sign01_p0, sign01_p0 = reg.vload_deinterleave(sign_source, offset_p0 * 2)
                unused_sign23_p0, sign23_p0 = reg.vload_deinterleave(sign_source, offset_p0 * 2 + 256)
                unused_signbytes_p0, signbytes_p0 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(sign01_p0, dtypes.uint8), reg.vreinterpret_lanes(sign23_p0, dtypes.uint8)
                )
                signs_p0 = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes_p0, sign_mask8, mask=full8), dtypes.uint32)
            block_scales_p0, reciprocal_p0 = compute_scale(self, maxima_p0, full, False)
            if const_expr(self.bf16):
                bits16_p0 = reg.vshr(reg.vreinterpret(reciprocal_p0, dtypes.uint32), 16, mask=full)
                r16_p0 = reg.vpack(bits16_p0, dtypes.uint16, part="lower")
                index16_p0 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
                broadcast_p0 = reg.vreinterpret(reg.vgather_reg(r16_p0, index16_p0), dtypes.bfloat16)
                a_p0 = reg.vmul(reg.vreinterpret(aa_p0, dtypes.bfloat16), broadcast_p0, mask=full16)
                b_p0 = reg.vmul(reg.vreinterpret(ab_p0, dtypes.bfloat16), broadcast_p0, mask=full16)
                v0_p0 = reg.vcast(a_p0, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v2_p0 = reg.vcast(a_p0, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                v1_p0 = reg.vcast(b_p0, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v3_p0 = reg.vcast(b_p0, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                invalid_reciprocal_p0 = reg.vreinterpret_lanes(broadcast_p0, dtypes.uint32)
            else:
                reciprocal_p0 = reg.vgather_reg(reciprocal_p0, scale_index)
                v0_p0 = reg.vmul(reg.vreinterpret(a0_p0, dtypes.float32), reciprocal_p0, mask=full)
                v1_p0 = reg.vmul(reg.vreinterpret(a1_p0, dtypes.float32), reciprocal_p0, mask=full)
                v2_p0 = reg.vmul(reg.vreinterpret(a2_p0, dtypes.float32), reciprocal_p0, mask=full)
                v3_p0 = reg.vmul(reg.vreinterpret(a3_p0, dtypes.float32), reciprocal_p0, mask=full)
                invalid_reciprocal_p0 = reciprocal_p0
            offset_p1 = part_p1 * 256
            if const_expr(self.bf16):
                a_p1, b_p1 = reg.vload_deinterleave(x, offset_p1)
                abs_mask_p1 = reg.vdups(32767, dtypes.uint16, mask=full16)
                aa_p1 = reg.vbitwise_and(reg.vreinterpret(a_p1, dtypes.uint16), abs_mask_p1, mask=full16)
                ab_p1 = reg.vbitwise_and(reg.vreinterpret(b_p1, dtypes.uint16), abs_mask_p1, mask=full16)
                maxima16_p1 = reg.vreduce_max_datablock(reg.vmax(aa_p1, ab_p1, mask=full16), mask=full16)
                maxima_p1 = reg.vshl(reg.vunpack(maxima16_p1, dtypes.uint32), 16, mask=full)
                unused_signbytes_p1, signbytes_p1 = reg.vload_deinterleave(sign_source, offset_p1 * 2)
                signs_p1 = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes_p1, sign_mask8, mask=full8), dtypes.uint32)
            else:
                a_p1, b_p1 = reg.vload_deinterleave(x, offset_p1)
                c_p1, d_p1 = reg.vload_deinterleave(x, offset_p1 + 128)
                v0_p1, v2_p1 = reg.vdeinterleave(a_p1, c_p1)
                v1_p1, v3_p1 = reg.vdeinterleave(b_p1, d_p1)
                a0_p1 = reg.vreinterpret(reg.vabs(v0_p1, mask=full), dtypes.uint32)
                a1_p1 = reg.vreinterpret(reg.vabs(v1_p1, mask=full), dtypes.uint32)
                a2_p1 = reg.vreinterpret(reg.vabs(v2_p1, mask=full), dtypes.uint32)
                a3_p1 = reg.vreinterpret(reg.vabs(v3_p1, mask=full), dtypes.uint32)
                pair0_p1 = reg.vmax(a0_p1, a1_p1, mask=full)
                pair1_p1 = reg.vmax(a2_p1, a3_p1, mask=full)
                maxima_p1 = reg.vreduce_max_datablock(reg.vmax(pair0_p1, pair1_p1, mask=full), mask=full)
                unused_sign01_p1, sign01_p1 = reg.vload_deinterleave(sign_source, offset_p1 * 2)
                unused_sign23_p1, sign23_p1 = reg.vload_deinterleave(sign_source, offset_p1 * 2 + 256)
                unused_signbytes_p1, signbytes_p1 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(sign01_p1, dtypes.uint8), reg.vreinterpret_lanes(sign23_p1, dtypes.uint8)
                )
                signs_p1 = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes_p1, sign_mask8, mask=full8), dtypes.uint32)
            block_scales_p1, reciprocal_p1 = compute_scale(self, maxima_p1, full, False)
            if const_expr(self.bf16):
                bits16_p1 = reg.vshr(reg.vreinterpret(reciprocal_p1, dtypes.uint32), 16, mask=full)
                r16_p1 = reg.vpack(bits16_p1, dtypes.uint16, part="lower")
                index16_p1 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
                broadcast_p1 = reg.vreinterpret(reg.vgather_reg(r16_p1, index16_p1), dtypes.bfloat16)
                a_p1 = reg.vmul(reg.vreinterpret(aa_p1, dtypes.bfloat16), broadcast_p1, mask=full16)
                b_p1 = reg.vmul(reg.vreinterpret(ab_p1, dtypes.bfloat16), broadcast_p1, mask=full16)
                v0_p1 = reg.vcast(a_p1, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v2_p1 = reg.vcast(a_p1, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                v1_p1 = reg.vcast(b_p1, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v3_p1 = reg.vcast(b_p1, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                invalid_reciprocal_p1 = reg.vreinterpret_lanes(broadcast_p1, dtypes.uint32)
            else:
                reciprocal_p1 = reg.vgather_reg(reciprocal_p1, scale_index)
                v0_p1 = reg.vmul(reg.vreinterpret(a0_p1, dtypes.float32), reciprocal_p1, mask=full)
                v1_p1 = reg.vmul(reg.vreinterpret(a1_p1, dtypes.float32), reciprocal_p1, mask=full)
                v2_p1 = reg.vmul(reg.vreinterpret(a2_p1, dtypes.float32), reciprocal_p1, mask=full)
                v3_p1 = reg.vmul(reg.vreinterpret(a3_p1, dtypes.float32), reciprocal_p1, mask=full)
                invalid_reciprocal_p1 = reciprocal_p1
            raw_p0 = reg.vreinterpret(reg.vload(scratch, part_p0 * 32), dtypes.uint32)
            raw_bytes_p0 = reg.vreinterpret_lanes(raw_p0, dtypes.uint8)
            word02_p0 = reg.vreinterpret_lanes(reg.vgather_reg(raw_bytes_p0, pair_index02), dtypes.uint32)
            word13_p0 = reg.vreinterpret_lanes(reg.vgather_reg(raw_bytes_p0, pair_index13), dtypes.uint32)
            r0_p0 = reg.vbitwise_and(word02_p0, random_mask32, mask=full)
            r1_p0 = reg.vbitwise_and(word13_p0, random_mask32, mask=full)
            r2_p0 = reg.vshr(word02_p0, 16, mask=full)
            r3_p0 = reg.vshr(word13_p0, 16, mask=full)
            c0_p0 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v0_p0, dtypes.uint32), r0_p0, self.mantissa, self.bias, full
            )
            c1_p0 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v1_p0, dtypes.uint32), r1_p0, self.mantissa, self.bias, full
            )
            c2_p0 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v2_p0, dtypes.uint32), r2_p0, self.mantissa, self.bias, full
            )
            c3_p0 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v3_p0, dtypes.uint32), r3_p0, self.mantissa, self.bias, full
            )
            unused_code01_p0, code01_p0 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c0_p0, dtypes.uint16), reg.vreinterpret_lanes(c1_p0, dtypes.uint16)
            )
            unused_code23_p0, code23_p0 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c2_p0, dtypes.uint16), reg.vreinterpret_lanes(c3_p0, dtypes.uint16)
            )
            code_bytes_p0, unused_code_bytes_p0 = reg.vdeinterleave(
                reg.vreinterpret_lanes(code01_p0, dtypes.uint8), reg.vreinterpret_lanes(code23_p0, dtypes.uint8)
            )
            packed_p0 = reg.vreinterpret_lanes(reg.vgather_reg(code_bytes_p0, pack_index8), dtypes.uint32)
            magnitude8_p0 = reg.vadds(
                reg.vreinterpret_lanes(packed_p0, dtypes.uint8), -(128 - self.bias << self.mantissa) & 255, mask=full8
            )
            saturated_p0 = reg.vmins(magnitude8_p0, self.max_code, mask=reg.vnes(magnitude8_p0, 255, mask=full8))
            packed_p0 = reg.vbitwise_or(reg.vreinterpret_lanes(saturated_p0, dtypes.uint32), signs_p0, mask=full)
            invalid_p0 = reg.veqs(
                reg.vreinterpret(invalid_reciprocal_p0, dtypes.uint32), self.invalid_packed_reciprocal, mask=full
            )
            packed_p0 = reg.vdups(2139062143, dtypes.uint32, mask=invalid_p0, mode="merging", merge=packed_p0)
            reg.vstore(y, offset_p0, reg.vreinterpret_lanes(packed_p0, dtypes.uint8), full8)
            scale_bytes_p0 = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(block_scales_p0, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes_p0, scale_cursor)
            raw_p1 = reg.vreinterpret(reg.vload(scratch, part_p1 * 32), dtypes.uint32)
            raw_bytes_p1 = reg.vreinterpret_lanes(raw_p1, dtypes.uint8)
            word02_p1 = reg.vreinterpret_lanes(reg.vgather_reg(raw_bytes_p1, pair_index02), dtypes.uint32)
            word13_p1 = reg.vreinterpret_lanes(reg.vgather_reg(raw_bytes_p1, pair_index13), dtypes.uint32)
            r0_p1 = reg.vbitwise_and(word02_p1, random_mask32, mask=full)
            r1_p1 = reg.vbitwise_and(word13_p1, random_mask32, mask=full)
            r2_p1 = reg.vshr(word02_p1, 16, mask=full)
            r3_p1 = reg.vshr(word13_p1, 16, mask=full)
            c0_p1 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v0_p1, dtypes.uint32), r0_p1, self.mantissa, self.bias, full
            )
            c1_p1 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v1_p1, dtypes.uint32), r1_p1, self.mantissa, self.bias, full
            )
            c2_p1 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v2_p1, dtypes.uint32), r2_p1, self.mantissa, self.bias, full
            )
            c3_p1 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v3_p1, dtypes.uint32), r3_p1, self.mantissa, self.bias, full
            )
            unused_code01_p1, code01_p1 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c0_p1, dtypes.uint16), reg.vreinterpret_lanes(c1_p1, dtypes.uint16)
            )
            unused_code23_p1, code23_p1 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c2_p1, dtypes.uint16), reg.vreinterpret_lanes(c3_p1, dtypes.uint16)
            )
            code_bytes_p1, unused_code_bytes_p1 = reg.vdeinterleave(
                reg.vreinterpret_lanes(code01_p1, dtypes.uint8), reg.vreinterpret_lanes(code23_p1, dtypes.uint8)
            )
            packed_p1 = reg.vreinterpret_lanes(reg.vgather_reg(code_bytes_p1, pack_index8), dtypes.uint32)
            magnitude8_p1 = reg.vadds(
                reg.vreinterpret_lanes(packed_p1, dtypes.uint8), -(128 - self.bias << self.mantissa) & 255, mask=full8
            )
            saturated_p1 = reg.vmins(magnitude8_p1, self.max_code, mask=reg.vnes(magnitude8_p1, 255, mask=full8))
            packed_p1 = reg.vbitwise_or(reg.vreinterpret_lanes(saturated_p1, dtypes.uint32), signs_p1, mask=full)
            invalid_p1 = reg.veqs(
                reg.vreinterpret(invalid_reciprocal_p1, dtypes.uint32), self.invalid_packed_reciprocal, mask=full
            )
            packed_p1 = reg.vdups(2139062143, dtypes.uint32, mask=invalid_p1, mode="merging", merge=packed_p1)
            reg.vstore(y, offset_p1, reg.vreinterpret_lanes(packed_p1, dtypes.uint8), full8)
            scale_bytes_p1 = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(block_scales_p1, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes_p1, scale_cursor)
        for part in range(parts // 2 * 2, parts):
            offset = part * 256
            if const_expr(self.bf16):
                a, b = reg.vload_deinterleave(x, offset)
                abs_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
                aa = reg.vbitwise_and(reg.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
                ab = reg.vbitwise_and(reg.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
                maxima16 = reg.vreduce_max_datablock(reg.vmax(aa, ab, mask=full16), mask=full16)
                maxima = reg.vshl(reg.vunpack(maxima16, dtypes.uint32), 16, mask=full)
                unused_signbytes, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
                signs = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes, sign_mask8, mask=full8), dtypes.uint32)
            else:
                a, b = reg.vload_deinterleave(x, offset)
                c, d = reg.vload_deinterleave(x, offset + 128)
                v0, v2 = reg.vdeinterleave(a, c)
                v1, v3 = reg.vdeinterleave(b, d)
                a0 = reg.vreinterpret(reg.vabs(v0, mask=full), dtypes.uint32)
                a1 = reg.vreinterpret(reg.vabs(v1, mask=full), dtypes.uint32)
                a2 = reg.vreinterpret(reg.vabs(v2, mask=full), dtypes.uint32)
                a3 = reg.vreinterpret(reg.vabs(v3, mask=full), dtypes.uint32)
                pair0 = reg.vmax(a0, a1, mask=full)
                pair1 = reg.vmax(a2, a3, mask=full)
                maxima = reg.vreduce_max_datablock(reg.vmax(pair0, pair1, mask=full), mask=full)
                unused_sign01, sign01 = reg.vload_deinterleave(sign_source, offset * 2)
                unused_sign23, sign23 = reg.vload_deinterleave(sign_source, offset * 2 + 256)
                unused_signbytes, signbytes = reg.vdeinterleave(
                    reg.vreinterpret_lanes(sign01, dtypes.uint8), reg.vreinterpret_lanes(sign23, dtypes.uint8)
                )
                signs = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes, sign_mask8, mask=full8), dtypes.uint32)
            block_scales, reciprocal = compute_scale(self, maxima, full, False)
            if const_expr(self.bf16):
                bits16 = reg.vshr(reg.vreinterpret(reciprocal, dtypes.uint32), 16, mask=full)
                r16 = reg.vpack(bits16, dtypes.uint16, part="lower")
                index16 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
                broadcast = reg.vreinterpret(reg.vgather_reg(r16, index16), dtypes.bfloat16)
                a = reg.vmul(reg.vreinterpret(aa, dtypes.bfloat16), broadcast, mask=full16)
                b = reg.vmul(reg.vreinterpret(ab, dtypes.bfloat16), broadcast, mask=full16)
                v0 = reg.vcast(a, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v2 = reg.vcast(a, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                v1 = reg.vcast(b, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v3 = reg.vcast(b, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                invalid_reciprocal = reg.vreinterpret_lanes(broadcast, dtypes.uint32)
            else:
                reciprocal = reg.vgather_reg(reciprocal, scale_index)
                v0 = reg.vmul(reg.vreinterpret(a0, dtypes.float32), reciprocal, mask=full)
                v1 = reg.vmul(reg.vreinterpret(a1, dtypes.float32), reciprocal, mask=full)
                v2 = reg.vmul(reg.vreinterpret(a2, dtypes.float32), reciprocal, mask=full)
                v3 = reg.vmul(reg.vreinterpret(a3, dtypes.float32), reciprocal, mask=full)
                invalid_reciprocal = reciprocal
            raw = reg.vreinterpret(reg.vload(scratch, part * 32), dtypes.uint32)
            raw_bytes = reg.vreinterpret_lanes(raw, dtypes.uint8)
            word02 = reg.vreinterpret_lanes(reg.vgather_reg(raw_bytes, pair_index02), dtypes.uint32)
            word13 = reg.vreinterpret_lanes(reg.vgather_reg(raw_bytes, pair_index13), dtypes.uint32)
            r0 = reg.vbitwise_and(word02, random_mask32, mask=full)
            r1 = reg.vbitwise_and(word13, random_mask32, mask=full)
            r2 = reg.vshr(word02, 16, mask=full)
            r3 = reg.vshr(word13, 16, mask=full)
            c0 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v0, dtypes.uint32), r0, self.mantissa, self.bias, full)
            c1 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v1, dtypes.uint32), r1, self.mantissa, self.bias, full)
            c2 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v2, dtypes.uint32), r2, self.mantissa, self.bias, full)
            c3 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v3, dtypes.uint32), r3, self.mantissa, self.bias, full)
            unused_code01, code01 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c0, dtypes.uint16), reg.vreinterpret_lanes(c1, dtypes.uint16)
            )
            unused_code23, code23 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c2, dtypes.uint16), reg.vreinterpret_lanes(c3, dtypes.uint16)
            )
            code_bytes, unused_code_bytes = reg.vdeinterleave(
                reg.vreinterpret_lanes(code01, dtypes.uint8), reg.vreinterpret_lanes(code23, dtypes.uint8)
            )
            packed = reg.vreinterpret_lanes(reg.vgather_reg(code_bytes, pack_index8), dtypes.uint32)
            magnitude8 = reg.vadds(
                reg.vreinterpret_lanes(packed, dtypes.uint8), -(128 - self.bias << self.mantissa) & 255, mask=full8
            )
            saturated = reg.vmins(magnitude8, self.max_code, mask=reg.vnes(magnitude8, 255, mask=full8))
            packed = reg.vbitwise_or(reg.vreinterpret_lanes(saturated, dtypes.uint32), signs, mask=full)
            invalid = reg.veqs(
                reg.vreinterpret(invalid_reciprocal, dtypes.uint32), self.invalid_packed_reciprocal, mask=full
            )
            packed = reg.vdups(2139062143, dtypes.uint32, mask=invalid, mode="merging", merge=packed)
            reg.vstore(y, offset, reg.vreinterpret_lanes(packed, dtypes.uint8), full8)
            scale_bytes = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(block_scales, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes, scale_cursor)
        reg.vsqueeze_and_storeunalign_finalize(scales, 0, scale_cursor)
        # Finish encoded output/scale writes before DMA or the next tile reuses UB.
        reg.vmem_bar("vst_vld")


@jit
def _quantize_fp32_output_alias_batched(
    self, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
):
    sign_source = reinterpret(x, dtypes.uint8 if self.bf16 else dtypes.uint16, shape=(1, self.tile_elements * 2))
    parts = count // 256
    batches = (count + 1023) // 1024
    scratch = reinterpret(
        y, dtypes.int32, shape=(1, self.tile_elements // 8 + 32), offset=self.tile_elements // 2 - 128
    )
    _produce_fp32_random_in_output(scratch, start, batches, seed, block_offset)
    with vf(mode="simd"):
        full = reg.full_mask()
        full16 = reg.full_mask(elem_bits=16)
        full8 = reg.full_mask(elem_bits=8)
        ids = reg.varange(0, dtypes.uint32)
        scale_index = reg.vshr(ids, 3, mask=full)
        sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
        pack_ids8 = reg.varange(0, dtypes.uint8)
        pack_group8 = reg.vbitwise_and(pack_ids8, reg.vdups(3, dtypes.uint8, mask=full8), mask=full8)
        pack_index8 = reg.vbitwise_or(
            reg.vshl(pack_group8, 6, mask=full8), reg.vshr(pack_ids8, 2, mask=full8), mask=full8
        )
        random_mask32 = reg.vdups(65535, dtypes.uint32, mask=full)
        scale_mask32 = reg.update_mask(8, elem_bits=32)[0]
        pair_ids = reg.varange(0, dtypes.uint8)
        pair_base = reg.vshl(reg.vshr(pair_ids, 3, mask=full8), 2, mask=full8)
        pair_position = reg.vbitwise_and(
            reg.vshr(reg.vadds(pair_ids, 1, mask=full8), 1, mask=full8),
            reg.vdups(3, dtypes.uint8, mask=full8),
            mask=full8,
        )
        pair_index02 = reg.vbitwise_or(pair_base, pair_position, mask=full8)
        pair_index13 = reg.vgather_reg(
            pair_index02, reg.vbitwise_xor(pair_ids, reg.vdups(1, dtypes.uint8, mask=full8), mask=full8)
        )
        for pair in range(parts // 2):
            part_p0 = pair * 2
            part_p1 = part_p0 + 1
            offset_p0 = part_p0 * 256
            if const_expr(self.bf16):
                a_p0, b_p0 = reg.vload_deinterleave(x, offset_p0)
                abs_mask_p0 = reg.vdups(32767, dtypes.uint16, mask=full16)
                aa_p0 = reg.vbitwise_and(reg.vreinterpret(a_p0, dtypes.uint16), abs_mask_p0, mask=full16)
                ab_p0 = reg.vbitwise_and(reg.vreinterpret(b_p0, dtypes.uint16), abs_mask_p0, mask=full16)
                maxima16_p0 = reg.vreduce_max_datablock(reg.vmax(aa_p0, ab_p0, mask=full16), mask=full16)
                maxima_p0 = reg.vshl(reg.vunpack(maxima16_p0, dtypes.uint32), 16, mask=full)
                unused_signbytes_p0, signbytes_p0 = reg.vload_deinterleave(sign_source, offset_p0 * 2)
                signs_p0 = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes_p0, sign_mask8, mask=full8), dtypes.uint32)
            else:
                a_p0, b_p0 = reg.vload_deinterleave(x, offset_p0)
                c_p0, d_p0 = reg.vload_deinterleave(x, offset_p0 + 128)
                v0_p0, v2_p0 = reg.vdeinterleave(a_p0, c_p0)
                v1_p0, v3_p0 = reg.vdeinterleave(b_p0, d_p0)
                a0_p0 = reg.vreinterpret(reg.vabs(v0_p0, mask=full), dtypes.uint32)
                a1_p0 = reg.vreinterpret(reg.vabs(v1_p0, mask=full), dtypes.uint32)
                a2_p0 = reg.vreinterpret(reg.vabs(v2_p0, mask=full), dtypes.uint32)
                a3_p0 = reg.vreinterpret(reg.vabs(v3_p0, mask=full), dtypes.uint32)
                pair0_p0 = reg.vmax(a0_p0, a1_p0, mask=full)
                pair1_p0 = reg.vmax(a2_p0, a3_p0, mask=full)
                maxima_p0 = reg.vreduce_max_datablock(reg.vmax(pair0_p0, pair1_p0, mask=full), mask=full)
                unused_sign01_p0, sign01_p0 = reg.vload_deinterleave(sign_source, offset_p0 * 2)
                unused_sign23_p0, sign23_p0 = reg.vload_deinterleave(sign_source, offset_p0 * 2 + 256)
                unused_signbytes_p0, signbytes_p0 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(sign01_p0, dtypes.uint8), reg.vreinterpret_lanes(sign23_p0, dtypes.uint8)
                )
                signs_p0 = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes_p0, sign_mask8, mask=full8), dtypes.uint32)
            block_scales_p0, reciprocal_p0 = compute_scale(self, maxima_p0, full, False)
            if const_expr(self.bf16):
                bits16_p0 = reg.vshr(reg.vreinterpret(reciprocal_p0, dtypes.uint32), 16, mask=full)
                r16_p0 = reg.vpack(bits16_p0, dtypes.uint16, part="lower")
                index16_p0 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
                broadcast_p0 = reg.vreinterpret(reg.vgather_reg(r16_p0, index16_p0), dtypes.bfloat16)
                a_p0 = reg.vmul(reg.vreinterpret(aa_p0, dtypes.bfloat16), broadcast_p0, mask=full16)
                b_p0 = reg.vmul(reg.vreinterpret(ab_p0, dtypes.bfloat16), broadcast_p0, mask=full16)
                v0_p0 = reg.vcast(a_p0, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v2_p0 = reg.vcast(a_p0, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                v1_p0 = reg.vcast(b_p0, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v3_p0 = reg.vcast(b_p0, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                invalid_reciprocal_p0 = reg.vreinterpret_lanes(broadcast_p0, dtypes.uint32)
            else:
                reciprocal_p0 = reg.vgather_reg(reciprocal_p0, scale_index)
                v0_p0 = reg.vmul(reg.vreinterpret(a0_p0, dtypes.float32), reciprocal_p0, mask=full)
                v1_p0 = reg.vmul(reg.vreinterpret(a1_p0, dtypes.float32), reciprocal_p0, mask=full)
                v2_p0 = reg.vmul(reg.vreinterpret(a2_p0, dtypes.float32), reciprocal_p0, mask=full)
                v3_p0 = reg.vmul(reg.vreinterpret(a3_p0, dtypes.float32), reciprocal_p0, mask=full)
                invalid_reciprocal_p0 = reciprocal_p0
            offset_p1 = part_p1 * 256
            if const_expr(self.bf16):
                a_p1, b_p1 = reg.vload_deinterleave(x, offset_p1)
                abs_mask_p1 = reg.vdups(32767, dtypes.uint16, mask=full16)
                aa_p1 = reg.vbitwise_and(reg.vreinterpret(a_p1, dtypes.uint16), abs_mask_p1, mask=full16)
                ab_p1 = reg.vbitwise_and(reg.vreinterpret(b_p1, dtypes.uint16), abs_mask_p1, mask=full16)
                maxima16_p1 = reg.vreduce_max_datablock(reg.vmax(aa_p1, ab_p1, mask=full16), mask=full16)
                maxima_p1 = reg.vshl(reg.vunpack(maxima16_p1, dtypes.uint32), 16, mask=full)
                unused_signbytes_p1, signbytes_p1 = reg.vload_deinterleave(sign_source, offset_p1 * 2)
                signs_p1 = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes_p1, sign_mask8, mask=full8), dtypes.uint32)
            else:
                a_p1, b_p1 = reg.vload_deinterleave(x, offset_p1)
                c_p1, d_p1 = reg.vload_deinterleave(x, offset_p1 + 128)
                v0_p1, v2_p1 = reg.vdeinterleave(a_p1, c_p1)
                v1_p1, v3_p1 = reg.vdeinterleave(b_p1, d_p1)
                a0_p1 = reg.vreinterpret(reg.vabs(v0_p1, mask=full), dtypes.uint32)
                a1_p1 = reg.vreinterpret(reg.vabs(v1_p1, mask=full), dtypes.uint32)
                a2_p1 = reg.vreinterpret(reg.vabs(v2_p1, mask=full), dtypes.uint32)
                a3_p1 = reg.vreinterpret(reg.vabs(v3_p1, mask=full), dtypes.uint32)
                pair0_p1 = reg.vmax(a0_p1, a1_p1, mask=full)
                pair1_p1 = reg.vmax(a2_p1, a3_p1, mask=full)
                maxima_p1 = reg.vreduce_max_datablock(reg.vmax(pair0_p1, pair1_p1, mask=full), mask=full)
                unused_sign01_p1, sign01_p1 = reg.vload_deinterleave(sign_source, offset_p1 * 2)
                unused_sign23_p1, sign23_p1 = reg.vload_deinterleave(sign_source, offset_p1 * 2 + 256)
                unused_signbytes_p1, signbytes_p1 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(sign01_p1, dtypes.uint8), reg.vreinterpret_lanes(sign23_p1, dtypes.uint8)
                )
                signs_p1 = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes_p1, sign_mask8, mask=full8), dtypes.uint32)
            raw_p0 = reg.vreinterpret(reg.vload(scratch, part_p0 * 32), dtypes.uint32)
            raw_bytes_p0 = reg.vreinterpret_lanes(raw_p0, dtypes.uint8)
            raw_p1 = reg.vreinterpret(reg.vload(scratch, part_p1 * 32), dtypes.uint32)
            raw_bytes_p1 = reg.vreinterpret_lanes(raw_p1, dtypes.uint8)
            block_scales_p1, reciprocal_p1 = compute_scale(self, maxima_p1, full, False)
            if const_expr(self.bf16):
                bits16_p1 = reg.vshr(reg.vreinterpret(reciprocal_p1, dtypes.uint32), 16, mask=full)
                r16_p1 = reg.vpack(bits16_p1, dtypes.uint16, part="lower")
                index16_p1 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
                broadcast_p1 = reg.vreinterpret(reg.vgather_reg(r16_p1, index16_p1), dtypes.bfloat16)
                a_p1 = reg.vmul(reg.vreinterpret(aa_p1, dtypes.bfloat16), broadcast_p1, mask=full16)
                b_p1 = reg.vmul(reg.vreinterpret(ab_p1, dtypes.bfloat16), broadcast_p1, mask=full16)
                v0_p1 = reg.vcast(a_p1, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v2_p1 = reg.vcast(a_p1, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                v1_p1 = reg.vcast(b_p1, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v3_p1 = reg.vcast(b_p1, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                invalid_reciprocal_p1 = reg.vreinterpret_lanes(broadcast_p1, dtypes.uint32)
            else:
                reciprocal_p1 = reg.vgather_reg(reciprocal_p1, scale_index)
                v0_p1 = reg.vmul(reg.vreinterpret(a0_p1, dtypes.float32), reciprocal_p1, mask=full)
                v1_p1 = reg.vmul(reg.vreinterpret(a1_p1, dtypes.float32), reciprocal_p1, mask=full)
                v2_p1 = reg.vmul(reg.vreinterpret(a2_p1, dtypes.float32), reciprocal_p1, mask=full)
                v3_p1 = reg.vmul(reg.vreinterpret(a3_p1, dtypes.float32), reciprocal_p1, mask=full)
                invalid_reciprocal_p1 = reciprocal_p1
            word02_p0 = reg.vreinterpret_lanes(reg.vgather_reg(raw_bytes_p0, pair_index02), dtypes.uint32)
            word13_p0 = reg.vreinterpret_lanes(reg.vgather_reg(raw_bytes_p0, pair_index13), dtypes.uint32)
            r0_p0 = reg.vbitwise_and(word02_p0, random_mask32, mask=full)
            r1_p0 = reg.vbitwise_and(word13_p0, random_mask32, mask=full)
            r2_p0 = reg.vshr(word02_p0, 16, mask=full)
            r3_p0 = reg.vshr(word13_p0, 16, mask=full)
            c0_p0 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v0_p0, dtypes.uint32), r0_p0, self.mantissa, self.bias, full
            )
            c1_p0 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v1_p0, dtypes.uint32), r1_p0, self.mantissa, self.bias, full
            )
            c2_p0 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v2_p0, dtypes.uint32), r2_p0, self.mantissa, self.bias, full
            )
            c3_p0 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v3_p0, dtypes.uint32), r3_p0, self.mantissa, self.bias, full
            )
            unused_code01_p0, code01_p0 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c0_p0, dtypes.uint16), reg.vreinterpret_lanes(c1_p0, dtypes.uint16)
            )
            unused_code23_p0, code23_p0 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c2_p0, dtypes.uint16), reg.vreinterpret_lanes(c3_p0, dtypes.uint16)
            )
            code_bytes_p0, unused_code_bytes_p0 = reg.vdeinterleave(
                reg.vreinterpret_lanes(code01_p0, dtypes.uint8), reg.vreinterpret_lanes(code23_p0, dtypes.uint8)
            )
            packed_p0 = reg.vreinterpret_lanes(reg.vgather_reg(code_bytes_p0, pack_index8), dtypes.uint32)
            magnitude8_p0 = reg.vadds(
                reg.vreinterpret_lanes(packed_p0, dtypes.uint8), -(128 - self.bias << self.mantissa) & 255, mask=full8
            )
            saturated_p0 = reg.vmins(magnitude8_p0, self.max_code, mask=reg.vnes(magnitude8_p0, 255, mask=full8))
            packed_p0 = reg.vbitwise_or(reg.vreinterpret_lanes(saturated_p0, dtypes.uint32), signs_p0, mask=full)
            invalid_p0 = reg.veqs(
                reg.vreinterpret(invalid_reciprocal_p0, dtypes.uint32), self.invalid_packed_reciprocal, mask=full
            )
            packed_p0 = reg.vdups(2139062143, dtypes.uint32, mask=invalid_p0, mode="merging", merge=packed_p0)
            reg.vstore(y, offset_p0, reg.vreinterpret_lanes(packed_p0, dtypes.uint8), full8)
            reg.vstore(self.scale32, part_p0 * 8, block_scales_p0, scale_mask32)
            word02_p1 = reg.vreinterpret_lanes(reg.vgather_reg(raw_bytes_p1, pair_index02), dtypes.uint32)
            word13_p1 = reg.vreinterpret_lanes(reg.vgather_reg(raw_bytes_p1, pair_index13), dtypes.uint32)
            r0_p1 = reg.vbitwise_and(word02_p1, random_mask32, mask=full)
            r1_p1 = reg.vbitwise_and(word13_p1, random_mask32, mask=full)
            r2_p1 = reg.vshr(word02_p1, 16, mask=full)
            r3_p1 = reg.vshr(word13_p1, 16, mask=full)
            c0_p1 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v0_p1, dtypes.uint32), r0_p1, self.mantissa, self.bias, full
            )
            c1_p1 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v1_p1, dtypes.uint32), r1_p1, self.mantissa, self.bias, full
            )
            c2_p1 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v2_p1, dtypes.uint32), r2_p1, self.mantissa, self.bias, full
            )
            c3_p1 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v3_p1, dtypes.uint32), r3_p1, self.mantissa, self.bias, full
            )
            unused_code01_p1, code01_p1 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c0_p1, dtypes.uint16), reg.vreinterpret_lanes(c1_p1, dtypes.uint16)
            )
            unused_code23_p1, code23_p1 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c2_p1, dtypes.uint16), reg.vreinterpret_lanes(c3_p1, dtypes.uint16)
            )
            code_bytes_p1, unused_code_bytes_p1 = reg.vdeinterleave(
                reg.vreinterpret_lanes(code01_p1, dtypes.uint8), reg.vreinterpret_lanes(code23_p1, dtypes.uint8)
            )
            packed_p1 = reg.vreinterpret_lanes(reg.vgather_reg(code_bytes_p1, pack_index8), dtypes.uint32)
            magnitude8_p1 = reg.vadds(
                reg.vreinterpret_lanes(packed_p1, dtypes.uint8), -(128 - self.bias << self.mantissa) & 255, mask=full8
            )
            saturated_p1 = reg.vmins(magnitude8_p1, self.max_code, mask=reg.vnes(magnitude8_p1, 255, mask=full8))
            packed_p1 = reg.vbitwise_or(reg.vreinterpret_lanes(saturated_p1, dtypes.uint32), signs_p1, mask=full)
            invalid_p1 = reg.veqs(
                reg.vreinterpret(invalid_reciprocal_p1, dtypes.uint32), self.invalid_packed_reciprocal, mask=full
            )
            packed_p1 = reg.vdups(2139062143, dtypes.uint32, mask=invalid_p1, mode="merging", merge=packed_p1)
            reg.vstore(y, offset_p1, reg.vreinterpret_lanes(packed_p1, dtypes.uint8), full8)
            reg.vstore(self.scale32, part_p1 * 8, block_scales_p1, scale_mask32)
        for part in range(parts // 2 * 2, parts):
            offset = part * 256
            if const_expr(self.bf16):
                a, b = reg.vload_deinterleave(x, offset)
                abs_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
                aa = reg.vbitwise_and(reg.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
                ab = reg.vbitwise_and(reg.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
                maxima16 = reg.vreduce_max_datablock(reg.vmax(aa, ab, mask=full16), mask=full16)
                maxima = reg.vshl(reg.vunpack(maxima16, dtypes.uint32), 16, mask=full)
                unused_signbytes, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
                signs = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes, sign_mask8, mask=full8), dtypes.uint32)
            else:
                a, b = reg.vload_deinterleave(x, offset)
                c, d = reg.vload_deinterleave(x, offset + 128)
                v0, v2 = reg.vdeinterleave(a, c)
                v1, v3 = reg.vdeinterleave(b, d)
                a0 = reg.vreinterpret(reg.vabs(v0, mask=full), dtypes.uint32)
                a1 = reg.vreinterpret(reg.vabs(v1, mask=full), dtypes.uint32)
                a2 = reg.vreinterpret(reg.vabs(v2, mask=full), dtypes.uint32)
                a3 = reg.vreinterpret(reg.vabs(v3, mask=full), dtypes.uint32)
                pair0 = reg.vmax(a0, a1, mask=full)
                pair1 = reg.vmax(a2, a3, mask=full)
                maxima = reg.vreduce_max_datablock(reg.vmax(pair0, pair1, mask=full), mask=full)
                unused_sign01, sign01 = reg.vload_deinterleave(sign_source, offset * 2)
                unused_sign23, sign23 = reg.vload_deinterleave(sign_source, offset * 2 + 256)
                unused_signbytes, signbytes = reg.vdeinterleave(
                    reg.vreinterpret_lanes(sign01, dtypes.uint8), reg.vreinterpret_lanes(sign23, dtypes.uint8)
                )
                signs = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes, sign_mask8, mask=full8), dtypes.uint32)
            block_scales, reciprocal = compute_scale(self, maxima, full, False)
            if const_expr(self.bf16):
                bits16 = reg.vshr(reg.vreinterpret(reciprocal, dtypes.uint32), 16, mask=full)
                r16 = reg.vpack(bits16, dtypes.uint16, part="lower")
                index16 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
                broadcast = reg.vreinterpret(reg.vgather_reg(r16, index16), dtypes.bfloat16)
                a = reg.vmul(reg.vreinterpret(aa, dtypes.bfloat16), broadcast, mask=full16)
                b = reg.vmul(reg.vreinterpret(ab, dtypes.bfloat16), broadcast, mask=full16)
                v0 = reg.vcast(a, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v2 = reg.vcast(a, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                v1 = reg.vcast(b, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                v3 = reg.vcast(b, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                invalid_reciprocal = reg.vreinterpret_lanes(broadcast, dtypes.uint32)
            else:
                reciprocal = reg.vgather_reg(reciprocal, scale_index)
                v0 = reg.vmul(reg.vreinterpret(a0, dtypes.float32), reciprocal, mask=full)
                v1 = reg.vmul(reg.vreinterpret(a1, dtypes.float32), reciprocal, mask=full)
                v2 = reg.vmul(reg.vreinterpret(a2, dtypes.float32), reciprocal, mask=full)
                v3 = reg.vmul(reg.vreinterpret(a3, dtypes.float32), reciprocal, mask=full)
                invalid_reciprocal = reciprocal
            raw = reg.vreinterpret(reg.vload(scratch, part * 32), dtypes.uint32)
            raw_bytes = reg.vreinterpret_lanes(raw, dtypes.uint8)
            word02 = reg.vreinterpret_lanes(reg.vgather_reg(raw_bytes, pair_index02), dtypes.uint32)
            word13 = reg.vreinterpret_lanes(reg.vgather_reg(raw_bytes, pair_index13), dtypes.uint32)
            r0 = reg.vbitwise_and(word02, random_mask32, mask=full)
            r1 = reg.vbitwise_and(word13, random_mask32, mask=full)
            r2 = reg.vshr(word02, 16, mask=full)
            r3 = reg.vshr(word13, 16, mask=full)
            c0 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v0, dtypes.uint32), r0, self.mantissa, self.bias, full)
            c1 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v1, dtypes.uint32), r1, self.mantissa, self.bias, full)
            c2 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v2, dtypes.uint32), r2, self.mantissa, self.bias, full)
            c3 = _sr_magnitude_q16_unbiased_sum(reg.vreinterpret(v3, dtypes.uint32), r3, self.mantissa, self.bias, full)
            unused_code01, code01 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c0, dtypes.uint16), reg.vreinterpret_lanes(c1, dtypes.uint16)
            )
            unused_code23, code23 = reg.vdeinterleave(
                reg.vreinterpret_lanes(c2, dtypes.uint16), reg.vreinterpret_lanes(c3, dtypes.uint16)
            )
            code_bytes, unused_code_bytes = reg.vdeinterleave(
                reg.vreinterpret_lanes(code01, dtypes.uint8), reg.vreinterpret_lanes(code23, dtypes.uint8)
            )
            packed = reg.vreinterpret_lanes(reg.vgather_reg(code_bytes, pack_index8), dtypes.uint32)
            magnitude8 = reg.vadds(
                reg.vreinterpret_lanes(packed, dtypes.uint8), -(128 - self.bias << self.mantissa) & 255, mask=full8
            )
            saturated = reg.vmins(magnitude8, self.max_code, mask=reg.vnes(magnitude8, 255, mask=full8))
            packed = reg.vbitwise_or(reg.vreinterpret_lanes(saturated, dtypes.uint32), signs, mask=full)
            invalid = reg.veqs(
                reg.vreinterpret(invalid_reciprocal, dtypes.uint32), self.invalid_packed_reciprocal, mask=full
            )
            packed = reg.vdups(2139062143, dtypes.uint32, mask=invalid, mode="merging", merge=packed)
            reg.vstore(y, offset, reg.vreinterpret_lanes(packed, dtypes.uint8), full8)
            reg.vstore(self.scale32, part * 8, block_scales, scale_mask32)
        # Publish scale32 stores before the same VF reloads and packs scale bytes.
        reg.vmem_bar("vst_vld")
        for scale_offset in range(0, parts * 8, 64):
            pack_mask = reg.update_mask(parts * 8 - scale_offset, elem_bits=32)[0]
            scale_codes = reg.vload(self.scale32, scale_offset)
            reg.vstore_pack(scales, scale_offset, scale_codes, pack_mask, pack_mode=reg.PackMode.B32_TO_B8)
        # Finish encoded output/scale writes before DMA or the next tile reuses UB.
        reg.vmem_bar("vst_vld")


@jit
def _run_fp32_output_alias(self, x, y, scales, quant_length, seed, block_offset):
    columns = x.shape[1]
    pairs = columns // 64
    core_pairs = (pairs + get_block_num() - 1) // get_block_num()
    if core_pairs >= 4:
        core_pairs = (core_pairs + 3) // 4 * 4
    if const_expr(self.quantization.bf16_guard_tail):
        if pairs >= get_block_num() * 64:
            core_pairs = (core_pairs + 15) // 16 * 16
    core_columns = core_pairs * 64
    core_start = get_block_idx() * core_columns
    scale_factor = 2
    if const_expr(self.quantization.scale_alg != 0):
        if scales.shape[1] == columns // 16:
            scale_factor = 4
    if core_start < columns:
        core_x = tile_slice(x, make_tiler((1, core_columns), alignment=(1, 64)), (0, get_block_idx()))
        core_y = tile_slice(y, make_tiler((1, core_columns), alignment=(1, 64)), (0, get_block_idx()))
        core_s = tile_slice(scales, make_tiler((1, core_pairs * scale_factor), alignment=(1, 2)), (0, get_block_idx()))
        core_count = columns - core_start
        if core_count > core_columns:
            core_count = core_columns
        tiles = (core_count + self.quantization.tile_elements - 1) // self.quantization.tile_elements
        for tile in range(tiles):
            count = core_count - tile * self.quantization.tile_elements
            if count > self.quantization.tile_elements:
                count = self.quantization.tile_elements
            input_slot = self.input.next_write()
            mem_copy(
                tile_slice(input_slot, make_tiler((1, self.quantization.tile_elements), alignment=(1, 64)), (0, 0)),
                tile_slice(core_x, make_tiler((1, self.quantization.tile_elements), alignment=(1, 64)), (0, tile)),
            )
            input_tile = tile_slice(
                self.input.next_read(), make_tiler((1, self.quantization.tile_elements), alignment=(1, 64)), (0, 0)
            )
            output_tile = self.output.next_write()
            scale_tile = self.scale.next_write()
            start = core_start + tile * self.quantization.tile_elements
            if const_expr(self.quantization.scale_alg == 0):
                _quantize_fp32_output_alias_batched(
                    self.quantization,
                    input_tile,
                    output_tile,
                    scale_tile,
                    start,
                    count,
                    scale_factor,
                    quant_length,
                    seed,
                    block_offset,
                )
            else:
                _quantize_fp32_output_alias(
                    self.quantization,
                    input_tile,
                    output_tile,
                    scale_tile,
                    start,
                    count,
                    scale_factor,
                    quant_length,
                    seed,
                    block_offset,
                )
            output_copy = self.output.next_read()
            _copy_to_gm(
                tile_slice(core_y, make_tiler((1, self.quantization.tile_elements), alignment=(1, 64)), (0, tile)),
                output_copy,
            )
            scale_copy = self.scale.next_read()
            _copy_to_gm(
                tile_slice(core_s, make_tiler((1, self.max_pairs * scale_factor), alignment=(1, 2)), (0, tile)),
                scale_copy,
            )


@jit
def dispatch_tail(self, x, y, scales, quant_length, seed, block_offset):
    if const_expr(self.bf16_full_tiles):
        _run_bf16_full_tiles(self, x, y, scales, quant_length, seed, block_offset)
    elif const_expr(self.output_rng_alias):
        _run_fp32_output_alias(self, x, y, scales, quant_length, seed, block_offset)
    else:
        _dispatch_standard_tail(self, x, y, scales, quant_length, seed, block_offset)


def _select_bf16_compact_tile_policy(numel, dtype, dst_dtype, layout, quant_length, blocks, ub_bytes):
    if (
        dtype != dtypes.bfloat16
        or dst_dtype != torch.float8_e4m3fn
        or layout is not Layout.TAIL
        or quant_length % PACKED_K_ALIGNMENT_ELEMENTS
    ):
        return 0
    # Amortize compact tiling only after layout and format admission.
    if numel < BF16_COMPACT_TILE_MIN_ELEMENTS:
        return 0
    parent = _make_tiling(dtype, ub_bytes, tail_axis=True)
    compact = _make_tiling(dtype, ub_bytes, tail_axis=True, bf16_compact_mode=BF16_COMPACT_TILE_KI_ELEMENTS)
    if (
        numel > blocks * parent.tile_elements
        and numel % (blocks * parent.tile_elements) == 0
        and numel % (blocks * compact.tile_elements) != 0
    ):
        return 0
    return BF16_COMPACT_TILE_KI_ELEMENTS


@jit
def _produce_bf16_four_panel_random(random, start, batches, seed, block_offset, tail_simd=False):
    key0, key1 = _split_uint64(seed)
    group_low, group_high = _split_uint64(dtypes.uint64(start) // 32)
    offset_low, offset_high = _split_uint64(block_offset)
    base_low = dtypes.uint32(offset_low // 4 + offset_high % 4 * 1073741824)
    base_high = dtypes.uint32(offset_high // 4)
    linear_group_count = dtypes.int64(batches) * 32
    first_group = dtypes.uint64(start) // 32
    draw = first_group // 16384
    last_draw = (first_group + dtypes.uint64(linear_group_count) - 1) // 16384
    if draw == last_draw:
        draw_low, draw_high = _split_uint64(draw)
        counter_low = dtypes.uint32(base_low) + dtypes.uint32(draw_low)
        counter_high = (
            dtypes.uint32(base_high) + dtypes.uint32(draw_high) + dtypes.uint32(counter_low < dtypes.uint32(base_low))
        )
        first_low, first_high = _split_uint64(dtypes.uint64(counter_low) * dtypes.uint64(PHILOX_M1))
        uniform2 = dtypes.uint32(counter_high) ^ dtypes.uint32(key1)
        second_low, second_high = _split_uint64(dtypes.uint64(uniform2) * dtypes.uint64(PHILOX_M1))
        next_key0 = dtypes.uint32(key0) + dtypes.uint32(PHILOX_W0)
        next_key1 = dtypes.uint32(key1) + dtypes.uint32(PHILOX_W1)
        uniform_second0 = dtypes.uint32(second_high) ^ dtypes.uint32(first_low) ^ dtypes.uint32(next_key0)
        third_low, third_high = _split_uint64(dtypes.uint64(uniform_second0) * dtypes.uint64(PHILOX_M0))
        uniform_initial = dtypes.uint32(first_high) ^ dtypes.uint32(key0)
        uniform_x0 = dtypes.uint32(second_low) ^ dtypes.uint32(key0) + dtypes.uint32(2 * PHILOX_W0 & UINT32_MASK)
        uniform_x2 = dtypes.uint32(third_high) ^ dtypes.uint32(key1) + dtypes.uint32(2 * PHILOX_W1 & UINT32_MASK)
        if linear_group_count % 128 == 0:
            with vf(mode="simd"):
                full = reg.full_mask()
                ids = reg.varange(0, dtypes.uint32)
                m0 = reg.vdups(PHILOX_M0, dtypes.uint32, mask=full)
                m1 = reg.vdups(PHILOX_M1, dtypes.uint32, mask=full)
                uinitial = reg.vdups(uniform_initial, dtypes.uint32, mask=full)
                uk1 = reg.vdups(next_key1, dtypes.uint32, mask=full)
                ux0 = reg.vdups(uniform_x0, dtypes.uint32, mask=full)
                ux2 = reg.vdups(uniform_x2, dtypes.uint32, mask=full)
                third = reg.vdups(third_low, dtypes.uint32, mask=full)
                for base in range(0, linear_group_count, 64):
                    sub0 = reg.vbitwise_and(
                        reg.vadds(ids, dtypes.uint32(group_low) + dtypes.uint32(base) + dtypes.uint32(0), mask=full),
                        reg.vdups(16383, dtypes.uint32, mask=full),
                        mask=full,
                    )
                    first0 = reg.vbitwise_xor(sub0, uinitial, mask=full)
                    varlo0, varhi0 = reg.vmull(first0, m0, dtypes.uint32, mask=full)
                    second0 = reg.vbitwise_xor(varhi0, uk1, mask=full)
                    p1lo0, p1hi0 = reg.vmull(second0, m1, dtypes.uint32, mask=full)
                    s0_0 = reg.vbitwise_xor(p1hi0, ux0, mask=full)
                    s0_1 = p1lo0
                    s0_2 = reg.vbitwise_xor(varlo0, ux2, mask=full)
                    s0_3 = third
                    k0 = reg.vdups(
                        dtypes.uint32(key0) + dtypes.uint32((3 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    k1 = reg.vdups(
                        dtypes.uint32(key1) + dtypes.uint32((3 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
                    lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
                    s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
                    s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
                    s0_1 = lo1_0
                    s0_3 = lo0_0
                    k0 = reg.vdups(
                        dtypes.uint32(key0) + dtypes.uint32((4 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    k1 = reg.vdups(
                        dtypes.uint32(key1) + dtypes.uint32((4 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
                    lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
                    s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
                    s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
                    s0_1 = lo1_0
                    s0_3 = lo0_0
                    k0 = reg.vdups(
                        dtypes.uint32(key0) + dtypes.uint32((5 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    k1 = reg.vdups(
                        dtypes.uint32(key1) + dtypes.uint32((5 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
                    lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
                    s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
                    s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
                    s0_1 = lo1_0
                    s0_3 = lo0_0
                    k0 = reg.vdups(
                        dtypes.uint32(key0) + dtypes.uint32((6 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    k1 = reg.vdups(
                        dtypes.uint32(key1) + dtypes.uint32((6 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
                    lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
                    s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
                    s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
                    s0_1 = lo1_0
                    s0_3 = lo0_0
                    k0 = reg.vdups(
                        dtypes.uint32(key0) + dtypes.uint32((7 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    k1 = reg.vdups(
                        dtypes.uint32(key1) + dtypes.uint32((7 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
                    lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
                    s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
                    s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
                    s0_1 = lo1_0
                    s0_3 = lo0_0
                    k0 = reg.vdups(
                        dtypes.uint32(key0) + dtypes.uint32((8 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    k1 = reg.vdups(
                        dtypes.uint32(key1) + dtypes.uint32((8 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
                    lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
                    s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
                    s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
                    s0_1 = lo1_0
                    s0_3 = lo0_0
                    k0 = reg.vdups(
                        dtypes.uint32(key0) + dtypes.uint32((9 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    k1 = reg.vdups(
                        dtypes.uint32(key1) + dtypes.uint32((9 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full
                    )
                    lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
                    lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
                    s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
                    s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
                    s0_1 = lo1_0
                    s0_3 = lo0_0
                    _store_random_words_aos(random.buffer, (base + 0) * 4, s0_0, s0_1, s0_2, s0_3, full)
                # Publish Philox scratch stores before SR VFs gather random words.
                reg.vmem_bar("vst_vld")
        elif const_expr(tail_simd):
            padded_group_count = (linear_group_count + 63) // 64 * 64
            padded_last_draw = (first_group + dtypes.uint64(padded_group_count) - 1) // 16384
            if padded_group_count <= random.capacity_bytes // 16 and padded_last_draw == draw:
                _produce_bf16_uniform_tail_simd(
                    random,
                    padded_group_count,
                    group_low,
                    uniform_initial,
                    next_key1,
                    uniform_x0,
                    uniform_x2,
                    third_low,
                    key0,
                    key1,
                )
            else:
                with vf(mode="simt", thread=256):
                    for group in range(simt.thread_idx()[0], linear_group_count, 256):
                        subsequence = (dtypes.uint32(group_low) + dtypes.uint32(group)) % 16384
                        first_c0 = subsequence ^ dtypes.uint32(uniform_initial)
                        variable_low = dtypes.uint32(first_c0) * dtypes.uint32(PHILOX_M0)
                        variable_high = _simt_umulhi(first_c0, PHILOX_M0)
                        second_c2 = variable_high ^ dtypes.uint32(next_key1)
                        p1lo = dtypes.uint32(second_c2) * dtypes.uint32(PHILOX_M1)
                        p1hi = _simt_umulhi(second_c2, PHILOX_M1)
                        c0 = p1hi ^ dtypes.uint32(uniform_x0)
                        c1 = p1lo
                        c2 = variable_low ^ dtypes.uint32(uniform_x2)
                        c3 = dtypes.uint32(third_low)
                        for step in range(3, 10):
                            p0lo = dtypes.uint32(c0) * dtypes.uint32(PHILOX_M0)
                            p0hi = _simt_umulhi(c0, PHILOX_M0)
                            p1lo = dtypes.uint32(c2) * dtypes.uint32(PHILOX_M1)
                            p1hi = _simt_umulhi(c2, PHILOX_M1)
                            k0 = dtypes.uint32(key0) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W0)
                            k1 = dtypes.uint32(key1) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W1)
                            c0, c1, c2, c3 = (p1hi ^ c1 ^ k0, p1lo, p0hi ^ c3 ^ k1, p0lo)
                        random.buffer[0, group * 4] = dtypes.int32(c0)
                        random.buffer[0, group * 4 + 1] = dtypes.int32(c1)
                        random.buffer[0, group * 4 + 2] = dtypes.int32(c2)
                        random.buffer[0, group * 4 + 3] = dtypes.int32(c3)
        else:
            with vf(mode="simt", thread=256):
                for group in range(simt.thread_idx()[0], linear_group_count, 256):
                    subsequence = (dtypes.uint32(group_low) + dtypes.uint32(group)) % 16384
                    first_c0 = subsequence ^ dtypes.uint32(uniform_initial)
                    variable_low = dtypes.uint32(first_c0) * dtypes.uint32(PHILOX_M0)
                    variable_high = _simt_umulhi(first_c0, PHILOX_M0)
                    second_c2 = variable_high ^ dtypes.uint32(next_key1)
                    p1lo = dtypes.uint32(second_c2) * dtypes.uint32(PHILOX_M1)
                    p1hi = _simt_umulhi(second_c2, PHILOX_M1)
                    c0 = p1hi ^ dtypes.uint32(uniform_x0)
                    c1 = p1lo
                    c2 = variable_low ^ dtypes.uint32(uniform_x2)
                    c3 = dtypes.uint32(third_low)
                    for step in range(3, 10):
                        p0lo = dtypes.uint32(c0) * dtypes.uint32(PHILOX_M0)
                        p0hi = _simt_umulhi(c0, PHILOX_M0)
                        p1lo = dtypes.uint32(c2) * dtypes.uint32(PHILOX_M1)
                        p1hi = _simt_umulhi(c2, PHILOX_M1)
                        k0 = dtypes.uint32(key0) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W0)
                        k1 = dtypes.uint32(key1) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W1)
                        c0, c1, c2, c3 = (p1hi ^ c1 ^ k0, p1lo, p0hi ^ c3 ^ k1, p0lo)
                    random.buffer[0, group * 4] = dtypes.int32(c0)
                    random.buffer[0, group * 4 + 1] = dtypes.int32(c1)
                    random.buffer[0, group * 4 + 2] = dtypes.int32(c2)
                    random.buffer[0, group * 4 + 3] = dtypes.int32(c3)
    else:
        with vf(mode="simt", thread=256):
            for group in range(simt.thread_idx()[0], linear_group_count, 256):
                gl = dtypes.uint32(group_low) + dtypes.uint32(group)
                gh = dtypes.uint32(group_high) + dtypes.uint32(gl < dtypes.uint32(group_low))
                draw_low = dtypes.uint32(gl // 16384) + dtypes.uint32(gh % 16384) * dtypes.uint32(262144)
                draw_high = dtypes.uint32(gh // 16384)
                local_counter_low = dtypes.uint32(base_low) + dtypes.uint32(draw_low)
                local_counter_high = (
                    dtypes.uint32(base_high)
                    + dtypes.uint32(draw_high)
                    + dtypes.uint32(local_counter_low < dtypes.uint32(base_low))
                )
                c0 = dtypes.uint32(0)
                c1 = dtypes.uint32(gl % 16384)
                c2 = local_counter_low
                c3 = local_counter_high
                for step in range(10):
                    p0lo = dtypes.uint32(c0) * dtypes.uint32(PHILOX_M0)
                    p0hi = _simt_umulhi(c0, PHILOX_M0)
                    p1lo = dtypes.uint32(c2) * dtypes.uint32(PHILOX_M1)
                    p1hi = _simt_umulhi(c2, PHILOX_M1)
                    k0 = dtypes.uint32(key0) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W0)
                    k1 = dtypes.uint32(key1) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W1)
                    c0, c1, c2, c3 = (p1hi ^ c1 ^ k0, p1lo, p0hi ^ c3 ^ k1, p0lo)
                random.buffer[0, group * 4] = dtypes.int32(c0)
                random.buffer[0, group * 4 + 1] = dtypes.int32(c1)
                random.buffer[0, group * 4 + 2] = dtypes.int32(c2)
                random.buffer[0, group * 4 + 3] = dtypes.int32(c3)


@jit
def _encode_bf16_clean_in_vf(self, x, y, first, stop, sign_source, random_bytes):
    full16 = reg.full_mask(elem_bits=16)
    full8 = reg.full_mask(elem_bits=8)
    byte_mask16 = reg.vdups(255, dtypes.uint16, mask=full16)
    sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
    random_mask = reg.vdups((1 << 7 - self.mantissa) - 1, dtypes.uint16, mask=full16)
    for part in range(first, stop):
        offset = part * 256
        av, bv = reg.vload_deinterleave(x, offset)
        raw = reg.vload(random_bytes, part * 128)
        expanded = reg.vunpack(raw, dtypes.uint16)
        ra = reg.vbitwise_and(expanded, random_mask, mask=full16)
        reverse = reg.vshr(expanded, 1 + self.mantissa, mask=full16)
        ca = reg.vshr(reg.vadd(reg.vreinterpret(av, dtypes.uint16), ra, mask=full16), 7 - self.mantissa, mask=full16)
        cb = reg.vshr(
            reg.vadd(reg.vreinterpret(bv, dtypes.uint16), reverse, mask=full16), 7 - self.mantissa, mask=full16
        )
        paired = reg.vbitwise_or(
            reg.vbitwise_and(ca, byte_mask16, mask=full16), reg.vshl(cb, 8, mask=full16), mask=full16
        )
        codes8 = reg.vreinterpret_lanes(paired, dtypes.uint8)
        codes8 = reg.vadds(codes8, -(127 - self.bias << self.mantissa) & 255, mask=full8)
        codes8 = reg.vmins(codes8, self.max_code, mask=full8)
        unused_low, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
        codes8 = reg.vbitwise_or(codes8, reg.vbitwise_and(signbytes, sign_mask8, mask=full8), mask=full8)
        reg.vstore(y, offset, codes8, full8)


@jit
def _encode_bf16_half_q7_in_vf(self, x, y, first, stop, sign_source, random_bytes):
    full16 = reg.full_mask(elem_bits=16)
    full8 = reg.full_mask(elem_bits=8)
    byte_mask16 = reg.vdups(255, dtypes.uint16, mask=full16)
    sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
    ids_byte = reg.varange(0, dtypes.uint8)
    pairs = reg.vshr(ids_byte, 1, mask=full8)
    base = reg.vshl(reg.vshr(pairs, 2, mask=full8), 2, mask=full8)
    byte = reg.vbitwise_and(ids_byte, reg.vdups(1, dtypes.uint8, mask=full8), mask=full8)
    position = reg.vbitwise_and(
        reg.vadd(reg.vadds(pairs, 3, mask=full8), byte, mask=full8), reg.vdups(3, dtypes.uint8, mask=full8), mask=full8
    )
    pair_index = reg.vbitwise_or(base, position, mask=full8)
    field_mask = reg.vdups(4095, dtypes.uint16, mask=full16)
    absolute_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
    for part in range(first, stop):
        offset = part * 256
        av, bv = reg.vload_deinterleave(x, offset)
        raw = reg.vload(random_bytes, part * 128)
        window = reg.vreinterpret_lanes(reg.vgather_reg(raw, pair_index), dtypes.uint16)
        if const_expr(self.mantissa == 3):
            ra = reg.vbitwise_and(window, field_mask, mask=full16)
        else:
            ra = reg.vbitwise_and(reg.vshr(window, 1, mask=full16), field_mask, mask=full16)
        reverse = reg.vshr(window, 4, mask=full16)
        ca = _sr_bf16_half_q7(av, ra, absolute_mask, full16)
        cb = _sr_bf16_half_q7(bv, reverse, absolute_mask, full16)
        paired = reg.vbitwise_or(
            reg.vbitwise_and(ca, byte_mask16, mask=full16), reg.vshl(cb, 8, mask=full16), mask=full16
        )
        codes8 = reg.vreinterpret_lanes(paired, dtypes.uint8)
        codes8 = reg.vadds(codes8, -(16 - self.bias << self.mantissa) & 255, mask=full8)
        codes8 = reg.vmins(codes8, self.max_code, mask=full8)
        unused_low, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
        codes8 = reg.vbitwise_or(codes8, reg.vbitwise_and(signbytes, sign_mask8, mask=full8), mask=full8)
        reg.vstore(y, offset, codes8, full8)


@jit
def _encode_bf16_half_q7_in_vf_random7(self, x, y, first, stop, sign_source, random_bytes):
    full16 = reg.full_mask(elem_bits=16)
    full8 = reg.full_mask(elem_bits=8)
    byte_mask16 = reg.vdups(255, dtypes.uint16, mask=full16)
    sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
    ids_byte = reg.varange(0, dtypes.uint8)
    pairs = reg.vshr(ids_byte, 1, mask=full8)
    base = reg.vshl(reg.vshr(pairs, 2, mask=full8), 2, mask=full8)
    byte = reg.vbitwise_and(ids_byte, reg.vdups(1, dtypes.uint8, mask=full8), mask=full8)
    position = reg.vbitwise_and(
        reg.vadd(reg.vadds(pairs, 3, mask=full8), byte, mask=full8), reg.vdups(3, dtypes.uint8, mask=full8), mask=full8
    )
    pair_index = reg.vbitwise_or(base, position, mask=full8)
    field_mask = reg.vdups(4095, dtypes.uint16, mask=full16)
    if const_expr(self.mantissa == 3):
        random7_mask = reg.vdups(127, dtypes.uint16, mask=full16)
    absolute_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
    for part in range(first, stop):
        offset = part * 256
        av, bv = reg.vload_deinterleave(x, offset)
        raw = reg.vload(random_bytes, part * 128)
        window = reg.vreinterpret_lanes(reg.vgather_reg(raw, pair_index), dtypes.uint16)
        if const_expr(self.mantissa == 3):
            ra7 = reg.vbitwise_and(reg.vshr(window, 5, mask=full16), random7_mask, mask=full16)
            rb7 = reg.vshr(window, 9, mask=full16)
            ca = _sr_bf16_half_q7_random7(av, ra7, absolute_mask, full16)
            cb = _sr_bf16_half_q7_random7(bv, rb7, absolute_mask, full16)
        else:
            ra = reg.vbitwise_and(reg.vshr(window, 1, mask=full16), field_mask, mask=full16)
            reverse = reg.vshr(window, 4, mask=full16)
            ca = _sr_bf16_half_q7(av, ra, absolute_mask, full16)
            cb = _sr_bf16_half_q7(bv, reverse, absolute_mask, full16)
        paired = reg.vbitwise_or(
            reg.vbitwise_and(ca, byte_mask16, mask=full16), reg.vshl(cb, 8, mask=full16), mask=full16
        )
        codes8 = reg.vreinterpret_lanes(paired, dtypes.uint8)
        codes8 = reg.vadds(codes8, -(16 - self.bias << self.mantissa) & 255, mask=full8)
        codes8 = reg.vmins(codes8, self.max_code, mask=full8)
        unused_low, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
        codes8 = reg.vbitwise_or(codes8, reg.vbitwise_and(signbytes, sign_mask8, mask=full8), mask=full8)
        reg.vstore(y, offset, codes8, full8)


@jit
def _encode_bf16_middle_in_vf(self, x, y, first, stop, sign_source, random_bytes):
    full16 = reg.full_mask(elem_bits=16)
    full8 = reg.full_mask(elem_bits=8)
    byte_mask16 = reg.vdups(255, dtypes.uint16, mask=full16)
    sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
    ids_byte = reg.varange(0, dtypes.uint8)
    pairs = reg.vshr(ids_byte, 1, mask=full8)
    base = reg.vshl(reg.vshr(pairs, 2, mask=full8), 2, mask=full8)
    byte = reg.vbitwise_and(ids_byte, reg.vdups(1, dtypes.uint8, mask=full8), mask=full8)
    position = reg.vbitwise_and(
        reg.vadd(reg.vadds(pairs, 3, mask=full8), byte, mask=full8), reg.vdups(3, dtypes.uint8, mask=full8), mask=full8
    )
    pair_index = reg.vbitwise_or(base, position, mask=full8)
    field_mask = reg.vdups(4095, dtypes.uint16, mask=full16)
    absolute_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
    multiplier = reg.vdups(2.0 ** (self.bias + self.mantissa + 11), dtypes.bfloat16, mask=full16)
    for part in range(first, stop):
        offset = part * 256
        av, bv = reg.vload_deinterleave(x, offset)
        raw = reg.vload(random_bytes, part * 128)
        window = reg.vreinterpret_lanes(reg.vgather_reg(raw, pair_index), dtypes.uint16)
        if const_expr(self.mantissa == 3):
            ra = reg.vbitwise_and(window, field_mask, mask=full16)
        else:
            ra = reg.vbitwise_and(reg.vshr(window, 1, mask=full16), field_mask, mask=full16)
        reverse = reg.vshr(window, 4, mask=full16)
        ca = _sr_bf16_middle_q12_direct(self, av, ra, absolute_mask, multiplier, full16)
        cb = _sr_bf16_middle_q12_direct(self, bv, reverse, absolute_mask, multiplier, full16)
        paired = reg.vbitwise_or(
            reg.vbitwise_and(ca, byte_mask16, mask=full16), reg.vshl(cb, 8, mask=full16), mask=full16
        )
        codes8 = reg.vreinterpret_lanes(paired, dtypes.uint8)
        codes8 = reg.vadds(codes8, -(127 - self.bias << self.mantissa) & 255, mask=full8)
        codes8 = reg.vmins(codes8, self.max_code, mask=full8)
        unused_low, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
        codes8 = reg.vbitwise_or(codes8, reg.vbitwise_and(signbytes, sign_mask8, mask=full8), mask=full8)
        reg.vstore(y, offset, codes8, full8)


@jit
def _encode_bf16_adjacent_bundles(self, random, x, y, first, flag_0, flag_1):
    sign_source = reinterpret(x, dtypes.uint8, shape=(1, self.tile_elements * 2))
    random_bytes = reinterpret(random.buffer, dtypes.uint8, shape=(1, random.capacity_bytes))
    with vf(mode="simd"):
        if flag_0 >= (128 - self.bias) * 128:
            _encode_bf16_clean_in_vf(self, x, y, first + 0, first + 16, sign_source, random_bytes)
        elif flag_0 >= BF16_HALF_Q7_MIN_NORMALIZED_BITS:
            if const_expr(self.scale_alg == 0 and self.mantissa == 3):
                _encode_bf16_half_q7_in_vf_random7(self, x, y, first + 0, first + 16, sign_source, random_bytes)
            else:
                _encode_bf16_half_q7_in_vf(self, x, y, first + 0, first + 16, sign_source, random_bytes)
        else:
            _encode_bf16_middle_in_vf(self, x, y, first + 0, first + 16, sign_source, random_bytes)
        if flag_1 >= (128 - self.bias) * 128:
            _encode_bf16_clean_in_vf(self, x, y, first + 16, first + 32, sign_source, random_bytes)
        elif flag_1 >= BF16_HALF_Q7_MIN_NORMALIZED_BITS:
            if const_expr(self.scale_alg == 0 and self.mantissa == 3):
                _encode_bf16_half_q7_in_vf_random7(self, x, y, first + 16, first + 32, sign_source, random_bytes)
            else:
                _encode_bf16_half_q7_in_vf(self, x, y, first + 16, first + 32, sign_source, random_bytes)
        else:
            _encode_bf16_middle_in_vf(self, x, y, first + 16, first + 32, sign_source, random_bytes)
        # Complete encoded byte stores before subsequent VFs reuse UB or DMA consumes this output slot.
        reg.vmem_bar("vst_vld")


@jit
def _quantize_bf16_four_panel(self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset):
    parts = count // 256
    batches = (count + 1023) // 1024
    _produce_bf16_four_panel_random(
        random, start, batches, seed, block_offset, tail_simd=self.bf16_tail_simd and (not self.bf16_four_panel)
    )
    if const_expr(self.bf16_full_tiles):
        _normalize_bf16_vector_scale_e2b(self, x, scales, parts, scale_factor)
        if const_expr(self.mantissa == 3):
            _encode_bf16_classified_bundles_8(self, random, x, y, parts, scales)
        else:
            _encode_bf16_classified_bundles_16(self, random, x, y, parts, scales)
    else:
        if const_expr(self.scale_alg == 0 and self.mantissa == 3):
            _normalize_bf16_four_panels_batched(self, x, scales, parts, scale_factor)
        else:
            _normalize_bf16_four_panels(self, x, scales, parts, scale_factor)
        _encode_bf16_classified_bundles(self, random, x, y, parts)


@jit
def _normalize_bf16_vector_scale_e2b(self, x, scales, parts, scale_factor):
    """Derive scales in batches, then normalize from a separate reciprocal buffer."""
    scale16 = reinterpret(self.scale32, dtypes.uint16, shape=(1, self.tile_elements // 16))
    with vf(mode="simd"):
        full16 = reg.full_mask(elem_bits=16)
        eight16 = reg.update_mask(8, elem_bits=16)[0]
        abs_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
        for part in range(parts):
            a, b = reg.vload_deinterleave(x, part * 256)
            aa = reg.vbitwise_and(reg.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
            ab = reg.vbitwise_and(reg.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
            maxima = reg.vreduce_max_datablock(reg.vmax(aa, ab, mask=full16), mask=full16)
            reg.vstore(scale16, part * 16, maxima, eight16)
        # Publish per-panel maxima before the next VF gathers them to derive scales.
        reg.vmem_bar("vst_vld")
    with vf(mode="simd"):
        full16 = reg.full_mask(elem_bits=16)
        ids = reg.varange(0, dtypes.uint16)
        panel_base = reg.vshl(reg.vshr(ids, 3, mask=full16), 4, mask=full16)
        panel_lane = reg.vbitwise_and(ids, reg.vdups(7, dtypes.uint16, mask=full16), mask=full16)
        sparse_index = reg.vbitwise_or(panel_base, panel_lane, mask=full16)
        for scale_offset in range(0, parts * 8, 128):
            indices = reg.vadds(sparse_index, scale_offset * 2, mask=full16)
            maxima = reg.vgather(scale16, indices, mask=full16)
            if const_expr(self.scale_alg == 0):
                exponent = reg.vshr(maxima, 7, mask=full16)
                invalid = reg.vges(exponent, 255, mask=full16)
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                code = reg.vadds(reg.vmaxs(exponent, scale_shift, mask=full16), -scale_shift & 65535, mask=full16)
                code = reg.vdups(255, dtypes.uint16, mask=invalid, mode="merging", merge=code)
                recip = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code, mask=full16),
                    7,
                    mask=reg.vnes(code, 0, mask=full16),
                )
            else:
                # Exact alg1 BF16 bit formula from the existing four-panel path.
                scale_shift = 15 if const_expr(self.mantissa == 2) else 8
                active = reg.vgts(maxima, scale_shift << 7 | 96, mask=full16)
                code = reg.vshr(reg.vadds(maxima, 31 - (scale_shift << 7) & 65535, mask=full16), 7, mask=active)
                code = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima, 32640, mask=full16), mode="merging", merge=code
                )
                recip = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code, mask=full16), 7, mask=full16
                )
            reg.vstore_pack(scales, scale_offset, code, full16, pack_mode=reg.PackMode.B16_TO_B8)
            if const_expr(not (self.scale_alg == 0 and self.mantissa == 3)):
                reg.vscatter(scale16, code, indices, mask=full16)
            reg.vstore(self.recip_scale, scale_offset, recip, full16)
        # Publish scale codes and reciprocals before normalization broadcasts reciprocal values.
        reg.vmem_bar("vst_vld")
    with vf(mode="simd"):
        full16 = reg.full_mask(elem_bits=16)
        one16 = reg.update_mask(1, elem_bits=16)[0]
        abs_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
        tile_minimum = reg.vdups(65535, dtypes.uint16, mask=one16)
        group_parts = 8 if const_expr(self.bf16_full_tiles and self.scale_alg in (0, 1) and self.mantissa == 3) else 16
        for first16 in range(0, parts, group_parts):
            group_min = reg.vdups(65535, dtypes.uint16, mask=full16)
            group_recip = reg.vdups(0, dtypes.uint16, mask=full16)
            for part in range(first16, first16 + group_parts):
                offset = part * 256
                a, b = reg.vload_deinterleave(x, offset, width="b16")
                recip = reg.vload_broadcast(self.recip_scale, part * 8, mode="datablock")
                group_recip = reg.vmax(group_recip, recip, mask=full16)
                broadcast = reg.vreinterpret(recip, dtypes.bfloat16)
                av = reg.vmul(a, broadcast, mask=full16)
                bv = reg.vmul(b, broadcast, mask=full16)
                reg.vstore_interleave(x, offset, av, bv)
                aa = reg.vbitwise_and(reg.vreinterpret(av, dtypes.uint16), abs_mask, mask=full16)
                ab = reg.vbitwise_and(reg.vreinterpret(bv, dtypes.uint16), abs_mask, mask=full16)
                group_min = reg.vmin(group_min, reg.vmin(aa, ab, mask=full16), mask=full16)
            minimum = reg.vreduce_min(group_min, mask=full16)
            maximum_recip = reg.vreduce_max(group_recip, mask=full16)
            minimum = reg.vdups(
                0,
                dtypes.uint16,
                mask=reg.veqs(maximum_recip, 65408, mask=full16),
                mode="merging",
                merge=minimum,
            )
            reg.vstore_first(self.norm_flags, first16 * self.norm_flag_stride, minimum)
            if const_expr(not (self.bf16_full_tiles and self.mantissa == 3)):
                tile_minimum = reg.vmin(tile_minimum, minimum, mask=one16)
        if const_expr(not (self.bf16_full_tiles and self.mantissa == 3)):
            reg.vstore_first(self.norm_flags, self.norm_tile_flag, tile_minimum)
        # Publish normalized x and class minima before scalar classification and encoding reloads.
        reg.vmem_bar("vst_vld")


@jit
def _normalize_bf16_four_panels(self, x, scales, parts, scale_factor):
    """Normalize four panels per iteration and publish group minima/scale bytes."""
    scale16 = reinterpret(self.scale32, dtypes.uint16, shape=(1, self.tile_elements // 16))
    with vf(mode="simd"):
        full = reg.full_mask()
        full16 = reg.full_mask(elem_bits=16)
        full8 = reg.full_mask(elem_bits=8)
        eight16 = reg.update_mask(8, elem_bits=16)[0]
        abs_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
        index16 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
        pack_ids8 = reg.varange(0, dtypes.uint8)
        scale_byte_slot = reg.vbitwise_and(pack_ids8, reg.vdups(1, dtypes.uint8, mask=full8), mask=full8)
        scale_byte_mask = reg.mask_and(
            reg.vlts(pack_ids8, 16, mask=full8),
            reg.vlts(scale_byte_slot, dtypes.uint8(scale_factor // 2), mask=full8),
            exec_mask=full8,
        )
        scale_cursor = reg.vstore_unalign_begin(scales)
        one16 = reg.update_mask(1, elem_bits=16)[0]
        tile_minimum = reg.vdups(65535, dtypes.uint16, mask=one16)
        for first4 in range(0, parts, 4):
            group_min = reg.vdups(65535, dtypes.uint16, mask=full16)
            group_invalid = reg.vdups(0, dtypes.uint16, mask=full16)
            part_p0 = first4 + 0
            offset_p0 = part_p0 * 256
            a_p0, b_p0 = reg.vload_deinterleave(x, offset_p0)
            aa_p0 = reg.vbitwise_and(reg.vreinterpret(a_p0, dtypes.uint16), abs_mask, mask=full16)
            ab_p0 = reg.vbitwise_and(reg.vreinterpret(b_p0, dtypes.uint16), abs_mask, mask=full16)
            maxima16_p0 = reg.vreduce_max_datablock(reg.vmax(aa_p0, ab_p0, mask=full16), mask=full16)
            part_p1 = first4 + 1
            offset_p1 = part_p1 * 256
            a_p1, b_p1 = reg.vload_deinterleave(x, offset_p1)
            aa_p1 = reg.vbitwise_and(reg.vreinterpret(a_p1, dtypes.uint16), abs_mask, mask=full16)
            ab_p1 = reg.vbitwise_and(reg.vreinterpret(b_p1, dtypes.uint16), abs_mask, mask=full16)
            maxima16_p1 = reg.vreduce_max_datablock(reg.vmax(aa_p1, ab_p1, mask=full16), mask=full16)
            part_p2 = first4 + 2
            offset_p2 = part_p2 * 256
            a_p2, b_p2 = reg.vload_deinterleave(x, offset_p2)
            aa_p2 = reg.vbitwise_and(reg.vreinterpret(a_p2, dtypes.uint16), abs_mask, mask=full16)
            ab_p2 = reg.vbitwise_and(reg.vreinterpret(b_p2, dtypes.uint16), abs_mask, mask=full16)
            maxima16_p2 = reg.vreduce_max_datablock(reg.vmax(aa_p2, ab_p2, mask=full16), mask=full16)
            part_p3 = first4 + 3
            offset_p3 = part_p3 * 256
            a_p3, b_p3 = reg.vload_deinterleave(x, offset_p3)
            aa_p3 = reg.vbitwise_and(reg.vreinterpret(a_p3, dtypes.uint16), abs_mask, mask=full16)
            ab_p3 = reg.vbitwise_and(reg.vreinterpret(b_p3, dtypes.uint16), abs_mask, mask=full16)
            maxima16_p3 = reg.vreduce_max_datablock(reg.vmax(aa_p3, ab_p3, mask=full16), mask=full16)
            if const_expr(self.scale_alg == 0):
                scale_shift_p0 = 15 if const_expr(self.mantissa == 2) else 8
                exponent16_p0 = reg.vshr(maxima16_p0, 7, mask=full16)
                invalid16_p0 = reg.vges(exponent16_p0, 255, mask=full16)
                code16_p0 = reg.vadds(
                    reg.vmaxs(exponent16_p0, scale_shift_p0, mask=full16), -scale_shift_p0 & 65535, mask=full16
                )
                code16_p0 = reg.vdups(255, dtypes.uint16, mask=invalid16_p0, mode="merging", merge=code16_p0)
                r16_p0 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16_p0, mask=full16),
                    7,
                    mask=reg.vnes(code16_p0, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift_p0 = 15 if const_expr(self.mantissa == 2) else 8
                active16_p0 = reg.vgts(maxima16_p0, scale_shift_p0 << 7 | 96, mask=full16)
                code16_p0 = reg.vshr(
                    reg.vadds(maxima16_p0, 31 - (scale_shift_p0 << 7) & 65535, mask=full16), 7, mask=active16_p0
                )
                code16_p0 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16_p0, 32640, mask=full16), mode="merging", merge=code16_p0
                )
                r16_p0 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16_p0, mask=full16), 7, mask=full16
                )
            else:
                maxima_p0 = reg.vshl(reg.vunpack(maxima16_p0, dtypes.uint32), 16, mask=full)
                block_scales_p0, reciprocal_p0 = compute_scale(self, maxima_p0, full, False)
                bits16_p0 = reg.vshr(reg.vreinterpret(reciprocal_p0, dtypes.uint32), 16, mask=full)
                r16_p0 = reg.vpack(bits16_p0, dtypes.uint16, part="lower")
                code16_p0 = reg.vpack(block_scales_p0, dtypes.uint16, part="lower")
            if const_expr(self.scale_alg == 0):
                scale_shift_p1 = 15 if const_expr(self.mantissa == 2) else 8
                exponent16_p1 = reg.vshr(maxima16_p1, 7, mask=full16)
                invalid16_p1 = reg.vges(exponent16_p1, 255, mask=full16)
                code16_p1 = reg.vadds(
                    reg.vmaxs(exponent16_p1, scale_shift_p1, mask=full16), -scale_shift_p1 & 65535, mask=full16
                )
                code16_p1 = reg.vdups(255, dtypes.uint16, mask=invalid16_p1, mode="merging", merge=code16_p1)
                r16_p1 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16_p1, mask=full16),
                    7,
                    mask=reg.vnes(code16_p1, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift_p1 = 15 if const_expr(self.mantissa == 2) else 8
                active16_p1 = reg.vgts(maxima16_p1, scale_shift_p1 << 7 | 96, mask=full16)
                code16_p1 = reg.vshr(
                    reg.vadds(maxima16_p1, 31 - (scale_shift_p1 << 7) & 65535, mask=full16), 7, mask=active16_p1
                )
                code16_p1 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16_p1, 32640, mask=full16), mode="merging", merge=code16_p1
                )
                r16_p1 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16_p1, mask=full16), 7, mask=full16
                )
            else:
                maxima_p1 = reg.vshl(reg.vunpack(maxima16_p1, dtypes.uint32), 16, mask=full)
                block_scales_p1, reciprocal_p1 = compute_scale(self, maxima_p1, full, False)
                bits16_p1 = reg.vshr(reg.vreinterpret(reciprocal_p1, dtypes.uint32), 16, mask=full)
                r16_p1 = reg.vpack(bits16_p1, dtypes.uint16, part="lower")
                code16_p1 = reg.vpack(block_scales_p1, dtypes.uint16, part="lower")
            if const_expr(self.scale_alg == 0):
                scale_shift_p2 = 15 if const_expr(self.mantissa == 2) else 8
                exponent16_p2 = reg.vshr(maxima16_p2, 7, mask=full16)
                invalid16_p2 = reg.vges(exponent16_p2, 255, mask=full16)
                code16_p2 = reg.vadds(
                    reg.vmaxs(exponent16_p2, scale_shift_p2, mask=full16), -scale_shift_p2 & 65535, mask=full16
                )
                code16_p2 = reg.vdups(255, dtypes.uint16, mask=invalid16_p2, mode="merging", merge=code16_p2)
                r16_p2 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16_p2, mask=full16),
                    7,
                    mask=reg.vnes(code16_p2, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift_p2 = 15 if const_expr(self.mantissa == 2) else 8
                active16_p2 = reg.vgts(maxima16_p2, scale_shift_p2 << 7 | 96, mask=full16)
                code16_p2 = reg.vshr(
                    reg.vadds(maxima16_p2, 31 - (scale_shift_p2 << 7) & 65535, mask=full16), 7, mask=active16_p2
                )
                code16_p2 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16_p2, 32640, mask=full16), mode="merging", merge=code16_p2
                )
                r16_p2 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16_p2, mask=full16), 7, mask=full16
                )
            else:
                maxima_p2 = reg.vshl(reg.vunpack(maxima16_p2, dtypes.uint32), 16, mask=full)
                block_scales_p2, reciprocal_p2 = compute_scale(self, maxima_p2, full, False)
                bits16_p2 = reg.vshr(reg.vreinterpret(reciprocal_p2, dtypes.uint32), 16, mask=full)
                r16_p2 = reg.vpack(bits16_p2, dtypes.uint16, part="lower")
                code16_p2 = reg.vpack(block_scales_p2, dtypes.uint16, part="lower")
            if const_expr(self.scale_alg == 0):
                scale_shift_p3 = 15 if const_expr(self.mantissa == 2) else 8
                exponent16_p3 = reg.vshr(maxima16_p3, 7, mask=full16)
                invalid16_p3 = reg.vges(exponent16_p3, 255, mask=full16)
                code16_p3 = reg.vadds(
                    reg.vmaxs(exponent16_p3, scale_shift_p3, mask=full16), -scale_shift_p3 & 65535, mask=full16
                )
                code16_p3 = reg.vdups(255, dtypes.uint16, mask=invalid16_p3, mode="merging", merge=code16_p3)
                r16_p3 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16_p3, mask=full16),
                    7,
                    mask=reg.vnes(code16_p3, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift_p3 = 15 if const_expr(self.mantissa == 2) else 8
                active16_p3 = reg.vgts(maxima16_p3, scale_shift_p3 << 7 | 96, mask=full16)
                code16_p3 = reg.vshr(
                    reg.vadds(maxima16_p3, 31 - (scale_shift_p3 << 7) & 65535, mask=full16), 7, mask=active16_p3
                )
                code16_p3 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16_p3, 32640, mask=full16), mode="merging", merge=code16_p3
                )
                r16_p3 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16_p3, mask=full16), 7, mask=full16
                )
            else:
                maxima_p3 = reg.vshl(reg.vunpack(maxima16_p3, dtypes.uint32), 16, mask=full)
                block_scales_p3, reciprocal_p3 = compute_scale(self, maxima_p3, full, False)
                bits16_p3 = reg.vshr(reg.vreinterpret(reciprocal_p3, dtypes.uint32), 16, mask=full)
                r16_p3 = reg.vpack(bits16_p3, dtypes.uint16, part="lower")
                code16_p3 = reg.vpack(block_scales_p3, dtypes.uint16, part="lower")
            group_invalid = reg.vmax(group_invalid, code16_p0, mask=eight16)
            broadcast_p0 = reg.vreinterpret(reg.vgather_reg(r16_p0, index16), dtypes.bfloat16)
            av_p0 = reg.vmul(a_p0, broadcast_p0, mask=full16)
            bv_p0 = reg.vmul(b_p0, broadcast_p0, mask=full16)
            reg.vstore_interleave(x, offset_p0, av_p0, bv_p0)
            aa_norm_p0 = reg.vbitwise_and(reg.vreinterpret(av_p0, dtypes.uint16), abs_mask, mask=full16)
            ab_norm_p0 = reg.vbitwise_and(reg.vreinterpret(bv_p0, dtypes.uint16), abs_mask, mask=full16)
            panel_min_p0 = reg.vmin(aa_norm_p0, ab_norm_p0, mask=full16)
            group_min = reg.vmin(group_min, panel_min_p0, mask=full16)
            reg.vstore(scale16, part_p0 * 16, code16_p0, eight16)
            scale_bytes_p0 = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(code16_p0, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes_p0, scale_cursor)
            group_invalid = reg.vmax(group_invalid, code16_p1, mask=eight16)
            broadcast_p1 = reg.vreinterpret(reg.vgather_reg(r16_p1, index16), dtypes.bfloat16)
            av_p1 = reg.vmul(a_p1, broadcast_p1, mask=full16)
            bv_p1 = reg.vmul(b_p1, broadcast_p1, mask=full16)
            reg.vstore_interleave(x, offset_p1, av_p1, bv_p1)
            aa_norm_p1 = reg.vbitwise_and(reg.vreinterpret(av_p1, dtypes.uint16), abs_mask, mask=full16)
            ab_norm_p1 = reg.vbitwise_and(reg.vreinterpret(bv_p1, dtypes.uint16), abs_mask, mask=full16)
            panel_min_p1 = reg.vmin(aa_norm_p1, ab_norm_p1, mask=full16)
            group_min = reg.vmin(group_min, panel_min_p1, mask=full16)
            reg.vstore(scale16, part_p1 * 16, code16_p1, eight16)
            scale_bytes_p1 = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(code16_p1, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes_p1, scale_cursor)
            group_invalid = reg.vmax(group_invalid, code16_p2, mask=eight16)
            broadcast_p2 = reg.vreinterpret(reg.vgather_reg(r16_p2, index16), dtypes.bfloat16)
            av_p2 = reg.vmul(a_p2, broadcast_p2, mask=full16)
            bv_p2 = reg.vmul(b_p2, broadcast_p2, mask=full16)
            reg.vstore_interleave(x, offset_p2, av_p2, bv_p2)
            aa_norm_p2 = reg.vbitwise_and(reg.vreinterpret(av_p2, dtypes.uint16), abs_mask, mask=full16)
            ab_norm_p2 = reg.vbitwise_and(reg.vreinterpret(bv_p2, dtypes.uint16), abs_mask, mask=full16)
            panel_min_p2 = reg.vmin(aa_norm_p2, ab_norm_p2, mask=full16)
            group_min = reg.vmin(group_min, panel_min_p2, mask=full16)
            reg.vstore(scale16, part_p2 * 16, code16_p2, eight16)
            scale_bytes_p2 = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(code16_p2, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes_p2, scale_cursor)
            group_invalid = reg.vmax(group_invalid, code16_p3, mask=eight16)
            broadcast_p3 = reg.vreinterpret(reg.vgather_reg(r16_p3, index16), dtypes.bfloat16)
            av_p3 = reg.vmul(a_p3, broadcast_p3, mask=full16)
            bv_p3 = reg.vmul(b_p3, broadcast_p3, mask=full16)
            reg.vstore_interleave(x, offset_p3, av_p3, bv_p3)
            aa_norm_p3 = reg.vbitwise_and(reg.vreinterpret(av_p3, dtypes.uint16), abs_mask, mask=full16)
            ab_norm_p3 = reg.vbitwise_and(reg.vreinterpret(bv_p3, dtypes.uint16), abs_mask, mask=full16)
            panel_min_p3 = reg.vmin(aa_norm_p3, ab_norm_p3, mask=full16)
            group_min = reg.vmin(group_min, panel_min_p3, mask=full16)
            reg.vstore(scale16, part_p3 * 16, code16_p3, eight16)
            scale_bytes_p3 = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(code16_p3, dtypes.uint8), mask=scale_byte_mask
            )
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes_p3, scale_cursor)
            minimum = reg.vreduce_min(group_min, mask=full16)
            maximum_scale = reg.vreduce_max(group_invalid, mask=eight16)
            minimum = reg.vdups(
                0, dtypes.uint16, mask=reg.veqs(maximum_scale, 255, mask=full16), mode="merging", merge=minimum
            )
            reg.vstore_first(self.norm_flags, first4 * self.norm_flag_stride, minimum)
            tile_minimum = reg.vmin(tile_minimum, minimum, mask=one16)
        reg.vsqueeze_and_storeunalign_finalize(scales, 0, scale_cursor)
        if const_expr(self.bf16_compact_mode != BF16_COMPACT_TILE_KI_ELEMENTS):
            reg.vstore_first(self.norm_flags, self.norm_tile_flag, tile_minimum)
        # Publish normalized x, scale16 and norm_flags before the scalar classifier
        # reads minima and subsequent VFs reload x/scales from UB.
        reg.vmem_bar("vst_vld")


@jit
def _normalize_bf16_four_panels_batched(self, x, scales, parts, scale_factor):
    """Normalize four panels per iteration and publish group minima/scale bytes."""
    scale16 = reinterpret(self.scale32, dtypes.uint16, shape=(1, self.tile_elements // 16))
    with vf(mode="simd"):
        full = reg.full_mask()
        full16 = reg.full_mask(elem_bits=16)
        eight16 = reg.update_mask(8, elem_bits=16)[0]
        abs_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
        index16 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
        one16 = reg.update_mask(1, elem_bits=16)[0]
        tile_minimum = reg.vdups(65535, dtypes.uint16, mask=one16)
        for first4 in range(0, parts, 4):
            group_min = reg.vdups(65535, dtypes.uint16, mask=full16)
            group_min_01 = group_min
            group_min_23 = group_min
            part_p0 = first4 + 0
            offset_p0 = part_p0 * 256
            a_p0, b_p0 = reg.vload_deinterleave(x, offset_p0)
            aa_p0 = reg.vbitwise_and(reg.vreinterpret(a_p0, dtypes.uint16), abs_mask, mask=full16)
            ab_p0 = reg.vbitwise_and(reg.vreinterpret(b_p0, dtypes.uint16), abs_mask, mask=full16)
            maxima16_p0 = reg.vreduce_max_datablock(reg.vmax(aa_p0, ab_p0, mask=full16), mask=full16)
            part_p1 = first4 + 1
            offset_p1 = part_p1 * 256
            a_p1, b_p1 = reg.vload_deinterleave(x, offset_p1)
            aa_p1 = reg.vbitwise_and(reg.vreinterpret(a_p1, dtypes.uint16), abs_mask, mask=full16)
            ab_p1 = reg.vbitwise_and(reg.vreinterpret(b_p1, dtypes.uint16), abs_mask, mask=full16)
            maxima16_p1 = reg.vreduce_max_datablock(reg.vmax(aa_p1, ab_p1, mask=full16), mask=full16)
            part_p2 = first4 + 2
            offset_p2 = part_p2 * 256
            a_p2, b_p2 = reg.vload_deinterleave(x, offset_p2)
            aa_p2 = reg.vbitwise_and(reg.vreinterpret(a_p2, dtypes.uint16), abs_mask, mask=full16)
            ab_p2 = reg.vbitwise_and(reg.vreinterpret(b_p2, dtypes.uint16), abs_mask, mask=full16)
            maxima16_p2 = reg.vreduce_max_datablock(reg.vmax(aa_p2, ab_p2, mask=full16), mask=full16)
            part_p3 = first4 + 3
            offset_p3 = part_p3 * 256
            a_p3, b_p3 = reg.vload_deinterleave(x, offset_p3)
            aa_p3 = reg.vbitwise_and(reg.vreinterpret(a_p3, dtypes.uint16), abs_mask, mask=full16)
            ab_p3 = reg.vbitwise_and(reg.vreinterpret(b_p3, dtypes.uint16), abs_mask, mask=full16)
            maxima16_p3 = reg.vreduce_max_datablock(reg.vmax(aa_p3, ab_p3, mask=full16), mask=full16)
            if const_expr(self.scale_alg == 0):
                scale_shift_p0 = 15 if const_expr(self.mantissa == 2) else 8
                exponent16_p0 = reg.vshr(maxima16_p0, 7, mask=full16)
                invalid16_p0 = reg.vges(exponent16_p0, 255, mask=full16)
                code16_p0 = reg.vadds(
                    reg.vmaxs(exponent16_p0, scale_shift_p0, mask=full16), -scale_shift_p0 & 65535, mask=full16
                )
                code16_p0 = reg.vdups(255, dtypes.uint16, mask=invalid16_p0, mode="merging", merge=code16_p0)
                r16_p0 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16_p0, mask=full16),
                    7,
                    mask=reg.vnes(code16_p0, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift_p0 = 15 if const_expr(self.mantissa == 2) else 8
                active16_p0 = reg.vgts(maxima16_p0, scale_shift_p0 << 7 | 96, mask=full16)
                code16_p0 = reg.vshr(
                    reg.vadds(maxima16_p0, 31 - (scale_shift_p0 << 7) & 65535, mask=full16), 7, mask=active16_p0
                )
                code16_p0 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16_p0, 32640, mask=full16), mode="merging", merge=code16_p0
                )
                r16_p0 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16_p0, mask=full16), 7, mask=full16
                )
            else:
                maxima_p0 = reg.vshl(reg.vunpack(maxima16_p0, dtypes.uint32), 16, mask=full)
                block_scales_p0, reciprocal_p0 = compute_scale(self, maxima_p0, full, False)
                bits16_p0 = reg.vshr(reg.vreinterpret(reciprocal_p0, dtypes.uint32), 16, mask=full)
                r16_p0 = reg.vpack(bits16_p0, dtypes.uint16, part="lower")
                code16_p0 = reg.vpack(block_scales_p0, dtypes.uint16, part="lower")
            if const_expr(self.scale_alg == 0):
                scale_shift_p1 = 15 if const_expr(self.mantissa == 2) else 8
                exponent16_p1 = reg.vshr(maxima16_p1, 7, mask=full16)
                invalid16_p1 = reg.vges(exponent16_p1, 255, mask=full16)
                code16_p1 = reg.vadds(
                    reg.vmaxs(exponent16_p1, scale_shift_p1, mask=full16), -scale_shift_p1 & 65535, mask=full16
                )
                code16_p1 = reg.vdups(255, dtypes.uint16, mask=invalid16_p1, mode="merging", merge=code16_p1)
                r16_p1 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16_p1, mask=full16),
                    7,
                    mask=reg.vnes(code16_p1, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift_p1 = 15 if const_expr(self.mantissa == 2) else 8
                active16_p1 = reg.vgts(maxima16_p1, scale_shift_p1 << 7 | 96, mask=full16)
                code16_p1 = reg.vshr(
                    reg.vadds(maxima16_p1, 31 - (scale_shift_p1 << 7) & 65535, mask=full16), 7, mask=active16_p1
                )
                code16_p1 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16_p1, 32640, mask=full16), mode="merging", merge=code16_p1
                )
                r16_p1 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16_p1, mask=full16), 7, mask=full16
                )
            else:
                maxima_p1 = reg.vshl(reg.vunpack(maxima16_p1, dtypes.uint32), 16, mask=full)
                block_scales_p1, reciprocal_p1 = compute_scale(self, maxima_p1, full, False)
                bits16_p1 = reg.vshr(reg.vreinterpret(reciprocal_p1, dtypes.uint32), 16, mask=full)
                r16_p1 = reg.vpack(bits16_p1, dtypes.uint16, part="lower")
                code16_p1 = reg.vpack(block_scales_p1, dtypes.uint16, part="lower")
            if const_expr(self.scale_alg == 0):
                scale_shift_p2 = 15 if const_expr(self.mantissa == 2) else 8
                exponent16_p2 = reg.vshr(maxima16_p2, 7, mask=full16)
                invalid16_p2 = reg.vges(exponent16_p2, 255, mask=full16)
                code16_p2 = reg.vadds(
                    reg.vmaxs(exponent16_p2, scale_shift_p2, mask=full16), -scale_shift_p2 & 65535, mask=full16
                )
                code16_p2 = reg.vdups(255, dtypes.uint16, mask=invalid16_p2, mode="merging", merge=code16_p2)
                r16_p2 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16_p2, mask=full16),
                    7,
                    mask=reg.vnes(code16_p2, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift_p2 = 15 if const_expr(self.mantissa == 2) else 8
                active16_p2 = reg.vgts(maxima16_p2, scale_shift_p2 << 7 | 96, mask=full16)
                code16_p2 = reg.vshr(
                    reg.vadds(maxima16_p2, 31 - (scale_shift_p2 << 7) & 65535, mask=full16), 7, mask=active16_p2
                )
                code16_p2 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16_p2, 32640, mask=full16), mode="merging", merge=code16_p2
                )
                r16_p2 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16_p2, mask=full16), 7, mask=full16
                )
            else:
                maxima_p2 = reg.vshl(reg.vunpack(maxima16_p2, dtypes.uint32), 16, mask=full)
                block_scales_p2, reciprocal_p2 = compute_scale(self, maxima_p2, full, False)
                bits16_p2 = reg.vshr(reg.vreinterpret(reciprocal_p2, dtypes.uint32), 16, mask=full)
                r16_p2 = reg.vpack(bits16_p2, dtypes.uint16, part="lower")
                code16_p2 = reg.vpack(block_scales_p2, dtypes.uint16, part="lower")
            if const_expr(self.scale_alg == 0):
                scale_shift_p3 = 15 if const_expr(self.mantissa == 2) else 8
                exponent16_p3 = reg.vshr(maxima16_p3, 7, mask=full16)
                invalid16_p3 = reg.vges(exponent16_p3, 255, mask=full16)
                code16_p3 = reg.vadds(
                    reg.vmaxs(exponent16_p3, scale_shift_p3, mask=full16), -scale_shift_p3 & 65535, mask=full16
                )
                code16_p3 = reg.vdups(255, dtypes.uint16, mask=invalid16_p3, mode="merging", merge=code16_p3)
                r16_p3 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16_p3, mask=full16),
                    7,
                    mask=reg.vnes(code16_p3, 0, mask=full16),
                )
            elif const_expr(self.scale_alg == 1 and self.max_low_bound_bits == 0):
                scale_shift_p3 = 15 if const_expr(self.mantissa == 2) else 8
                active16_p3 = reg.vgts(maxima16_p3, scale_shift_p3 << 7 | 96, mask=full16)
                code16_p3 = reg.vshr(
                    reg.vadds(maxima16_p3, 31 - (scale_shift_p3 << 7) & 65535, mask=full16), 7, mask=active16_p3
                )
                code16_p3 = reg.vdups(
                    255, dtypes.uint16, mask=reg.vges(maxima16_p3, 32640, mask=full16), mode="merging", merge=code16_p3
                )
                r16_p3 = reg.vshl(
                    reg.vsub(reg.vdups(254, dtypes.uint16, mask=full16), code16_p3, mask=full16), 7, mask=full16
                )
            else:
                maxima_p3 = reg.vshl(reg.vunpack(maxima16_p3, dtypes.uint32), 16, mask=full)
                block_scales_p3, reciprocal_p3 = compute_scale(self, maxima_p3, full, False)
                bits16_p3 = reg.vshr(reg.vreinterpret(reciprocal_p3, dtypes.uint32), 16, mask=full)
                r16_p3 = reg.vpack(bits16_p3, dtypes.uint16, part="lower")
                code16_p3 = reg.vpack(block_scales_p3, dtypes.uint16, part="lower")
            group_invalid = code16_p0
            broadcast_p0 = reg.vreinterpret(reg.vgather_reg(r16_p0, index16), dtypes.bfloat16)
            av_p0 = reg.vmul(a_p0, broadcast_p0, mask=full16)
            bv_p0 = reg.vmul(b_p0, broadcast_p0, mask=full16)
            reg.vstore_interleave(x, offset_p0, av_p0, bv_p0)
            aa_norm_p0 = reg.vbitwise_and(reg.vreinterpret(av_p0, dtypes.uint16), abs_mask, mask=full16)
            ab_norm_p0 = reg.vbitwise_and(reg.vreinterpret(bv_p0, dtypes.uint16), abs_mask, mask=full16)
            panel_min_p0 = reg.vmin(aa_norm_p0, ab_norm_p0, mask=full16)
            if const_expr(self.scale_alg == 0 and self.mantissa == 3):
                group_min_01 = panel_min_p0
            else:
                group_min = reg.vmin(group_min, panel_min_p0, mask=full16)
            reg.vstore(scale16, part_p0 * 16, code16_p0, eight16)
            group_invalid = reg.vmax(group_invalid, code16_p1, mask=eight16)
            broadcast_p1 = reg.vreinterpret(reg.vgather_reg(r16_p1, index16), dtypes.bfloat16)
            av_p1 = reg.vmul(a_p1, broadcast_p1, mask=full16)
            bv_p1 = reg.vmul(b_p1, broadcast_p1, mask=full16)
            reg.vstore_interleave(x, offset_p1, av_p1, bv_p1)
            aa_norm_p1 = reg.vbitwise_and(reg.vreinterpret(av_p1, dtypes.uint16), abs_mask, mask=full16)
            ab_norm_p1 = reg.vbitwise_and(reg.vreinterpret(bv_p1, dtypes.uint16), abs_mask, mask=full16)
            panel_min_p1 = reg.vmin(aa_norm_p1, ab_norm_p1, mask=full16)
            if const_expr(self.scale_alg == 0 and self.mantissa == 3):
                group_min_01 = reg.vmin(group_min_01, panel_min_p1, mask=full16)
            else:
                group_min = reg.vmin(group_min, panel_min_p1, mask=full16)
            reg.vstore(scale16, part_p1 * 16, code16_p1, eight16)
            group_invalid = reg.vmax(group_invalid, code16_p2, mask=eight16)
            broadcast_p2 = reg.vreinterpret(reg.vgather_reg(r16_p2, index16), dtypes.bfloat16)
            av_p2 = reg.vmul(a_p2, broadcast_p2, mask=full16)
            bv_p2 = reg.vmul(b_p2, broadcast_p2, mask=full16)
            reg.vstore_interleave(x, offset_p2, av_p2, bv_p2)
            aa_norm_p2 = reg.vbitwise_and(reg.vreinterpret(av_p2, dtypes.uint16), abs_mask, mask=full16)
            ab_norm_p2 = reg.vbitwise_and(reg.vreinterpret(bv_p2, dtypes.uint16), abs_mask, mask=full16)
            panel_min_p2 = reg.vmin(aa_norm_p2, ab_norm_p2, mask=full16)
            if const_expr(self.scale_alg == 0 and self.mantissa == 3):
                group_min_23 = panel_min_p2
            else:
                group_min = reg.vmin(group_min, panel_min_p2, mask=full16)
            reg.vstore(scale16, part_p2 * 16, code16_p2, eight16)
            group_invalid = reg.vmax(group_invalid, code16_p3, mask=eight16)
            broadcast_p3 = reg.vreinterpret(reg.vgather_reg(r16_p3, index16), dtypes.bfloat16)
            av_p3 = reg.vmul(a_p3, broadcast_p3, mask=full16)
            bv_p3 = reg.vmul(b_p3, broadcast_p3, mask=full16)
            reg.vstore_interleave(x, offset_p3, av_p3, bv_p3)
            aa_norm_p3 = reg.vbitwise_and(reg.vreinterpret(av_p3, dtypes.uint16), abs_mask, mask=full16)
            ab_norm_p3 = reg.vbitwise_and(reg.vreinterpret(bv_p3, dtypes.uint16), abs_mask, mask=full16)
            panel_min_p3 = reg.vmin(aa_norm_p3, ab_norm_p3, mask=full16)
            if const_expr(self.scale_alg == 0 and self.mantissa == 3):
                group_min_23 = reg.vmin(group_min_23, panel_min_p3, mask=full16)
            else:
                group_min = reg.vmin(group_min, panel_min_p3, mask=full16)
            reg.vstore(scale16, part_p3 * 16, code16_p3, eight16)
            if const_expr(self.scale_alg == 0 and self.mantissa == 3):
                minimum = reg.vreduce_min(reg.vmin(group_min_01, group_min_23, mask=full16), mask=full16)
            else:
                minimum = reg.vreduce_min(group_min, mask=full16)
            maximum_scale = reg.vreduce_max(group_invalid, mask=eight16)
            minimum = reg.vdups(
                0, dtypes.uint16, mask=reg.veqs(maximum_scale, 255, mask=full16), mode="merging", merge=minimum
            )
            reg.vstore_first(self.norm_flags, first4 * self.norm_flag_stride, minimum)
            tile_minimum = reg.vmin(tile_minimum, minimum, mask=one16)
        # Publish scale16 stores before gathering each panel's first eight codes
        # into aligned scale-byte output stores.
        reg.vmem_bar("vst_vld")
        pack_ids16 = reg.varange(0, dtypes.uint16)
        panel_base16 = reg.vshl(reg.vshr(pack_ids16, 3, mask=full16), 4, mask=full16)
        panel_lane16 = reg.vbitwise_and(pack_ids16, reg.vdups(7, dtypes.uint16, mask=full16), mask=full16)
        pack_index16 = reg.vbitwise_or(panel_base16, panel_lane16, mask=full16)
        for scale_offset in range(0, parts * 8, 128):
            pack_mask = reg.update_mask(parts * 8 - scale_offset, elem_bits=16)[0]
            indices = reg.vadds(pack_index16, scale_offset * 2, mask=full16)
            packed_codes = reg.vgather(scale16, indices, mask=pack_mask)
            reg.vstore_pack(scales, scale_offset, packed_codes, pack_mask, pack_mode=reg.PackMode.B16_TO_B8)
        if const_expr(self.bf16_compact_mode != BF16_COMPACT_TILE_KI_ELEMENTS):
            reg.vstore_first(self.norm_flags, self.norm_tile_flag, tile_minimum)
        # Publish normalized x, scale bytes and norm_flags before the scalar
        # classifier reads minima and subsequent VFs reload x/scales from UB.
        reg.vmem_bar("vst_vld")


@jit
def _encode_bf16_adjacent_eight(self, random, x, y, first, flag_0, flag_1, flag_2, flag_3):
    """Use one VF for four eight-panel classes; caller excludes general groups."""
    sign_source = reinterpret(x, dtypes.uint8, shape=(1, self.tile_elements * 2))
    random_bytes = reinterpret(random.buffer, dtypes.uint8, shape=(1, random.capacity_bytes))
    with vf(mode="simd"):
        if flag_0 >= (128 - self.bias) * 128:
            _encode_bf16_clean_in_vf(self, x, y, first + 0, first + 8, sign_source, random_bytes)
        elif flag_0 >= BF16_HALF_Q7_MIN_NORMALIZED_BITS:
            _encode_bf16_half_q7_in_vf_random7(self, x, y, first + 0, first + 8, sign_source, random_bytes)
        else:
            _encode_bf16_middle_in_vf(self, x, y, first + 0, first + 8, sign_source, random_bytes)
        if flag_1 >= (128 - self.bias) * 128:
            _encode_bf16_clean_in_vf(self, x, y, first + 8, first + 16, sign_source, random_bytes)
        elif flag_1 >= BF16_HALF_Q7_MIN_NORMALIZED_BITS:
            _encode_bf16_half_q7_in_vf_random7(self, x, y, first + 8, first + 16, sign_source, random_bytes)
        else:
            _encode_bf16_middle_in_vf(self, x, y, first + 8, first + 16, sign_source, random_bytes)
        if flag_2 >= (128 - self.bias) * 128:
            _encode_bf16_clean_in_vf(self, x, y, first + 16, first + 24, sign_source, random_bytes)
        elif flag_2 >= BF16_HALF_Q7_MIN_NORMALIZED_BITS:
            _encode_bf16_half_q7_in_vf_random7(self, x, y, first + 16, first + 24, sign_source, random_bytes)
        else:
            _encode_bf16_middle_in_vf(self, x, y, first + 16, first + 24, sign_source, random_bytes)
        if flag_3 >= (128 - self.bias) * 128:
            _encode_bf16_clean_in_vf(self, x, y, first + 24, first + 32, sign_source, random_bytes)
        elif flag_3 >= BF16_HALF_Q7_MIN_NORMALIZED_BITS:
            _encode_bf16_half_q7_in_vf_random7(self, x, y, first + 24, first + 32, sign_source, random_bytes)
        else:
            _encode_bf16_middle_in_vf(self, x, y, first + 24, first + 32, sign_source, random_bytes)

        # Finish encoded byte stores before DMA reads or the next tile reuses the output slot.
        reg.vmem_bar("vst_vld")


@jit
def _encode_bf16_classified_bundles_8(self, random, x, y, parts, scales):
    """Retain exact eight-panel class selection without the whole-tile shortcut."""
    for bundle in range(0, parts, 32):
        flag_0 = dtypes.int64(self.norm_flags[0, bundle * self.norm_flag_stride])
        flag_1 = dtypes.int64(self.norm_flags[0, (bundle + 8) * self.norm_flag_stride])
        flag_2 = dtypes.int64(self.norm_flags[0, (bundle + 16) * self.norm_flag_stride])
        flag_3 = dtypes.int64(self.norm_flags[0, (bundle + 24) * self.norm_flag_stride])
        minimum = flag_0
        if flag_1 < minimum:
            minimum = flag_1
        if flag_2 < minimum:
            minimum = flag_2
        if flag_3 < minimum:
            minimum = flag_3
        if minimum >= (128 - self.bias) * 128:
            _emit_nibble_clean_part(self, random, x, y, bundle, bundle + 32)
        elif minimum >= (123 - self.bias - self.mantissa) * 128:
            _encode_bf16_adjacent_eight(self, random, x, y, bundle, flag_0, flag_1, flag_2, flag_3)
        else:
            for subfirst in range(bundle, bundle + 32, 8):
                flag = dtypes.int64(self.norm_flags[0, subfirst * self.norm_flag_stride])
                if flag >= (128 - self.bias) * 128:
                    _emit_nibble_clean_part(self, random, x, y, subfirst, subfirst + 8)
                elif flag >= BF16_HALF_Q7_MIN_NORMALIZED_BITS:
                    _emit_nibble_half_q7_part(self, random, x, y, subfirst, subfirst + 8)
                elif flag >= (123 - self.bias - self.mantissa) * 128:
                    _emit_nibble_middle_part(self, random, x, y, subfirst, subfirst + 8)
                else:
                    _emit_nibble_general_part(self, random, x, y, subfirst, subfirst + 8, scales)


@jit
def _encode_bf16_classified_bundles_16(self, random, x, y, parts, scales):
    """Consume one exact minimum per 16 panels on the complete-tile route."""
    if dtypes.int64(self.norm_flags[0, self.norm_tile_flag]) >= (128 - self.bias) * 128:
        _emit_nibble_clean_part(self, random, x, y, 0, parts)
    else:
        for bundle in range(0, parts, 32):
            flag_0 = dtypes.int64(self.norm_flags[0, bundle * self.norm_flag_stride])
            flag_1 = dtypes.int64(self.norm_flags[0, (bundle + 16) * self.norm_flag_stride])
            bundle_min = flag_0
            if flag_1 < bundle_min:
                bundle_min = flag_1
            if bundle_min >= (123 - self.bias - self.mantissa) * 128:
                _encode_bf16_adjacent_bundles(self, random, x, y, bundle, flag_0, flag_1)
            else:
                for first in range(bundle, bundle + 32, 16):
                    flag_min = dtypes.int64(self.norm_flags[0, first * self.norm_flag_stride])
                    if flag_min >= (128 - self.bias) * 128:
                        _emit_nibble_clean_part(self, random, x, y, first, first + 16)
                    elif flag_min >= BF16_HALF_Q7_MIN_NORMALIZED_BITS:
                        _emit_nibble_half_q7_part(self, random, x, y, first, first + 16)
                    elif flag_min >= (123 - self.bias - self.mantissa) * 128:
                        _emit_nibble_middle_part(self, random, x, y, first, first + 16)
                    else:
                        _emit_nibble_general_part(self, random, x, y, first, first + 16, scales)


@jit
def _encode_bf16_classified_bundles(self, random, x, y, parts):
    """Consume normalized input and minima; select exact SR math per bundle."""
    if (
        const_expr(self.bf16_compact_mode != BF16_COMPACT_TILE_KI_ELEMENTS)
        and dtypes.int64(self.norm_flags[0, self.norm_tile_flag]) >= (128 - self.bias) * 128
    ):
        _emit_nibble_clean_part(self, random, x, y, 0, parts)
    else:
        for bundle in range(0, parts, 32):
            if bundle + 32 <= parts:
                flag_0 = dtypes.int64(self.norm_flags[0, (bundle + 0) * self.norm_flag_stride])
                for cursor in range(bundle + 4, bundle + 16, 4):
                    value = dtypes.int64(self.norm_flags[0, cursor * self.norm_flag_stride])
                    if value < flag_0:
                        flag_0 = value
                flag_1 = dtypes.int64(self.norm_flags[0, (bundle + 16) * self.norm_flag_stride])
                for cursor in range(bundle + 20, bundle + 32, 4):
                    value = dtypes.int64(self.norm_flags[0, cursor * self.norm_flag_stride])
                    if value < flag_1:
                        flag_1 = value
                bundle_min = flag_0
                if flag_1 < bundle_min:
                    bundle_min = flag_1
                if bundle_min >= (123 - self.bias - self.mantissa) * 128:
                    _encode_bf16_adjacent_bundles(self, random, x, y, bundle, flag_0, flag_1)
                else:
                    for first in range(bundle, bundle + 32, 16):
                        stop = first + 16
                        if stop > parts:
                            stop = parts
                        flag_min = dtypes.int64(self.norm_flags[0, first * self.norm_flag_stride])
                        for cursor in range(first + 4, stop, 4):
                            flag = dtypes.int64(self.norm_flags[0, cursor * self.norm_flag_stride])
                            if flag < flag_min:
                                flag_min = flag
                        if flag_min >= (128 - self.bias) * 128:
                            _emit_nibble_clean_part(self, random, x, y, first, stop)
                        elif flag_min >= BF16_HALF_Q7_MIN_NORMALIZED_BITS:
                            _emit_nibble_half_q7_part(self, random, x, y, first, stop)
                        elif flag_min >= (123 - self.bias - self.mantissa) * 128:
                            _emit_nibble_middle_part(self, random, x, y, first, stop)
                        else:
                            _emit_nibble_general_part(self, random, x, y, first, stop)
            else:
                for first in range(bundle, parts, 16):
                    stop = first + 16
                    if stop > parts:
                        stop = parts
                    flag_min = dtypes.int64(self.norm_flags[0, first * self.norm_flag_stride])
                    for cursor in range(first + 4, stop, 4):
                        flag = dtypes.int64(self.norm_flags[0, cursor * self.norm_flag_stride])
                        if flag < flag_min:
                            flag_min = flag
                    if flag_min >= (128 - self.bias) * 128:
                        _emit_nibble_clean_part(self, random, x, y, first, stop)
                    elif flag_min >= BF16_HALF_Q7_MIN_NORMALIZED_BITS:
                        _emit_nibble_half_q7_part(self, random, x, y, first, stop)
                    elif flag_min >= (123 - self.bias - self.mantissa) * 128:
                        _emit_nibble_middle_part(self, random, x, y, first, stop)
                    else:
                        _emit_nibble_general_part(self, random, x, y, first, stop)


def _select_bf16_compact_four_panel_policy(
    numel, dtype, dst_dtype, scale_alg, bound_bits, layout, quant_length, blocks, ub_bytes
):
    if (
        dtype != dtypes.bfloat16
        or dst_dtype != torch.float8_e4m3fn
        or scale_alg != 0
        or (bound_bits != 0)
        or (layout is not Layout.TAIL)
        or quant_length % PACKED_K_ALIGNMENT_ELEMENTS
    ):
        return False
    # The 36K path needs enough work to amortize its per-core tile setup.
    # Tail-axis K is a runtime stride; it is not a performance whitelist.
    if numel < BF16_COMPACT_MIN_ELEMENTS or numel < blocks * BF16_COMPACT_MIN_ELEMENTS_PER_AIV:
        return False
    parent = _make_tiling(dtype, ub_bytes, tail_axis=True)
    compact = _make_tiling(dtype, ub_bytes, tail_axis=True, bf16_compact_mode=BF16_COMPACT_TILE_KI_ELEMENTS)
    if compact.tile_elements != BF16_COMPACT_TILE_KI_ELEMENTS * 1024:
        return False
    if (
        numel > blocks * parent.tile_elements
        and (numel % (blocks * parent.tile_elements) == 0)
        and (numel % (blocks * compact.tile_elements) != 0)
    ):
        return False
    return True


@jit
def _produce_bf16_uniform_tail_simd(
    random, group_count, group_low, uniform_initial, next_key1, uniform_x0, uniform_x2, third_low, key0, key1
):
    with vf(mode="simd"):
        full = reg.full_mask()
        ids = reg.varange(0, dtypes.uint32)
        m0 = reg.vdups(PHILOX_M0, dtypes.uint32, mask=full)
        m1 = reg.vdups(PHILOX_M1, dtypes.uint32, mask=full)
        uinitial = reg.vdups(uniform_initial, dtypes.uint32, mask=full)
        uk1 = reg.vdups(next_key1, dtypes.uint32, mask=full)
        ux0 = reg.vdups(uniform_x0, dtypes.uint32, mask=full)
        ux2 = reg.vdups(uniform_x2, dtypes.uint32, mask=full)
        third = reg.vdups(third_low, dtypes.uint32, mask=full)
        for base in range(0, group_count, 64):
            sub0 = reg.vbitwise_and(
                reg.vadds(ids, dtypes.uint32(group_low) + dtypes.uint32(base) + dtypes.uint32(0), mask=full),
                reg.vdups(16383, dtypes.uint32, mask=full),
                mask=full,
            )
            first0 = reg.vbitwise_xor(sub0, uinitial, mask=full)
            varlo0, varhi0 = reg.vmull(first0, m0, dtypes.uint32, mask=full)
            second0 = reg.vbitwise_xor(varhi0, uk1, mask=full)
            p1lo0, p1hi0 = reg.vmull(second0, m1, dtypes.uint32, mask=full)
            s0_0 = reg.vbitwise_xor(p1hi0, ux0, mask=full)
            s0_1 = p1lo0
            s0_2 = reg.vbitwise_xor(varlo0, ux2, mask=full)
            s0_3 = third
            k0 = reg.vdups(dtypes.uint32(key0) + dtypes.uint32((3 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full)
            k1 = reg.vdups(dtypes.uint32(key1) + dtypes.uint32((3 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full)
            lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
            lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
            s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
            s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
            s0_1 = lo1_0
            s0_3 = lo0_0
            k0 = reg.vdups(dtypes.uint32(key0) + dtypes.uint32((4 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full)
            k1 = reg.vdups(dtypes.uint32(key1) + dtypes.uint32((4 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full)
            lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
            lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
            s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
            s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
            s0_1 = lo1_0
            s0_3 = lo0_0
            k0 = reg.vdups(dtypes.uint32(key0) + dtypes.uint32((5 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full)
            k1 = reg.vdups(dtypes.uint32(key1) + dtypes.uint32((5 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full)
            lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
            lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
            s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
            s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
            s0_1 = lo1_0
            s0_3 = lo0_0
            k0 = reg.vdups(dtypes.uint32(key0) + dtypes.uint32((6 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full)
            k1 = reg.vdups(dtypes.uint32(key1) + dtypes.uint32((6 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full)
            lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
            lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
            s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
            s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
            s0_1 = lo1_0
            s0_3 = lo0_0
            k0 = reg.vdups(dtypes.uint32(key0) + dtypes.uint32((7 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full)
            k1 = reg.vdups(dtypes.uint32(key1) + dtypes.uint32((7 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full)
            lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
            lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
            s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
            s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
            s0_1 = lo1_0
            s0_3 = lo0_0
            k0 = reg.vdups(dtypes.uint32(key0) + dtypes.uint32((8 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full)
            k1 = reg.vdups(dtypes.uint32(key1) + dtypes.uint32((8 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full)
            lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
            lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
            s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
            s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
            s0_1 = lo1_0
            s0_3 = lo0_0
            k0 = reg.vdups(dtypes.uint32(key0) + dtypes.uint32((9 * PHILOX_W0) & UINT32_MASK), dtypes.uint32, mask=full)
            k1 = reg.vdups(dtypes.uint32(key1) + dtypes.uint32((9 * PHILOX_W1) & UINT32_MASK), dtypes.uint32, mask=full)
            lo0_0, hi0_0 = reg.vmull(s0_0, m0, dtypes.uint32, mask=full)
            lo1_0, hi1_0 = reg.vmull(s0_2, m1, dtypes.uint32, mask=full)
            s0_0 = reg.vbitwise_xor(hi1_0, reg.vbitwise_xor(s0_1, k0, mask=full), mask=full)
            s0_2 = reg.vbitwise_xor(hi0_0, reg.vbitwise_xor(s0_3, k1, mask=full), mask=full)
            s0_1 = lo1_0
            s0_3 = lo0_0
            _store_random_words_aos(random.buffer, (base + 0) * 4, s0_0, s0_1, s0_2, s0_3, full)
        # Publish Philox scratch stores before SR VFs gather random words.
        reg.vmem_bar("vst_vld")


def _bf16_four_panel_legal(dtype, dst_dtype, scale_alg, bound_bits, layout, quant_length):
    return (
        dtype == dtypes.bfloat16
        and dst_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        and scale_alg in (0, 1)
        and bound_bits == 0
        and layout is Layout.TAIL
        and quant_length % PACKED_K_ALIGNMENT_ELEMENTS == 0
    )


def _select_bf16_four_panel_policy(numel, dtype, dst_dtype, scale_alg, bound_bits, layout, quant_length, blocks):
    if not _bf16_four_panel_legal(dtype, dst_dtype, scale_alg, bound_bits, layout, quant_length):
        return False
    # Preserve the existing partial-wave work threshold. Exact complete waves
    # below this threshold are admitted separately by the full-tile guard.
    min_work_per_aiv = (
        BF16_E5_FLOOR_MIN_ELEMENTS_PER_AIV
        if dst_dtype == torch.float8_e5m2 and scale_alg == 0
        else BF16_FOUR_PANEL_MIN_ELEMENTS_PER_AIV
    )
    return numel >= blocks * min_work_per_aiv


def _select_bf16_full_tiles_policy(
    numel, dtype, dst_dtype, scale_alg, bound_bits, plan, ub_bytes, compact_mode, four_panel
):
    if (
        dtype != dtypes.bfloat16
        or dst_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
        or scale_alg not in (0, 1)
        or bound_bits != 0
        or plan.layout is not Layout.TAIL
        or not four_panel
        or compact_mode != 0
        or plan.quant_length % PACKED_K_ALIGNMENT_ELEMENTS
        or plan.kernel_shape != (1, numel)
    ):
        return False
    tiling = _make_tiling(dtype, ub_bytes, tail_axis=True)
    tile = tiling.tile_elements
    # Complete equal waves retain the existing round-robin addresses and order.
    if tile != BF16_TILE_LIMIT_ELEMENTS or numel % (plan.blocks * tile):
        return False
    flags, reciprocals = _bf16_normalization_slots(
        tile, full_tiles=True, e5m2=dst_dtype == torch.float8_e5m2, scale_alg=scale_alg
    )
    actual_bytes = tiling.buffer_bytes - _bf16_guard_buffer_bytes(tile) + 2 * (flags + reciprocals)
    return actual_bytes <= ub_bytes - SIMT_UB_RESERVE


def _select_bf16_lower_wave_four_panel_policy(
    numel, dtype, dst_dtype, scale_alg, bound_bits, plan, ub_bytes, compact_mode
):
    # All four BF16 format/algorithm routes use the same complete-wave floor.
    # The existing higher-work selector handles later waves and partial waves.
    # Reuse the full-tile guard to preserve resource and layout fallbacks.
    if not _select_bf16_full_tiles_policy(
        numel, dtype, dst_dtype, scale_alg, bound_bits, plan, ub_bytes, compact_mode, True
    ):
        return False
    return numel <= plan.blocks * BF16_TILE_LIMIT_ELEMENTS * BF16_FULL_TILE_MAX_LOWER_WAVES


@jit
def _run_bf16_full_tiles(self, x, y, scales, quant_length, seed, block_offset):
    # Host guard proves flattened complete waves; buffers and random mapping are unchanged.
    tile_elements = self.quantization.tile_elements
    tasks = x.shape[1] // tile_elements
    for task in range(get_block_idx(), tasks, get_block_num()):
        input_slot = self.input.next_write()
        mem_copy(
            tile_slice(input_slot, make_tiler((1, tile_elements), alignment=(1, 64)), (0, 0)),
            tile_slice(x, make_tiler((1, tile_elements), alignment=(1, 64)), (0, task)),
        )
        input_tile = tile_slice(self.input.next_read(), make_tiler((1, tile_elements), alignment=(1, 64)), (0, 0))
        output_tile = self.output.next_write()
        scale_tile = self.scale.next_write()
        _quantize_bf16_four_panel(
            self.quantization,
            self.random.words,
            input_tile,
            output_tile,
            scale_tile,
            task * tile_elements,
            tile_elements,
            2,
            quant_length,
            seed,
            block_offset,
        )
        _copy_to_gm(
            tile_slice(y, make_tiler((1, tile_elements), alignment=(1, 64)), (0, task)), self.output.next_read()
        )
        _copy_to_gm(
            tile_slice(scales, make_tiler((1, tile_elements // 32), alignment=(1, 2)), (0, task)),
            self.scale.next_read(),
        )


@jit
def _quantize_fp32_four_pair_256(
    self, random, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset
):
    sign_source = reinterpret(x, dtypes.uint8 if self.bf16 else dtypes.uint16, shape=(1, self.tile_elements * 2))
    parts = count // 256
    batches = (count + 1023) // 1024
    prepare_random_words_aos(self, random, start, batches, quant_length, seed, block_offset)
    with vf(mode="simd"):
        full = reg.full_mask()
        full16 = reg.full_mask(elem_bits=16)
        full8 = reg.full_mask(elem_bits=8)
        ids = reg.varange(0, dtypes.uint32)
        scale_index = reg.vshr(ids, 3, mask=full)
        sign_mask8 = reg.vdups(128, dtypes.uint8, mask=full8)
        pack_ids8 = reg.varange(0, dtypes.uint8)
        pack_group8 = reg.vbitwise_and(pack_ids8, reg.vdups(3, dtypes.uint8, mask=full8), mask=full8)
        pack_index8 = reg.vbitwise_or(
            reg.vshl(pack_group8, 6, mask=full8), reg.vshr(pack_ids8, 2, mask=full8), mask=full8
        )
        random_mask32 = reg.vdups(65535, dtypes.uint32, mask=full)
        scale_byte_slot = reg.vbitwise_and(pack_ids8, reg.vdups(3, dtypes.uint8, mask=full8), mask=full8)
        scale_byte_mask = reg.mask_and(
            reg.vlts(pack_ids8, 32, mask=full8),
            reg.vlts(scale_byte_slot, dtypes.uint8(scale_factor // 2), mask=full8),
            exec_mask=full8,
        )
        scale_cursor = reg.vstore_unalign_begin(scales)
        if const_expr(not self.bf16):
            for pair in range(parts // 2):
                part_p0 = pair * 2
                part_p1 = part_p0 + 1
                offset_p0 = part_p0 * 256
                if const_expr(self.bf16):
                    a_p0, b_p0 = reg.vload_deinterleave(x, offset_p0)
                    abs_mask_p0 = reg.vdups(32767, dtypes.uint16, mask=full16)
                    aa_p0 = reg.vbitwise_and(reg.vreinterpret(a_p0, dtypes.uint16), abs_mask_p0, mask=full16)
                    ab_p0 = reg.vbitwise_and(reg.vreinterpret(b_p0, dtypes.uint16), abs_mask_p0, mask=full16)
                    maxima16_p0 = reg.vreduce_max_datablock(reg.vmax(aa_p0, ab_p0, mask=full16), mask=full16)
                    maxima_p0 = reg.vshl(reg.vunpack(maxima16_p0, dtypes.uint32), 16, mask=full)
                    unused_signbytes_p0, signbytes_p0 = reg.vload_deinterleave(sign_source, offset_p0 * 2)
                    signs_p0 = reg.vreinterpret_lanes(
                        reg.vbitwise_and(signbytes_p0, sign_mask8, mask=full8), dtypes.uint32
                    )
                else:
                    a_p0, b_p0 = reg.vload_deinterleave(x, offset_p0)
                    c_p0, d_p0 = reg.vload_deinterleave(x, offset_p0 + 128)
                    v0_p0, v2_p0 = reg.vdeinterleave(a_p0, c_p0)
                    v1_p0, v3_p0 = reg.vdeinterleave(b_p0, d_p0)
                    a0_p0 = reg.vreinterpret(reg.vabs(v0_p0, mask=full), dtypes.uint32)
                    a1_p0 = reg.vreinterpret(reg.vabs(v1_p0, mask=full), dtypes.uint32)
                    a2_p0 = reg.vreinterpret(reg.vabs(v2_p0, mask=full), dtypes.uint32)
                    a3_p0 = reg.vreinterpret(reg.vabs(v3_p0, mask=full), dtypes.uint32)
                    pair0_p0 = reg.vmax(a0_p0, a1_p0, mask=full)
                    pair1_p0 = reg.vmax(a2_p0, a3_p0, mask=full)
                    maxima_p0 = reg.vreduce_max_datablock(reg.vmax(pair0_p0, pair1_p0, mask=full), mask=full)
                    unused_sign01_p0, sign01_p0 = reg.vload_deinterleave(sign_source, offset_p0 * 2)
                    unused_sign23_p0, sign23_p0 = reg.vload_deinterleave(sign_source, offset_p0 * 2 + 256)
                    unused_signbytes_p0, signbytes_p0 = reg.vdeinterleave(
                        reg.vreinterpret_lanes(sign01_p0, dtypes.uint8), reg.vreinterpret_lanes(sign23_p0, dtypes.uint8)
                    )
                    signs_p0 = reg.vreinterpret_lanes(
                        reg.vbitwise_and(signbytes_p0, sign_mask8, mask=full8), dtypes.uint32
                    )
                offset_p1 = part_p1 * 256
                if const_expr(self.bf16):
                    a_p1, b_p1 = reg.vload_deinterleave(x, offset_p1)
                    abs_mask_p1 = reg.vdups(32767, dtypes.uint16, mask=full16)
                    aa_p1 = reg.vbitwise_and(reg.vreinterpret(a_p1, dtypes.uint16), abs_mask_p1, mask=full16)
                    ab_p1 = reg.vbitwise_and(reg.vreinterpret(b_p1, dtypes.uint16), abs_mask_p1, mask=full16)
                    maxima16_p1 = reg.vreduce_max_datablock(reg.vmax(aa_p1, ab_p1, mask=full16), mask=full16)
                    maxima_p1 = reg.vshl(reg.vunpack(maxima16_p1, dtypes.uint32), 16, mask=full)
                    unused_signbytes_p1, signbytes_p1 = reg.vload_deinterleave(sign_source, offset_p1 * 2)
                    signs_p1 = reg.vreinterpret_lanes(
                        reg.vbitwise_and(signbytes_p1, sign_mask8, mask=full8), dtypes.uint32
                    )
                else:
                    a_p1, b_p1 = reg.vload_deinterleave(x, offset_p1)
                    c_p1, d_p1 = reg.vload_deinterleave(x, offset_p1 + 128)
                    v0_p1, v2_p1 = reg.vdeinterleave(a_p1, c_p1)
                    v1_p1, v3_p1 = reg.vdeinterleave(b_p1, d_p1)
                    a0_p1 = reg.vreinterpret(reg.vabs(v0_p1, mask=full), dtypes.uint32)
                    a1_p1 = reg.vreinterpret(reg.vabs(v1_p1, mask=full), dtypes.uint32)
                    a2_p1 = reg.vreinterpret(reg.vabs(v2_p1, mask=full), dtypes.uint32)
                    a3_p1 = reg.vreinterpret(reg.vabs(v3_p1, mask=full), dtypes.uint32)
                    pair0_p1 = reg.vmax(a0_p1, a1_p1, mask=full)
                    pair1_p1 = reg.vmax(a2_p1, a3_p1, mask=full)
                    maxima_p1 = reg.vreduce_max_datablock(reg.vmax(pair0_p1, pair1_p1, mask=full), mask=full)
                    unused_sign01_p1, sign01_p1 = reg.vload_deinterleave(sign_source, offset_p1 * 2)
                    unused_sign23_p1, sign23_p1 = reg.vload_deinterleave(sign_source, offset_p1 * 2 + 256)
                    unused_signbytes_p1, signbytes_p1 = reg.vdeinterleave(
                        reg.vreinterpret_lanes(sign01_p1, dtypes.uint8), reg.vreinterpret_lanes(sign23_p1, dtypes.uint8)
                    )
                    signs_p1 = reg.vreinterpret_lanes(
                        reg.vbitwise_and(signbytes_p1, sign_mask8, mask=full8), dtypes.uint32
                    )
                block_scales_p0, reciprocal_p0 = compute_scale(self, maxima_p0, full, False)
                if const_expr(self.bf16):
                    bits16_p0 = reg.vshr(reg.vreinterpret(reciprocal_p0, dtypes.uint32), 16, mask=full)
                    r16_p0 = reg.vpack(bits16_p0, dtypes.uint16, part="lower")
                    index16_p0 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
                    broadcast_p0 = reg.vreinterpret(reg.vgather_reg(r16_p0, index16_p0), dtypes.bfloat16)
                    a_p0 = reg.vmul(reg.vreinterpret(aa_p0, dtypes.bfloat16), broadcast_p0, mask=full16)
                    b_p0 = reg.vmul(reg.vreinterpret(ab_p0, dtypes.bfloat16), broadcast_p0, mask=full16)
                    v0_p0 = reg.vcast(a_p0, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                    v2_p0 = reg.vcast(a_p0, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                    v1_p0 = reg.vcast(b_p0, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                    v3_p0 = reg.vcast(b_p0, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                    invalid_reciprocal_p0 = reg.vreinterpret_lanes(broadcast_p0, dtypes.uint32)
                else:
                    reciprocal_p0 = reg.vgather_reg(reciprocal_p0, scale_index)
                    v0_p0 = reg.vmul(reg.vreinterpret(a0_p0, dtypes.float32), reciprocal_p0, mask=full)
                    v1_p0 = reg.vmul(reg.vreinterpret(a1_p0, dtypes.float32), reciprocal_p0, mask=full)
                    v2_p0 = reg.vmul(reg.vreinterpret(a2_p0, dtypes.float32), reciprocal_p0, mask=full)
                    v3_p0 = reg.vmul(reg.vreinterpret(a3_p0, dtypes.float32), reciprocal_p0, mask=full)
                    invalid_reciprocal_p0 = reciprocal_p0
                block_scales_p1, reciprocal_p1 = compute_scale(self, maxima_p1, full, False)
                if const_expr(self.bf16):
                    bits16_p1 = reg.vshr(reg.vreinterpret(reciprocal_p1, dtypes.uint32), 16, mask=full)
                    r16_p1 = reg.vpack(bits16_p1, dtypes.uint16, part="lower")
                    index16_p1 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
                    broadcast_p1 = reg.vreinterpret(reg.vgather_reg(r16_p1, index16_p1), dtypes.bfloat16)
                    a_p1 = reg.vmul(reg.vreinterpret(aa_p1, dtypes.bfloat16), broadcast_p1, mask=full16)
                    b_p1 = reg.vmul(reg.vreinterpret(ab_p1, dtypes.bfloat16), broadcast_p1, mask=full16)
                    v0_p1 = reg.vcast(a_p1, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                    v2_p1 = reg.vcast(a_p1, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                    v1_p1 = reg.vcast(b_p1, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                    v3_p1 = reg.vcast(b_p1, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                    invalid_reciprocal_p1 = reg.vreinterpret_lanes(broadcast_p1, dtypes.uint32)
                else:
                    reciprocal_p1 = reg.vgather_reg(reciprocal_p1, scale_index)
                    v0_p1 = reg.vmul(reg.vreinterpret(a0_p1, dtypes.float32), reciprocal_p1, mask=full)
                    v1_p1 = reg.vmul(reg.vreinterpret(a1_p1, dtypes.float32), reciprocal_p1, mask=full)
                    v2_p1 = reg.vmul(reg.vreinterpret(a2_p1, dtypes.float32), reciprocal_p1, mask=full)
                    v3_p1 = reg.vmul(reg.vreinterpret(a3_p1, dtypes.float32), reciprocal_p1, mask=full)
                    invalid_reciprocal_p1 = reciprocal_p1
                word_p0 = reg.vreinterpret(reg.vload(random.buffer, part_p0 * 64), dtypes.uint32)
                reversed_word_p0 = _rotate32_words8(word_p0, full)
                r0_p0 = reg.vshr(word_p0, 16, mask=full)
                r1_p0 = reg.vbitwise_and(reversed_word_p0, random_mask32, mask=full)
                r2_p0 = reg.vbitwise_and(word_p0, random_mask32, mask=full)
                r3_p0 = reg.vshr(reversed_word_p0, 16, mask=full)
                c0_p0 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v0_p0, dtypes.uint32), r0_p0, self.mantissa, self.bias, full
                )
                c1_p0 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v1_p0, dtypes.uint32), r1_p0, self.mantissa, self.bias, full
                )
                c2_p0 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v2_p0, dtypes.uint32), r2_p0, self.mantissa, self.bias, full
                )
                c3_p0 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v3_p0, dtypes.uint32), r3_p0, self.mantissa, self.bias, full
                )
                unused_code01_p0, code01_p0 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(c0_p0, dtypes.uint16), reg.vreinterpret_lanes(c1_p0, dtypes.uint16)
                )
                unused_code23_p0, code23_p0 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(c2_p0, dtypes.uint16), reg.vreinterpret_lanes(c3_p0, dtypes.uint16)
                )
                code_bytes_p0, unused_code_bytes_p0 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(code01_p0, dtypes.uint8), reg.vreinterpret_lanes(code23_p0, dtypes.uint8)
                )
                packed_p0 = reg.vreinterpret_lanes(reg.vgather_reg(code_bytes_p0, pack_index8), dtypes.uint32)
                magnitude8_p0 = reg.vadds(
                    reg.vreinterpret_lanes(packed_p0, dtypes.uint8),
                    -(128 - self.bias << self.mantissa) & 255,
                    mask=full8,
                )
                saturated_p0 = reg.vmins(magnitude8_p0, self.max_code, mask=reg.vnes(magnitude8_p0, 255, mask=full8))
                packed_p0 = reg.vbitwise_or(reg.vreinterpret_lanes(saturated_p0, dtypes.uint32), signs_p0, mask=full)
                invalid_p0 = reg.veqs(
                    reg.vreinterpret(invalid_reciprocal_p0, dtypes.uint32), self.invalid_packed_reciprocal, mask=full
                )
                packed_p0 = reg.vdups(2139062143, dtypes.uint32, mask=invalid_p0, mode="merging", merge=packed_p0)
                reg.vstore(y, offset_p0, reg.vreinterpret_lanes(packed_p0, dtypes.uint8), full8)
                scale_bytes_p0 = reg.vsqueeze_and_storeunalign_init(
                    reg.vreinterpret_lanes(block_scales_p0, dtypes.uint8), mask=scale_byte_mask
                )
                reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes_p0, scale_cursor)
                word_p1 = reg.vreinterpret(reg.vload(random.buffer, part_p1 * 64), dtypes.uint32)
                reversed_word_p1 = _rotate32_words8(word_p1, full)
                r0_p1 = reg.vshr(word_p1, 16, mask=full)
                r1_p1 = reg.vbitwise_and(reversed_word_p1, random_mask32, mask=full)
                r2_p1 = reg.vbitwise_and(word_p1, random_mask32, mask=full)
                r3_p1 = reg.vshr(reversed_word_p1, 16, mask=full)
                c0_p1 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v0_p1, dtypes.uint32), r0_p1, self.mantissa, self.bias, full
                )
                c1_p1 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v1_p1, dtypes.uint32), r1_p1, self.mantissa, self.bias, full
                )
                c2_p1 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v2_p1, dtypes.uint32), r2_p1, self.mantissa, self.bias, full
                )
                c3_p1 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v3_p1, dtypes.uint32), r3_p1, self.mantissa, self.bias, full
                )
                unused_code01_p1, code01_p1 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(c0_p1, dtypes.uint16), reg.vreinterpret_lanes(c1_p1, dtypes.uint16)
                )
                unused_code23_p1, code23_p1 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(c2_p1, dtypes.uint16), reg.vreinterpret_lanes(c3_p1, dtypes.uint16)
                )
                code_bytes_p1, unused_code_bytes_p1 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(code01_p1, dtypes.uint8), reg.vreinterpret_lanes(code23_p1, dtypes.uint8)
                )
                packed_p1 = reg.vreinterpret_lanes(reg.vgather_reg(code_bytes_p1, pack_index8), dtypes.uint32)
                magnitude8_p1 = reg.vadds(
                    reg.vreinterpret_lanes(packed_p1, dtypes.uint8),
                    -(128 - self.bias << self.mantissa) & 255,
                    mask=full8,
                )
                saturated_p1 = reg.vmins(magnitude8_p1, self.max_code, mask=reg.vnes(magnitude8_p1, 255, mask=full8))
                packed_p1 = reg.vbitwise_or(reg.vreinterpret_lanes(saturated_p1, dtypes.uint32), signs_p1, mask=full)
                invalid_p1 = reg.veqs(
                    reg.vreinterpret(invalid_reciprocal_p1, dtypes.uint32), self.invalid_packed_reciprocal, mask=full
                )
                packed_p1 = reg.vdups(2139062143, dtypes.uint32, mask=invalid_p1, mode="merging", merge=packed_p1)
                reg.vstore(y, offset_p1, reg.vreinterpret_lanes(packed_p1, dtypes.uint8), full8)
                scale_bytes_p1 = reg.vsqueeze_and_storeunalign_init(
                    reg.vreinterpret_lanes(block_scales_p1, dtypes.uint8), mask=scale_byte_mask
                )
                reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes_p1, scale_cursor)
            for part in range(parts // 2 * 2, parts):
                offset = part * 256
                if const_expr(self.bf16):
                    a, b = reg.vload_deinterleave(x, offset)
                    abs_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
                    aa = reg.vbitwise_and(reg.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
                    ab = reg.vbitwise_and(reg.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
                    maxima16 = reg.vreduce_max_datablock(reg.vmax(aa, ab, mask=full16), mask=full16)
                    maxima = reg.vshl(reg.vunpack(maxima16, dtypes.uint32), 16, mask=full)
                    unused_signbytes, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
                    signs = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes, sign_mask8, mask=full8), dtypes.uint32)
                else:
                    a, b = reg.vload_deinterleave(x, offset)
                    c, d = reg.vload_deinterleave(x, offset + 128)
                    v0, v2 = reg.vdeinterleave(a, c)
                    v1, v3 = reg.vdeinterleave(b, d)
                    a0 = reg.vreinterpret(reg.vabs(v0, mask=full), dtypes.uint32)
                    a1 = reg.vreinterpret(reg.vabs(v1, mask=full), dtypes.uint32)
                    a2 = reg.vreinterpret(reg.vabs(v2, mask=full), dtypes.uint32)
                    a3 = reg.vreinterpret(reg.vabs(v3, mask=full), dtypes.uint32)
                    pair0 = reg.vmax(a0, a1, mask=full)
                    pair1 = reg.vmax(a2, a3, mask=full)
                    maxima = reg.vreduce_max_datablock(reg.vmax(pair0, pair1, mask=full), mask=full)
                    unused_sign01, sign01 = reg.vload_deinterleave(sign_source, offset * 2)
                    unused_sign23, sign23 = reg.vload_deinterleave(sign_source, offset * 2 + 256)
                    unused_signbytes, signbytes = reg.vdeinterleave(
                        reg.vreinterpret_lanes(sign01, dtypes.uint8), reg.vreinterpret_lanes(sign23, dtypes.uint8)
                    )
                    signs = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes, sign_mask8, mask=full8), dtypes.uint32)
                block_scales, reciprocal = compute_scale(self, maxima, full, False)
                if const_expr(self.bf16):
                    bits16 = reg.vshr(reg.vreinterpret(reciprocal, dtypes.uint32), 16, mask=full)
                    r16 = reg.vpack(bits16, dtypes.uint16, part="lower")
                    index16 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
                    broadcast = reg.vreinterpret(reg.vgather_reg(r16, index16), dtypes.bfloat16)
                    a = reg.vmul(reg.vreinterpret(aa, dtypes.bfloat16), broadcast, mask=full16)
                    b = reg.vmul(reg.vreinterpret(ab, dtypes.bfloat16), broadcast, mask=full16)
                    v0 = reg.vcast(a, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                    v2 = reg.vcast(a, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                    v1 = reg.vcast(b, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                    v3 = reg.vcast(b, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                    invalid_reciprocal = reg.vreinterpret_lanes(broadcast, dtypes.uint32)
                else:
                    reciprocal = reg.vgather_reg(reciprocal, scale_index)
                    v0 = reg.vmul(reg.vreinterpret(a0, dtypes.float32), reciprocal, mask=full)
                    v1 = reg.vmul(reg.vreinterpret(a1, dtypes.float32), reciprocal, mask=full)
                    v2 = reg.vmul(reg.vreinterpret(a2, dtypes.float32), reciprocal, mask=full)
                    v3 = reg.vmul(reg.vreinterpret(a3, dtypes.float32), reciprocal, mask=full)
                    invalid_reciprocal = reciprocal
                word = reg.vreinterpret(reg.vload(random.buffer, part * 64), dtypes.uint32)
                reversed_word = _rotate32_words8(word, full)
                r0 = reg.vshr(word, 16, mask=full)
                r1 = reg.vbitwise_and(reversed_word, random_mask32, mask=full)
                r2 = reg.vbitwise_and(word, random_mask32, mask=full)
                r3 = reg.vshr(reversed_word, 16, mask=full)
                c0 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v0, dtypes.uint32), r0, self.mantissa, self.bias, full
                )
                c1 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v1, dtypes.uint32), r1, self.mantissa, self.bias, full
                )
                c2 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v2, dtypes.uint32), r2, self.mantissa, self.bias, full
                )
                c3 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v3, dtypes.uint32), r3, self.mantissa, self.bias, full
                )
                unused_code01, code01 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(c0, dtypes.uint16), reg.vreinterpret_lanes(c1, dtypes.uint16)
                )
                unused_code23, code23 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(c2, dtypes.uint16), reg.vreinterpret_lanes(c3, dtypes.uint16)
                )
                code_bytes, unused_code_bytes = reg.vdeinterleave(
                    reg.vreinterpret_lanes(code01, dtypes.uint8), reg.vreinterpret_lanes(code23, dtypes.uint8)
                )
                packed = reg.vreinterpret_lanes(reg.vgather_reg(code_bytes, pack_index8), dtypes.uint32)
                magnitude8 = reg.vadds(
                    reg.vreinterpret_lanes(packed, dtypes.uint8), -(128 - self.bias << self.mantissa) & 255, mask=full8
                )
                saturated = reg.vmins(magnitude8, self.max_code, mask=reg.vnes(magnitude8, 255, mask=full8))
                packed = reg.vbitwise_or(reg.vreinterpret_lanes(saturated, dtypes.uint32), signs, mask=full)
                invalid = reg.veqs(
                    reg.vreinterpret(invalid_reciprocal, dtypes.uint32), self.invalid_packed_reciprocal, mask=full
                )
                packed = reg.vdups(2139062143, dtypes.uint32, mask=invalid, mode="merging", merge=packed)
                reg.vstore(y, offset, reg.vreinterpret_lanes(packed, dtypes.uint8), full8)
                scale_bytes = reg.vsqueeze_and_storeunalign_init(
                    reg.vreinterpret_lanes(block_scales, dtypes.uint8), mask=scale_byte_mask
                )
                reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes, scale_cursor)
        else:
            for part in range(parts):
                offset = part * 256
                if const_expr(self.bf16):
                    a, b = reg.vload_deinterleave(x, offset)
                    abs_mask = reg.vdups(32767, dtypes.uint16, mask=full16)
                    aa = reg.vbitwise_and(reg.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
                    ab = reg.vbitwise_and(reg.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
                    maxima16 = reg.vreduce_max_datablock(reg.vmax(aa, ab, mask=full16), mask=full16)
                    maxima = reg.vshl(reg.vunpack(maxima16, dtypes.uint32), 16, mask=full)
                    unused_signbytes, signbytes = reg.vload_deinterleave(sign_source, offset * 2)
                    signs = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes, sign_mask8, mask=full8), dtypes.uint32)
                else:
                    a, b = reg.vload_deinterleave(x, offset)
                    c, d = reg.vload_deinterleave(x, offset + 128)
                    v0, v2 = reg.vdeinterleave(a, c)
                    v1, v3 = reg.vdeinterleave(b, d)
                    a0 = reg.vreinterpret(reg.vabs(v0, mask=full), dtypes.uint32)
                    a1 = reg.vreinterpret(reg.vabs(v1, mask=full), dtypes.uint32)
                    a2 = reg.vreinterpret(reg.vabs(v2, mask=full), dtypes.uint32)
                    a3 = reg.vreinterpret(reg.vabs(v3, mask=full), dtypes.uint32)
                    pair0 = reg.vmax(a0, a1, mask=full)
                    pair1 = reg.vmax(a2, a3, mask=full)
                    maxima = reg.vreduce_max_datablock(reg.vmax(pair0, pair1, mask=full), mask=full)
                    unused_sign01, sign01 = reg.vload_deinterleave(sign_source, offset * 2)
                    unused_sign23, sign23 = reg.vload_deinterleave(sign_source, offset * 2 + 256)
                    unused_signbytes, signbytes = reg.vdeinterleave(
                        reg.vreinterpret_lanes(sign01, dtypes.uint8), reg.vreinterpret_lanes(sign23, dtypes.uint8)
                    )
                    signs = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes, sign_mask8, mask=full8), dtypes.uint32)
                block_scales, reciprocal = compute_scale(self, maxima, full, False)
                if const_expr(self.bf16):
                    bits16 = reg.vshr(reg.vreinterpret(reciprocal, dtypes.uint32), 16, mask=full)
                    r16 = reg.vpack(bits16, dtypes.uint16, part="lower")
                    index16 = reg.vshr(reg.varange(0, dtypes.uint16), 4, mask=full16)
                    broadcast = reg.vreinterpret(reg.vgather_reg(r16, index16), dtypes.bfloat16)
                    a = reg.vmul(reg.vreinterpret(aa, dtypes.bfloat16), broadcast, mask=full16)
                    b = reg.vmul(reg.vreinterpret(ab, dtypes.bfloat16), broadcast, mask=full16)
                    v0 = reg.vcast(a, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                    v2 = reg.vcast(a, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                    v1 = reg.vcast(b, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ZERO)
                    v3 = reg.vcast(b, dtypes.float32, mask=full16, reg_layout=reg.RegLayout.ONE)
                    invalid_reciprocal = reg.vreinterpret_lanes(broadcast, dtypes.uint32)
                else:
                    reciprocal = reg.vgather_reg(reciprocal, scale_index)
                    v0 = reg.vmul(reg.vreinterpret(a0, dtypes.float32), reciprocal, mask=full)
                    v1 = reg.vmul(reg.vreinterpret(a1, dtypes.float32), reciprocal, mask=full)
                    v2 = reg.vmul(reg.vreinterpret(a2, dtypes.float32), reciprocal, mask=full)
                    v3 = reg.vmul(reg.vreinterpret(a3, dtypes.float32), reciprocal, mask=full)
                    invalid_reciprocal = reciprocal
                word = reg.vreinterpret(reg.vload(random.buffer, part * 64), dtypes.uint32)
                reversed_word = _rotate32_words8(word, full)
                r0 = reg.vshr(word, 16, mask=full)
                r1 = reg.vbitwise_and(reversed_word, random_mask32, mask=full)
                r2 = reg.vbitwise_and(word, random_mask32, mask=full)
                r3 = reg.vshr(reversed_word, 16, mask=full)
                c0 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v0, dtypes.uint32), r0, self.mantissa, self.bias, full
                )
                c1 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v1, dtypes.uint32), r1, self.mantissa, self.bias, full
                )
                c2 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v2, dtypes.uint32), r2, self.mantissa, self.bias, full
                )
                c3 = _sr_magnitude_q16_unbiased_sum(
                    reg.vreinterpret(v3, dtypes.uint32), r3, self.mantissa, self.bias, full
                )
                unused_code01, code01 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(c0, dtypes.uint16), reg.vreinterpret_lanes(c1, dtypes.uint16)
                )
                unused_code23, code23 = reg.vdeinterleave(
                    reg.vreinterpret_lanes(c2, dtypes.uint16), reg.vreinterpret_lanes(c3, dtypes.uint16)
                )
                code_bytes, unused_code_bytes = reg.vdeinterleave(
                    reg.vreinterpret_lanes(code01, dtypes.uint8), reg.vreinterpret_lanes(code23, dtypes.uint8)
                )
                packed = reg.vreinterpret_lanes(reg.vgather_reg(code_bytes, pack_index8), dtypes.uint32)
                magnitude8 = reg.vadds(
                    reg.vreinterpret_lanes(packed, dtypes.uint8), -(128 - self.bias << self.mantissa) & 255, mask=full8
                )
                saturated = reg.vmins(magnitude8, self.max_code, mask=reg.vnes(magnitude8, 255, mask=full8))
                packed = reg.vbitwise_or(reg.vreinterpret_lanes(saturated, dtypes.uint32), signs, mask=full)
                invalid = reg.veqs(
                    reg.vreinterpret(invalid_reciprocal, dtypes.uint32), self.invalid_packed_reciprocal, mask=full
                )
                packed = reg.vdups(2139062143, dtypes.uint32, mask=invalid, mode="merging", merge=packed)
                reg.vstore(y, offset, reg.vreinterpret_lanes(packed, dtypes.uint8), full8)
                scale_bytes = reg.vsqueeze_and_storeunalign_init(
                    reg.vreinterpret_lanes(block_scales, dtypes.uint8), mask=scale_byte_mask
                )
                reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes, scale_cursor)
        reg.vsqueeze_and_storeunalign_finalize(scales, 0, scale_cursor)
        # Finish encoded output/scale writes before DMA or the next tile reuses UB.
        reg.vmem_bar("vst_vld")


def _select_fp32_four_pair_policy(numel, dtype, layout, quant_length, blocks, ub_bytes):
    """Pair panels from one tile per AIV, keeping eight-field mapping priority."""
    if (
        dtype != dtypes.float32
        or layout is not Layout.TAIL
        or quant_length % PACKED_K_ALIGNMENT_ELEMENTS
        or blocks <= 0
    ):
        return False
    if _select_fp32_eight_policy(numel, dtype, layout, quant_length):
        return False
    tile = _make_tiling(dtype, ub_bytes).tile_elements
    return numel >= blocks * tile


@jit
def _fp32_crossdraw_uniform_three(counter_low, counter_high, key0, key1):
    first_low, first_high = _split_uint64(dtypes.uint64(counter_low) * dtypes.uint64(PHILOX_M1))
    uniform2 = dtypes.uint32(counter_high) ^ dtypes.uint32(key1)
    second_low, second_high = _split_uint64(dtypes.uint64(uniform2) * dtypes.uint64(PHILOX_M1))
    uniform_second0 = (
        dtypes.uint32(second_high) ^ dtypes.uint32(first_low) ^ dtypes.uint32(key0) + dtypes.uint32(PHILOX_W0)
    )
    third_low, third_high = _split_uint64(dtypes.uint64(uniform_second0) * dtypes.uint64(PHILOX_M0))
    initial = dtypes.uint32(first_high) ^ dtypes.uint32(key0)
    x0 = dtypes.uint32(second_low) ^ dtypes.uint32(key0) + dtypes.uint32(2 * PHILOX_W0 & UINT32_MASK)
    x2 = dtypes.uint32(third_high) ^ dtypes.uint32(key1) + dtypes.uint32(2 * PHILOX_W1 & UINT32_MASK)
    return (initial, x0, x2, third_low)


@jit
def _fp32_random_two_draws(scratch, draw, linear_group_count, group_low, base_low, base_high, key0, key1):
    draw_low, draw_high = _split_uint64(draw)
    counter0 = dtypes.uint32(base_low) + dtypes.uint32(draw_low)
    counter_high0 = (
        dtypes.uint32(base_high) + dtypes.uint32(draw_high) + dtypes.uint32(counter0 < dtypes.uint32(base_low))
    )
    counter1 = dtypes.uint32(counter0) + dtypes.uint32(1)
    counter_high1 = dtypes.uint32(counter_high0) + dtypes.uint32(counter1 < counter0)
    initial0, x00, x20, third0 = _fp32_crossdraw_uniform_three(counter0, counter_high0, key0, key1)
    initial1, x01, x21, third1 = _fp32_crossdraw_uniform_three(counter1, counter_high1, key0, key1)
    next_key1 = dtypes.uint32(key1) + dtypes.uint32(PHILOX_W1)
    boundary = 16384 - dtypes.uint32(group_low) % 16384
    with vf(mode="simt", thread=256):
        for group in range(simt.thread_idx()[0], linear_group_count, 256):
            uniform_initial, uniform_x0, uniform_x2, third_low = (initial0, x00, x20, third0)
            if dtypes.uint32(group) >= dtypes.uint32(boundary):
                uniform_initial, uniform_x0, uniform_x2, third_low = (initial1, x01, x21, third1)
            subsequence = (dtypes.uint32(group_low) + dtypes.uint32(group)) % 16384
            first_c0 = subsequence ^ dtypes.uint32(uniform_initial)
            variable_low = dtypes.uint32(first_c0) * dtypes.uint32(PHILOX_M0)
            variable_high = _simt_umulhi(first_c0, PHILOX_M0)
            second_c2 = variable_high ^ dtypes.uint32(next_key1)
            p1lo = dtypes.uint32(second_c2) * dtypes.uint32(PHILOX_M1)
            p1hi = _simt_umulhi(second_c2, PHILOX_M1)
            c0 = p1hi ^ dtypes.uint32(uniform_x0)
            c1 = p1lo
            c2 = variable_low ^ dtypes.uint32(uniform_x2)
            c3 = dtypes.uint32(third_low)
            for step in range(3, 10):
                p0lo = dtypes.uint32(c0) * dtypes.uint32(PHILOX_M0)
                p0hi = _simt_umulhi(c0, PHILOX_M0)
                p1lo = dtypes.uint32(c2) * dtypes.uint32(PHILOX_M1)
                p1hi = _simt_umulhi(c2, PHILOX_M1)
                k0 = dtypes.uint32(key0) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W0)
                k1 = dtypes.uint32(key1) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W1)
                c0, c1, c2, c3 = (p1hi ^ c1 ^ k0, p1lo, p0hi ^ c3 ^ k1, p0lo)
            scratch[0, group * 4] = dtypes.int32(c0)
            scratch[0, group * 4 + 1] = dtypes.int32(c1)
            scratch[0, group * 4 + 2] = dtypes.int32(c2)
            scratch[0, group * 4 + 3] = dtypes.int32(c3)


@jit
def _prepare_fp4_crossdraw_aos_linear(random, start, batches, seed, block_offset):
    key0, key1 = _split_uint64(seed)
    group_low, group_high = _split_uint64(dtypes.uint64(start) // 16)
    offset_low, offset_high = _split_uint64(block_offset)
    base_low = dtypes.uint32(offset_low // 4 + offset_high % 4 * 1073741824)
    base_high = dtypes.uint32(offset_high // 4)
    linear_group_count = dtypes.int64(batches) * 64
    first_group = dtypes.uint64(start) // 16
    draw = first_group // 16384
    last_draw = (first_group + dtypes.uint64(linear_group_count) - 1) // 16384
    if draw == last_draw:
        draw_low, draw_high = _split_uint64(draw)
        counter_low = dtypes.uint32(base_low) + dtypes.uint32(draw_low)
        counter_high = (
            dtypes.uint32(base_high) + dtypes.uint32(draw_high) + dtypes.uint32(counter_low < dtypes.uint32(base_low))
        )
        first_low, first_high = _split_uint64(dtypes.uint64(counter_low) * dtypes.uint64(PHILOX_M1))
        uniform2 = dtypes.uint32(counter_high) ^ dtypes.uint32(key1)
        second_low, second_high = _split_uint64(dtypes.uint64(uniform2) * dtypes.uint64(PHILOX_M1))
        next_key0 = dtypes.uint32(key0) + dtypes.uint32(PHILOX_W0)
        next_key1 = dtypes.uint32(key1) + dtypes.uint32(PHILOX_W1)
        uniform_second0 = dtypes.uint32(second_high) ^ dtypes.uint32(first_low) ^ dtypes.uint32(next_key0)
        third_low, third_high = _split_uint64(dtypes.uint64(uniform_second0) * dtypes.uint64(PHILOX_M0))
        uniform_initial = dtypes.uint32(first_high) ^ dtypes.uint32(key0)
        uniform_x0 = dtypes.uint32(second_low) ^ dtypes.uint32(key0) + dtypes.uint32(2 * PHILOX_W0 & UINT32_MASK)
        uniform_x2 = dtypes.uint32(third_high) ^ dtypes.uint32(key1) + dtypes.uint32(2 * PHILOX_W1 & UINT32_MASK)
        with vf(mode="simt", thread=256):
            for group in range(simt.thread_idx()[0], linear_group_count, 256):
                subsequence = (dtypes.uint32(group_low) + dtypes.uint32(group)) % 16384
                first_c0 = subsequence ^ dtypes.uint32(uniform_initial)
                variable_low = dtypes.uint32(first_c0) * dtypes.uint32(PHILOX_M0)
                variable_high = _simt_umulhi(first_c0, PHILOX_M0)
                second_c2 = variable_high ^ dtypes.uint32(next_key1)
                p1lo = dtypes.uint32(second_c2) * dtypes.uint32(PHILOX_M1)
                p1hi = _simt_umulhi(second_c2, PHILOX_M1)
                c0 = p1hi ^ dtypes.uint32(uniform_x0)
                c1 = p1lo
                c2 = variable_low ^ dtypes.uint32(uniform_x2)
                c3 = dtypes.uint32(third_low)
                for step in range(3, 10):
                    p0lo = dtypes.uint32(c0) * dtypes.uint32(PHILOX_M0)
                    p0hi = _simt_umulhi(c0, PHILOX_M0)
                    p1lo = dtypes.uint32(c2) * dtypes.uint32(PHILOX_M1)
                    p1hi = _simt_umulhi(c2, PHILOX_M1)
                    k0 = dtypes.uint32(key0) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W0)
                    k1 = dtypes.uint32(key1) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W1)
                    c0, c1, c2, c3 = (p1hi ^ c1 ^ k0, p1lo, p0hi ^ c3 ^ k1, p0lo)
                random.buffer[0, group * 4] = dtypes.int32(c0)
                random.buffer[0, group * 4 + 1] = dtypes.int32(c1)
                random.buffer[0, group * 4 + 2] = dtypes.int32(c2)
                random.buffer[0, group * 4 + 3] = dtypes.int32(c3)
    elif last_draw == draw + 1:
        _fp32_random_two_draws(random.buffer, draw, linear_group_count, group_low, base_low, base_high, key0, key1)
    else:
        with vf(mode="simt", thread=256):
            for group in range(simt.thread_idx()[0], linear_group_count, 256):
                gl = dtypes.uint32(group_low) + dtypes.uint32(group)
                gh = dtypes.uint32(group_high) + dtypes.uint32(gl < dtypes.uint32(group_low))
                draw_low = dtypes.uint32(gl // 16384) + dtypes.uint32(gh % 16384) * dtypes.uint32(262144)
                draw_high = dtypes.uint32(gh // 16384)
                local_counter_low = dtypes.uint32(base_low) + dtypes.uint32(draw_low)
                local_counter_high = (
                    dtypes.uint32(base_high)
                    + dtypes.uint32(draw_high)
                    + dtypes.uint32(local_counter_low < dtypes.uint32(base_low))
                )
                c0 = dtypes.uint32(0)
                c1 = dtypes.uint32(gl % 16384)
                c2 = local_counter_low
                c3 = local_counter_high
                for step in range(10):
                    p0lo = dtypes.uint32(c0) * dtypes.uint32(PHILOX_M0)
                    p0hi = _simt_umulhi(c0, PHILOX_M0)
                    p1lo = dtypes.uint32(c2) * dtypes.uint32(PHILOX_M1)
                    p1hi = _simt_umulhi(c2, PHILOX_M1)
                    k0 = dtypes.uint32(key0) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W0)
                    k1 = dtypes.uint32(key1) + dtypes.uint32(step) * dtypes.uint32(PHILOX_W1)
                    c0, c1, c2, c3 = (p1hi ^ c1 ^ k0, p1lo, p0hi ^ c3 ^ k1, p0lo)
                random.buffer[0, group * 4] = dtypes.int32(c0)
                random.buffer[0, group * 4 + 1] = dtypes.int32(c1)
                random.buffer[0, group * 4 + 2] = dtypes.int32(c2)
                random.buffer[0, group * 4 + 3] = dtypes.int32(c3)


def dynamic_mx_quant_sr_fwd(
    input,
    axis,
    round_mode,
    dst_type,
    block_size,
    scale_alg,
    dst_type_max,
    max_low_bound,
):
    """Positional wrapper for the mojo_opset dispatch layer."""
    return dynamic_mx_quant_sr(
        input,
        axis=axis,
        round_mode=round_mode,
        dst_type=dst_type,
        block_size=block_size,
        scale_alg=scale_alg,
        dst_type_max=dst_type_max,
        max_low_bound=max_low_bound,
    )
