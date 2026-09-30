# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the repository.

"""Dynamic MXFP8 SR: SIMD numerical work, SIMT native bit reversal only.

One public operator and seven dynamic layout specializations. Standard AOT
owns binary management; this sample keeps bounded, generation-safe Program
references. Compilation requires stable process configuration.
"""

import math
import os
import struct
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from threading import Lock
from types import MappingProxyType
from typing import NamedTuple

import cannbotdsl
import torch
from cannbotdsl import Buffer, Dim, TensorSpec, dtypes, get_mem_size
from cannbotdsl.channel import Channel
from cannbotdsl.core.compiler import cache as dsl_cache, cache_key
from cannbotdsl.core.compiler.abi import HOST_ABI_VERSION
from cannbotdsl.core.utils.env import env
from cannbotdsl.lang import const_expr, host, jit, kernel, vf
from cannbotdsl.ops import reg, scalar, simt
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.tensor import MemLoc, Tensor, make_tiler, reinterpret, tile_slice

from ._rng_state import _reserve_rng_state, _rng_increment, _validate_rng_geometry


# Host execution plans and tiling constants


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


# Tail prefetch tuning, in elements/slots; not chip capacity constraints.
# 16K/32K is the measured BF16 proposal; 4K/32K is still experimental.
BF16_TILE_LIMIT_ELEMENTS = 16 * 1024
TAIL_PREFETCH_COMPUTE_ELEMENTS = 16 * 1024
TAIL_PREFETCH_DMA_ELEMENTS = 32 * 1024
TAIL_PREFETCH_TILES = TAIL_PREFETCH_DMA_ELEMENTS // TAIL_PREFETCH_COMPUTE_ELEMENTS
TAIL_PREFETCH_INPUT_DEPTH = 1
TAIL_PREFETCH_OUTPUT_DEPTH = 2


# Compilation identity and Program cache


SUPPORTED_ARCHITECTURES = frozenset({"dav-3510"})


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
    rng_abi_version: int = 2


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
class AuxiliaryCompileKey:
    role: str
    variant: bool
    identity: RuntimeCompileIdentity


@dataclass(frozen=True)
class CompileContext:
    key: CompileKey | AuxiliaryCompileKey
    bypass_reason: str | None


def _build_identity():
    """Concentrate version-specific internal SDK identity APIs here."""
    return (cannbotdsl.__version__, cache_key._runtime_identity(),
            cache_key._toolchain_version(), HOST_ABI_VERSION,
            cache_key._FORMAT_VERSION_TAG)


def _diagnostic_reason(config):
    reason = dsl_cache.bypass_reason(config, has_debug_plans=False)
    # This kernel has no debug-print plans. Frontend dumps are also observable
    # and must not disappear behind a hot outer Program reference.
    return reason or ("frontend_dumps" if config.frontend_dumps else None)


def _capture_runtime_identity(device):
    """One configuration/diagnostics path for quantization and state helpers."""
    config = env()
    arch = config.npu_arch
    if arch not in SUPPORTED_ARCHITECTURES:
        raise NotImplementedError(f"dynamic_mx_quant_sr does not support architecture {arch!r}")
    if device.type not in ("npu", "privateuseone") or device.index is None:
        raise ValueError("compile device must have a resolved NPU index")
    options = {name: value for name, value in vars(config).items()
               if name not in cache_key._KEY_IGNORED_ENV}
    version, native, compiler, abi, format_version = _build_identity()
    identity = RuntimeCompileIdentity(
        arch, "npu", device.index, version, native, compiler, abi, format_version,
        os.environ.get("CANNBOTDSL_CACHE_FORMAT_TAG", ""),
        cache_key._encode_env_snapshot(options),
    )
    return identity, _diagnostic_reason(config)


def capture_compile_context(input_dtype, e5m2, ub_bytes, scale_alg,
                            max_low_bound_bits, layout, device):
    """Capture one effective DSL configuration; no shape/runtime launch key."""
    identity, reason = _capture_runtime_identity(device)
    key = CompileKey(input_dtype, e5m2, ub_bytes, scale_alg, max_low_bound_bits,
                     Layout(layout), **asdict(identity))
    return CompileContext(key, reason)


def capture_auxiliary_context(role, variant, device):
    identity, reason = _capture_runtime_identity(device)
    return CompileContext(AuxiliaryCompileKey(role, variant, identity), reason)


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
            # Diagnostic AOT may return an empty-so translation stub. It must
            # neither consume a production hit nor merge/publish a normal key.
            return builder()
        with self._lock:
            program = self._programs.get(key)
            if program is not None and not program.closed:
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
            # Notify outside the lock: Future callbacks may reenter clear().
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
        # Eviction/clear release references only; no explicit Program.close()
        # or global DSL cache clear, so external Programs remain usable.
        del evicted
        pending.set_result(program)
        return program

    def clear(self):
        with self._lock:
            released = list(self._programs.values())
            self._generation += 1
            self._programs.clear()
            self._hits = self._misses = 0
        # Old in-flight owners still wake their existing waiters but cannot
        # refill this generation. New requests do not wait for those owners.
        del released

    def info(self):
        with self._lock:
            return CacheInfo(self._hits, self._misses, self._capacity,
                             len(self._programs), len(self._inflight), self._generation)


# Philox words, SIMT bit reverse and random fields


SIMT_THREADS = 512
PHILOX_ROUNDS = 10
# Random123 Philox4x32 constants, modulo 2**32 arithmetic.
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85


def broadcast_u32(value, mask):
    """Broadcast a scalar as masked u32 SIMD lanes, not a scalar cast."""
    return reg.vdups(value, dtypes.uint32, mask=mask)


def bitwise_and_scalar(value, bits, mask):
    """Bitwise AND with an explicit u32 SIMD scalar broadcast."""
    return reg.vbitwise_and(value, broadcast_u32(bits, mask), mask=mask)


def _split_uint64(value):
    """Clear the unsigned low word before exact signed division.

    UInt32 -> Int64 widens with zero extension. Clearing that low word
    leaves a multiple of 2**32 in signed range, so trunc division is exact
    for every 64-bit seed/offset bit pattern, including negative int64.
    """
    low = dtypes.uint32(value)
    cleared = dtypes.int64(value) - dtypes.int64(low)
    return low, dtypes.uint32(cleared // 4294967296)


def _pair_add(low0, high0, low1, high1, mask):
    carry, low = reg.vaddco(low0, low1, mask=mask)
    _, high = reg.vaddc(high0, high1, carry, mask=mask)
    return low, high


def _nvfp4_counter_coordinates(row_low, row_high, col_low, col_high,
                              axis_groups, block_offset, mask):
    """TE rowwise 128x128 task/thread coordinates, with full-width counters."""
    row_task_low = reg.vbitwise_or(reg.vshr(row_low, 7, mask=mask),
                                  reg.vshl(row_high, 25, mask=mask), mask=mask)
    row_task_high = reg.vshr(row_high, 7, mask=mask)
    col_task_low = reg.vbitwise_or(reg.vshr(col_low, 3, mask=mask),
                                  reg.vshl(col_high, 29, mask=mask), mask=mask)
    col_task_high = reg.vshr(col_high, 3, mask=mask)
    grid_low, grid_high = _split_uint64((axis_groups + 7) // 8)
    task_low, task_high = reg.vmull(row_task_low, broadcast_u32(grid_low, mask),
                                   dtypes.uint32, mask=mask)
    task_high = reg.vadd(task_high, reg.vmuls(row_task_low, grid_high, mask=mask), mask=mask)
    task_high = reg.vadd(task_high, reg.vmuls(row_task_high, grid_low, mask=mask), mask=mask)
    task_low, task_high = _pair_add(task_low, task_high, col_task_low, col_task_high, mask)
    lane = reg.vbitwise_or(
        reg.vshl(bitwise_and_scalar(row_low, 15, mask), 3, mask=mask),
        bitwise_and_scalar(col_low, 7, mask), mask=mask)
    subsequence_low = reg.vbitwise_or(reg.vshl(task_low, 7, mask=mask), lane, mask=mask)
    subsequence_high = reg.vbitwise_or(reg.vshl(task_high, 7, mask=mask),
                                     reg.vshr(task_low, 25, mask=mask), mask=mask)
    draw = bitwise_and_scalar(reg.vshr(row_low, 4, mask=mask), 7, mask)
    offset_low, offset_high = _split_uint64(block_offset)
    counter_low, counter_high = _pair_add(
        broadcast_u32(offset_low, mask), broadcast_u32(offset_high, mask),
        draw, broadcast_u32(0, mask), mask)
    return counter_low, counter_high, subsequence_low, subsequence_high


@jit
def _nvfp4_counter_contiguous(row_base, col_base, ids, axis_groups,
                             reciprocal, block_offset, mask):
    """Map one normalized start plus 64 consecutive logical Philox groups."""
    zero = broadcast_u32(0, mask)
    row_low_bits, row_high_bits = _split_uint64(row_base)
    counter_low = zero
    counter_high = zero
    subsequence_low = zero
    subsequence_high = zero
    row_increment = zero
    col_low = zero
    col_high = zero
    if axis_groups == 128:
        # With 128 random groups per row, flattening and the TE rowwise
        # mapping reduce to fixed bit fields:
        #   subsequence = (row >> 7) << 11 | (col >> 3) << 7
        #                 | (row & 15) << 3 | (col & 7)
        # This preserves the NVFP4 mapping while avoiding the generic u64
        # compare/subtract and task-grid multiply on every Philox batch.
        linear_col = reg.vadds(ids, dtypes.uint32(col_base), mask=mask)
        row_increment = reg.vshr(linear_col, 7, mask=mask)
        col_low = bitwise_and_scalar(linear_col, 127, mask)
        row_low, row_high = _pair_add(
            broadcast_u32(row_low_bits, mask), broadcast_u32(row_high_bits, mask),
            row_increment, zero, mask)
        row_task_low = reg.vshl(
            reg.vshr(row_low, 7, mask=mask), 11, mask=mask)
        col_task_low = reg.vshl(
            reg.vshr(col_low, 3, mask=mask), 7, mask=mask)
        lane = reg.vbitwise_or(
            reg.vshl(bitwise_and_scalar(row_low, 15, mask), 3, mask=mask),
            bitwise_and_scalar(col_low, 7, mask), mask=mask)
        subsequence_low = reg.vbitwise_or(
            reg.vbitwise_or(row_task_low, col_task_low, mask=mask),
            lane, mask=mask)
        subsequence_high = reg.vbitwise_or(
            reg.vshl(row_high, 4, mask=mask),
            reg.vshr(row_low, 28, mask=mask), mask=mask)
        draw = bitwise_and_scalar(reg.vshr(row_low, 4, mask=mask), 7, mask)
        offset_low, offset_high = _split_uint64(block_offset)
        counter_low, counter_high = _pair_add(
            broadcast_u32(offset_low, mask), broadcast_u32(offset_high, mask),
            draw, zero, mask)
    else:
        one = broadcast_u32(1, mask)
        col_low_bits, col_high_bits = _split_uint64(col_base)
        row_increment = zero
        col_low = zero
        col_high = zero
        if axis_groups == 1:
            row_increment = ids
        elif axis_groups < 64:
            numerator = reg.vadds(ids, dtypes.uint32(col_base), mask=mask)
            _, quotient = reg.vmull(
                numerator, broadcast_u32(reciprocal, mask),
                dtypes.uint32, mask=mask)
            remainder = reg.vsub(
                numerator,
                reg.vmuls(quotient, dtypes.uint32(axis_groups), mask=mask),
                mask=mask)
            fix = reg.vselect(one, zero, cond_mask=reg.vges(
                remainder, dtypes.uint32(axis_groups), mask=mask))
            row_increment = reg.vadd(quotient, fix, mask=mask)
            col_low = reg.vsub(
                remainder,
                reg.vmuls(fix, dtypes.uint32(axis_groups), mask=mask),
                mask=mask)
        else:
            axis_low, axis_high = _split_uint64(axis_groups)
            col_low, col_high = _pair_add(
                broadcast_u32(col_low_bits, mask),
                broadcast_u32(col_high_bits, mask), ids, zero, mask)
            high_equal = reg.veqs(col_high, axis_high, mask=mask)
            wrapped = reg.mask_or(
                reg.vgts(col_high, axis_high, mask=mask),
                reg.mask_and(
                    high_equal, reg.vges(col_low, axis_low, mask=mask),
                    exec_mask=mask),
                exec_mask=mask)
            row_increment = reg.vselect(one, zero, cond_mask=wrapped)
            subtract_low = reg.vmuls(row_increment, axis_low, mask=mask)
            subtract_high = reg.vmuls(row_increment, axis_high, mask=mask)
            borrow = reg.vselect(
                one, zero,
                cond_mask=reg.vlt(col_low, subtract_low, mask=mask))
            col_low = reg.vsub(col_low, subtract_low, mask=mask)
            col_high = reg.vsub(
                reg.vsub(col_high, subtract_high, mask=mask),
                borrow, mask=mask)
        row_low, row_high = _pair_add(
            broadcast_u32(row_low_bits, mask),
            broadcast_u32(row_high_bits, mask), row_increment, zero, mask)
        counter_low, counter_high, subsequence_low, subsequence_high = (
            _nvfp4_counter_coordinates(
                row_low, row_high, col_low, col_high,
                axis_groups, block_offset, mask))
    return counter_low, counter_high, subsequence_low, subsequence_high


def _nvfp4_counter_rows(row_base, ids, col_group, axis_groups, block_offset, mask):
    """Wide layouts visit consecutive original logical rows at one column."""
    row_low_bits, row_high_bits = _split_uint64(row_base)
    col_low_bits, col_high_bits = _split_uint64(col_group)
    row_low, row_high = _pair_add(
        broadcast_u32(row_low_bits, mask), broadcast_u32(row_high_bits, mask),
        ids, broadcast_u32(0, mask), mask)
    return _nvfp4_counter_coordinates(
        row_low, row_high, broadcast_u32(col_low_bits, mask),
        broadcast_u32(col_high_bits, mask), axis_groups, block_offset, mask)


def _rng_geometry(start, quant_length):
    return _rng_group_geometry(dtypes.int64(start) // 16, quant_length)


@jit
def _rng_group_geometry(group, quant_length):
    """Nonnegative scalar division stays outside all SIMD VF loops."""
    axis_groups = dtypes.int64(quant_length) // 16
    group = dtypes.int64(group)
    row_base, col_base = group // axis_groups, group % axis_groups
    reciprocal = dtypes.uint32(0)
    if axis_groups > 1 and axis_groups < 64:
        reciprocal = dtypes.uint32(4294967296 // axis_groups)
    return axis_groups, reciprocal, row_base, col_base


def _advance_group(row_base, col_base, step_q, step_r, axis_groups):
    total = col_base + step_r
    carry = dtypes.int64(total >= axis_groups)
    return row_base + step_q + carry, total - carry * axis_groups


def _cursor_coordinates(row_low, row_high, col_low, col_high):
    """Rebuild temporary coordinates from explicitly zero-extended words.

    gc8c8581 narrows scalar VF loop-carried values to u32, even after an
    int64 cast. Carry four u32 words through loops instead; these temporary
    int64 coordinates retain the validated nonnegative geometry domain.
    """
    row = dtypes.int64(dtypes.uint64(dtypes.uint32(row_low))
                      + dtypes.uint64(dtypes.uint32(row_high)) * 4294967296)
    col = dtypes.int64(dtypes.uint64(dtypes.uint32(col_low))
                      + dtypes.uint64(dtypes.uint32(col_high)) * 4294967296)
    return row, col


def _advance_cursor_words(row, col, step_q, step_r, axis_groups):
    row, col = _advance_group(row, col, step_q, step_r, axis_groups)
    row_low, row_high = _split_uint64(row)
    col_low, col_high = _split_uint64(col)
    return row_low, row_high, col_low, col_high


def _philox_counter10(c0, c1, c2, c3, key0, key1, mask):
    """Full ten-round Philox4x32 for runtime counter and 64-bit seed key."""
    m0 = broadcast_u32(PHILOX_M0, mask)
    m1 = broadcast_u32(PHILOX_M1, mask)
    for step in range(PHILOX_ROUNDS):
        lo0, hi0 = reg.vmull(c0, m0, dtypes.uint32, mask=mask)
        lo1, hi1 = reg.vmull(c2, m1, dtypes.uint32, mask=mask)
        next0 = reg.vbitwise_xor(hi1, c1, mask=mask)
        next2 = reg.vbitwise_xor(hi0, c3, mask=mask)
        # Scalar u32 anchors ensure seed+round*W wraps modulo 2**32.
        k0 = broadcast_u32(dtypes.uint32(key0) + ((step * PHILOX_W0) & 0xFFFFFFFF), mask)
        k1 = broadcast_u32(dtypes.uint32(key1) + ((step * PHILOX_W1) & 0xFFFFFFFF), mask)
        c0 = reg.vbitwise_xor(next0, k0, mask=mask)
        c2 = reg.vbitwise_xor(next2, k1, mask=mask)
        c1 = lo1
        c3 = lo0
    return c0, c1, c2, c3


def _store_random_words(scratch, offset, w0, w1, w2, w3, mask):
    """Store one Philox result in the existing four-plane SoA order."""
    reg.vstore(scratch, offset, reg.vreinterpret(w0, dtypes.int32), mask)
    reg.vstore(scratch, offset + 64, reg.vreinterpret(w1, dtypes.int32), mask)
    reg.vstore(scratch, offset + 128, reg.vreinterpret(w2, dtypes.int32), mask)
    reg.vstore(scratch, offset + 192, reg.vreinterpret(w3, dtypes.int32), mask)


def _philox_first_round(c0, c1, c2, c3, m0, m1, key0, key1, mask):
    """Common full first round; mapping remains in its own counter helper."""
    lo0, hi0 = reg.vmull(c0, m0, dtypes.uint32, mask=mask)
    lo1, hi1 = reg.vmull(c2, m1, dtypes.uint32, mask=mask)
    next0 = reg.vbitwise_xor(reg.vbitwise_xor(hi1, c1, mask=mask), broadcast_u32(key0, mask), mask=mask)
    next2 = reg.vbitwise_xor(reg.vbitwise_xor(hi0, c3, mask=mask), broadcast_u32(key1, mask), mask=mask)
    return next0, lo1, next2, lo0


def _generate_random_words_keyed(scratch, row_base, col_base, axis_groups, reciprocal,
                           offset, key0, key1, block_offset):
    mask = reg.full_mask()
    c0, c1, c2, c3 = _nvfp4_counter_contiguous(
        row_base, col_base, reg.varange(0, dtypes.uint32), axis_groups,
        reciprocal, block_offset, mask)
    w0, w1, w2, w3 = _philox_counter10(c0, c1, c2, c3, key0, key1, mask)
    _store_random_words(scratch, offset, w0, w1, w2, w3, mask)


def _generate_random_words(scratch, row_base, col_base, axis_groups, reciprocal,
                           offset, seed, block_offset):
    key0, key1 = _split_uint64(seed)
    _generate_random_words_keyed(scratch, row_base, col_base, axis_groups, reciprocal,
                offset, key0, key1, block_offset)


def _generate_random_words_aos_keyed(scratch, row_base, col_base, axis_groups, reciprocal,
                               offset, key0, key1, block_offset):
    mask = reg.full_mask()
    c0, c1, c2, c3 = _nvfp4_counter_contiguous(
        row_base, col_base, reg.varange(0, dtypes.uint32), axis_groups,
        reciprocal, block_offset, mask)
    w0, w1, w2, w3 = _philox_counter10(c0, c1, c2, c3, key0, key1, mask)
    _store_random_words_aos(scratch, offset, w0, w1, w2, w3, mask)


def _generate_random_words_aos(scratch, row_base, col_base, axis_groups, reciprocal,
                           offset, seed, block_offset):
    key0, key1 = _split_uint64(seed)
    _generate_random_words_aos_keyed(scratch, row_base, col_base, axis_groups, reciprocal,
                offset, key0, key1, block_offset)


def _philox_counter_first(row_base, col_base, axis_groups, reciprocal,
                          ids, m0, m1, key0, key1, block_offset, mask):
    """Full first round for each normalized half of a paired group batch."""
    c0, c1, c2, c3 = _nvfp4_counter_contiguous(
        row_base, col_base, ids, axis_groups, reciprocal, block_offset, mask)
    return _philox_first_round(c0, c1, c2, c3, m0, m1, key0, key1, mask)


@jit
def _philox_counter_first_same_row(
        base_low, base_high, counter_low, counter_high, lane_start,
        batch_offset, ids, m0, m1, key0, key1, mask):
    """One TE rowwise base plus a small offset for one 64-group batch."""
    position = reg.vadds(
        ids, dtypes.uint32(lane_start + batch_offset), mask=mask)
    column_task_delta = reg.vshl(reg.vshr(position, 3, mask=mask), 7, mask=mask)
    column_lane = bitwise_and_scalar(position, 7, mask)
    subsequence_delta = reg.vbitwise_or(
        column_task_delta, column_lane, mask=mask)
    subsequence_low, subsequence_high = _pair_add(
        base_low, base_high, subsequence_delta,
        broadcast_u32(0, mask), mask)
    return _philox_first_round(
        counter_low, counter_high, subsequence_low, subsequence_high,
        m0, m1, key0, key1, mask)


def _philox_pair_three(a0, a1, a2, a3, b0, b1, b2, b3, step, m0, m1, seed0, seed1, mask):
    """Three rounds for two independent groups, sharing each round's keys."""
    # This Python loop unfolds three rounds. The enclosing VF loop supplies
    # runtime steps 1, 4 and 7, so only three pairs of keys are needed at once.
    for offset in range(3):
        round_id = dtypes.uint32(step) + offset
        key0 = broadcast_u32(seed0 + round_id * PHILOX_W0, mask)
        key1 = broadcast_u32(seed1 + round_id * PHILOX_W1, mask)
        lo_a0, hi_a0 = reg.vmull(a0, m0, dtypes.uint32, mask=mask)
        lo_a1, hi_a1 = reg.vmull(a2, m1, dtypes.uint32, mask=mask)
        lo_b0, hi_b0 = reg.vmull(b0, m0, dtypes.uint32, mask=mask)
        lo_b1, hi_b1 = reg.vmull(b2, m1, dtypes.uint32, mask=mask)
        # Precombine each low word with its round key while the high product resolves.
        a0 = reg.vbitwise_xor(hi_a1, reg.vbitwise_xor(a1, key0, mask=mask), mask=mask)
        a2 = reg.vbitwise_xor(hi_a0, reg.vbitwise_xor(a3, key1, mask=mask), mask=mask)
        a1, a3 = lo_a1, lo_a0
        b0 = reg.vbitwise_xor(hi_b1, reg.vbitwise_xor(b1, key0, mask=mask), mask=mask)
        b2 = reg.vbitwise_xor(hi_b0, reg.vbitwise_xor(b3, key1, mask=mask), mask=mask)
        b1, b3 = lo_b1, lo_b0
    return a0, a1, a2, a3, b0, b1, b2, b3


def _philox_quad_three(
        a0, a1, a2, a3, b0, b1, b2, b3,
        c0, c1, c2, c3, d0, d1, d2, d3,
        step, m0, m1, seed0, seed1, mask):
    """Advance four independent Philox groups through three full rounds."""
    for offset in range(3):
        round_id = dtypes.uint32(step) + offset
        key0 = broadcast_u32(seed0 + round_id * PHILOX_W0, mask)
        key1 = broadcast_u32(seed1 + round_id * PHILOX_W1, mask)
        lo_a0, hi_a0 = reg.vmull(a0, m0, dtypes.uint32, mask=mask)
        lo_a1, hi_a1 = reg.vmull(a2, m1, dtypes.uint32, mask=mask)
        lo_b0, hi_b0 = reg.vmull(b0, m0, dtypes.uint32, mask=mask)
        lo_b1, hi_b1 = reg.vmull(b2, m1, dtypes.uint32, mask=mask)
        lo_c0, hi_c0 = reg.vmull(c0, m0, dtypes.uint32, mask=mask)
        lo_c1, hi_c1 = reg.vmull(c2, m1, dtypes.uint32, mask=mask)
        lo_d0, hi_d0 = reg.vmull(d0, m0, dtypes.uint32, mask=mask)
        lo_d1, hi_d1 = reg.vmull(d2, m1, dtypes.uint32, mask=mask)
        a0 = reg.vbitwise_xor(hi_a1, reg.vbitwise_xor(a1, key0, mask=mask), mask=mask)
        a2 = reg.vbitwise_xor(hi_a0, reg.vbitwise_xor(a3, key1, mask=mask), mask=mask)
        a1, a3 = lo_a1, lo_a0
        b0 = reg.vbitwise_xor(hi_b1, reg.vbitwise_xor(b1, key0, mask=mask), mask=mask)
        b2 = reg.vbitwise_xor(hi_b0, reg.vbitwise_xor(b3, key1, mask=mask), mask=mask)
        b1, b3 = lo_b1, lo_b0
        c0 = reg.vbitwise_xor(hi_c1, reg.vbitwise_xor(c1, key0, mask=mask), mask=mask)
        c2 = reg.vbitwise_xor(hi_c0, reg.vbitwise_xor(c3, key1, mask=mask), mask=mask)
        c1, c3 = lo_c1, lo_c0
        d0 = reg.vbitwise_xor(hi_d1, reg.vbitwise_xor(d1, key0, mask=mask), mask=mask)
        d2 = reg.vbitwise_xor(hi_d0, reg.vbitwise_xor(d3, key1, mask=mask), mask=mask)
        d1, d3 = lo_d1, lo_d0
    return (a0, a1, a2, a3, b0, b1, b2, b3,
            c0, c1, c2, c3, d0, d1, d2, d3)


def _store_random_words_aos(scratch, offset, w0, w1, w2, w3, mask):
    a0, a1 = reg.vinterleave(w0, w2)
    b0, b1 = reg.vinterleave(w1, w3)
    # Interleave during the store; retain the same AoS word order.
    reg.vstore_interleave(scratch, offset, a0, b0)
    reg.vstore_interleave(scratch, offset + 128, a1, b1)


def _reverse32_scalar(value):
    # The current DSL exposes this operation as a native SIMT scalar intrinsic.
    return simt.brev(value)


def _random_fields_from_words(ids, scratch, reversed_words, segment, mask):
    # 每 1024 个输入对应 4×64 个随机字；word 的四个输出依次取
    # high、reverse16(high)、low、reverse16(low)，仅第 1/3 lane 反转。
    slot = bitwise_and_scalar(reg.vshr(ids, 2, mask=mask), 3, mask)
    index = reg.vadd(reg.vshl(slot, 6, mask=mask), reg.vshr(ids, 4, mask=mask), mask=mask)
    index = reg.vadds(index, (segment // 16) * 256 + (segment % 16) * 4, mask=mask)
    word = reg.vreinterpret(reg.vgather(scratch, index, mask=mask), dtypes.uint32)
    reverse = reg.vreinterpret(reg.vgather(reversed_words, index, mask=mask), dtypes.uint32)
    lane = bitwise_and_scalar(ids, 3, mask)
    even = reg.veqs(bitwise_and_scalar(lane, 1, mask), 0, mask=mask)
    selected = reg.vselect(word, reverse, cond_mask=even)
    # reverse32 同时交换两个半字，因此第 0/3 个 lane 取高半字。
    high_half = reg.mask_or(reg.veqs(lane, 0, mask=mask), reg.veqs(lane, 3, mask=mask), exec_mask=mask)
    return reg.vselect(reg.vshr(selected, 16, mask=mask), bitwise_and_scalar(selected, 0xFFFF, mask), cond_mask=high_half)


@jit
def reverse_random_words(self, batches):
    # A complete 4096-word RNG tile gives eight independent words per
    # thread. Load them together, then reverse/store without a SIMT loop.
    # Short exact batches also avoid a SIMT loop. Other runtime counts,
    # including larger BF16 tiles, retain the general path.
    if batches * 256 == SIMT_THREADS * 8:
        with vf(mode="simt", thread=SIMT_THREADS):
            index = simt.thread_idx()[0]
            word0 = self.random_words[0, index + SIMT_THREADS * 0]
            word1 = self.random_words[0, index + SIMT_THREADS * 1]
            word2 = self.random_words[0, index + SIMT_THREADS * 2]
            word3 = self.random_words[0, index + SIMT_THREADS * 3]
            word4 = self.random_words[0, index + SIMT_THREADS * 4]
            word5 = self.random_words[0, index + SIMT_THREADS * 5]
            word6 = self.random_words[0, index + SIMT_THREADS * 6]
            word7 = self.random_words[0, index + SIMT_THREADS * 7]
            self.reversed_words[0, index + SIMT_THREADS * 0] = _reverse32_scalar(word0)
            self.reversed_words[0, index + SIMT_THREADS * 1] = _reverse32_scalar(word1)
            self.reversed_words[0, index + SIMT_THREADS * 2] = _reverse32_scalar(word2)
            self.reversed_words[0, index + SIMT_THREADS * 3] = _reverse32_scalar(word3)
            self.reversed_words[0, index + SIMT_THREADS * 4] = _reverse32_scalar(word4)
            self.reversed_words[0, index + SIMT_THREADS * 5] = _reverse32_scalar(word5)
            self.reversed_words[0, index + SIMT_THREADS * 6] = _reverse32_scalar(word6)
            self.reversed_words[0, index + SIMT_THREADS * 7] = _reverse32_scalar(word7)
    elif batches == 8:
        with vf(mode="simt", thread=512):
            index = simt.thread_idx()[0]
            word0 = self.random_words[0, index + 0]
            word1 = self.random_words[0, index + 512]
            word2 = self.random_words[0, index + 1024]
            word3 = self.random_words[0, index + 1536]
            self.reversed_words[0, index + 0] = _reverse32_scalar(word0)
            self.reversed_words[0, index + 512] = _reverse32_scalar(word1)
            self.reversed_words[0, index + 1024] = _reverse32_scalar(word2)
            self.reversed_words[0, index + 1536] = _reverse32_scalar(word3)
    elif batches == 4:
        with vf(mode="simt", thread=512):
            index = simt.thread_idx()[0]
            word0 = self.random_words[0, index + 0]
            word1 = self.random_words[0, index + 512]
            self.reversed_words[0, index + 0] = _reverse32_scalar(word0)
            self.reversed_words[0, index + 512] = _reverse32_scalar(word1)
    # Exact 5/7/12-batch tiles avoid the general SIMT loop.
    elif batches == 12:
        with vf(mode="simt", thread=512):
            index = simt.thread_idx()[0]
            word0 = self.random_words[0, index + 0]
            word1 = self.random_words[0, index + 512]
            word2 = self.random_words[0, index + 1024]
            word3 = self.random_words[0, index + 1536]
            word4 = self.random_words[0, index + 2048]
            word5 = self.random_words[0, index + 2560]
            self.reversed_words[0, index + 0] = _reverse32_scalar(word0)
            self.reversed_words[0, index + 512] = _reverse32_scalar(word1)
            self.reversed_words[0, index + 1024] = _reverse32_scalar(word2)
            self.reversed_words[0, index + 1536] = _reverse32_scalar(word3)
            self.reversed_words[0, index + 2048] = _reverse32_scalar(word4)
            self.reversed_words[0, index + 2560] = _reverse32_scalar(word5)
    elif batches == 7:
        with vf(mode="simt", thread=512):
            index = simt.thread_idx()[0]
            word0 = self.random_words[0, index]
            word1 = self.random_words[0, index + 512]
            word2 = self.random_words[0, index + 1024]
            self.reversed_words[0, index] = _reverse32_scalar(word0)
            self.reversed_words[0, index + 512] = _reverse32_scalar(word1)
            self.reversed_words[0, index + 1024] = _reverse32_scalar(word2)
            if index < 256:
                word3 = self.random_words[0, index + 1536]
                self.reversed_words[0, index + 1536] = _reverse32_scalar(word3)
    elif batches == 5:
        with vf(mode="simt", thread=256):
            index = simt.thread_idx()[0]
            word0 = self.random_words[0, index + 0]
            word1 = self.random_words[0, index + 256]
            word2 = self.random_words[0, index + 512]
            word3 = self.random_words[0, index + 768]
            word4 = self.random_words[0, index + 1024]
            self.reversed_words[0, index + 0] = _reverse32_scalar(word0)
            self.reversed_words[0, index + 256] = _reverse32_scalar(word1)
            self.reversed_words[0, index + 512] = _reverse32_scalar(word2)
            self.reversed_words[0, index + 768] = _reverse32_scalar(word3)
            self.reversed_words[0, index + 1024] = _reverse32_scalar(word4)
    elif batches == 2:
        with vf(mode="simt", thread=512):
            index = simt.thread_idx()[0]
            word0 = self.random_words[0, index + 0]
            self.reversed_words[0, index + 0] = _reverse32_scalar(word0)
    elif batches == 1:
        with vf(mode="simt", thread=256):
            index = simt.thread_idx()[0]
            word0 = self.random_words[0, index + 0]
            self.reversed_words[0, index + 0] = _reverse32_scalar(word0)
    else:
        with vf(mode="simt", thread=SIMT_THREADS):
            for index in range(simt.thread_idx()[0], batches * 256, SIMT_THREADS):
                word = self.random_words[0, index]
                self.reversed_words[0, index] = _reverse32_scalar(word)


@jit
def _finish_philox_quad(
        self, a0, a1, a2, a3, b0, b1, b2, b3,
        c0, c1, c2, c3, d0, d1, d2, d3,
        m0, m1, key0, key1, quad, mask):
    """Finish full Philox10 and write four 64-group random batches."""
    for step in range(1, 10, 3):
        (a0, a1, a2, a3, b0, b1, b2, b3,
         c0, c1, c2, c3, d0, d1, d2, d3) = _philox_quad_three(
            a0, a1, a2, a3, b0, b1, b2, b3,
            c0, c1, c2, c3, d0, d1, d2, d3,
            step, m0, m1, key0, key1, mask)
    _store_random_words_aos(self.random_words, quad * 1024, a0, a1, a2, a3, mask)
    _store_random_words_aos(self.random_words, quad * 1024 + 256, b0, b1, b2, b3, mask)
    _store_random_words_aos(self.random_words, quad * 1024 + 512, c0, c1, c2, c3, mask)
    _store_random_words_aos(self.random_words, quad * 1024 + 768, d0, d1, d2, d3, mask)


@jit
def prepare_random_words_aos(self, start, batches, quant_length, seed, block_offset):
    key0, key1 = _split_uint64(seed)
    axis_groups, reciprocal, row_base, col_base = _rng_geometry(start, quant_length)
    row_low, row_high = _split_uint64(row_base)
    col_low, col_high = _split_uint64(col_base)
    step_q = dtypes.int64(64) // axis_groups
    step_r = dtypes.int64(64) % axis_groups
    pair_q = dtypes.int64(128) // axis_groups
    pair_r = dtypes.int64(128) % axis_groups
    quad_q = dtypes.int64(256) // axis_groups
    quad_r = dtypes.int64(256) % axis_groups
    if batches < 4:
        with vf(mode="simd"):
            rl, rh, cl, ch = row_low, row_high, col_low, col_high
            for batch in range(batches):
                row_cursor, col_cursor = _cursor_coordinates(rl, rh, cl, ch)
                _generate_random_words_aos_keyed(
                    self.random_words, row_cursor, col_cursor, axis_groups, reciprocal,
                    batch * 256, key0, key1, block_offset)
                rl, rh, cl, ch = _advance_cursor_words(
                    row_cursor, col_cursor, step_q, step_r, axis_groups)
            # Publish SIMD Philox stores in random_words before SIMT bit reverse reads.
            reg.vmem_bar("vst_vld")
    elif batches == 16 and axis_groups >= 256:
        with vf(mode="simd"):
            mask = reg.full_mask()
            ids = reg.varange(0, dtypes.uint32)
            m0 = broadcast_u32(PHILOX_M0, mask)
            m1 = broadcast_u32(PHILOX_M1, mask)
            rl, rh, cl, ch = row_low, row_high, col_low, col_high
            for quad in range(4):
                row0, col0 = _cursor_coordinates(rl, rh, cl, ch)
                row1, col1 = _advance_group(row0, col0, step_q, step_r, axis_groups)
                row2, col2 = _advance_group(row0, col0, pair_q, pair_r, axis_groups)
                row3, col3 = _advance_group(row2, col2, step_q, step_r, axis_groups)
                empty = broadcast_u32(0, mask)
                a0, a1, a2, a3 = empty, empty, empty, empty
                b0, b1, b2, b3 = empty, empty, empty, empty
                c0, c1, c2, c3 = empty, empty, empty, empty
                d0, d1, d2, d3 = empty, empty, empty, empty
                if col0 <= axis_groups - 256:
                    # All four batches remain in one original row. Compute
                    # their shared TE task base and draw once as u32 pairs.
                    # Round the starting column down to a task boundary.
                    # The existing TE vector helper performs one full u64
                    # row-task vmull and pair_add for this quad, including
                    # the offset/draw counter carry.
                    lane_start = col0 % 8
                    aligned_col = col0 - lane_start
                    row_low_bits, row_high_bits = _split_uint64(row0)
                    col_low_bits, col_high_bits = _split_uint64(aligned_col)
                    counter_low, counter_high, sub_low, sub_high = (
                        _nvfp4_counter_coordinates(
                            broadcast_u32(row_low_bits, mask),
                            broadcast_u32(row_high_bits, mask),
                            broadcast_u32(col_low_bits, mask),
                            broadcast_u32(col_high_bits, mask),
                            axis_groups, block_offset, mask))
                    a0, a1, a2, a3 = _philox_counter_first_same_row(
                        sub_low, sub_high, counter_low, counter_high,
                        lane_start, 0, ids, m0, m1, key0, key1, mask)
                    b0, b1, b2, b3 = _philox_counter_first_same_row(
                        sub_low, sub_high, counter_low, counter_high,
                        lane_start, 64, ids, m0, m1, key0, key1, mask)
                    c0, c1, c2, c3 = _philox_counter_first_same_row(
                        sub_low, sub_high, counter_low, counter_high,
                        lane_start, 128, ids, m0, m1, key0, key1, mask)
                    d0, d1, d2, d3 = _philox_counter_first_same_row(
                        sub_low, sub_high, counter_low, counter_high,
                        lane_start, 192, ids, m0, m1, key0, key1, mask)
                else:
                    a0, a1, a2, a3 = _philox_counter_first(
                        row0, col0, axis_groups, reciprocal, ids,
                        m0, m1, key0, key1, block_offset, mask)
                    b0, b1, b2, b3 = _philox_counter_first(
                        row1, col1, axis_groups, reciprocal, ids,
                        m0, m1, key0, key1, block_offset, mask)
                    c0, c1, c2, c3 = _philox_counter_first(
                        row2, col2, axis_groups, reciprocal, ids,
                        m0, m1, key0, key1, block_offset, mask)
                    d0, d1, d2, d3 = _philox_counter_first(
                        row3, col3, axis_groups, reciprocal, ids,
                        m0, m1, key0, key1, block_offset, mask)
                _finish_philox_quad(
                    self, a0, a1, a2, a3, b0, b1, b2, b3,
                    c0, c1, c2, c3, d0, d1, d2, d3,
                    m0, m1, key0, key1, quad, mask)
                rl, rh, cl, ch = _advance_cursor_words(
                    row0, col0, quad_q, quad_r, axis_groups)
            reg.vmem_bar("vst_vld")
    else:
        with vf(mode="simd"):
            mask = reg.full_mask()
            ids = reg.varange(0, dtypes.uint32)
            m0 = broadcast_u32(PHILOX_M0, mask)
            m1 = broadcast_u32(PHILOX_M1, mask)
            rl, rh, cl, ch = row_low, row_high, col_low, col_high
            for pair in range(batches // 2):
                row_cursor, col_cursor = _cursor_coordinates(rl, rh, cl, ch)
                second_row, second_col = _advance_group(
                    row_cursor, col_cursor, step_q, step_r, axis_groups)
                a0, a1, a2, a3 = _philox_counter_first(
                    row_cursor, col_cursor, axis_groups, reciprocal,
                    ids, m0, m1, key0, key1, block_offset, mask)
                b0, b1, b2, b3 = _philox_counter_first(
                    second_row, second_col, axis_groups, reciprocal,
                    ids, m0, m1, key0, key1, block_offset, mask)
                for step in range(1, 10, 3):
                    a0, a1, a2, a3, b0, b1, b2, b3 = _philox_pair_three(
                        a0, a1, a2, a3, b0, b1, b2, b3, step, m0, m1, key0, key1, mask)
                _store_random_words_aos(self.random_words, pair * 512, a0, a1, a2, a3, mask)
                _store_random_words_aos(self.random_words, pair * 512 + 256, b0, b1, b2, b3, mask)
                rl, rh, cl, ch = _advance_cursor_words(
                    row_cursor, col_cursor, pair_q, pair_r, axis_groups)
            # Publish SIMD Philox stores in random_words before SIMT bit reverse reads.
            reg.vmem_bar("vst_vld")
        if batches % 2 != 0:
            tail_batch = batches // 2 * 2
            _, _, tail_row, tail_col = _rng_geometry(
                dtypes.int64(start) + dtypes.int64(tail_batch) * 1024, quant_length)
            with vf(mode="simd"):
                _generate_random_words_aos_keyed(
                    self.random_words, tail_row, tail_col, axis_groups, reciprocal,
                    tail_batch * 256, key0, key1, block_offset)
                # Publish SIMD Philox stores in random_words before SIMT bit reverse reads.
                reg.vmem_bar("vst_vld")


@jit
def prepare_random_words(self, start, batches, quant_length, seed, block_offset):
    key0, key1 = _split_uint64(seed)
    axis_groups, reciprocal, row_base, col_base = _rng_geometry(start, quant_length)
    row_low, row_high = _split_uint64(row_base)
    col_low, col_high = _split_uint64(col_base)
    step_q = dtypes.int64(64) // axis_groups
    step_r = dtypes.int64(64) % axis_groups
    with vf(mode="simd"):
        rl, rh, cl, ch = row_low, row_high, col_low, col_high
        for batch in range(batches):
            row_cursor, col_cursor = _cursor_coordinates(rl, rh, cl, ch)
            _generate_random_words_keyed(
                self.random_words, row_cursor, col_cursor, axis_groups, reciprocal,
                batch * 256, key0, key1, block_offset)
            rl, rh, cl, ch = _advance_cursor_words(
                row_cursor, col_cursor, step_q, step_r, axis_groups)
        # Publish SIMD Philox stores in random_words before SIMT bit reverse reads.
        reg.vmem_bar("vst_vld")


# Scale computation and stochastic quantization


def _sr_fp32_to_fp8_q16(value, r16, mantissa, bias, max_code, mask):
    """Encode base/discarded16 together; retain Algorithm 19's exact carry."""
    bits = reg.vreinterpret(value, dtypes.uint32)
    sign = bitwise_and_scalar(reg.vshr(bits, 24, mask=mask), 0x80, mask)
    absolute = bitwise_and_scalar(bits, 0x7FFFFFFF, mask)
    threshold = (128 - bias) << 23
    normal_q = reg.vsub(
        reg.vshr(absolute, 7 - mantissa, mask=mask),
        broadcast_u32((127 - bias) << (mantissa + 16), mask), mask=mask,
    )
    # Only subnormal lanes consume this conversion. Out-of-range normal
    # lanes are discarded by vselect; NaNs are canonicalized below.
    absolute_value = reg.vreinterpret(absolute, dtypes.float32)
    scaled = reg.vmuls(absolute_value, float(2 ** (bias + mantissa + 15)), mask=mask)
    sub_q = reg.vreinterpret(
        reg.vcast(scaled, dtypes.int32, mask=mask, rounding=reg.RoundingMode.RZ), dtypes.uint32,
    )
    q = reg.vselect(normal_q, sub_q, cond_mask=reg.vges(absolute, threshold, mask=mask))
    code = reg.vshr(reg.vadd(q, bitwise_and_scalar(r16, 0xFFFF, mask), mask=mask), 16, mask=mask)
    code = reg.vbitwise_or(reg.vmins(code, max_code, mask=mask), sign, mask=mask)
    return reg.vselect(broadcast_u32(0x7F, mask), code, cond_mask=reg.vgts(absolute, 0x7F800000, mask=mask))


def _sr_fp32_to_fp8(value, r16, mantissa, bias, max_code, mask):
    """FP32 bits + explicit low-16 random fields -> FP8 low-byte codes."""
    return _sr_fp32_to_fp8_q16(value, r16, mantissa, bias, max_code, mask)


def _scale_alg0(amax_bits, inv_max_bits, mask, canonical_invalid=True):
    """ASC ComputeScaleOcp: floor exponent; zero reciprocal when scale is zero."""
    shift = 15 if inv_max_bits == 0x37924925 else 8
    exponent = reg.vshr(amax_bits, 23, mask=mask)
    invalid = reg.vges(exponent, 255, mask=mask)
    code = reg.vadds(reg.vmaxs(exponent, shift, mask=mask), (-shift) & 0xFFFFFFFF, mask=mask)
    # Preserve finite lanes; fill invalid codes without a constant vector.
    code = reg.vdups(255, dtypes.uint32, mask=invalid, mode="merging", merge=code)
    # Finite FP8 scale codes cannot reach 254 (max exponent minus 8/15).
    # Masked shifts zero inactive lanes, including the code-0 multiplier.
    reciprocal = reg.vshl(reg.vsub(broadcast_u32(254, mask), code, mask=mask), 23,
                         mask=reg.vnes(code, 0, mask=mask))
    if canonical_invalid:
        reciprocal = reg.vdups(0x7F810000, dtypes.uint32, mask=invalid, mode="merging", merge=reciprocal)
    return code, reg.vreinterpret(reciprocal, dtypes.float32)


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
    shift = 15 if inv_max_bits == 0x37924925 else 8
    invalid = reg.vges(amax_bits, 0x7F800000, mask=mask)
    # Active finite lanes cannot underflow after this bias adjustment;
    # inactive/invalid lanes are replaced by the masks below.
    code = reg.vshr(reg.vadds(amax_bits,
        (0x1FFFFF - (shift << 23)) & 0xFFFFFFFF, mask=mask), 23,
        mask=reg.vgts(amax_bits, (shift << 23) | 0x600001, mask=mask))
    # Preserve finite lanes; fill invalid codes without a constant vector.
    code = reg.vdups(255, dtypes.uint32, mask=invalid, mode="merging", merge=code)
    recip = reg.vshl(reg.vsub(broadcast_u32(254, mask), code, mask=mask), 23, mask=mask)
    if canonical_invalid:
        recip = reg.vdups(0x7F810000, dtypes.uint32, mask=invalid, mode="merging", merge=recip)
    # amax==0 means the complete block contains only signed zeros. Multiplying
    # those by the finite code-0 multiplier preserves their output bytes.
    return code, reg.vreinterpret(recip, dtypes.float32)


def _sr_magnitude_q16_unbiased_sum(absolute, r16, mantissa, bias, mask):
    # Normal lanes double exactly. Small lanes align the significand to the
    # FP8 subnormal grid with a half-ULP downward bias before FP32 rounding.
    value = reg.vreinterpret(absolute, dtypes.float32)
    magic = 2.0 ** (1 - bias) - 2.0 ** (-bias - 23)
    mapped = reg.vadd(value, reg.vmaxs(value, magic, mask=mask), mask=mask)
    # The common integer-code bias is removed once after byte packing.
    q = reg.vshr(reg.vreinterpret(mapped, dtypes.uint32), 7 - mantissa, mask=mask)
    return reg.vadd(q, r16, mask=mask)


def compute_scale(self, maxima, mask, canonical_invalid=True):
    """Shared scale rule; packed SR handles invalid output bytes itself."""
    if const_expr(self.scale_alg == 1 and self.max_low_bound_bits != 0):
        original_zero = reg.veqs(maxima, 0, mask=mask)
        maxima = reg.vmaxs(maxima, self.max_low_bound_bits, mask=mask)
        maxima = reg.vselect(broadcast_u32(0, mask), maxima, cond_mask=original_zero)
    if const_expr(self.scale_alg == 0):
        return _scale_alg0(maxima, self.inv_max_bits, mask, canonical_invalid)
    return _scale_alg1(maxima, self.inv_max_bits, mask, canonical_invalid)


@jit
def store_scale(self, scales, count, scale_factor, segments, ids, full):
    # SIMD wrote scale32; packing reads those u32 E8M0 codes.
    reg.vmem_bar("vst_vld")
    scale_count = segments * 2
    if scale_factor == 4:
        scale_count = (count // 32) * 2
    for offset in range(0, scale_count, 64):
        scale_mask = reg.update_mask(scale_count - offset, elem_bits=32)[0]
        scale_ids = reg.vadds(ids, offset, mask=full)
        if scale_factor == 4:
            scale_codes = reg.vgather(self.scale32, reg.vshr(scale_ids, 1, mask=full), mask=scale_mask)
            even = reg.veqs(bitwise_and_scalar(ids, 1, full), 0, mask=full)
            scale_codes = reg.vselect(scale_codes, broadcast_u32(0, full), cond_mask=even)
            reg.vstore_pack(scales, offset, scale_codes, scale_mask, pack_mode=reg.PackMode.B32_TO_B8)
        else:
            scale_codes = reg.vgather(self.scale32, scale_ids, mask=scale_mask)
            reg.vstore_pack(scales, offset, scale_codes, scale_mask, pack_mode=reg.PackMode.B32_TO_B8)
    # SIMD packed output scales; subsequent MTE reads them for GM.
    reg.vmem_bar("vst_vld")


@jit
def quantize_tile(self, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset):
    segments = (count + 63) // 64
    batches = (count + 1023) // 1024
    prepare_random_words(self, start, batches, quant_length, seed, block_offset)
    # 随机字按同一核 SIMD → SIMT → SIMD 传递，仅 SIMT 产生反转后的副本。
    reverse_random_words(self, batches)
    # One VF produces the whole batch. Per-segment VFs would rotate the
    # output Channel slot early, splitting one DMA tile across two slots.
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
            absolute = bitwise_and_scalar(reg.vreinterpret(values, dtypes.uint32), 0x7FFFFFFF, full)
            max0 = reg.vdup(reg.vreduce_max(absolute, mask=lower), mask=full)
            max1 = reg.vdup(reg.vreduce_max(absolute, mask=upper), mask=full)
            maxima = reg.vselect(max0, max1, cond_mask=lower)
            block_scales, reciprocal = compute_scale(self, maxima, full)
            normalized = reg.vmul(values, reciprocal, mask=full)
            if const_expr(self.bf16):
                rounded = reg.vcast(normalized, dtypes.bfloat16, mask=full, rounding=reg.RoundingMode.RN)
                normalized = reg.vcast(rounded, dtypes.float32, mask=full)
            # RNG follows the unpadded logical input, never UB pitch/grid.
            random16 = _random_fields_from_words(ids, self.random_words, self.reversed_words, segment, full)
            codes = _sr_fp32_to_fp8(normalized, random16, self.mantissa, self.bias, self.max_code, full)
            reg.vstore_pack(y, offset, codes, active, pack_mode=reg.PackMode.B32_TO_B8)
            reg.vstore_first(self.scale32, segment * 2, block_scales)
            upper_scale = reg.vgather_reg(block_scales, broadcast_u32(32, full))
            reg.vstore_first(self.scale32, segment * 2 + 1, upper_scale)
        store_scale(self, scales, count, scale_factor, segments, ids, full)


@jit
def quantize_tile_256(self, x, y, scales, start, count, scale_factor, quant_length, seed, block_offset):
    sign_source = reinterpret(x, dtypes.uint8 if self.bf16 else dtypes.uint16, shape=(1, self.tile_elements * 2))
    parts = count // 256
    batches = (count + 1023) // 1024
    prepare_random_words_aos(self, start, batches, quant_length, seed, block_offset)
    # 随机字按同一核 SIMD → SIMT → SIMD 传递，仅 SIMT 产生反转后的副本。
    reverse_random_words(self, batches)
    # One producer for each complete DMA tile; keep y -> scales order.
    with vf(mode="simd"):
        full = reg.full_mask()
        full16 = reg.full_mask(elem_bits=16)
        full8 = reg.full_mask(elem_bits=8)
        ids = reg.varange(0, dtypes.uint32)
        scale_index = reg.vshr(ids, 3, mask=full)
        sign_mask8 = reg.vdups(0x80, dtypes.uint8, mask=full8)
        # Gather order maps four 64-byte groups back to interleaved FP8.
        pack_ids8 = reg.varange(0, dtypes.uint8)
        pack_group8 = reg.vbitwise_and(pack_ids8, reg.vdups(3, dtypes.uint8, mask=full8), mask=full8)
        pack_index8 = reg.vbitwise_or(reg.vshl(pack_group8, 6, mask=full8), reg.vshr(pack_ids8, 2, mask=full8), mask=full8)
        random_mask32 = broadcast_u32(0xFFFF, full)
        # Each uint32 scale contributes its low byte, plus a zero byte
        # for K32 padding. Stream these bytes directly to the output.
        scale_byte_slot = reg.vbitwise_and(pack_ids8, reg.vdups(3, dtypes.uint8, mask=full8), mask=full8)
        scale_byte_mask = reg.mask_and(reg.vlts(pack_ids8, 32, mask=full8),
            reg.vlts(scale_byte_slot, dtypes.uint8(scale_factor // 2), mask=full8), exec_mask=full8)
        scale_cursor = reg.vstore_unalign_begin(scales)
        for part in range(parts):
            offset = part * 256
            if const_expr(self.bf16):
                a, b = reg.vload_deinterleave(x, offset)
                abs_mask = reg.vdups(0x7FFF, dtypes.uint16, mask=full16)
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
                unused_signbytes, signbytes = reg.vdeinterleave(reg.vreinterpret_lanes(sign01, dtypes.uint8), reg.vreinterpret_lanes(sign23, dtypes.uint8))
                signs = reg.vreinterpret_lanes(reg.vbitwise_and(signbytes, sign_mask8, mask=full8), dtypes.uint32)
            block_scales, reciprocal = compute_scale(self, maxima, full, False)
            if const_expr(self.bf16):
                # Pack selects the destination register half, not the
                # source halfword: first move BF16 bits into the low16.
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
                # Expand each invalid scale to all 32-bit packed output words.
                invalid_reciprocal = reg.vreinterpret_lanes(broadcast, dtypes.uint32)
            else:
                reciprocal = reg.vgather_reg(reciprocal, scale_index)
                v0 = reg.vmul(reg.vreinterpret(a0, dtypes.float32), reciprocal, mask=full)
                v1 = reg.vmul(reg.vreinterpret(a1, dtypes.float32), reciprocal, mask=full)
                v2 = reg.vmul(reg.vreinterpret(a2, dtypes.float32), reciprocal, mask=full)
                v3 = reg.vmul(reg.vreinterpret(a3, dtypes.float32), reciprocal, mask=full)
                invalid_reciprocal = reciprocal
            word = reg.vreinterpret(reg.vload(self.random_words, part * 64), dtypes.uint32)
            reversed_word = reg.vreinterpret(reg.vload(self.reversed_words, part * 64), dtypes.uint32)
            r0 = reg.vshr(word, 16, mask=full)
            r1 = reg.vbitwise_and(reversed_word, random_mask32, mask=full)
            r2 = reg.vbitwise_and(word, random_mask32, mask=full)
            r3 = reg.vshr(reversed_word, 16, mask=full)
            c0 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v0, dtypes.uint32), r0,
                self.mantissa, self.bias, full)
            c1 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v1, dtypes.uint32), r1,
                self.mantissa, self.bias, full)
            c2 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v2, dtypes.uint32), r2,
                self.mantissa, self.bias, full)
            c3 = _sr_magnitude_q16_unbiased_sum(
                reg.vreinterpret(v3, dtypes.uint32), r3,
                self.mantissa, self.bias, full)
            # Select bits 23:16 directly; avoid four right shifts
            # and the following per-word shifts/ORs.
            unused_code01, code01 = reg.vdeinterleave(reg.vreinterpret_lanes(c0, dtypes.uint16), reg.vreinterpret_lanes(c1, dtypes.uint16))
            unused_code23, code23 = reg.vdeinterleave(reg.vreinterpret_lanes(c2, dtypes.uint16), reg.vreinterpret_lanes(c3, dtypes.uint16))
            code_bytes, unused_code_bytes = reg.vdeinterleave(reg.vreinterpret_lanes(code01, dtypes.uint8), reg.vreinterpret_lanes(code23, dtypes.uint8))
            packed = reg.vreinterpret_lanes(reg.vgather_reg(code_bytes, pack_index8), dtypes.uint32)
            # Every packed word belongs to one block32. Its scale
            # marks all four bytes invalid together for Inf/NaN input.
            # Q16 bias contains whole code units; remove it once
            # per packed byte after all four carry additions.
            magnitude8 = reg.vadds(reg.vreinterpret_lanes(packed, dtypes.uint8),
                                  (-((128 - self.bias) << self.mantissa)) & 255, mask=full8)
            # Masked min zeroes inactive lanes, including the -1
            # Q16 sentinel (255 after packing), and saturates others.
            saturated = reg.vmins(magnitude8, self.max_code,
                                 mask=reg.vnes(magnitude8, 255, mask=full8))
            packed = reg.vbitwise_or(reg.vreinterpret_lanes(saturated, dtypes.uint32), signs, mask=full)
            # The fast scale path leaves code 255's multiplier as
            # -Inf; override all four output bytes explicitly here.
            invalid = reg.veqs(reg.vreinterpret(invalid_reciprocal, dtypes.uint32), self.invalid_packed_reciprocal, mask=full)
            packed = reg.vdups(0x7F7F7F7F, dtypes.uint32, mask=invalid, mode="merging", merge=packed)
            reg.vstore(y, offset, reg.vreinterpret_lanes(packed, dtypes.uint8), full8)
            scale_bytes = reg.vsqueeze_and_storeunalign_init(
                reg.vreinterpret_lanes(block_scales, dtypes.uint8), mask=scale_byte_mask)
            reg.vsqueeze_and_storeunalign(scales, 0, scale_bytes, scale_cursor)
        reg.vsqueeze_and_storeunalign_finalize(scales, 0, scale_cursor)
        # SIMD writes packed output/scale; subsequent MTE copies consume them.
        reg.vmem_bar("vst_vld")


@jit
def quantize_short_rows(self, x, y, scales, start, rows, columns, pitch_shift, seed, block_offset):
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
    step_q, step_r = 64 // axis_groups, 64 % axis_groups
    with vf(mode="simd"):
        rl, rh, cl, ch = row_low, row_high, col_low, col_high
        for batch in range(batches):
            batch_row, batch_col = _cursor_coordinates(rl, rh, cl, ch)
            _generate_random_words_aos_keyed(self.random_words, batch_row, batch_col,
                                       axis_groups, reciprocal, batch * 256, key0, key1, block_offset)
            rl, rh, cl, ch = _advance_cursor_words(
                batch_row, batch_col, step_q, step_r, axis_groups)
        # SIMD stores Philox words; the following SIMT VF reverses this UB.
        reg.vmem_bar("vst_vld")
    # 随机字按同一核 SIMD → SIMT → SIMD 传递，仅 SIMT 产生反转后的副本。
    reverse_random_words(self, batches)
    # The single VF owns the complete output/scale Channel transactions.
    with vf(mode="simd"):
        full = reg.full_mask()
        ids = reg.varange(0, dtypes.uint32)
        eight = reg.update_mask(8, elem_bits=32)[0]
        scale_index = reg.vshr(ids, 3, mask=full)
        zero = broadcast_u32(0, full)
        wide_pitch = reg.veqs(broadcast_u32(pitch_bits, full), 7, mask=full)
        for part in range(parts):
            virtual4 = reg.vadds(ids, part * 64, mask=full)
            row = reg.vselect(reg.vshr(virtual4, 5, mask=full),
                             reg.vshr(virtual4, 4, mask=full), cond_mask=wide_pitch)
            column4 = reg.vbitwise_and(virtual4, broadcast_u32(column_mask, full), mask=full)
            logical4 = reg.vadd(reg.vmuls(row, columns4, mask=full), column4, mask=full)
            valid = reg.mask_and(reg.vlts(row, row_count, mask=full),
                                reg.vlts(column4, columns4, mask=full), exec_mask=full)
            if const_expr(self.bf16):
                index = reg.vshl(logical4, 1, mask=full)
                a = reg.vgather(packed_x, index, mask=valid)
                b = reg.vgather(packed_x, reg.vadds(index, 1, mask=full), mask=valid)
                a = reg.vselect(a, zero, cond_mask=valid)
                b = reg.vselect(b, zero, cond_mask=valid)
                v0 = reg.vreinterpret(reg.vshl(a, 16, mask=full), dtypes.float32)
                v1 = reg.vreinterpret(bitwise_and_scalar(a, 0xFFFF0000, full), dtypes.float32)
                v2 = reg.vreinterpret(reg.vshl(b, 16, mask=full), dtypes.float32)
                v3 = reg.vreinterpret(bitwise_and_scalar(b, 0xFFFF0000, full), dtypes.float32)
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
            a0 = bitwise_and_scalar(reg.vreinterpret(v0, dtypes.uint32), 0x7FFFFFFF, full)
            a1 = bitwise_and_scalar(reg.vreinterpret(v1, dtypes.uint32), 0x7FFFFFFF, full)
            a2 = bitwise_and_scalar(reg.vreinterpret(v2, dtypes.uint32), 0x7FFFFFFF, full)
            a3 = bitwise_and_scalar(reg.vreinterpret(v3, dtypes.uint32), 0x7FFFFFFF, full)
            maxima = reg.vreduce_max_datablock(
                reg.vmax(reg.vmax(a0, a1, mask=full), reg.vmax(a2, a3, mask=full), mask=full), mask=full)
            block_scales, reciprocal = compute_scale(self, maxima, full)
            reciprocal = reg.vgather_reg(reciprocal, scale_index)
            v0 = reg.vmul(v0, reciprocal, mask=full)
            v1 = reg.vmul(v1, reciprocal, mask=full)
            v2 = reg.vmul(v2, reciprocal, mask=full)
            v3 = reg.vmul(v3, reciprocal, mask=full)
            if const_expr(self.bf16):
                v0 = reg.vcast(reg.vcast(v0, dtypes.bfloat16, mask=full, rounding=reg.RoundingMode.RN), dtypes.float32, mask=full)
                v1 = reg.vcast(reg.vcast(v1, dtypes.bfloat16, mask=full, rounding=reg.RoundingMode.RN), dtypes.float32, mask=full)
                v2 = reg.vcast(reg.vcast(v2, dtypes.bfloat16, mask=full, rounding=reg.RoundingMode.RN), dtypes.float32, mask=full)
                v3 = reg.vcast(reg.vcast(v3, dtypes.bfloat16, mask=full, rounding=reg.RoundingMode.RN), dtypes.float32, mask=full)
            # RNG index follows compact input, never the padded row pitch.
            word = reg.vreinterpret(reg.vgather(self.random_words, logical4, mask=valid), dtypes.uint32)
            reversed_word = reg.vreinterpret(reg.vgather(self.reversed_words, logical4, mask=valid), dtypes.uint32)
            rab = reg.vshr(word, 16, mask=full)
            ref = bitwise_and_scalar(word, 0xFFFF, full)
            c0 = _sr_fp32_to_fp8(v0, rab, self.mantissa, self.bias, self.max_code, full)
            c1 = _sr_fp32_to_fp8(v1, bitwise_and_scalar(reversed_word, 0xFFFF, full), self.mantissa, self.bias, self.max_code, full)
            c2 = _sr_fp32_to_fp8(v2, ref, self.mantissa, self.bias, self.max_code, full)
            c3 = _sr_fp32_to_fp8(v3, reg.vshr(reversed_word, 16, mask=full), self.mantissa, self.bias, self.max_code, full)
            lo = reg.vbitwise_or(c0, reg.vshl(c1, 8, mask=full), mask=full)
            hi = reg.vbitwise_or(reg.vshl(c2, 16, mask=full), reg.vshl(c3, 24, mask=full), mask=full)
            reg.vscatter(packed_y, reg.vbitwise_or(lo, hi, mask=full), logical4, mask=valid)
            reg.vstore(self.scale32, part * 8, block_scales, eight)
        # SIMD writes scale32; subsequent SIMD loads pack those E8M0 codes.
        reg.vmem_bar("vst_vld")
        for offset in range(0, scale_count, 64):
            scale_codes = reg.vload(self.scale32, offset)
            scale_mask = reg.update_mask(scale_count - offset, elem_bits=32)[0]
            reg.vstore_pack(scales, offset, scale_codes, scale_mask, pack_mode=reg.PackMode.B32_TO_B8)
        # SIMD writes packed output/scale; subsequent MTE copies consume them.
        reg.vmem_bar("vst_vld")


@jit
def quantize_wide64(self, x, y, scales, row, group, quant_length, width, d_offset, tile_rows, seed, block_offset):
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
            c0, c1, c2, c3 = _nvfp4_counter_rows(row_base, lane, col_group + counter, axis_groups, block_offset, full)
            w0, w1, w2, w3 = _philox_counter10(c0, c1, c2, c3, key0, key1, full)
            reg.vstore(self.random_words, counter * 256, reg.vreinterpret(w0, dtypes.int32), full)
            reg.vstore(self.random_words, counter * 256 + 64, reg.vreinterpret(w1, dtypes.int32), full)
            reg.vstore(self.random_words, counter * 256 + 128, reg.vreinterpret(w2, dtypes.int32), full)
            reg.vstore(self.random_words, counter * 256 + 192, reg.vreinterpret(w3, dtypes.int32), full)
        # SIMD stores Philox words; the following SIMT VF reverses this UB.
        reg.vmem_bar("vst_vld")
    reverse_random_words(self, tile_rows // 16)
    with vf(mode="simd"):
        full = reg.full_mask()
        for half in range(tile_rows // 32):
            maxima = broadcast_u32(0, full)
            for inner in range(32):
                offset = (half * 32 + inner) * width + d_offset
                if const_expr(self.bf16):
                    loaded = reg.vload_unpack(x, offset, unpack_mode=reg.UnpackMode.B16_TO_B32)
                    values = reg.vcast(loaded, dtypes.float32, mask=full)
                else:
                    values = reg.vload(x, offset)
                absolute = bitwise_and_scalar(reg.vreinterpret(values, dtypes.uint32), 0x7FFFFFFF, full)
                maxima = reg.vmax(maxima, absolute, mask=full)
            scale_codes, reciprocal = compute_scale(self, maxima, full)
            reg.vstore(self.scale32, half * 64, scale_codes, full)
            for word in range(8):
                slot = half * 512 + word * 64
                original = reg.vreinterpret(reg.vload(self.random_words, slot), dtypes.uint32)
                reversed_word = reg.vreinterpret(reg.vload(self.reversed_words, slot), dtypes.uint32)
                first = half * 32 + word * 4
                store_wide_row(self, x, y, first * width + d_offset, reciprocal,
                                     reg.vshr(original, 16, mask=full), full)
                store_wide_row(self, x, y, (first + 1) * width + d_offset, reciprocal,
                                     bitwise_and_scalar(reversed_word, 0xFFFF, full), full)
                store_wide_row(self, x, y, (first + 2) * width + d_offset, reciprocal,
                                     bitwise_and_scalar(original, 0xFFFF, full), full)
                store_wide_row(self, x, y, (first + 3) * width + d_offset, reciprocal,
                                     reg.vshr(reversed_word, 16, mask=full), full)
        # SIMD writes scale32; subsequent SIMD loads pack those E8M0 codes.
        reg.vmem_bar("vst_vld")
        pair_mask = reg.update_mask(64, elem_bits=16)[0]
        for pair in range(tile_rows // 64):
            first_scale = reg.vload(self.scale32, pair * 128)
            second_scale = reg.vload(self.scale32, pair * 128 + 64)
            code_pairs = reg.vbitwise_or(first_scale,
                                       reg.vshl(second_scale, 8, mask=full), mask=full)
            packed = reg.vpack(code_pairs, dtypes.uint16, part="lower")
            # A uint8 interleave would write 512 B, overlapping the next
            # D chunk. Store exactly 64 uint16 scale pairs (128 B).
            reg.vstore(scale_pairs, pair * width + d_offset, packed, pair_mask)
        # SIMD writes packed output/scale; subsequent MTE copies consume them.
        reg.vmem_bar("vst_vld")


@jit
def quantize_wide32(self, x, y, scales, row, group, quant_length, width, d_offset, tile_rows, seed, block_offset):
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
            c0, c1, c2, c3 = _nvfp4_counter_rows(row_base, lane, col_group + counter, axis_groups, block_offset, full)
            w0, w1, w2, w3 = _philox_counter10(c0, c1, c2, c3, key0, key1, full)
            reg.vstore(self.random_words, counter * 4 * lanes, reg.vreinterpret(w0, dtypes.int32), full)
            reg.vstore(self.random_words, counter * 4 * lanes + lanes, reg.vreinterpret(w1, dtypes.int32), full)
            reg.vstore(self.random_words, counter * 4 * lanes + 2 * lanes, reg.vreinterpret(w2, dtypes.int32), full)
            reg.vstore(self.random_words, counter * 4 * lanes + 3 * lanes, reg.vreinterpret(w3, dtypes.int32), full)
        # SIMD stores Philox words; the following SIMT VF reverses this UB.
        reg.vmem_bar("vst_vld")
    reverse_random_words(self, tile_rows * lanes // 1024)
    with vf(mode="simd"):
        full = reg.update_mask(lanes, elem_bits=32)[0]
        for half in range(tile_rows // 32):
            if const_expr(not self.bf16):
                full64 = reg.full_mask()
                lane64 = reg.varange(0, dtypes.uint32)
                maxima64 = broadcast_u32(0, full64)
                for inner in range(16):
                    offset = (half * 32 + inner * 2) * width + d_offset
                    values64 = reg.vload(x, offset)
                    absolute64 = bitwise_and_scalar(reg.vreinterpret(values64, dtypes.uint32), 0x7FFFFFFF, full64)
                    maxima64 = reg.vmax(maxima64, absolute64, mask=full64)
                swap32 = reg.vbitwise_xor(lane64, broadcast_u32(32, full64), mask=full64)
                maxima = reg.vmax(maxima64, reg.vgather_reg(maxima64, swap32), mask=full64)
            else:
                maxima = broadcast_u32(0, full)
                for inner in range(32):
                    offset = (half * 32 + inner) * width + d_offset
                    loaded = reg.vload_unpack(x, offset, unpack_mode=reg.UnpackMode.B16_TO_B32)
                    values = reg.vcast(loaded, dtypes.float32, mask=full)
                    absolute = bitwise_and_scalar(reg.vreinterpret(values, dtypes.uint32), 0x7FFFFFFF, full)
                    maxima = reg.vmax(maxima, absolute, mask=full)
            scale_codes, reciprocal = compute_scale(self, maxima, full)
            reg.vstore(self.scale32, half * lanes, scale_codes, full)
            for word in range(8):
                slot = half * 8 * lanes + word * lanes
                original = reg.vreinterpret(reg.vload(self.random_words, slot), dtypes.uint32)
                reversed_word = reg.vreinterpret(reg.vload(self.reversed_words, slot), dtypes.uint32)
                first = half * 32 + word * 4
                store_wide_row(self, x, y, first * width + d_offset, reciprocal,
                                     reg.vshr(original, 16, mask=full), full)
                store_wide_row(self, x, y, (first + 1) * width + d_offset, reciprocal,
                                     bitwise_and_scalar(reversed_word, 0xFFFF, full), full)
                store_wide_row(self, x, y, (first + 2) * width + d_offset, reciprocal,
                                     bitwise_and_scalar(original, 0xFFFF, full), full)
                store_wide_row(self, x, y, (first + 3) * width + d_offset, reciprocal,
                                     reg.vshr(reversed_word, 16, mask=full), full)
        # SIMD writes scale32; subsequent SIMD loads pack those E8M0 codes.
        reg.vmem_bar("vst_vld")
        pair_mask = reg.update_mask(lanes, elem_bits=16)[0]
        for pair in range(tile_rows // 64):
            first_scale = reg.vload(self.scale32, pair * 2 * lanes)
            second_scale = reg.vload(self.scale32, pair * 2 * lanes + lanes)
            code_pairs = reg.vbitwise_or(first_scale,
                                       reg.vshl(second_scale, 8, mask=full), mask=full)
            packed = reg.vpack(code_pairs, dtypes.uint16, part="lower")
            reg.vstore(scale_pairs, pair * width + d_offset, packed, pair_mask)
        # SIMD writes packed output/scale; subsequent MTE copies consume them.
        reg.vmem_bar("vst_vld")


@jit
def load_wide8_bf16_aligned(self, x, offset, mask):
    # BF16 rows occupy 16 bytes. vload_unpack requires a 32-byte UB
    # address, so load the containing row pair and select its eight lanes.
    aligned = (offset // 16) * 16
    loaded = reg.vload_unpack(x, aligned, unpack_mode=reg.UnpackMode.B16_TO_B32)
    sixteen = reg.update_mask(16, elem_bits=32)[0]
    values = reg.vcast(loaded, dtypes.float32, mask=sixteen)
    ids = reg.varange(0, dtypes.uint32)
    indices = reg.vadd(ids, broadcast_u32(offset % 16, mask), mask=mask)
    return reg.vgather_reg(values, indices)


@jit
def load_wide8_bf16_pair(self, x, offset, mask):
    # Two adjacent eight-element BF16 rows share one aligned 32-byte load.
    loaded = reg.vload_unpack(x, offset, unpack_mode=reg.UnpackMode.B16_TO_B32)
    sixteen = reg.update_mask(16, elem_bits=32)[0]
    values = reg.vcast(loaded, dtypes.float32, mask=sixteen)
    lanes = reg.varange(0, dtypes.uint32)
    upper = reg.vadd(lanes, broadcast_u32(8, mask), mask=mask)
    return reg.vgather_reg(values, lanes), reg.vgather_reg(values, upper)


@jit
def quantize_wide8(self, x, y, scales, row, group, quant_length, width, d_offset, tile_rows, seed, block_offset):
    """Quantize one S tile by compile-time physical D lanes without GM transposes."""
    lanes = self.wide_lanes
    row_base = dtypes.int64(row) * dtypes.int64(width) + dtypes.int64(d_offset)
    col_group = dtypes.int64(group) * dtypes.int64(tile_rows) // 16
    axis_groups = dtypes.int64(quant_length) // 16
    key0, key1 = _split_uint64(seed)
    with vf(mode="simd"):
        full = reg.update_mask(lanes, elem_bits=32)[0]
        lane = reg.varange(0, dtypes.uint32)
        for counter in range(tile_rows // 16):
            c0, c1, c2, c3 = _nvfp4_counter_rows(row_base, lane, col_group + counter, axis_groups, block_offset, full)
            w0, w1, w2, w3 = _philox_counter10(c0, c1, c2, c3, key0, key1, full)
            reg.vstore(self.random_words, counter * 4 * lanes, reg.vreinterpret(w0, dtypes.int32), full)
            reg.vstore(self.random_words, counter * 4 * lanes + lanes, reg.vreinterpret(w1, dtypes.int32), full)
            reg.vstore(self.random_words, counter * 4 * lanes + 2 * lanes, reg.vreinterpret(w2, dtypes.int32), full)
            reg.vstore(self.random_words, counter * 4 * lanes + 3 * lanes, reg.vreinterpret(w3, dtypes.int32), full)
        # SIMD stores Philox words; the following SIMT VF reverses this UB.
        reg.vmem_bar("vst_vld")
    reverse_random_words(self, tile_rows * lanes // 1024)
    with vf(mode="simd"):
        full = reg.update_mask(lanes, elem_bits=32)[0]
        y_cursor = reg.vstore_unalign_begin(y)
        for half in range(tile_rows // 32):
            if const_expr(lanes == 16 and not self.bf16):
                full64 = reg.full_mask()
                lane64 = reg.varange(0, dtypes.uint32)
                maxima64 = broadcast_u32(0, full64)
                for inner in range(8):
                    offset = (half * 32 + inner * 4) * width + d_offset
                    values64 = reg.vload(x, offset)
                    absolute64 = bitwise_and_scalar(reg.vreinterpret(values64, dtypes.uint32), 0x7FFFFFFF, full64)
                    maxima64 = reg.vmax(maxima64, absolute64, mask=full64)
                swap16 = reg.vbitwise_xor(lane64, broadcast_u32(16, full64), mask=full64)
                maxima64 = reg.vmax(maxima64, reg.vgather_reg(maxima64, swap16), mask=full64)
                swap32 = reg.vbitwise_xor(lane64, broadcast_u32(32, full64), mask=full64)
                maxima = reg.vmax(maxima64, reg.vgather_reg(maxima64, swap32), mask=full64)
            else:
                maxima = broadcast_u32(0, full)
                if const_expr(self.bf16 and lanes == 8):
                    for pair in range(16):
                        offset = (half * 32 + pair * 2) * width + d_offset
                        first, second = load_wide8_bf16_pair(self, x, offset, full)
                        first_abs = bitwise_and_scalar(reg.vreinterpret(first, dtypes.uint32), 0x7FFFFFFF, full)
                        second_abs = bitwise_and_scalar(reg.vreinterpret(second, dtypes.uint32), 0x7FFFFFFF, full)
                        maxima = reg.vmax(maxima, first_abs, mask=full)
                        maxima = reg.vmax(maxima, second_abs, mask=full)
                else:
                    for inner in range(32):
                        offset = (half * 32 + inner) * width + d_offset
                        if const_expr(self.bf16):
                            loaded = reg.vload_unpack(x, offset, unpack_mode=reg.UnpackMode.B16_TO_B32)
                            values = reg.vcast(loaded, dtypes.float32, mask=full)
                        else:
                            values = reg.vload(x, offset)
                        absolute = bitwise_and_scalar(reg.vreinterpret(values, dtypes.uint32), 0x7FFFFFFF, full)
                        maxima = reg.vmax(maxima, absolute, mask=full)
            scale_codes, reciprocal = compute_scale(self, maxima, full)
            reg.vstore(self.scale32, half * lanes, scale_codes, full)
            for word in range(8):
                slot = half * 8 * lanes + word * lanes
                original = reg.vreinterpret(reg.vload(self.random_words, slot), dtypes.uint32)
                reversed_word = reg.vreinterpret(reg.vload(self.reversed_words, slot), dtypes.uint32)
                first = half * 32 + word * 4
                store_wide8_row_unaligned(self, x, y, first * width + d_offset, reciprocal,
                                     reg.vshr(original, 16, mask=full), full, y_cursor)
                store_wide8_row_unaligned(self, x, y, (first + 1) * width + d_offset, reciprocal,
                                     bitwise_and_scalar(reversed_word, 0xFFFF, full), full, y_cursor)
                store_wide8_row_unaligned(self, x, y, (first + 2) * width + d_offset, reciprocal,
                                     bitwise_and_scalar(original, 0xFFFF, full), full, y_cursor)
                store_wide8_row_unaligned(self, x, y, (first + 3) * width + d_offset, reciprocal,
                                     reg.vshr(reversed_word, 16, mask=full), full, y_cursor)
        reg.vsqueeze_and_storeunalign_finalize(y, 0, y_cursor)
        # SIMD writes scale32; subsequent SIMD loads pack those E8M0 codes.
        reg.vmem_bar("vst_vld")
        pair_mask = reg.update_mask(lanes * 2, elem_bits=8)[0]
        scale_cursor = reg.vstore_unalign_begin(scales)
        for pair in range(tile_rows // 64):
            first_scale = reg.vload(self.scale32, pair * 2 * lanes)
            second_scale = reg.vload(self.scale32, pair * 2 * lanes + lanes)
            code_pairs = reg.vbitwise_or(first_scale,
                                       reg.vshl(second_scale, 8, mask=full), mask=full)
            packed = reg.vpack(code_pairs, dtypes.uint16, part="lower")
            packed_bytes = reg.vreinterpret_lanes(packed, dtypes.uint8)
            squeezed = reg.vsqueeze_and_storeunalign_init(packed_bytes, mask=pair_mask)
            reg.vsqueeze_and_storeunalign(scales, 0, squeezed, scale_cursor)
        reg.vsqueeze_and_storeunalign_finalize(scales, 0, scale_cursor)
        # SIMD writes packed output/scale; subsequent MTE copies consume them.
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
    codes = _sr_fp32_to_fp8(normalized, random16, self.mantissa, self.bias, self.max_code, mask)
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
    codes = _sr_fp32_to_fp8(normalized, random16, self.mantissa, self.bias, self.max_code, mask)
    packed16 = reg.vpack(codes, dtypes.uint16, part="lower")
    packed8 = reg.vpack(packed16, dtypes.uint8, part="lower")
    byte_mask = reg.update_mask(self.wide_lanes, elem_bits=8)[0]
    squeezed = reg.vsqueeze_and_storeunalign_init(packed8, mask=byte_mask)
    reg.vsqueeze_and_storeunalign(y, 0, squeezed, y_cursor)


# Tail-axis DMA and stage scheduling


@jit
def run_short_rows(self, x, y, scales, quant_length, seed, block_offset):
    columns = x.shape[1]
    rows = x.shape[0]
    pitch_shift = 6
    max_rows = self.tile_elements // 64
    if columns > 64:
        pitch_shift = 7
        max_rows = self.tile_elements // 128
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
        input_slot = self.input.produce()
        mem_copy(tile_slice(input_slot, make_tiler((1, self.tile_elements), alignment=(1, 16)), (0, 0)), tile_slice(flat_x, make_tiler((1, batch_rows * columns), alignment=(1, 16)), (0, task)))
        input_tile = tile_slice(self.input.consume(), make_tiler((1, self.tile_elements), alignment=(1, 16)), (0, 0))
        output_tile = self.output.produce()
        if const_expr(self.tail_prefetch):
            output_tile = tile_slice(output_tile, make_tiler((1, self.tile_elements), alignment=(1, 16)), (0, 0))
        scale_tile = self.scale.produce()
        if const_expr(self.tail_prefetch):
            scale_tile = tile_slice(scale_tile, make_tiler((1, self.tile_elements // 16), alignment=(1, 16)), (0, 0))
        quantize_short_rows(self, input_tile, output_tile, scale_tile, task * batch_rows * columns, count, columns, pitch_shift, seed, block_offset)
        output_copy = self.output.consume()
        if const_expr(self.tail_prefetch):
            output_copy = tile_slice(output_copy, make_tiler((1, self.tile_elements), alignment=(1, 16)), (0, 0))
        mem_copy(tile_slice(flat_y, make_tiler((1, batch_rows * columns), alignment=(1, 16)), (0, task)), output_copy)
        scale_copy = self.scale.consume()
        if const_expr(self.tail_prefetch):
            scale_copy = tile_slice(scale_copy, make_tiler((1, self.tile_elements // 16), alignment=(1, 16)), (0, 0))
        mem_copy(tile_slice(flat_scales, make_tiler((1, batch_rows * scale_columns), alignment=(1, 2)), (0, task)), scale_copy)


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
    tile_columns = pair_batch * 64
    column_tiles = (pairs + pair_batch - 1) // pair_batch
    tasks = x.shape[0] * column_tiles
    for task in range(get_block_idx(), tasks, get_block_num()):
        row = task // column_tiles
        tile = task % column_tiles
        count = columns - tile * tile_columns
        if count > tile_columns:
            count = tile_columns
        input_slot = self.input.produce()
        mem_copy(tile_slice(input_slot, make_tiler((1, self.tile_elements), alignment=(1, 64)), (0, 0)), tile_slice(x, make_tiler((1, tile_columns), alignment=(1, 64)), (row, tile)))
        input_tile = tile_slice(self.input.consume(), make_tiler((1, self.tile_elements), alignment=(1, 64)), (0, 0))
        output_tile = self.output.produce()
        if const_expr(self.tail_prefetch):
            output_tile = tile_slice(output_tile, make_tiler((1, self.tile_elements), alignment=(1, 16)), (0, 0))
        scale_tile = self.scale.produce()
        if const_expr(self.tail_prefetch):
            scale_tile = tile_slice(scale_tile, make_tiler((1, self.tile_elements // 16), alignment=(1, 16)), (0, 0))
        if count >= 256 and count % 256 == 0:
            quantize_tile_256(self, input_tile, output_tile, scale_tile, row * columns + tile * tile_columns, count, scale_factor, quant_length, seed, block_offset)
        else:
            quantize_tile(self, input_tile, output_tile, scale_tile, row * columns + tile * tile_columns, count, scale_factor, quant_length, seed, block_offset)
        output_copy = self.output.consume()
        if const_expr(self.tail_prefetch):
            output_copy = tile_slice(output_copy, make_tiler((1, self.tile_elements), alignment=(1, 16)), (0, 0))
        mem_copy(tile_slice(y, make_tiler((1, tile_columns), alignment=(1, 64)), (row, tile)), output_copy)
        scale_copy = self.scale.consume()
        if const_expr(self.tail_prefetch):
            scale_copy = tile_slice(scale_copy, make_tiler((1, self.tile_elements // 16), alignment=(1, 16)), (0, 0))
        mem_copy(tile_slice(scales, make_tiler((1, pair_batch * scale_factor), alignment=(1, 2)), (row, tile)), scale_copy)


@jit
def run_coalesced(self, x, y, scales, quant_length, seed, block_offset):
    columns = x.shape[1]
    pairs = columns // 64
    core_pairs = (pairs + get_block_num() - 1) // get_block_num()
    if core_pairs >= 4:
        core_pairs = (core_pairs + 3) // 4 * 4
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
        tiles = (core_count + self.tile_elements - 1) // self.tile_elements
        for tile in range(tiles):
            count = core_count - tile * self.tile_elements
            if count > self.tile_elements:
                count = self.tile_elements
            input_slot = self.input.produce()
            mem_copy(tile_slice(input_slot, make_tiler((1, self.tile_elements), alignment=(1, 64)), (0, 0)), tile_slice(core_x, make_tiler((1, self.tile_elements), alignment=(1, 64)), (0, tile)))
            input_tile = tile_slice(self.input.consume(), make_tiler((1, self.tile_elements), alignment=(1, 64)), (0, 0))
            output_tile = self.output.produce()
            if const_expr(self.tail_prefetch):
                output_tile = tile_slice(output_tile, make_tiler((1, self.tile_elements), alignment=(1, 16)), (0, 0))
            scale_tile = self.scale.produce()
            if const_expr(self.tail_prefetch):
                scale_tile = tile_slice(scale_tile, make_tiler((1, self.tile_elements // 16), alignment=(1, 16)), (0, 0))
            start = core_start + tile * self.tile_elements
            if count >= 256 and count % 256 == 0:
                quantize_tile_256(self, input_tile, output_tile, scale_tile, start, count, scale_factor, quant_length, seed, block_offset)
            else:
                quantize_tile(self, input_tile, output_tile, scale_tile, start, count, scale_factor, quant_length, seed, block_offset)
            output_copy = self.output.consume()
            if const_expr(self.tail_prefetch):
                output_copy = tile_slice(output_copy, make_tiler((1, self.tile_elements), alignment=(1, 16)), (0, 0))
            mem_copy(tile_slice(core_y, make_tiler((1, self.tile_elements), alignment=(1, 64)), (0, tile)), output_copy)
            scale_copy = self.scale.consume()
            if const_expr(self.tail_prefetch):
                scale_copy = tile_slice(scale_copy, make_tiler((1, self.tile_elements // 16), alignment=(1, 16)), (0, 0))
            mem_copy(tile_slice(core_s, make_tiler((1, self.max_pairs * scale_factor), alignment=(1, 2)), (0, tile)), scale_copy)


@jit
def run_prefetch32k(self, x, y, scales, quant_length, seed, block_offset):
    # Full32K chunks only. Two original16K compute bodies share one output slot.
    columns = x.shape[1]
    chunks = columns // (TAIL_PREFETCH_TILES * self.tile_elements)
    scale_factor = 2
    if scales.shape[1] == columns // 16:
        scale_factor = 4
    for task in range(get_block_idx(), chunks, get_block_num()):
        mem_copy(self.input.produce(), tile_slice(x, make_tiler((1, TAIL_PREFETCH_TILES * self.tile_elements), alignment=(1, 64)), (0, task)))
        input_slot = self.input.consume()
        output_slot = self.output.produce()
        scale_slot = self.scale.produce()
        # Literal half0/half1 input and output origins preserve reinterpret contracts.
        input_tile = tile_slice(input_slot, make_tiler((1, self.tile_elements), alignment=(1, 64)), (0, 0))
        output_tile = tile_slice(output_slot, make_tiler((1, self.tile_elements), alignment=(1, 64)), (0, 0))
        scale_tile = tile_slice(scale_slot, make_tiler((1, self.tile_elements // 16), alignment=(1, 2)), (0, 0))
        output_index = TAIL_PREFETCH_TILES * task + 0
        start = output_index * self.tile_elements
        quantize_tile_256(self, input_tile, output_tile, scale_tile,
                          start, self.tile_elements, scale_factor, quant_length, seed, block_offset)
        input_tile = tile_slice(input_slot, make_tiler((1, self.tile_elements), alignment=(1, 64)), (0, 1))
        output_tile = tile_slice(output_slot, make_tiler((1, self.tile_elements), alignment=(1, 64)), (0, 1))
        output_index = TAIL_PREFETCH_TILES * task + 1
        start = output_index * self.tile_elements
        # The DSL permits static UB leaves only. Each runtime path runs one
        # second16K compute, with a static scale origin matching actual bytes.
        if scale_factor == 4:
            scale_tile = tile_slice(scale_slot, make_tiler((1, self.tile_elements // 16), alignment=(1, 2)), (0, 1))
            quantize_tile_256(self, input_tile, output_tile, scale_tile,
                              start, self.tile_elements, scale_factor, quant_length, seed, block_offset)
        else:
            scale_tile = tile_slice(scale_slot, make_tiler((1, self.tile_elements // 32), alignment=(1, 2)), (0, 1))
            quantize_tile_256(self, input_tile, output_tile, scale_tile,
                              start, self.tile_elements, scale_factor, quant_length, seed, block_offset)
        mem_copy(tile_slice(y, make_tiler((1, TAIL_PREFETCH_TILES * self.tile_elements), alignment=(1, 64)), (0, task)), self.output.consume())
        mem_copy(tile_slice(scales, make_tiler((1, TAIL_PREFETCH_TILES * self.max_pairs * scale_factor), alignment=(1, 2)), (0, task)), self.scale.consume())


@jit
def dispatch_tail(self, x, y, scales, quant_length, seed, block_offset):
    """One geometry dispatcher for GM and eager scalar RNG carriers."""
    if (const_expr(self.tail_prefetch) and x.shape[0] == 1
            and x.shape[1] % (get_block_num() * TAIL_PREFETCH_TILES * self.tile_elements) == 0):
        run_prefetch32k(self, x, y, scales, quant_length, seed, block_offset)
    elif (x.shape[0] == 1 and x.shape[1] % 64 == 0
            and x.shape[1] > get_block_num() * self.tile_elements
            and x.shape[1] % (get_block_num() * self.tile_elements) != 0):
        run_coalesced(self, x, y, scales, quant_length, seed, block_offset)
    elif x.shape[0] > 1 and x.shape[1] < 128:
        run_short_rows(self, x, y, scales, quant_length, seed, block_offset)
    else:
        run_tail(self, x, y, scales, quant_length, seed, block_offset)


# Non-tail-axis DMA and stage scheduling


@jit
def _stage_run_wide_non_tail(self, x: Tensor, y: Tensor, scales: Tensor,
                      quant_length: int, width: int, seed, block_offset):
    base_tasks = x.shape[0] * (quant_length // 64)
    tile_rows = 64
    if base_tasks >= 256 and quant_length % 256 == 0 and width == 64:
        tile_rows = 256
    elif base_tasks >= 256 and quant_length % 128 == 0:
        tile_rows = 128
    groups = quant_length // tile_rows
    tasks = x.shape[0] * groups
    for task in range(get_block_idx(), tasks, get_block_num()):
        row = task // groups
        group = task % groups
        tile = tile_rows * width
        mem_copy(self.input.produce(), tile_slice(x, make_tiler((1, tile), alignment=(1, 64)), (row, group)))
        x_tile = self.input.consume()
        y_tile = self.output.produce()
        scale_tile = self.scale.produce()
        for d_chunk in range(width // 64):
            quantize_wide64(self, x_tile, y_tile, scale_tile,
                                 row, group, quant_length, width, d_chunk * 64, tile_rows, seed, block_offset)
        mem_copy(tile_slice(y, make_tiler((1, tile), alignment=(1, 64)), (row, group)),
                 self.output.consume())
        mem_copy(tile_slice(scales, make_tiler((1, tile_rows * width // 32), alignment=(1, 128)), (row, group)),
                 self.scale.consume())


@jit
def _stage_run_wide32_non_tail(self, x: Tensor, y: Tensor, scales: Tensor,
                      quant_length: int, width: int, seed, block_offset):
    base_tasks = x.shape[0] * (quant_length // 64)
    tile_rows = 64
    if base_tasks >= 256 and quant_length % 512 == 0 and width == 32:
        tile_rows = 512
    elif base_tasks >= 256 and quant_length % 256 == 0 and width == 32:
        tile_rows = 256
    elif base_tasks >= 256 and quant_length % 128 == 0:
        tile_rows = 128
    groups = quant_length // tile_rows
    tasks = x.shape[0] * groups
    for task in range(get_block_idx(), tasks, get_block_num()):
        row = task // groups
        group = task % groups
        tile = tile_rows * width
        mem_copy(self.input.produce(), tile_slice(x, make_tiler((1, tile), alignment=(1, 64)), (row, group)))
        x_tile = self.input.consume()
        y_tile = self.output.produce()
        scale_tile = self.scale.produce()
        quantize_wide32(self, x_tile, y_tile, scale_tile,
                             row, group, quant_length, width, 0, tile_rows, seed, block_offset)
        mem_copy(tile_slice(y, make_tiler((1, tile), alignment=(1, 64)), (row, group)),
                 self.output.consume())
        mem_copy(tile_slice(scales, make_tiler((1, tile_rows * width // 32), alignment=(1, 64)), (row, group)),
                 self.scale.consume())


@jit
def _stage_run_wide8_non_tail(self, x: Tensor, y: Tensor, scales: Tensor,
                      quant_length: int, width: int, seed, block_offset):
    # 256 rows provide a 64-byte scale DMA tile for width 8.
    tile_rows = 256
    groups = quant_length // tile_rows
    tasks = x.shape[0] * groups
    for task in range(get_block_idx(), tasks, get_block_num()):
        row = task // groups
        group = task % groups
        tile = tile_rows * width
        mem_copy(self.input.produce(), tile_slice(x, make_tiler((1, tile), alignment=(1, 64)), (row, group)))
        x_tile = self.input.consume()
        y_tile = self.output.produce()
        scale_tile = self.scale.produce()
        quantize_wide8(self, x_tile, y_tile, scale_tile,
                             row, group, quant_length, width, 0, tile_rows, seed, block_offset)
        mem_copy(tile_slice(y, make_tiler((1, tile), alignment=(1, 64)), (row, group)),
                 self.output.consume())
        mem_copy(tile_slice(scales, make_tiler((1, tile_rows * width // 32), alignment=(1, 64)), (row, group)),
                 self.scale.consume())


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
            mem_copy(self.input.produce(), tile_slice(x0, make_tiler((1, tile_columns), alignment=(1, 64)), (row, tile)))
        else:
            mem_copy(self.input.produce(), tile_slice(x1, make_tiler((1, tile_columns), alignment=(1, 64)), (row, tile)))
        input_tile = self.input.consume()
        output_tile = self.output.produce()
        scale_tile = self.scale.produce()
        start = logical_row * columns + tile * tile_columns
        if count >= 256 and count % 256 == 0:
            quantize_tile_256(self, input_tile, output_tile, scale_tile, start, count, 2, quant_length, seed, block_offset)
        else:
            quantize_tile(self, input_tile, output_tile, scale_tile, start, count, 2, quant_length, seed, block_offset)
        mem_copy(tile_slice(y, make_tiler((1, tile_columns), alignment=(1, 64)), (logical_row, tile)),
                 self.output.consume())
        mem_copy(tile_slice(scales, make_tiler((1, pair_batch * 2), alignment=(1, 2)), (logical_row, tile)),
                 self.scale.consume())


@jit
def _stage_run_non_tail_fused(self, x0: Tensor, x1: Tensor, y: Tensor, scales: Tensor, quant_length: int, seed, block_offset):
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
        mem_copy(self.input.produce(), tile_slice(x0, make_tiler((1, tile_columns), alignment=(1, 64)), (row, tile)))
        x_tile = self.input.consume()
        y_tile = self.output.produce()
        s_tile = self.scale.produce()
        quantize_tile_256(self, x_tile, y_tile, s_tile, row * 2 * columns + tile * tile_columns, count, 2, quant_length, seed, block_offset)
        mem_copy(self.saved_output, self.output.consume())
        mem_copy(self.saved_scale, self.scale.consume())
        mem_copy(self.input.produce(), tile_slice(x1, make_tiler((1, tile_columns), alignment=(1, 64)), (row, tile)))
        x_tile = self.input.consume()
        y_tile = self.output.produce()
        s_tile = self.scale.produce()
        quantize_tile_256(self, x_tile, y_tile, s_tile, (row * 2 + 1) * columns + tile * tile_columns, count, 2, quant_length, seed, block_offset)
        second_y = self.output.consume()
        second_s = self.scale.consume()
        first_pairs = reinterpret(self.saved_scale, dtypes.uint16, shape=(1, self.tile_elements // 32))
        second_pairs = reinterpret(second_s, dtypes.uint16, shape=(1, self.tile_elements // 32))
        with vf(mode="simd"):
            for offset in range(0, count, 256):
                a = reg.vload(self.saved_output, offset)
                b = reg.vload(second_y, offset)
                reg.vstore_interleave(self.interleaved_output, offset * 2, a, b)
            for offset in range(0, count // 64, 128):
                a = reg.vload(first_pairs, offset)
                b = reg.vload(second_pairs, offset)
                reg.vstore_interleave(self.interleaved_scale, offset * 2, a, b)
        mem_copy(tile_slice(y, make_tiler((1, tile_columns * 2), alignment=(1, 512)), (row, tile)),
                 self.interleaved_output)
        scale_bytes = reinterpret(self.interleaved_scale, dtypes.uint8,
                                  shape=(1, self.tile_elements // 8))
        mem_copy(tile_slice(scales, make_tiler((1, pair_batch * 4), alignment=(1, 16)), (row, tile)),
                 scale_bytes)


# Kernel resource ownership and stage dispatch


SIMT_UB_RESERVE = 40 * 1024


@jit
def _read_rng_state(rng_state):
    # Two exact 8-byte GM reads, bypassing stale scalar DCache on graph replay.
    # No UB allocation or full-vector access to this 16-byte state carrier.
    seed = dtypes.uint64(scalar.load_bypass(rng_state.ptr(0)))
    block_offset = dtypes.uint64(scalar.load_bypass(rng_state.ptr(1)))
    return seed, block_offset


@kernel
class _DynamicMxQuantSrKernel:
    def __init__(self, input_dtype, e5m2, tile, scale_capacity, scale_alg=1, max_low_bound_bits=0,
                 fused_non_tail=False, wide_lanes=8, tail_prefetch=False):
        self.wide_lanes = wide_lanes
        self.scale_alg = scale_alg
        self.max_low_bound_bits = max_low_bound_bits
        self.bf16 = input_dtype == dtypes.bfloat16
        self.tail_prefetch = tail_prefetch and self.bf16 and tile == TAIL_PREFETCH_COMPUTE_ELEMENTS and not fused_non_tail
        self.invalid_packed_reciprocal = 0xFF80FF80 if self.bf16 else 0xFF800000
        self.mantissa = 2 if e5m2 else 3
        self.bias = 15 if e5m2 else 7
        self.max_code = 0x7B if e5m2 else 0x7E
        self.inv_max_bits = 0x37924925 if e5m2 else 0x3B124925
        input_tile = TAIL_PREFETCH_DMA_ELEMENTS if self.tail_prefetch else tile
        input_depth = TAIL_PREFETCH_INPUT_DEPTH if self.tail_prefetch else 2
        self.input = Channel(MemLoc.UB, (1, input_tile), input_dtype, depth=input_depth)
        output_tile = TAIL_PREFETCH_DMA_ELEMENTS if self.tail_prefetch else tile
        output_depth = TAIL_PREFETCH_OUTPUT_DEPTH
        self.output = Channel(MemLoc.UB, (1, output_tile), dtypes.uint8, depth=output_depth)
        self.scale = Channel(MemLoc.UB, (1, output_tile // 16), dtypes.uint8, depth=output_depth)
        self.scale32 = Buffer(MemLoc.UB, (1, scale_capacity), dtypes.uint32)
        self.random_words = Buffer(MemLoc.UB, (1, tile // 4), dtypes.int32)
        self.reversed_words = Buffer(MemLoc.UB, (1, tile // 4), dtypes.int32)
        if fused_non_tail:
            self.saved_output = Buffer(MemLoc.UB, (1, tile), dtypes.uint8)
            self.saved_scale = Buffer(MemLoc.UB, (1, tile // 16), dtypes.uint8)
            self.interleaved_output = Buffer(MemLoc.UB, (1, 2 * tile), dtypes.uint8)
            self.interleaved_scale = Buffer(MemLoc.UB, (1, tile // 16), dtypes.uint16)
        self.tile_elements = tile
        self.max_pairs = tile // 64

    def __call__(self, x: Tensor, y: Tensor, scales: Tensor,
                 quant_length: int, rng_state: Tensor):
        seed, block_offset = _read_rng_state(rng_state)
        dispatch_tail(self, x, y, scales, quant_length, seed, block_offset)

    def run_tail_scalar_rng(self, x: Tensor, y: Tensor, scales: Tensor,
                            quant_length: int, seed_bits, block_offset_bits):
        """Private eager experiment; graph capture uses the GM-state entry."""
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


# Host tiling, launch preparation and public interface


__all__ = ["dynamic_mx_quant_sr"]

# Path support/alignment constraints, not performance cutoffs (units: elements).
STRIDE2_MIN_QUANT_LENGTH = 512
STRIDE2_ALIGNMENT = 64
FUSED_STRIDE2_ALIGNMENT = 256
SMALL_WIDE_ALIGNMENT = 256
WIDE_ALIGNMENT = 64
PHILOX_BATCH_ELEMENTS = 1024
MAX_AIV_BUDGET = 64

# Existing tuned values, retained unchanged. Historical measurement provenance
# is incomplete; these are selection policy, not hardware capacity limits.
BF16_WIDE8_MAX_ELEMENTS = 128 * 1024
WIDE16_MAX_ELEMENTS = 2 * 1024 * 1024
BF16_WIDE16_SMALL_ELEMENTS = 256 * 1024
BF16_WIDE16_LONG_AXIS_ELEMENTS = 1024 * 1024
BF16_WIDE16_LONG_AXIS_LENGTH = 4096

@dataclass(frozen=True)
class BitReverseTiling:
    tile_elements: int
    scale_capacity: int
    buffer_bytes: int
    ub_bytes: int
    tail_prefetch: bool = False


def _prefetch_buffer_bytes(tile, scale_capacity):
    """Sum the same BF16 Channel/Buffer allocations as the owner Kernel."""
    transfer = TAIL_PREFETCH_DMA_ELEMENTS
    return (TAIL_PREFETCH_INPUT_DEPTH * transfer * 2
            + TAIL_PREFETCH_OUTPUT_DEPTH * transfer
            + TAIL_PREFETCH_OUTPUT_DEPTH * (transfer // 16)
            + 4 * scale_capacity + 2 * tile)


def _make_tiling(input_dtype, ub_bytes, fused_non_tail=False, tail_prefetch=True):
    if input_dtype not in (dtypes.float32, dtypes.bfloat16):
        raise TypeError("tiling requires float32 or bfloat16")
    if type(ub_bytes) is not int or ub_bytes <= SIMT_UB_RESERVE:
        raise ValueError("UB capacity must exceed the 40 KiB SIMT reserve")
    item = 4 if input_dtype == dtypes.float32 else 2
    budget = ub_bytes - SIMT_UB_RESERVE
    if tail_prefetch and input_dtype == dtypes.bfloat16 and not fused_non_tail:
        tile = TAIL_PREFETCH_COMPUTE_ELEMENTS
        scales = ((tile // 32 + 63) // 64) * 64
        used = _prefetch_buffer_bytes(tile, scales)
        if used <= budget:
            return BitReverseTiling(tile, scales, used, ub_bytes, tail_prefetch=True)
        # Preserve the original supported UB range with ordinary buffering.
    tile = budget // (2 * item + 4) // 1024 * 1024
    if input_dtype == dtypes.bfloat16:
        tile = min(tile, BF16_TILE_LIMIT_ELEMENTS)
    while tile >= 1024:
        scales = ((tile // 32 + 63) // 64) * 64
        used = 2 * tile * item + 2 * tile + 2 * (tile // 16) + 4 * scales + 2 * tile
        if fused_non_tail:
            used += tile + tile // 16 + 2 * tile + tile // 8
        if used <= budget:
            return BitReverseTiling(tile, scales, used, ub_bytes)
        tile -= 1024
    raise ValueError("UB remaining after the SIMT reserve cannot fit one tile")


@jit
def _new_kernel(config, tail_prefetch=False):
    return _DynamicMxQuantSrKernel(
        config.input_dtype, config.e5m2, config.tiling.tile_elements,
        config.tiling.scale_capacity, config.scale_alg, config.max_low_bound_bits,
        config.fused, config.wide_lanes,
        tail_prefetch and config.tiling.tail_prefetch,
    )


class _Launch:
    def __init__(self, input_dtype, e5m2, tiling, scale_alg, max_low_bound_bits,
                 *, fused=False, wide_lanes=8):
        self.input_dtype, self.e5m2, self.tiling = input_dtype, e5m2, tiling
        self.scale_alg, self.max_low_bound_bits = scale_alg, max_low_bound_bits
        self.fused, self.wide_lanes = fused, wide_lanes

    @host
    def run(self, x, y, scales, quant_length: int, blocks: int, rng_state):
        op = _new_kernel(self, tail_prefetch=True)
        op[blocks](x, y, scales, quant_length, rng_state)

    @host
    def run_tail_scalar_rng(self, x, y, scales, quant_length: int, blocks: int,
                            seed_bits, block_offset_bits):
        op = _new_kernel(self, tail_prefetch=True)
        op[blocks].run_tail_scalar_rng(x, y, scales, quant_length,
                                      seed_bits, block_offset_bits)


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


def _tail_specs(dtype):
    rows = Dim("rows", min=1)
    columns = Dim("columns", min=16 if dtype == dtypes.bfloat16 else 32, multiple_of=16)
    scales = Dim("scale_columns", min=2, multiple_of=2)
    return (TensorSpec((rows, columns), dtype), TensorSpec((rows, columns), dtypes.uint8),
            TensorSpec((rows, scales), dtypes.uint8), dtypes.int64, dtypes.int64,
            TensorSpec((2,), dtypes.int64))


def _tail_scalar_rng_specs(dtype):
    # Same runtime tensor geometry, original axis length and core count as GM.
    return (*_tail_specs(dtype)[:5], dtypes.int64, dtypes.int64)


def _tail_rng_signed64(value):
    """Preserve every unsigned seed/block-offset bit in an Int64 carrier."""
    if type(value) is not int or not 0 <= value < (1 << 64):
        raise ValueError("scalar RNG value must be a Python int in [0, 2**64)")
    return value if value < (1 << 63) else value - (1 << 64)


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
    return (source, source, TensorSpec((out_rows, out_columns), dtypes.uint8),
            TensorSpec((out_rows, scale_columns), dtypes.uint8), dtypes.int64, dtypes.int64,
            TensorSpec((2,), dtypes.int64))


def _wide_specs(dtype, *, column_alignment):
    rows = Dim("rows", min=1)
    columns = Dim("physical_columns", min=column_alignment, multiple_of=column_alignment)
    scale_alignment = column_alignment // 32
    scales = Dim("physical_scale_columns", min=scale_alignment, multiple_of=scale_alignment)
    return (TensorSpec((rows, columns), dtype), TensorSpec((rows, columns), dtypes.uint8),
            TensorSpec((rows, scales), dtypes.uint8), dtypes.int64, dtypes.int64, dtypes.int64,
            TensorSpec((2,), dtypes.int64))


@dataclass(frozen=True)
class _LayoutSpec:
    entry: str
    spec_factory: object
    wide_lanes: int = 8
    fused: bool = False


_LAYOUT_REGISTRY = MappingProxyType({
    Layout.TAIL: _LayoutSpec("run", _tail_specs),
    Layout.NON_TAIL: _LayoutSpec("run_non_tail", partial(_stride2_specs, fused=False)),
    Layout.NON_TAIL_FUSED: _LayoutSpec("run_non_tail_fused", partial(_stride2_specs, fused=True), fused=True),
    Layout.WIDE: _LayoutSpec("run_wide_non_tail", partial(_wide_specs, column_alignment=4096)),
    Layout.WIDE32: _LayoutSpec("run_wide32_non_tail", partial(_wide_specs, column_alignment=2048)),
    Layout.WIDE8: _LayoutSpec("run_wide8_non_tail", partial(_wide_specs, column_alignment=2048)),
    Layout.WIDE16: _LayoutSpec("run_wide8_non_tail", partial(_wide_specs, column_alignment=4096), wide_lanes=16),
})


def _build_program(key):
    """Exactly one standard AOT call; DSL manages its own binary cache."""
    spec = _LAYOUT_REGISTRY[key.layout]
    dtype = dtypes.bfloat16 if key.input_dtype == "bf16" else dtypes.float32
    tiling = _make_tiling(dtype, key.ub_bytes, spec.fused, tail_prefetch=key.layout is Layout.TAIL)
    launch = _Launch(dtype, key.e5m2, tiling, key.scale_alg, key.max_low_bound_bits,
                     fused=spec.fused, wide_lanes=spec.wide_lanes)
    return cannbotdsl.compile(getattr(launch, spec.entry), *spec.spec_factory(dtype))


_PROGRAM_CACHE = _ProgramCache(64)


def clear_caches():
    """Release this operator's references only; external Programs stay live."""
    _PROGRAM_CACHE.clear()
    from ._rng_state import clear_caches as clear_rng_programs
    clear_rng_programs()


def cache_info():
    return _PROGRAM_CACHE.info()


def _get_compiled_kernel(input_dtype, e5m2, ub_bytes, scale_alg=1, max_low_bound_bits=0,
                         *, layout=Layout.TAIL, device=None):
    if input_dtype not in (dtypes.float32, dtypes.bfloat16):
        raise TypeError("compile input dtype must be float32 or bfloat16")
    if device is None or device.index is None:
        device = torch.device("npu", torch.npu.current_device())
    context = capture_compile_context(
        "bf16" if input_dtype == dtypes.bfloat16 else "fp32", e5m2, ub_bytes,
        scale_alg, max_low_bound_bits, layout, device,
    )
    return _PROGRAM_CACHE.get_or_build(
        context.key, lambda: _build_program(context.key), bypass=context.bypass_reason is not None,
    )


def _resolve_block_dim(x, block_dim, *, quant_length=None):
    """Host-only AIV budget; refresh stream quota without specializing code."""
    if block_dim is not None:
        return block_dim
    stream = torch.npu.current_stream(x.device)
    info = cannbotdsl.get_platform_info(stream=stream)
    count = getattr(info, "vector_core_num", None)
    if type(count) is not int or count <= 0:
        raise RuntimeError(f"NPU {x.device} reports invalid vector_core_num={count!r}")
    # A Philox batch supplies 1024 inputs. Avoid spreading a small coalesced
    # input across cores that each launch SIMT for only a fraction of a batch.
    budget = min(count, MAX_AIV_BUDGET)
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


def _validate_attributes(
    input, axis, round_mode, dst_type, block_size, scale_alg, dst_type_max, max_low_bound
):
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
    if isinstance(dst_type_max, float) and not math.isfinite(dst_type_max):
        raise ValueError("dst_type_max must be finite")
    if dst_type_max != 0:
        raise NotImplementedError("dynamic_mx_quant_sr requires dst_type_max=0.0")
    if type(max_low_bound) not in (float, int) or not math.isfinite(max_low_bound) or max_low_bound < 0:
        raise ValueError("max_low_bound must be a finite non-negative number")
    if scale_alg != 1 and max_low_bound != 0:
        raise ValueError("max_low_bound must be 0 when scale_alg != 1")
    bound_bits = _encode_bound(max_low_bound)
    dst_dtype = torch.float8_e5m2 if dst_type == 23 else torch.float8_e4m3fn
    return axis, dst_dtype, bound_bits


def _supports_stride2(x, axis):
    return (x.ndim >= 2 and 0 <= axis < x.ndim - 1
            and math.prod(x.shape[axis + 1:]) == 2
            and x.shape[axis] >= STRIDE2_MIN_QUANT_LENGTH
            and x.shape[axis] % STRIDE2_ALIGNMENT == 0)


def _supports_small_wide(quant_length, width):
    return (width in (8, 16) and quant_length >= SMALL_WIDE_ALIGNMENT
            and quant_length % SMALL_WIDE_ALIGNMENT == 0)


def _select_layout(x, axis):
    """Separate path support from inherited element-count tuning policy."""
    if axis == x.ndim - 1:
        return Layout.TAIL
    if _supports_stride2(x, axis):
        return (Layout.NON_TAIL_FUSED if x.shape[axis] % FUSED_STRIDE2_ALIGNMENT == 0
                else Layout.NON_TAIL)
    quant_length, width = x.shape[axis], math.prod(x.shape[axis + 1:])
    elements = x.numel()
    if _supports_small_wide(quant_length, width):
        if width == 8 and (x.dtype == torch.float32 or elements <= BF16_WIDE8_MAX_ELEMENTS):
            return Layout.WIDE8
        if width == 16 and elements <= WIDE16_MAX_ELEMENTS and (
            x.dtype == torch.float32 or elements <= BF16_WIDE16_SMALL_ELEMENTS
            or (quant_length >= BF16_WIDE16_LONG_AXIS_LENGTH and elements <= BF16_WIDE16_LONG_AXIS_ELEMENTS)
            or quant_length == BF16_WIDE16_LONG_AXIS_LENGTH
        ):
            return Layout.WIDE16
    if width in (32, 64, 128) and quant_length >= WIDE_ALIGNMENT and quant_length % WIDE_ALIGNMENT == 0:
        return Layout.WIDE32 if width == 32 else Layout.WIDE
    return Layout.TRANSPOSE


def _make_execution_plan(x, axis, *, block_dim=None):
    layout = _select_layout(x, axis)
    shape = tuple(x.shape)
    quant_length = shape[axis]
    pairs = (quant_length + 63) // 64
    scale_shape = (*shape[:axis], pairs, *shape[axis + 1:], 2)
    budget = _resolve_block_dim(x, block_dim, quant_length=quant_length if layout is Layout.TRANSPOSE else None)
    width = math.prod(shape[axis + 1:])
    base_task_rows = 1
    if layout in (Layout.TAIL, Layout.TRANSPOSE):
        logical_rows = x.numel() // quant_length
        moved_shape = (*shape[:axis], *shape[axis + 1:], quant_length)
        storage_output_shape = shape if layout is Layout.TAIL else moved_shape
        storage_scale_shape = (*storage_output_shape[:-1], pairs, 2)
        coalesced = quant_length % 64 == 0 or quant_length == 32
        kernel_shape = (1, x.numel()) if coalesced else (logical_rows, quant_length)
        kernel_scale_shape = (1, logical_rows * pairs * 2) if coalesced else (logical_rows, pairs * 2)
        if layout is Layout.TRANSPOSE and coalesced:
            # Query was made on the physical source. The old fallback queried
            # its moved-axis carrier; preserve that carrier's small-work cap.
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
        blocks = min(budget, (x.numel() + PHILOX_BATCH_ELEMENTS - 1) // PHILOX_BATCH_ELEMENTS,
                     logical_rows * pairs)
    else:
        logical_rows = x.numel() // (quant_length * width)
        storage_output_shape, storage_scale_shape = shape, scale_shape
        kernel_shape = (logical_rows, quant_length * width)
        kernel_scale_shape = (logical_rows, pairs * width * 2)
        base_task_rows = 256 if layout in (Layout.WIDE8, Layout.WIDE16) else 64
        blocks = min(budget, logical_rows * (quant_length // base_task_rows))
    return ExecutionPlan(
        layout, axis, quant_length, width, logical_rows, shape, scale_shape,
        storage_output_shape, storage_scale_shape, kernel_shape, kernel_scale_shape,
        blocks, base_task_rows,
    )


def _kernel_views(x, encoded, scales, plan):
    return (x.view(plan.kernel_shape), encoded.view(plan.kernel_shape),
            scales.view(plan.kernel_scale_shape))


@dataclass(frozen=True)
class _PreparedLaunch:
    plan: ExecutionPlan
    program: object
    arguments: tuple
    encoded: torch.Tensor
    scales: torch.Tensor
    dst_dtype: torch.dtype

    def launch(self, *, rng_state=None):
        device = self.arguments[0].device
        if not self.program.so_path or self.program.closed:
            raise RuntimeError("SR launch requires a live executable AOT Program")
        if rng_state is not None:
            if not isinstance(rng_state, torch.Tensor):
                raise TypeError("rng_state must be an NPU int64[2] tensor")
            if (rng_state.device != device or rng_state.dtype != torch.int64
                    or rng_state.shape != (2,) or not rng_state.is_contiguous()):
                raise ValueError("rng_state must be contiguous int64[2] on the input device")
        with torch.npu.device(device):
            if rng_state is None:
                rng_state = _reserve_rng_state(device, _rng_increment(math.prod(self.plan.output_shape)))
            # Explicit replay state is trusted: value checks belong to its
            # constructor, not a capture-breaking device-to-host read here.
            self.program(*self.arguments, rng_state)

    def outputs(self):
        y, scales = self.encoded.view(self.dst_dtype), self.scales
        if self.plan.layout is Layout.NON_TAIL:
            y = y.movedim(1, 2).contiguous().reshape(self.plan.output_shape)
            scales = scales.movedim(1, 2).contiguous().reshape(self.plan.scale_shape)
        elif self.plan.layout is Layout.TRANSPOSE:
            y = y.movedim(-1, self.plan.axis).contiguous()
            scales = scales.movedim(-2, self.plan.axis).contiguous()
        return y, scales.view(torch.float8_e8m0fnu)


def _prepare_launch(x, dst_dtype, axis, scale_alg, bound_bits, *, block_dim=None):
    """Host setup shared with benchmarks; launch arguments retain runtime sizes."""
    with torch.npu.device(x.device):
        plan = _make_execution_plan(x, axis, block_dim=block_dim)
        _validate_rng_geometry(x.numel(), plan.quant_length)
        encoded = torch.empty(plan.storage_output_shape, dtype=torch.uint8, device=x.device)
        scales = torch.empty(plan.storage_scale_shape, dtype=torch.uint8, device=x.device)
        source = x.movedim(axis, -1).contiguous() if plan.layout is Layout.TRANSPOSE else x
        dtype = dtypes.bfloat16 if x.dtype == torch.bfloat16 else dtypes.float32
        program = _get_compiled_kernel(dtype, dst_dtype == torch.float8_e5m2, get_mem_size("ub"),
                                      scale_alg, bound_bits, layout=plan.compile_layout, device=x.device)
        if plan.layout in (Layout.TAIL, Layout.TRANSPOSE):
            arguments = (*_kernel_views(source, encoded, scales, plan), plan.quant_length, plan.blocks)
        elif plan.layout in (Layout.NON_TAIL, Layout.NON_TAIL_FUSED):
            source = x.reshape(-1, plan.quant_length, 2)
            arguments = (source[:, :, 0], source[:, :, 1], encoded.view(plan.kernel_shape),
                         scales.view(plan.kernel_scale_shape), plan.quant_length, plan.blocks)
        else:
            arguments = (*_kernel_views(source, encoded, scales, plan),
                         plan.quant_length, plan.width, plan.blocks)
    return _PreparedLaunch(plan, program, arguments, encoded, scales, dst_dtype)


def _dynamic_mx_quant_sr_impl(x, dst_dtype=torch.float8_e4m3fn, *, block_dim=None,
                             scale_alg=1, max_low_bound=0.0, rng_state=None):
    """Private tail-axis launch budget for tuning; no second public API."""
    _validate_input(x)
    if dst_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise TypeError("dst_dtype must be float8_e4m3fn or float8_e5m2")
    if block_dim is not None and (type(block_dim) is not int or not 1 <= block_dim <= MAX_AIV_BUDGET):
        raise ValueError("block_dim must be None or an integer in [1,64]")
    prepared = _prepare_launch(x, dst_dtype, x.ndim - 1, scale_alg, _encode_bound(max_low_bound), block_dim=block_dim)
    prepared.launch(rng_state=rng_state)
    return prepared.outputs()


def dynamic_mx_quant_sr(input, *, axis=-1, round_mode="stochastic", dst_type=24,
                        block_size=32, scale_alg=1, dst_type_max=0.0, max_low_bound=0.0):
    """Dynamic MXFP8 stochastic quantization, Torch-compatible argument names.

    FP32/BF16 -> E4M3FN (24)/E5M2 (23); any axis; block_size=32;
    scale_alg=0 (floor exponent) or 1 (ceil rounded product). max_low_bound
    extends the Torch interface for scale_alg=1. Returns input-shaped FP8 y
    and E8M0 scales: replace axis with ceil(K/64), then append dimension 2.

    Each call reserves from the input device's default framework generator.
    State restore replays the sequence; logical mapping is scheduling-independent.
    Four elements share fields [H,reverse16(H),L,reverse16(L)], not independent
    random samples. See README for the validated NPU Graph usage boundaries.
    The input must be contiguous/nonempty and on a supported dav-3510 NPU;
    axis length is a multiple of 16, at least 32 for FP32.
    """
    axis, dst_dtype, bound_bits = _validate_attributes(
        input, axis, round_mode, dst_type, block_size, scale_alg, dst_type_max, max_low_bound,
    )
    _validate_input(input, axis)
    prepared = _prepare_launch(input, dst_dtype, axis, scale_alg, bound_bits)
    prepared.launch()
    return prepared.outputs()

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
