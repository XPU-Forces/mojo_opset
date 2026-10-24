"""Shared inputs, byte oracles, fixtures and device probes for SR tests.

offset refers only to input storage offset, not RNG state. References always
receive axis/algorithm/bound, and BF16 values come from the actually cast CPU
input. The production module is imported through the Mojo provider package.
"""

import math

from concurrent.futures import Future
from dataclasses import dataclass
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import cannbotdsl
import numpy as np
import pytest
import torch

from cannbotdsl import Buffer
from cannbotdsl import dtypes
from cannbotdsl import get_mem_size
from cannbotdsl.lang import const_expr
from cannbotdsl.lang import host
from cannbotdsl.lang import kernel
from cannbotdsl.lang import vf
from cannbotdsl.ops import reg
from cannbotdsl.ops import scalar
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.ops.arch import get_block_num
from cannbotdsl.ops.info import PlatformInfo
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import MemLoc
from cannbotdsl.tensor import reinterpret
from cannbotdsl.tensor import tile_slice
from ml_dtypes import bfloat16

from mojo_opset.kernels.npu_a5_cannbotdsl import _rng_state as framework_rng
from mojo_opset.kernels.npu_a5_cannbotdsl import dynamic_mx_quant_sr as operator
from mojo_opset.kernels.torch import dynamic_mx_quant_sr_golden as golden

DTYPE_PAIRS = (
    (torch.float32, torch.float8_e4m3fn),
    (torch.bfloat16, torch.float8_e4m3fn),
    (torch.float32, torch.float8_e5m2),
    (torch.bfloat16, torch.float8_e5m2),
)


def sample():
    return operator


def input_numpy(cpu):
    values = cpu.float().numpy()
    return values.astype(bfloat16) if cpu.dtype == torch.bfloat16 else values


def expected_scale_shape(shape, axis=-1):
    axis %= len(shape)
    result = list(shape)
    result[axis] = (result[axis] + 63) // 64
    return (*result, 2)


def launch(x, dst, *, axis=-1, block_dim=None, scale_alg=1, max_low_bound=0.0, rng_seed=0, rng_block_offset=0):
    if dst not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise TypeError("dst must be float8_e4m3fn or float8_e5m2")
    if block_dim is not None and (type(block_dim) is not int or not 1 <= block_dim <= 0xFFFFFFFF):
        raise ValueError("block_dim must be None or a positive uint32 integer")
    op = sample()
    dst_type = 23 if dst == torch.float8_e5m2 else 24
    normalized_axis, dst_dtype, bound = op._validate_attributes(
        x,
        axis,
        "stochastic",
        dst_type,
        32,
        scale_alg,
        0.0,
        max_low_bound,
    )
    op._validate_input(x, normalized_axis)
    prepared = op._prepare_launch(x, dst_dtype, normalized_axis, scale_alg, bound, block_dim=block_dim)
    state = replay_state(x, rng_seed, rng_block_offset)
    prepared.launch(rng_state=state)
    return prepared.outputs()


def replay_state(x, seed=0, block_offset=0):
    rng = framework_rng
    return rng._make_rng_state(seed, block_offset, device=x.device)


def framework_state_words(state, device):
    """Inspect a reserved carrier after execution, without a helper kernel."""
    captured, secondary, seed, offset, intragraph = state
    assert not secondary
    if captured:
        assert seed.device == offset.device == device
        seed, offset = int(seed.item()), int(offset.item()) + intragraph
    return [int(seed) & ((1 << 64) - 1), int(offset) & ((1 << 64) - 1)]


def public_replay(input, *, rng_seed=0, rng_block_offset=0, **attributes):
    """Exercise the real public API at a known framework state, eager tests only.

    Restore the caller's generator after enqueueing; eager main-kernel scalar
    arguments carry the reserved values. Default-resampling/graph tests do not use
    this helper, so precision replay cannot hide a broken default lifecycle.
    """
    with torch.npu.device(input.device):
        saved = torch.npu.get_rng_state(input.device)
        generator = torch.npu.default_generators[input.device.index]
        try:
            generator.manual_seed(rng_seed)
            generator.set_offset(rng_block_offset)
            return sample().dynamic_mx_quant_sr(input=input, **attributes)
        finally:
            torch.npu.set_rng_state(saved, input.device)


def assert_result(cpu, result, axis, src, dst_code, algorithm, bound, *, rng_seed=0, rng_block_offset=0):
    y, scale = result
    dst = torch.float8_e5m2 if dst_code == 23 else torch.float8_e4m3fn
    expected = golden.dynamic_mx_quant_golden(
        input_numpy(cpu),
        str(dst).removeprefix("torch."),
        axis=axis,
        scale_alg=algorithm,
        max_low_bound=bound,
        seed=rng_seed,
        block_offset=rng_block_offset,
    )
    assert cpu.dtype == src and y.shape == cpu.shape and y.dtype == dst
    assert y.device.type in ("npu", "privateuseone") and scale.device == y.device
    assert scale.shape == expected_scale_shape(tuple(cpu.shape), axis)
    assert scale.dtype == torch.float8_e8m0fnu
    actual = tuple(value.view(torch.uint8).cpu().numpy() for value in result)
    for value, reference in zip(actual, expected):
        np.testing.assert_array_equal(value, reference)
    assert all(value.is_contiguous() for value in result)
    return actual


def check(
    shape,
    axis,
    src,
    dst_code,
    algorithm,
    bound=0.0,
    *,
    offset=0,
    cpu=None,
    block_dim=None,
    rng_seed=0,
    rng_block_offset=0,
):
    cpu = axis_input(shape, axis, src) if cpu is None else cpu
    assert tuple(cpu.shape) == shape and cpu.dtype == src
    dst = torch.float8_e5m2 if dst_code == 23 else torch.float8_e4m3fn
    return run_check(
        cpu,
        dst,
        block_dim,
        offset,
        axis=axis,
        scale_alg=algorithm,
        max_low_bound=bound,
        rng_seed=rng_seed,
        rng_block_offset=rng_block_offset,
    )


def run_check(
    cpu,
    dst,
    blocks=None,
    offset=0,
    expected=None,
    *,
    axis=-1,
    scale_alg=1,
    max_low_bound=0.0,
    rng_seed=0,
    rng_block_offset=0,
):
    import torch_npu  # noqa: F401  # registers the provider

    backing_cpu = torch.full((offset + cpu.numel() + 16,), 8192.0, dtype=cpu.dtype)
    backing_cpu[offset : offset + cpu.numel()].copy_(cpu.reshape(-1))
    backing = backing_cpu.npu()
    x = backing[offset : offset + cpu.numel()].reshape(cpu.shape)
    assert x.is_contiguous() and x.storage_offset() == offset
    result = launch(
        x,
        dst,
        axis=axis,
        block_dim=blocks,
        scale_alg=scale_alg,
        max_low_bound=max_low_bound,
        rng_seed=rng_seed,
        rng_block_offset=rng_block_offset,
    )
    dst_code = 23 if dst == torch.float8_e5m2 else 24
    if expected is None:
        actual = assert_result(
            cpu,
            result,
            axis,
            cpu.dtype,
            dst_code,
            scale_alg,
            max_low_bound,
            rng_seed=rng_seed,
            rng_block_offset=rng_block_offset,
        )
    else:
        assert result[0].shape == cpu.shape and result[0].dtype == dst
        assert result[1].shape == expected_scale_shape(tuple(cpu.shape), axis)
        actual = tuple(value.view(torch.uint8).cpu().numpy() for value in result)
        for value, reference in zip(actual, expected):
            np.testing.assert_array_equal(value, reference)
    # Compare raw bytes, including NaN payloads, signed zeros and both guards.
    np.testing.assert_array_equal(backing.view(torch.uint8).cpu().numpy(), backing_cpu.view(torch.uint8).numpy())
    return actual


def axis_input(shape, axis, src):
    generator = torch.Generator().manual_seed(20260924)
    cpu = (torch.randn(shape, generator=generator) * 7).to(src)
    logical = cpu.movedim(axis, -1).contiguous().reshape(-1, shape[axis])
    logical[0, :32] = 0
    logical[0, 1:32:2] = -0.0
    if logical.numel() >= 40:
        # Invalid values get their own group. Finite underflow is tested separately.
        logical.reshape(-1)[32:40] = torch.tensor(
            [float("inf"), -float("inf"), float("nan"), 0.5, -0.5, 1.125, -1.125, 0.0]
        ).to(src)
    return logical.reshape(cpu.movedim(axis, -1).shape).movedim(-1, axis).contiguous()


@dataclass(frozen=True)
class Case:
    name: str
    category: str
    shape: tuple | None = None
    blocks: int | None = 1
    tile_multiple: int = 0
    delta: int = 0
    pattern: str = "random"
    offset: int = 0

    def resolve_shape(self, tile):
        return self.shape if self.shape is not None else (1, self.tile_multiple * tile + self.delta)


def cases_for(src):
    cases = []
    inner = [32, 48, 64, 80, 96, 112, 128, 496, 512, 528, 1008, 1024, 1040, 2032, 2048, 2064, 4080, 4096, 4112]
    if src == torch.bfloat16:
        inner.insert(0, 16)
    cases += [Case(f"inner-k{k}", "vector_and_rng_boundaries", (3, k)) for k in inner]
    for multiple, delta in ((1, -16), (1, 0), (1, 16), (2, -16), (2, 0), (2, 16), (3, 48)):
        cases.append(Case(f"ub-{multiple}tile{delta:+}", "ub_capacity_transition", tile_multiple=multiple, delta=delta))
    for rows, blocks in ((1, 64), (7, 7), (31, 32), (32, 32), (33, 32), (63, 64), (64, 64), (65, 64), (129, 32)):
        cases.append(Case(f"rows{rows}-grid{blocks}", "dispatch_and_row_tails", (rows, 80), blocks))
    minimum = 16 if src == torch.bfloat16 else 32
    for i, shape in enumerate(((minimum,), (2, 3, 80), (2, 3, 4, 144), (1, 2, 3, 4, 64), (65, 32))):
        cases.append(Case(f"rank-layout-{i}", "rank_and_scale_layout", shape, (64, 7, 31, 33, 1)[i]))
    for offset in (1, 7, 16):
        for k in (80, 1040):
            cases.append(Case(f"offset{offset}-k{k}", "contiguous_storage_offset", (3, k), 7, offset=offset))
    for pattern in (
        "signed_zero",
        "wide_exponent",
        "scale_thresholds",
        "special_isolation",
        "sr_boundaries",
        "raw_bits",
    ):
        shape = (6, 96) if pattern == "special_isolation" else (16, 96)
        cases.append(Case(pattern, "numerical_boundaries", shape, 7, pattern=pattern))
    rng = np.random.default_rng(20260921)
    for i in range(8):
        prefixes = ((), (3,), (2, 3), (2, 1, 3), (1, 2, 1, 3))
        k = int(rng.integers(1 if src == torch.bfloat16 else 2, 161)) * 16
        cases.append(
            Case(
                f"generated-{i}",
                "deterministic_shape_sweep",
                prefixes[i % 5] + (k,),
                int(rng.choice([1, 2, 7, 31, 33, 63, 64])),
            )
        )
    return cases


def random_input(shape, src, seed=20260921):
    return (torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * 7).to(src)


def numerical_input(pattern, shape, src, dst):
    if pattern == "random":
        return random_input(shape, src)
    rng = np.random.default_rng(20260921)
    values = np.zeros(shape, dtype=np.float32)
    flat = values.reshape(-1)
    e5 = dst == torch.float8_e5m2
    maximum = 57344.0 if e5 else 448.0
    if pattern == "signed_zero":
        flat[1::2] = -0.0
    elif pattern == "wide_exponent":
        values = np.ldexp(rng.uniform(0.5, 1, shape).astype(np.float32), rng.integers(-145, 126, shape)) * rng.choice(
            [-1, 1], shape
        )
    elif pattern == "raw_bits":
        values = rng.integers(0, 2**32, shape, dtype=np.uint32).view(np.float32)
    elif pattern == "scale_thresholds":
        scalar = np.float32 if src == torch.float32 else bfloat16
        centers = [scalar(maximum * 2.0**-127), scalar(maximum), scalar(torch.finfo(src).tiny)]
        amplitudes = [float(np.nextafter(scalar(0), scalar(1))), 0.0]
        for center in centers:
            below = np.nextafter(center, scalar(0))
            above = np.nextafter(center, scalar(np.inf))
            above2 = np.nextafter(above, scalar(np.inf))
            amplitudes.extend(map(float, (below, center, above, above2)))
        amplitudes += [torch.finfo(src).max, torch.finfo(src).tiny / 2]
        for row, amplitude in enumerate(amplitudes):
            values[row, ::2] = amplitude
            values[row, 1::2] = -amplitude
    elif pattern == "special_isolation":
        signs = np.where(np.arange(32) % 2, -1.0, 1.0).astype(np.float32)
        values[0, 1::2] = -0.0
        for group, exponent in enumerate((-64, 0, 63)):
            values[1, group * 32 : (group + 1) * 32] = signs * np.float32(maximum * 2.0**exponent)
        for group, exponent in enumerate((-8, 0, 8)):
            values[2:5, group * 32 : (group + 1) * 32] = signs * np.float32(maximum * 2.0**exponent)
        values[2, 31] = np.nan
        values[3, 32] = -np.inf
        values[4, 95] = -np.nan
        values[5] = np.resize(np.array([maximum, 0, -0.0, 1.125, -1.125], np.float32), 96)
        values[5, ::32] = maximum
    elif pattern == "sr_boundaries":
        mantissa, emin = (2, -14) if e5 else (3, -6)
        half = 2.0 ** (emin - mantissa - 1)
        probes = [
            0.0,
            -0.0,
            half,
            -half,
            2.0**emin - half,
            2.0**emin,
            1 + 2.0 ** (-mantissa - 1),
            2 - 2.0 ** (-mantissa - 1),
        ]
        for shift in range(25):
            probes.extend([1.75 * 2.0 ** (emin - shift), -1.75 * 2.0 ** (emin - shift)])
        groups = values.reshape(-1, 32)
        groups[:, 0] = maximum
        groups[:, 1:] = np.resize(np.asarray(probes, np.float32), groups.shape[0] * 31).reshape(-1, 31)
    else:
        raise ValueError(pattern)
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(src)


def amax_bits(inv_bits):
    inv = np.uint32(inv_bits).view(np.float32)
    # Exact input-side neighborhoods of every product exponent transition,
    # especially the strictly-greater-than subnormal half threshold.
    targets = np.unique(
        np.concatenate(
            (
                np.array([0, 1, 0x3FFFFF, 0x400000, 0x400001, 0x7FFFFF], dtype=np.uint32),
                np.arange(1, 255, dtype=np.uint32) << np.uint32(23),
            )
        )
    )
    lo = np.zeros(targets.size, dtype=np.uint32)
    hi = np.full(targets.size, 0x7F800000, dtype=np.uint32)
    with np.errstate(under="ignore"):
        for _ in range(32):
            mid = lo + ((hi - lo) >> np.uint32(1))
            product = (mid.view(np.float32) * inv).view(np.uint32)
            take = product >= targets
            hi = np.where(take, mid, hi)
            lo = np.where(take, lo, mid + np.uint32(1))
    neighbors = lo.astype(np.int64)[:, None] + np.arange(-4, 5, dtype=np.int64)
    neighbors = neighbors[(neighbors >= 0) & (neighbors < 0x7F800000)].astype(np.uint32)
    rng = np.random.default_rng(20260922)
    bits = np.concatenate(
        (
            np.arange(32768, dtype=np.uint32) << np.uint32(16),
            np.arange(256, dtype=np.uint32),
            np.array([0x7FFFFF, 0x800000, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FFFFFFF], dtype=np.uint32),
            neighbors,
            rng.integers(0, 0x80000000, size=65536, dtype=np.uint32),
        )
    )
    return np.pad(bits, (0, (-bits.size) % 64)).reshape(-1, 64)


def dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def tile_elements(src):
    dtype = dtypes.bfloat16 if src == torch.bfloat16 else dtypes.float32
    return sample()._make_tiling(dtype, get_mem_size("ub")).tile_elements


def scalar_philox(counter, *, subsequence=0, seed=0):
    """Plain Python integer oracle, independent of NumPy vector packing."""
    mask = (1 << 32) - 1
    # ABI4 linear map; the caller supplies a fixed16384 stream and64-bit draw.
    words = (0, subsequence & mask, counter & mask, counter >> 32)
    key0, key1 = seed & mask, seed >> 32
    for _ in range(10):
        product0 = words[0] * 0xD2511F53
        product1 = words[2] * 0xCD9E8D57
        words = (
            (product1 >> 32) ^ words[1] ^ key0,
            product1 & mask,
            (product0 >> 32) ^ words[3] ^ key1,
            product0 & mask,
        )
        key0 = (key0 + 0x9E3779B9) & mask
        key1 = (key1 + 0xBB67AE85) & mask
    return words


def challenging_input(rows, columns, dtype):
    """Independent scales and special encodings, including partial rows."""
    values = torch.empty((rows, columns), dtype=torch.float32)
    for row in range(rows):
        for block, begin in enumerate(range(0, columns, 32)):
            end = min(begin + 32, columns)
            ramp = torch.linspace(-1.75, 2.25, end - begin)
            values[row, begin:end] = ramp * 2.0 ** (((row * 37 + block * 19) % 241) - 120)
    values[0] = 0.0
    values[0, 1::2] = -0.0
    if rows > 1:
        values[1, min(17, columns - 1)] = float("nan")
    if rows > 2:
        values[2, 0], values[2, -1] = float("inf"), -float("inf")
    return values.to(dtype)


def guarded_launch(cpu, dst, blocks, offset):
    """Run production carriers with input/output/scale guards on both sides."""
    count, shape = cpu.numel(), tuple(cpu.shape)
    scale_shape = expected_scale_shape(shape)
    scale_count = int(np.prod(scale_shape))
    backing_cpu = torch.full((offset + count + 32,), 13.0, dtype=cpu.dtype)
    backing_cpu[offset : offset + count].copy_(cpu.reshape(-1))
    backing = backing_cpu.npu()
    x = backing[offset : offset + count].reshape(shape)
    encoded_all = torch.full((offset + count + 32,), 0xCD, dtype=torch.uint8, device=x.device)
    scales_all = torch.full((offset + scale_count + 32,), 0xCD, dtype=torch.uint8, device=x.device)
    encoded = encoded_all[offset : offset + count].reshape(shape)
    scales = scales_all[offset : offset + scale_count].reshape(scale_shape)
    op = sample()
    plan = op._make_execution_plan(x, x.ndim - 1, block_dim=blocks)
    dtype = dtypes.float32 if cpu.dtype == torch.float32 else dtypes.bfloat16
    program = op._get_compiled_kernel(dtype, dst == torch.float8_e5m2, get_mem_size("ub"), device=x.device)
    program(*op._kernel_views(x, encoded, scales, plan), plan.quant_length, plan.blocks, replay_state(x))
    expected = golden.dynamic_mx_quant_golden(input_numpy(cpu), dtype_name(dst))
    for actual, reference in zip((encoded, scales), expected):
        np.testing.assert_array_equal(actual.cpu().numpy(), reference)
    np.testing.assert_array_equal(backing.view(torch.uint8).cpu().numpy(), backing_cpu.view(torch.uint8).numpy())
    for allocation, size in ((encoded_all, count), (scales_all, scale_count)):
        raw = allocation.cpu().numpy()
        assert np.all(raw[:offset] == 0xCD) and np.all(raw[offset + size :] == 0xCD)
    if (shape[-1] + 31) // 32 % 2:
        assert not scales.cpu().numpy()[..., -1, 1].any()
    return tuple(value.cpu().numpy() for value in (encoded, scales))


CARRY_BASE = (17 << 32) + 0xFFFFFFFE

# Test probes own their resources; production stages expose narrow buffer views.
# Explicit-random Q16 expected bytes still come from independent Algorithm 19.
# SIMD Philox and explicit-field SR probes.


@kernel
class RuntimeStateWordsProbe:
    def __init__(self):
        self.dst = RingBuffer(MemLoc.UB, (8, 64), dtypes.uint32, depth=1)

    def __call__(self, state, out, base, stride, quant_length):
        seed = dtypes.uint64(scalar.load_bypass(state.ptr(0)))
        block_offset = dtypes.uint64(scalar.load_bypass(state.ptr(1)))
        axis_groups, reciprocal, row_base, col_base = operator._rng_group_geometry(base, quant_length)
        key0, key1 = operator._split_uint64(seed)
        dst = self.dst.next_write()
        with vf(mode="simd"):
            mask = reg.full_mask()
            ids = reg.varange(0, dtypes.uint32)
            c0, c1, c2, c3 = (
                reg.vdups(0, dtypes.uint32, mask=mask),
                reg.vdups(0, dtypes.uint32, mask=mask),
                reg.vdups(0, dtypes.uint32, mask=mask),
                reg.vdups(0, dtypes.uint32, mask=mask),
            )
            if stride == 1:
                c0, c1, c2, c3 = operator._linear_counter_contiguous(
                    row_base, col_base, ids, axis_groups, reciprocal, block_offset, mask
                )
            else:
                # The wide-layout probe visits stride=original C/16.
                c0, c1, c2, c3 = operator._linear_counter_rows(row_base, ids, col_base, axis_groups, block_offset, mask)
            reg.vstore(dst, 0, c0, mask)
            reg.vstore(dst, 64, c1, mask)
            reg.vstore(dst, 128, c2, mask)
            reg.vstore(dst, 192, c3, mask)
            w0, w1, w2, w3 = operator._philox_counter10(c0, c1, c2, c3, key0, key1, mask)
            reg.vstore(dst, 256, w0, mask)
            reg.vstore(dst, 320, w1, mask)
            reg.vstore(dst, 384, w2, mask)
            reg.vstore(dst, 448, w3, mask)
        operator._copy_to_gm(out, self.dst.next_read())


@host
def run_runtime_state_words(state, out, base: int, stride: int, quant_length: int):
    op = RuntimeStateWordsProbe()
    op[1](state, out, base, stride, quant_length)


@kernel
class CounterWordsProbe:
    def __init__(self):
        self.src = RingBuffer(MemLoc.UB, (4, 64), dtypes.uint32, depth=1)
        self.dst = RingBuffer(MemLoc.UB, (4, 64), dtypes.uint32, depth=1)

    def __call__(self, counters, out):
        mem_copy(self.src.next_write(), counters)
        src = self.src.next_read()
        dst = self.dst.next_write()
        with vf(mode="simd"):
            mask = reg.full_mask()
            c0, c1, c2, c3 = operator._philox_counter10(
                reg.vload(src, 0), reg.vload(src, 64), reg.vload(src, 128), reg.vload(src, 192), 0, 0, mask
            )
            reg.vstore(dst, 0, c0, mask)
            reg.vstore(dst, 64, c1, mask)
            reg.vstore(dst, 128, c2, mask)
            reg.vstore(dst, 192, c3, mask)
            reg.vmem_bar("vst_vld")
        operator._copy_to_gm(out, self.dst.next_read())


@host
def run_counter_words(counters, out):
    op = CounterWordsProbe()
    op[1](counters, out)


@kernel
class ExplicitSRProbe:
    def __init__(self, mantissa, bias, max_code):
        self.mantissa, self.bias, self.max_code = (mantissa, bias, max_code)
        self.x = RingBuffer(MemLoc.UB, (1, 64), dtypes.float32, depth=1)
        self.random = RingBuffer(MemLoc.UB, (1, 64), dtypes.uint32, depth=1)
        self.y = RingBuffer(MemLoc.UB, (1, 64), dtypes.uint8, depth=1)

    def __call__(self, x, random, out):
        mem_copy(self.x.next_write(), x)
        mem_copy(self.random.next_write(), random)
        x = self.x.next_read()
        random = self.random.next_read()
        y = self.y.next_write()
        with vf(mode="simd"):
            mask = reg.full_mask()
            codes = operator._sr_fp32_to_fp8_q16(
                reg.vload(x, 0), reg.vload(random, 0), self.mantissa, self.bias, self.max_code, mask
            )
            reg.vstore_pack(y, 0, codes, mask, pack_mode=reg.PackMode.B32_TO_B8)
            reg.vmem_bar("vst_vld")
        operator._copy_to_gm(out, self.y.next_read())


class ExplicitSRLaunch:
    def __init__(self, dst):
        self.bias, self.mantissa, self.max_code = golden._FP8_SR_PARAMS[dst]

    @host
    def run(self, x, random, out):
        op = ExplicitSRProbe(self.mantissa, self.bias, self.max_code)
        op[1](x, random, out)


# Batched Philox and carry-boundary probes.


@kernel
class WordStreamProbe:
    def __init__(self, aos):
        self.aos = aos
        self.dst = RingBuffer(MemLoc.UB, (1, 256), dtypes.int32, depth=1)

    def __call__(self, out, base, quant_length):
        axis_groups, reciprocal, row_base, col_base = operator._rng_group_geometry(base, quant_length)
        dst = self.dst.next_write()
        with vf(mode="simd"):
            operator._generate_random_words_keyed(
                dst, row_base, col_base, axis_groups, reciprocal, 0, 0, 0, dtypes.uint64(0), aos=self.aos
            )
            reg.vmem_bar("vst_vld")
        operator._copy_to_gm(out, self.dst.next_read())


@host
def run_words_soa(out, base: int, quant_length: int):
    op = WordStreamProbe(False)
    op[1](out, base, quant_length)


@host
def run_words_aos(out, base: int, quant_length: int):
    op = WordStreamProbe(True)
    op[1](out, base, quant_length)


@kernel
class PairedWordProbe:
    def __init__(self):
        self.dst = RingBuffer(MemLoc.UB, (1, 512), dtypes.int32, depth=1)

    def __call__(self, out, base, quant_length):
        axis_groups, reciprocal, row_base, col_base = operator._rng_group_geometry(base, quant_length)
        next_row, next_col = operator._advance_group(
            row_base, col_base, 64 // axis_groups, 64 % axis_groups, axis_groups
        )
        dst = self.dst.next_write()
        with vf(mode="simd"):
            operator._generate_random_words_keyed(
                dst, row_base, col_base, axis_groups, reciprocal, 0, 0, 0, dtypes.uint64(0), aos=True
            )
            operator._generate_random_words_keyed(
                dst, next_row, next_col, axis_groups, reciprocal, 256, 0, 0, dtypes.uint64(0), aos=True
            )
            reg.vmem_bar("vst_vld")
        operator._copy_to_gm(out, self.dst.next_read())


@host
def run_paired_words(out, base: int, quant_length: int):
    op = PairedWordProbe()
    op[1](out, base, quant_length)


@kernel
class MultiBatchWordsProbe:
    """Expose production normalized cursors, paired halves and odd last batch."""

    def __init__(self, aos, word_capacity=12 * 256):
        self.aos = aos
        self.random_words = Buffer(MemLoc.UB, (1, word_capacity), dtypes.int32)
        self.random = operator._RandomWords(self.random_words, word_capacity * 4)

    def __call__(self, out, base, quant_length, batches):
        mem_copy(self.random_words, out)
        if const_expr(self.aos):
            operator.prepare_random_words_aos_linear(
                self.random, base * 16, batches, dtypes.uint64(0), dtypes.uint64(0)
            )
        else:
            operator.prepare_random_words(
                self.random, base * 16, batches, quant_length, dtypes.uint64(0), dtypes.uint64(0)
            )
        operator._copy_to_gm(out, self.random_words)


class MultiBatchWordsLaunch:
    def __init__(self, aos, word_capacity=12 * 256):
        self.aos = aos
        self.word_capacity = word_capacity

    @host
    def run(self, out, base: int, quant_length: int, batches: int):
        op = MultiBatchWordsProbe(self.aos, self.word_capacity)
        op[1](out, base, quant_length, batches)


# Q16 rounding and integer scale probes.


@kernel
class Q16Probe:
    def __init__(self, mantissa, bias, max_code):
        self.mantissa, self.bias, self.max_code = (mantissa, bias, max_code)
        self.x = RingBuffer(MemLoc.UB, (1, 64), dtypes.float32, depth=1)
        self.random = RingBuffer(MemLoc.UB, (1, 64), dtypes.uint32, depth=1)
        self.y = RingBuffer(MemLoc.UB, (1, 64), dtypes.uint8, depth=1)

    def __call__(self, x, random, out):
        for tile in range(get_block_idx(), x.shape[0], get_block_num()):
            mem_copy(self.x.next_write(), tile_slice(x, (1, 64), (tile, 0)))
            mem_copy(self.random.next_write(), tile_slice(random, (1, 64), (tile, 0)))
            x_tile = self.x.next_read()
            random_tile = self.random.next_read()
            y_tile = self.y.next_write()
            with vf(mode="simd"):
                mask = reg.full_mask()
                codes = operator._sr_fp32_to_fp8_q16(
                    reg.vload(x_tile, 0), reg.vload(random_tile, 0), self.mantissa, self.bias, self.max_code, mask
                )
                reg.vstore_pack(y_tile, 0, codes, mask, pack_mode=reg.PackMode.B32_TO_B8)
                reg.vmem_bar("vst_vld")
            operator._copy_to_gm(tile_slice(out, (1, 64), (tile, 0)), self.y.next_read())


class Q16Launch:
    def __init__(self, dst):
        self.bias, self.mantissa, self.max_code = golden._FP8_SR_PARAMS[dst]

    @host
    def run(self, x, random, out):
        op = Q16Probe(self.mantissa, self.bias, self.max_code)
        op[32](x, random, out)


def q16_inputs(family):
    random_edges = np.array([0, 1, 32767, 32768, 65534, 65535, 2882371584, 4294967295], dtype=np.uint32)
    if family == "bf16_all_bits":
        bits = np.arange(65536, dtype=np.uint32) << np.uint32(16)
    elif family == "exponent_edges":
        fractions = np.array(
            [0, 1, 7, 15, 16, 31, 32, 131071, 262143, 4194303, 4194304, 8388606, 8388607], dtype=np.uint32
        )
        positive = ((np.arange(256, dtype=np.uint32) << np.uint32(23))[:, None] | fractions).reshape(-1)
        bits = np.concatenate((positive, positive | np.uint32(2147483648)))
    else:
        rng = np.random.default_rng(20260921)
        bits = rng.integers(0, 2**32, size=65536, dtype=np.uint32)
        fields = rng.integers(0, 2**32, size=65536, dtype=np.uint32)
        return (bits.view(np.float32).reshape(-1, 64), fields.reshape(-1, 64))
    return (
        np.repeat(bits, random_edges.size).view(np.float32).reshape(-1, 64),
        np.tile(random_edges, bits.size).reshape(-1, 64),
    )


# Independent integer scale probes.


@kernel
class IntegerScaleProbe:
    def __init__(self, inv_bits):
        self.inv_bits = inv_bits
        self.x = RingBuffer(MemLoc.UB, (1, 64), dtypes.uint32, depth=1)
        self.codes = RingBuffer(MemLoc.UB, (1, 64), dtypes.uint32, depth=1)
        self.recip = RingBuffer(MemLoc.UB, (1, 64), dtypes.uint32, depth=1)

    def __call__(self, x, codes, recip):
        for tile in range(get_block_idx(), x.shape[0], get_block_num()):
            mem_copy(self.x.next_write(), tile_slice(x, (1, 64), (tile, 0)))
            x_tile = self.x.next_read()
            codes_tile = self.codes.next_write()
            recip_tile = self.recip.next_write()
            with vf(mode="simd"):
                mask = reg.full_mask()
                code, reciprocal = operator._scale_alg1_integer(reg.vload(x_tile, 0), self.inv_bits, mask)
                reg.vstore(codes_tile, 0, code, mask)
                reg.vstore(recip_tile, 0, reg.vreinterpret(reciprocal, dtypes.uint32), mask)
                reg.vmem_bar("vst_vld")
            operator._copy_to_gm(tile_slice(codes, (1, 64), (tile, 0)), self.codes.next_read())
            operator._copy_to_gm(tile_slice(recip, (1, 64), (tile, 0)), self.recip.next_read())


class IntegerScaleLaunch:
    def __init__(self, inv_bits):
        self.inv_bits = inv_bits

    @host
    def run(self, x, codes, recip):
        op = IntegerScaleProbe(self.inv_bits)
        op[64](x, codes, recip)


# Retained short_rows.py probes.


def random_bits_at(start, numel, *, quant_length):
    """Independent golden Philox fields beginning at a 16-element boundary."""
    assert start % 16 == 0
    counters = np.arange((numel + 15) // 16, dtype=np.uint64) + np.uint64(start // 16)
    words = golden._philox_blocks(counters, quant_length=quant_length).reshape(-1)
    high, low = (words >> np.uint32(16), words & np.uint32(65535))
    rotate = (words << np.uint32(8)) | (words >> np.uint32(24))
    return np.stack((high, rotate & np.uint32(65535), low, rotate >> np.uint32(16)), axis=-1).reshape(-1)[:numel]


@kernel
class ShortRowsProbe(operator._DynamicMxQuantSrKernel._cls):
    """Expose the production helper without reproducing its implementation."""

    def __init__(self, input_dtype, e5m2, rows, columns, pitch_shift):
        tiling = operator._make_tiling(input_dtype, get_mem_size("ub"), tail_axis=False)
        super().__init__(input_dtype, e5m2, tiling.tile_elements, tiling.scale_capacity)
        self.rows = rows
        self.columns = columns
        self.pitch_shift = pitch_shift

    def __call__(self, x, y, scales, start):
        mem_copy(self.input.next_write(), x)
        input_tile = self.input.next_read()
        output_tile = self.output.next_write()
        scale_tile = self.scale.next_write()
        operator.quantize_short_rows(
            self.quantization,
            self.random.words,
            input_tile,
            output_tile,
            scale_tile,
            start,
            self.rows,
            self.columns,
            self.pitch_shift,
            dtypes.uint64(0),
            dtypes.uint64(0),
        )
        operator._copy_to_gm(y, self.output.next_read())
        operator._copy_to_gm(scales, self.scale.next_read())


class ShortRowsLaunch:
    def __init__(self, input_dtype, e5m2, rows, columns, pitch_shift):
        self.args = (input_dtype, e5m2, rows, columns, pitch_shift)

    @host
    def run(self, x, y, scales, start: int):
        op = ShortRowsProbe(*self.args)
        op[1](x, y, scales, start)


@kernel
class ShortRowsWordsProbe(ShortRowsProbe._cls):
    """Observe the production short-row RNG scratch, not a copied RNG loop."""

    def __call__(self, x, y, scales, words, start):
        mem_copy(self.input.next_write(), x)
        input_tile = self.input.next_read()
        output_tile = self.output.next_write()
        scale_tile = self.scale.next_write()
        operator.quantize_short_rows(
            self.quantization,
            self.random.words,
            input_tile,
            output_tile,
            scale_tile,
            start,
            self.rows,
            self.columns,
            self.pitch_shift,
            dtypes.uint64(0),
            dtypes.uint64(0),
        )
        operator._copy_to_gm(y, self.output.next_read())
        operator._copy_to_gm(scales, self.scale.next_read())
        operator._copy_to_gm(words, tile_slice(self.random.words.buffer, words.shape, (0, 0)))


class ShortRowsWordsLaunch(ShortRowsLaunch):
    @host
    def run(self, x, y, scales, words, start: int):
        op = ShortRowsWordsProbe(*self.args)
        op[1](x, y, scales, words, start)


@kernel
class BatchedFieldProbe:
    """Real SIMD->SIMT->SIMD stages, with untouched-lane sentinels."""

    def __init__(self):
        self.random_words = Buffer(MemLoc.UB, (1, 256), dtypes.int32)
        self.random = operator._RandomWords(self.random_words, 1024)
        self.fields = RingBuffer(MemLoc.UB, (16, 64), dtypes.uint32, depth=1)

    def __call__(self, words_out, fields_out, counter_low, counter_high, count, quant_length):
        result = self.fields.next_write()
        mem_copy(result, fields_out)
        start = (counter_low + counter_high * 4294967296) * 16
        operator.prepare_random_words(self.random, start, 1, quant_length, dtypes.uint64(0), dtypes.uint64(0))
        with vf(mode="simd"):
            ids = reg.varange(0, dtypes.uint32)
            for segment in range((count + 63) // 64):
                active = reg.update_mask(count - segment * 64, elem_bits=32)[0]
                fields = operator._random_fields_from_words(ids, self.random_words, segment, active)
                reg.vstore(result, segment * 64, fields, active)
            reg.vmem_bar("vst_vld")
        operator._copy_to_gm(words_out, reinterpret(self.random_words, dtypes.uint32, shape=(4, 64)))
        operator._copy_to_gm(fields_out, self.fields.next_read())


@host
def run_batched_fields(words_out, fields_out, low: int, high: int, count: int, quant_length: int):
    probe = BatchedFieldProbe()
    probe[1](words_out, fields_out, low, high, count, quant_length)


def finite_scale_boundary_input(src, dst_code, algorithm):
    # E4M3 uses exponent shift 8, E5M2 uses 15. Build exact representable
    # neighbors of the largest amax whose E8M0 code is zero. BF16 needs
    # adjacent BF16 values, rather than FP32 neighbors that collapse on cast.
    shift = 15 if dst_code == 23 else 8
    fraction_bits = 7 if src == torch.bfloat16 else 23
    if algorithm == 0:
        last_zero = ((shift + 1) << fraction_bits) - 1
    elif src == torch.bfloat16:
        last_zero = (shift << 7) | 0x60
    else:
        last_zero = (shift << 23) | 0x600001
    bits = [
        0,
        1,
        last_zero - 1,
        last_zero,
        last_zero + 1,
        last_zero + 2,
        (shift + 2) << fraction_bits,
        127 << fraction_bits,
    ]
    raw = torch.tensor(bits, dtype=torch.int32)
    if src == torch.bfloat16:
        raw = raw << 16
    magnitudes = raw.view(torch.float32).to(src)
    cpu = magnitudes.repeat_interleave(32).reshape(8, 32)
    cpu[:, 1::2].neg_()
    return cpu


@kernel
class EightFieldWordsProbe:
    """Generate eight-field Philox words without a scale/SR workspace."""

    def __init__(self):
        self.output = RingBuffer(MemLoc.UB, (1, 512), dtypes.int32, depth=1)

    def __call__(self, out, start: int, seed_bits, offset_bits):
        words = operator._RandomWords(self.output.next_write(), 2048)
        operator._prepare_eight_field_aos_linear(words, start, 4, dtypes.uint64(seed_bits), dtypes.uint64(offset_bits))
        operator._copy_to_gm(out, self.output.next_read())


@host
def run_eight_field_words(out, start: int, seed_bits, offset_bits):
    op = EightFieldWordsProbe()
    op[1](out, start, seed_bits, offset_bits)


INPUT_DTYPES = [torch.float32, torch.bfloat16]


OUTPUT_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


def _rng_host():
    return framework_rng


@pytest.fixture
def rng_device():
    import torch_npu  # noqa: F401  # register NPU without changing other processes

    if not torch.npu.is_available():
        pytest.skip("Ascend NPU unavailable")

    device = torch.device("npu", 0)
    with torch.npu.device(device):
        saved = torch.npu.get_rng_state(device)
        try:
            yield device
        finally:
            torch.npu.synchronize(device)
            torch.npu.set_rng_state(saved, device)


def _state_words(state, device):
    assert state.device == device and state.dtype == torch.int64
    assert state.shape == (2,) and state.is_contiguous()
    return state.cpu().numpy().view(np.uint64).tolist()


def _cache_key(**changes):
    key = operator.CompileKey(
        input_dtype="fp32",
        e5m2=False,
        ub_bytes=253952,
        scale_alg=1,
        max_low_bound_bits=0,
        layout=operator.Layout.TAIL,
        npu_arch="dav-3510",
        device_type="npu",
        device_index=0,
        dsl_version="test-wheel",
        native_build="test-native",
        compiler_identity="test-bisheng",
        host_abi=9,
        cache_format="test-format",
        cache_format_tag="",
        codegen_options=b"test-options",
    )
    return replace(key, **changes)


def _fake_program(path="compiled.so"):
    return SimpleNamespace(so_path=path, closed=False)


def _observe_cache_waiter(monkeypatch):
    waiting = Event()
    module = operator

    class ObservedFuture(Future):
        def result(self, *args, **kwargs):
            waiting.set()
            return super().result(*args, **kwargs)

    monkeypatch.setattr(module, "Future", ObservedFuture)
    return waiting


def _compile_context(monkeypatch, **changes):
    module = operator
    monkeypatch.setenv("CANNBOTDSL_NPU_ARCH", "dav-3510")
    monkeypatch.setattr(module, "_build_identity", lambda: ("wheel", "native", "compiler", 9, "format"))
    arguments = dict(
        input_dtype="fp32",
        e5m2=False,
        ub_bytes=253952,
        scale_alg=1,
        max_low_bound_bits=0,
        layout=operator.Layout.TAIL,
        device=SimpleNamespace(type="privateuseone", index=1),
    )
    arguments.update(changes)
    return module.capture_compile_context(**arguments)


@kernel
class _ReviewFixedFieldProbe:
    def __init__(self):
        self.words = RingBuffer(MemLoc.UB, (1, 256), dtypes.int32, depth=1)
        self.output = RingBuffer(MemLoc.UB, (1, 64), dtypes.uint32, depth=1)

    def __call__(self, words, output):
        mem_copy(self.words.next_write(), words)
        words_tile = self.words.next_read()
        result = self.output.next_write()
        with vf(mode="simd"):
            full = reg.full_mask()
            fields = operator._random_fields_from_words(reg.varange(0, dtypes.uint32), words_tile, 0, full)
            reg.vstore(result, 0, fields, full)
            # SIMD writes result; the following MTE read copies it to GM.
            reg.vmem_bar("vst_vld")
        operator._copy_to_gm(output, self.output.next_read())


@host
def _review_run_fixed_fields(words, output):
    probe = _ReviewFixedFieldProbe()
    probe[1](words, output)


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


def _unexpected_device_work(*args, **kwargs):
    pytest.fail("invalid input reached NPU setup or compilation")


LARGE_LAYOUT_CASES = [
    pytest.param((128, 128, 32), id="width32-tile128"),
    pytest.param((64, 256, 32), id="width32-tile256"),
    pytest.param((32, 512, 32), id="width32-tile512"),
    pytest.param((128, 128, 64), id="width64-tile128"),
    pytest.param((64, 256, 64), id="width64-tile256"),
    pytest.param((128, 128, 128), id="width128-tile128"),
]


SHORT_CASES = [
    pytest.param(
        src,
        dst,
        columns,
        multiple,
        delta,
        id=f"{('fp32-e4', 'bf16-e4', 'fp32-e5', 'bf16-e5')[index]}-k{columns}-{multiple}batch{delta:+}",
    )
    for index, (src, dst) in enumerate(DTYPE_PAIRS)
    for columns in ((16, 48, 80, 112) if src == torch.bfloat16 else (48, 80, 112))
    for multiple, delta in ((1, -1), (1, 0), (1, 1), (2, 1))
]


def _resolver():
    sample = operator
    assert callable(getattr(sample, "_resolve_block_dim", None)), "automatic AIV launch policy missing"
    return sample._resolve_block_dim


def _platform(monkeypatch, counts):
    device, stream = object(), object()
    values = iter(counts)

    def current_stream(selected_device):
        assert selected_device is device, "must query the input tensor's device"
        return stream

    def platform_info(*, stream):
        assert stream is expected_stream, "must respect this stream's core quota"
        return PlatformInfo(cube_core_num=32, vector_core_num=next(values), available=True)

    expected_stream = stream
    monkeypatch.setattr(torch.npu, "current_stream", current_stream)
    monkeypatch.setattr(cannbotdsl, "get_platform_info", platform_info)
    return SimpleNamespace(device=device, shape=(64, 80), numel=lambda: 5120)


WHITEBOX_CASES = [
    pytest.param(src, dst, case, id=f"{dtype_name(src)}-{dtype_name(dst)}-{case.name}")
    for src, dst in DTYPE_PAIRS
    for case in cases_for(src)
]


def _cpu(rows):
    # Admitted Q7 region with both signs and all three fallback regions,
    # repeated across original4K regions; both scale algorithms are exercised.
    base = torch.tensor([2.0, 2.0**-16, 0.5, -0.75] * 8, dtype=torch.bfloat16)
    cpu = base.repeat(rows * 8).reshape(rows, 256)
    cpu[16:32, 1] = 2.0**-30
    cpu[32:48] = 0
    cpu[48, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")])
    return cpu


THREADS = 16384


def _scalar_philox_words(counter, subsequence, seed):
    mask = 2**32 - 1
    a, b = 0, subsequence & mask
    c, d = counter & mask, counter >> 32
    k, key_high = seed & mask, seed >> 32
    for _ in range(10):
        p, q = a * 0xD2511F53, c * 0xCD9E8D57
        a, b, c, d = ((q >> 32) ^ b ^ k) & mask, q & mask, ((p >> 32) ^ d ^ key_high) & mask, p & mask
        k, key_high = (k + 0x9E3779B9) & mask, (key_high + 0xBB67AE85) & mask
    return [a, b, c, d]


def reject_materialization(*args, **kwargs):
    raise AssertionError("public launch must not materialize RNG state")


class TensorGeometry:
    dtype = torch.bfloat16
    device = SimpleNamespace(type="npu", index=0)

    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)

    def numel(self):
        return math.prod(self.shape)


@pytest.fixture
def quota(monkeypatch):
    def install(count):
        stream = object()
        monkeypatch.setattr(torch.npu, "current_stream", lambda device: stream)

        def platform(*, stream):
            return SimpleNamespace(vector_core_num=count)

        monkeypatch.setattr(cannbotdsl, "get_platform_info", platform)
        monkeypatch.setattr(operator, "get_mem_size", lambda name: 253952)

    return install
