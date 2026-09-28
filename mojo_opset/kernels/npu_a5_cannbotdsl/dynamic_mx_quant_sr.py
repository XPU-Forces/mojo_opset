# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the repository.

"""SIMD MX Quant + SR; SIMT is used only to reverse Philox word bits.

SIMD produces the Philox words and consumes their original/reversed forms.
Scale, normalization, SR and FP8 packing stay in SIMD. The current DSL emits
static UB and native bit reversal; the sample-local build adapter validates
the UB layout. Build files live with the loaded program; set
CANNBOT_MX_QUANT_SR_KEEP_BUILD=1 to retain them for inspection.
"""
import importlib.util
import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cannbotdsl
import torch
from cannbotdsl import Buffer, Dim, TensorSpec, dtypes, get_mem_size
from cannbotdsl.channel import Channel
from cannbotdsl.lang import const_expr, host, jit, kernel, vf
from cannbotdsl.ops import reg as rr, simt
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.tensor import MemLoc, Tensor, reinterpret, tile_slice

__all__ = ["dynamic_mx_quant_sr"]

SIMT_UB_RESERVE = 40 * 1024
SIMT_THREADS = 512

# Prefer "simd"; some DSL builds use "raw" for the same SIMD VF mode.
try:
    vf(mode="simd")
except ValueError as exc:
    if "unsupported vf mode" not in str(exc):
        raise
    _SIMD_VF_MODE = "raw"
else:
    _SIMD_VF_MODE = "simd"


@dataclass(frozen=True)
class BitReverseTiling:
    tile_elements: int
    scale_capacity: int
    buffer_bytes: int
    ub_bytes: int


def _make_tiling(input_dtype, ub_bytes, fused_non_tail=False):
    if input_dtype not in (dtypes.float32, dtypes.bfloat16):
        raise TypeError("tiling requires float32 or bfloat16")
    if type(ub_bytes) is not int or ub_bytes <= SIMT_UB_RESERVE:
        raise ValueError("UB capacity must exceed the 40 KiB SIMT reserve")
    item = 4 if input_dtype == dtypes.float32 else 2
    budget = ub_bytes - SIMT_UB_RESERVE
    tile = budget // (2 * item + 4) // 1024 * 1024
    while tile >= 1024:
        scales = ((tile // 32 + 63) // 64) * 64
        # 输入/FP8/scale 双缓冲；两份随机字各 N 字节；scale32 暂存。
        used = 2 * tile * item + 2 * tile + 2 * (tile // 16) + 4 * scales + 2 * tile
        if fused_non_tail:
            # Saved channel 0, packed channel pair, and packed E8M0 pairs.
            used += tile + tile // 16 + 2 * tile + tile // 8
        if used <= budget:
            return BitReverseTiling(tile, scales, used, ub_bytes)
        tile -= 1024
    raise ValueError("UB remaining after the SIMT reserve cannot fit one tile")


def _u32(value, mask):
    return rr.vdups(value, dtypes.uint32, mask=mask)


def _and(value, bits, mask):
    return rr.vbitwise_and(value, _u32(bits, mask), mask=mask)


def _philox_counter10(c0, c1, mask):
    """Ten Philox rounds for counter=(low, high, 0, 0), key=(0, 0)."""
    m0 = _u32(0xD2511F53, mask)
    m1 = _u32(0xCD9E8D57, mask)
    # All callers supply (low, high, 0, 0) with a zero initial key.
    # Fold only the first round's operations on these known zero words.
    lo0, hi0 = rr.vmull(c0, m0, dtypes.uint32, mask=mask)
    c0, c1, c2, c3 = c1, _u32(0, mask), hi0, lo0
    for step in range(1, 10):
        lo0, hi0 = rr.vmull(c0, m0, dtypes.uint32, mask=mask)
        lo1, hi1 = rr.vmull(c2, m1, dtypes.uint32, mask=mask)
        next0 = rr.vbitwise_xor(hi1, c1, mask=mask)
        next2 = rr.vbitwise_xor(hi0, c3, mask=mask)
        c0 = rr.vbitwise_xor(next0, _u32((step * 0x9E3779B9) & 0xFFFFFFFF, mask), mask=mask)
        c2 = rr.vbitwise_xor(next2, _u32((step * 0xBB67AE85) & 0xFFFFFFFF, mask), mask=mask)
        c1 = lo1
        c3 = lo0
    return c0, c1, c2, c3


def _sr_fp32_to_fp8_q16(value, r16, mantissa, bias, max_code, mask):
    """Encode base/discarded16 together; retain Algorithm 19's exact carry."""
    bits = rr.vreinterpret(value, dtypes.uint32)
    sign = _and(rr.vshr(bits, 24, mask=mask), 0x80, mask)
    absolute = _and(bits, 0x7FFFFFFF, mask)
    threshold = (128 - bias) << 23
    normal_q = rr.vsub(
        rr.vshr(absolute, 7 - mantissa, mask=mask),
        _u32((127 - bias) << (mantissa + 16), mask), mask=mask,
    )
    # Only subnormal lanes consume this conversion. Out-of-range normal
    # lanes are discarded by vselect; NaNs are canonicalized below.
    absolute_value = rr.vreinterpret(absolute, dtypes.float32)
    scaled = rr.vmuls(absolute_value, float(2 ** (bias + mantissa + 15)), mask=mask)
    sub_q = rr.vreinterpret(
        rr.vcast(scaled, dtypes.int32, mask=mask, rounding=rr.RoundingMode.RZ), dtypes.uint32,
    )
    q = rr.vselect(normal_q, sub_q, cond_mask=rr.vges(absolute, threshold, mask=mask))
    code = rr.vshr(rr.vadd(q, _and(r16, 0xFFFF, mask), mask=mask), 16, mask=mask)
    code = rr.vbitwise_or(rr.vmins(code, max_code, mask=mask), sign, mask=mask)
    return rr.vselect(_u32(0x7F, mask), code, cond_mask=rr.vgts(absolute, 0x7F800000, mask=mask))


def _sr_fp32_to_fp8(value, r16, mantissa, bias, max_code, mask):
    """FP32 bits + explicit low-16 random fields -> FP8 low-byte codes."""
    return _sr_fp32_to_fp8_q16(value, r16, mantissa, bias, max_code, mask)


def _scale_alg0(amax_bits, inv_max_bits, mask, canonical_invalid=True):
    """ASC ComputeScaleOcp: floor exponent; zero reciprocal when scale is zero."""
    shift = 15 if inv_max_bits == 0x37924925 else 8
    exponent = rr.vshr(amax_bits, 23, mask=mask)
    invalid = rr.vges(exponent, 255, mask=mask)
    code = rr.vadds(rr.vmaxs(exponent, shift, mask=mask), (-shift) & 0xFFFFFFFF, mask=mask)
    # Preserve finite lanes; fill invalid codes without a constant vector.
    code = rr.vdups(255, dtypes.uint32, mask=invalid, mode="merging", merge=code)
    # Finite FP8 scale codes cannot reach 254 (max exponent minus 8/15).
    # Masked shifts zero inactive lanes, including the code-0 multiplier.
    reciprocal = rr.vshl(rr.vsub(_u32(254, mask), code, mask=mask), 23,
                         mask=rr.vnes(code, 0, mask=mask))
    if canonical_invalid:
        reciprocal = rr.vdups(0x7F810000, dtypes.uint32, mask=invalid, mode="merging", merge=reciprocal)
    return code, rr.vreinterpret(reciprocal, dtypes.float32)


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
    invalid = rr.vges(amax_bits, 0x7F800000, mask=mask)
    # Active finite lanes cannot underflow after this bias adjustment;
    # inactive/invalid lanes are replaced by the masks below.
    code = rr.vshr(rr.vadds(amax_bits,
        (0x1FFFFF - (shift << 23)) & 0xFFFFFFFF, mask=mask), 23,
        mask=rr.vgts(amax_bits, (shift << 23) | 0x600001, mask=mask))
    # Preserve finite lanes; fill invalid codes without a constant vector.
    code = rr.vdups(255, dtypes.uint32, mask=invalid, mode="merging", merge=code)
    recip = rr.vshl(rr.vsub(_u32(254, mask), code, mask=mask), 23, mask=mask)
    if canonical_invalid:
        recip = rr.vdups(0x7F810000, dtypes.uint32, mask=invalid, mode="merging", merge=recip)
    # amax==0 means the complete block contains only signed zeros. Multiplying
    # those by the finite code-0 multiplier preserves their output bytes.
    return code, rr.vreinterpret(recip, dtypes.float32)


def _generate_random_words(scratch, counter, offset):
    mask = rr.full_mask()
    base_low = _u32(dtypes.uint32(counter), mask)
    low = rr.vadd(base_low, rr.varange(0, dtypes.uint32), mask=mask)
    carry = rr.vselect(_u32(1, mask), _u32(0, mask), cond_mask=rr.vlt(low, base_low, mask=mask))
    high = rr.vadd(_u32(dtypes.uint32(counter // 4294967296), mask), carry, mask=mask)
    w0, w1, w2, w3 = _philox_counter10(low, high, mask)
    rr.vstore(scratch, offset, rr.vreinterpret(w0, dtypes.int32), mask)
    rr.vstore(scratch, offset + 64, rr.vreinterpret(w1, dtypes.int32), mask)
    rr.vstore(scratch, offset + 128, rr.vreinterpret(w2, dtypes.int32), mask)
    rr.vstore(scratch, offset + 192, rr.vreinterpret(w3, dtypes.int32), mask)


def _generate_random_words_aos(scratch, counter, offset):
    mask = rr.full_mask()
    base_low = _u32(dtypes.uint32(counter), mask)
    low = rr.vadd(base_low, rr.varange(0, dtypes.uint32), mask=mask)
    carry = rr.vselect(_u32(1, mask), _u32(0, mask), cond_mask=rr.vlt(low, base_low, mask=mask))
    high = rr.vadd(_u32(dtypes.uint32(counter // 4294967296), mask), carry, mask=mask)
    w0, w1, w2, w3 = _philox_counter10(low, high, mask)
    a0, a1 = rr.vinterleave(w0, w2)
    b0, b1 = rr.vinterleave(w1, w3)
    # Interleave during the store; retain the same AoS word order.
    rr.vstore_interleave(scratch, offset, a0, b0)
    rr.vstore_interleave(scratch, offset + 128, a1, b1)


def _philox_counter_first(counter, ids, m0, mask):
    """First Philox round; retain carry into the high 32 counter bits."""
    base = _u32(dtypes.uint32(counter), mask)
    low = rr.vadd(base, ids, mask=mask)
    carry = rr.vselect(_u32(1, mask), _u32(0, mask), cond_mask=rr.vlt(low, base, mask=mask))
    high = rr.vadd(_u32(dtypes.uint32(counter // 4294967296), mask), carry, mask=mask)
    product, upper = rr.vmull(low, m0, dtypes.uint32, mask=mask)
    return high, _u32(0, mask), upper, product


def _philox_pair_three(a0, a1, a2, a3, b0, b1, b2, b3, step, m0, m1, mask):
    """Three rounds for two independent groups, sharing each round's keys."""
    # This Python loop unfolds three rounds. The enclosing VF loop supplies
    # runtime steps 1, 4 and 7, so only three pairs of keys are needed at once.
    for offset in range(3):
        round_id = dtypes.uint32(step) + dtypes.uint32(offset)
        key0 = _u32(round_id * dtypes.uint32(0x9E3779B9), mask)
        key1 = _u32(round_id * dtypes.uint32(0xBB67AE85), mask)
        lo_a0, hi_a0 = rr.vmull(a0, m0, dtypes.uint32, mask=mask)
        lo_a1, hi_a1 = rr.vmull(a2, m1, dtypes.uint32, mask=mask)
        lo_b0, hi_b0 = rr.vmull(b0, m0, dtypes.uint32, mask=mask)
        lo_b1, hi_b1 = rr.vmull(b2, m1, dtypes.uint32, mask=mask)
        a0 = rr.vbitwise_xor(rr.vbitwise_xor(hi_a1, a1, mask=mask), key0, mask=mask)
        a2 = rr.vbitwise_xor(rr.vbitwise_xor(hi_a0, a3, mask=mask), key1, mask=mask)
        a1, a3 = lo_a1, lo_a0
        b0 = rr.vbitwise_xor(rr.vbitwise_xor(hi_b1, b1, mask=mask), key0, mask=mask)
        b2 = rr.vbitwise_xor(rr.vbitwise_xor(hi_b0, b3, mask=mask), key1, mask=mask)
        b1, b3 = lo_b1, lo_b0
    return a0, a1, a2, a3, b0, b1, b2, b3


def _store_random_words_aos(scratch, offset, w0, w1, w2, w3, mask):
    a0, a1 = rr.vinterleave(w0, w2)
    b0, b1 = rr.vinterleave(w1, w3)
    # Interleave during the store; retain the same AoS word order.
    rr.vstore_interleave(scratch, offset, a0, b0)
    rr.vstore_interleave(scratch, offset + 128, a1, b1)


def _reverse32_scalar(value):
    # The current DSL exposes this operation as a native SIMT scalar intrinsic.
    return simt.brev(value)


def _random_fields_from_words(ids, scratch, reversed_words, segment, mask):
    # 每 1024 个输入对应 4×64 个随机字；word 的四个输出依次取
    # reverse16(high)、high、reverse16(low)、low。
    slot = _and(rr.vshr(ids, 2, mask=mask), 3, mask)
    index = rr.vadd(rr.vshl(slot, 6, mask=mask), rr.vshr(ids, 4, mask=mask), mask=mask)
    index = rr.vadds(index, dtypes.uint32((segment // 16) * 256 + (segment % 16) * 4), mask=mask)
    word = rr.vreinterpret(rr.vgather(scratch, index, mask=mask), dtypes.uint32)
    reverse = rr.vreinterpret(rr.vgather(reversed_words, index, mask=mask), dtypes.uint32)
    lane = _and(ids, 3, mask)
    even = rr.veqs(_and(lane, 1, mask), 0, mask=mask)
    selected = rr.vselect(reverse, word, cond_mask=even)
    # reverse32 同时交换两个半字，因此第 1/2 个 lane 取高半字。
    high_half = rr.mask_or(rr.veqs(lane, 1, mask=mask), rr.veqs(lane, 2, mask=mask), exec_mask=mask)
    return rr.vselect(rr.vshr(selected, 16, mask=mask), _and(selected, 0xFFFF, mask), cond_mask=high_half)


def _sr_magnitude_q16_unbiased_sum(absolute, r16, mantissa, bias, mask):
    # Normal lanes double exactly. Small lanes align the significand to the
    # FP8 subnormal grid with a half-ULP downward bias before FP32 rounding.
    value = rr.vreinterpret(absolute, dtypes.float32)
    magic = float(2.0 ** (1 - bias) - 2.0 ** (-bias - 23))
    mapped = rr.vadd(value, rr.vmaxs(value, magic, mask=mask), mask=mask)
    # The common integer-code bias is removed once after byte packing.
    q = rr.vshr(rr.vreinterpret(mapped, dtypes.uint32), 7 - mantissa, mask=mask)
    return rr.vadd(q, r16, mask=mask)


@kernel
class _BitReverseKernel:
    def __init__(self, input_dtype, e5m2, tile, scale_capacity, scale_alg=1, max_low_bound_bits=0,
                 fused_non_tail=False, wide_lanes=8):
        self.wide_lanes = wide_lanes
        self.scale_alg = scale_alg
        self.max_low_bound_bits = max_low_bound_bits
        self.bf16 = input_dtype == dtypes.bfloat16
        self.mantissa = 2 if e5m2 else 3
        self.bias = 15 if e5m2 else 7
        self.max_code = 0x7B if e5m2 else 0x7E
        self.inv_max_bits = 0x37924925 if e5m2 else 0x3B124925
        self.input = Channel(MemLoc.UB, (1, tile), input_dtype, depth=2)
        self.output = Channel(MemLoc.UB, (1, tile), dtypes.uint8, depth=2)
        self.scale = Channel(MemLoc.UB, (1, tile // 16), dtypes.uint8, depth=2)
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

    def _compute_scale(self, maxima, mask, canonical_invalid=True):
        """Shared scale rule; packed SR handles invalid output bytes itself."""
        if const_expr(self.scale_alg == 1 and self.max_low_bound_bits != 0):
            original_zero = rr.veqs(maxima, 0, mask=mask)
            maxima = rr.vmaxs(maxima, self.max_low_bound_bits, mask=mask)
            maxima = rr.vselect(_u32(0, mask), maxima, cond_mask=original_zero)
        if const_expr(self.scale_alg == 0):
            return _scale_alg0(maxima, self.inv_max_bits, mask, canonical_invalid)
        return _scale_alg1(maxima, self.inv_max_bits, mask, canonical_invalid)

    @jit
    def _reverse_random_words(self, batches):
        # A complete 4096-word RNG tile gives eight independent words per
        # thread. Load them together, then reverse/store without a SIMT loop.
        # Short exact batches also avoid a SIMT loop. Other runtime counts,
        # including larger BF16 tiles, retain the general path.
        if batches * 256 == SIMT_THREADS * 8:
            with vf(mode="simt", thread=SIMT_THREADS):
                index = dtypes.int32(simt.thread_idx()[0])
                word0 = dtypes.int32(self.random_words[0, index + SIMT_THREADS * 0])
                word1 = dtypes.int32(self.random_words[0, index + SIMT_THREADS * 1])
                word2 = dtypes.int32(self.random_words[0, index + SIMT_THREADS * 2])
                word3 = dtypes.int32(self.random_words[0, index + SIMT_THREADS * 3])
                word4 = dtypes.int32(self.random_words[0, index + SIMT_THREADS * 4])
                word5 = dtypes.int32(self.random_words[0, index + SIMT_THREADS * 5])
                word6 = dtypes.int32(self.random_words[0, index + SIMT_THREADS * 6])
                word7 = dtypes.int32(self.random_words[0, index + SIMT_THREADS * 7])
                self.reversed_words[0, index + SIMT_THREADS * 0] = dtypes.int32(_reverse32_scalar(word0))
                self.reversed_words[0, index + SIMT_THREADS * 1] = dtypes.int32(_reverse32_scalar(word1))
                self.reversed_words[0, index + SIMT_THREADS * 2] = dtypes.int32(_reverse32_scalar(word2))
                self.reversed_words[0, index + SIMT_THREADS * 3] = dtypes.int32(_reverse32_scalar(word3))
                self.reversed_words[0, index + SIMT_THREADS * 4] = dtypes.int32(_reverse32_scalar(word4))
                self.reversed_words[0, index + SIMT_THREADS * 5] = dtypes.int32(_reverse32_scalar(word5))
                self.reversed_words[0, index + SIMT_THREADS * 6] = dtypes.int32(_reverse32_scalar(word6))
                self.reversed_words[0, index + SIMT_THREADS * 7] = dtypes.int32(_reverse32_scalar(word7))
        elif batches == 8:
            with vf(mode="simt", thread=512):
                index = dtypes.int32(simt.thread_idx()[0])
                word0 = dtypes.int32(self.random_words[0, index + 0])
                word1 = dtypes.int32(self.random_words[0, index + 512])
                word2 = dtypes.int32(self.random_words[0, index + 1024])
                word3 = dtypes.int32(self.random_words[0, index + 1536])
                self.reversed_words[0, index + 0] = dtypes.int32(_reverse32_scalar(word0))
                self.reversed_words[0, index + 512] = dtypes.int32(_reverse32_scalar(word1))
                self.reversed_words[0, index + 1024] = dtypes.int32(_reverse32_scalar(word2))
                self.reversed_words[0, index + 1536] = dtypes.int32(_reverse32_scalar(word3))
        elif batches == 4:
            with vf(mode="simt", thread=512):
                index = dtypes.int32(simt.thread_idx()[0])
                word0 = dtypes.int32(self.random_words[0, index + 0])
                word1 = dtypes.int32(self.random_words[0, index + 512])
                self.reversed_words[0, index + 0] = dtypes.int32(_reverse32_scalar(word0))
                self.reversed_words[0, index + 512] = dtypes.int32(_reverse32_scalar(word1))
        elif batches == 2:
            with vf(mode="simt", thread=512):
                index = dtypes.int32(simt.thread_idx()[0])
                word0 = dtypes.int32(self.random_words[0, index + 0])
                self.reversed_words[0, index + 0] = dtypes.int32(_reverse32_scalar(word0))
        elif batches == 1:
            with vf(mode="simt", thread=256):
                index = dtypes.int32(simt.thread_idx()[0])
                word0 = dtypes.int32(self.random_words[0, index + 0])
                self.reversed_words[0, index + 0] = dtypes.int32(_reverse32_scalar(word0))
        else:
            with vf(mode="simt", thread=SIMT_THREADS):
                for index in range(simt.thread_idx()[0], batches * 256, SIMT_THREADS):
                    word = dtypes.int32(self.random_words[0, index])
                    self.reversed_words[0, index] = dtypes.int32(_reverse32_scalar(word))

    @jit
    def _compute(self, x, y, scales, start, count, scale_factor):
        segments = (count + 63) // 64
        batches = (count + 1023) // 1024
        with vf(mode=_SIMD_VF_MODE):
            for batch in range(batches):
                _generate_random_words(self.random_words, start // 16 + batch * 64, batch * 256)
            rr.vmem_bar("vst_vld")
        # 随机字按同一核 SIMD → SIMT → SIMD 传递，仅 SIMT 产生反转后的副本。
        self._reverse_random_words(batches)
        # One VF produces the whole batch. Per-segment VFs would rotate the
        # output Channel slot early, splitting one DMA tile across two slots.
        with vf(mode=_SIMD_VF_MODE):
            full = rr.full_mask()
            ids = rr.varange(0, dtypes.uint32)
            lower = rr.vlts(ids, 32, mask=full)
            upper = rr.vges(ids, 32, mask=full)
            for segment in range(segments):
                offset = segment * 64
                active = rr.update_mask(count - offset, elem_bits=32)[0]
                if const_expr(self.bf16):
                    loaded = rr.vload_unpack(x, offset, unpack_mode=rr.UnpackMode.B16_TO_B32)
                    values = rr.vcast(loaded, dtypes.float32, mask=full)
                else:
                    values = rr.vload(x, offset)
                values = rr.vselect(values, rr.vdups(0.0, dtypes.float32, mask=full), cond_mask=active)
                absolute = _and(rr.vreinterpret(values, dtypes.uint32), 0x7FFFFFFF, full)
                max0 = rr.vdup(rr.vreduce_max(absolute, mask=lower), mask=full)
                max1 = rr.vdup(rr.vreduce_max(absolute, mask=upper), mask=full)
                maxima = rr.vselect(max0, max1, cond_mask=lower)
                block_scales, reciprocal = self._compute_scale(maxima, full)
                normalized = rr.vmul(values, reciprocal, mask=full)
                if const_expr(self.bf16):
                    rounded = rr.vcast(normalized, dtypes.bfloat16, mask=full, rounding=rr.RoundingMode.RN)
                    normalized = rr.vcast(rounded, dtypes.float32, mask=full)
                # RNG follows the unpadded logical input, never UB pitch/grid.
                random16 = _random_fields_from_words(ids, self.random_words, self.reversed_words, segment, full)
                codes = _sr_fp32_to_fp8(normalized, random16, self.mantissa, self.bias, self.max_code, full)
                rr.vstore_pack(y, offset, codes, active, pack_mode=rr.PackMode.B32_TO_B8)
                rr.vstore_first(self.scale32, segment * 2, block_scales)
                upper_scale = rr.vgather_reg(block_scales, _u32(32, full))
                rr.vstore_first(self.scale32, segment * 2 + 1, upper_scale)
            rr.vmem_bar("vst_vld")
            scale_count = segments * 2
            if scale_factor == 4:
                scale_count = (count // 32) * 2
            for offset in range(0, scale_count, 64):
                scale_mask = rr.update_mask(scale_count - offset, elem_bits=32)[0]
                scale_ids = rr.vadds(ids, dtypes.uint32(offset), mask=full)
                if scale_factor == 4:
                    scale_codes = rr.vgather(self.scale32, rr.vshr(scale_ids, 1, mask=full), mask=scale_mask)
                    even = rr.veqs(_and(ids, 1, full), 0, mask=full)
                    scale_codes = rr.vselect(scale_codes, _u32(0, full), cond_mask=even)
                    rr.vstore_pack(scales, offset, scale_codes, scale_mask, pack_mode=rr.PackMode.B32_TO_B8)
                else:
                    scale_codes = rr.vgather(self.scale32, scale_ids, mask=scale_mask)
                    rr.vstore_pack(scales, offset, scale_codes, scale_mask, pack_mode=rr.PackMode.B32_TO_B8)
            rr.vmem_bar("vst_vld")

    @jit
    def _prepare_random_words_aos(self, start, batches):
        if const_expr(self.bf16):
            with vf(mode=_SIMD_VF_MODE):
                for batch in range(batches):
                    _generate_random_words_aos(self.random_words, start // 16 + batch * 64, batch * 256)
                rr.vmem_bar("vst_vld")
        else:
            if batches < 4:
                with vf(mode=_SIMD_VF_MODE):
                    for batch in range(batches):
                        _generate_random_words_aos(self.random_words, start // 16 + batch * 64, batch * 256)
                    rr.vmem_bar("vst_vld")
            else:
                with vf(mode=_SIMD_VF_MODE):
                    mask = rr.full_mask()
                    ids = rr.varange(0, dtypes.uint32)
                    m0 = _u32(0xD2511F53, mask)
                    m1 = _u32(0xCD9E8D57, mask)
                    for pair in range(batches // 2):
                        counter = start // 16 + pair * 128
                        a0, a1, a2, a3 = _philox_counter_first(counter, ids, m0, mask)
                        b0, b1, b2, b3 = _philox_counter_first(counter + 64, ids, m0, mask)
                        for step in range(1, 10, 3):
                            a0, a1, a2, a3, b0, b1, b2, b3 = _philox_pair_three(
                                a0, a1, a2, a3, b0, b1, b2, b3, step, m0, m1, mask)
                        _store_random_words_aos(self.random_words, pair * 512, a0, a1, a2, a3, mask)
                        _store_random_words_aos(self.random_words, pair * 512 + 256, b0, b1, b2, b3, mask)
                    rr.vmem_bar("vst_vld")
                # Keep the unpaired batch separate so its unrolled constants do
                # not extend the paired VF's register lifetimes.
                if batches % 2 != 0:
                    with vf(mode=_SIMD_VF_MODE):
                        for batch in range(batches // 2 * 2, batches):
                            _generate_random_words_aos(self.random_words, start // 16 + batch * 64, batch * 256)
                        rr.vmem_bar("vst_vld")

    @jit
    def _compute_256(self, x, y, scales, start, count, scale_factor):
        sign_source = reinterpret(x, dtypes.uint8 if self.bf16 else dtypes.uint16, shape=(1, self.tile_elements * 2))
        parts = count // 256
        batches = (count + 1023) // 1024
        self._prepare_random_words_aos(start, batches)
        # 随机字按同一核 SIMD → SIMT → SIMD 传递，仅 SIMT 产生反转后的副本。
        self._reverse_random_words(batches)
        # One producer for each complete DMA tile; keep y -> scales order.
        with vf(mode=_SIMD_VF_MODE):
            full = rr.full_mask()
            full16 = rr.full_mask(elem_bits=16)
            full8 = rr.full_mask(elem_bits=8)
            ids = rr.varange(0, dtypes.uint32)
            eight = rr.update_mask(8, elem_bits=32)[0]
            scale_index = rr.vshr(ids, 3, mask=full)
            sign_mask8 = rr.vdups(0x80, dtypes.uint8, mask=full8)
            # Gather order maps four 64-byte groups back to interleaved FP8.
            pack_ids8 = rr.varange(0, dtypes.uint8)
            pack_group8 = rr.vbitwise_and(pack_ids8, rr.vdups(3, dtypes.uint8, mask=full8), mask=full8)
            pack_index8 = rr.vbitwise_or(rr.vshl(pack_group8, 6, mask=full8), rr.vshr(pack_ids8, 2, mask=full8), mask=full8)
            random_mask32 = _u32(0xFFFF, full)
            # Each uint32 scale contributes its low byte, plus a zero byte
            # for K32 padding. Stream these bytes directly to the output.
            scale_byte_slot = rr.vbitwise_and(pack_ids8, rr.vdups(3, dtypes.uint8, mask=full8), mask=full8)
            scale_byte_mask = rr.mask_and(rr.vlts(pack_ids8, 32, mask=full8),
                rr.vlts(scale_byte_slot, dtypes.uint8(scale_factor // 2), mask=full8), exec_mask=full8)
            scale_cursor = rr.vstore_unalign_begin(scales)
            for part in range(parts):
                offset = part * 256
                if const_expr(self.bf16):
                    a, b = rr.vload_deinterleave(x, offset)
                    abs_mask = rr.vdups(0x7FFF, dtypes.uint16, mask=full16)
                    aa = rr.vbitwise_and(rr.vreinterpret(a, dtypes.uint16), abs_mask, mask=full16)
                    ab = rr.vbitwise_and(rr.vreinterpret(b, dtypes.uint16), abs_mask, mask=full16)
                    maxima16 = rr.vreduce_max_datablock(rr.vmax(aa, ab, mask=full16), mask=full16)
                    maxima = rr.vshl(rr.vunpack(maxima16, dtypes.uint32), 16, mask=full)
                    unused_signbytes, signbytes = rr.vload_deinterleave(sign_source, offset * 2)
                    signs = rr.vreinterpret_lanes(rr.vbitwise_and(signbytes, sign_mask8, mask=full8), dtypes.uint32)
                else:
                    a, b = rr.vload_deinterleave(x, offset)
                    c, d = rr.vload_deinterleave(x, offset + 128)
                    v0, v2 = rr.vdeinterleave(a, c)
                    v1, v3 = rr.vdeinterleave(b, d)
                    a0 = rr.vreinterpret(rr.vabs(v0, mask=full), dtypes.uint32)
                    a1 = rr.vreinterpret(rr.vabs(v1, mask=full), dtypes.uint32)
                    a2 = rr.vreinterpret(rr.vabs(v2, mask=full), dtypes.uint32)
                    a3 = rr.vreinterpret(rr.vabs(v3, mask=full), dtypes.uint32)
                    pair0 = rr.vmax(a0, a1, mask=full)
                    pair1 = rr.vmax(a2, a3, mask=full)
                    maxima = rr.vreduce_max_datablock(rr.vmax(pair0, pair1, mask=full), mask=full)
                    unused_sign01, sign01 = rr.vload_deinterleave(sign_source, offset * 2)
                    unused_sign23, sign23 = rr.vload_deinterleave(sign_source, offset * 2 + 256)
                    unused_signbytes, signbytes = rr.vdeinterleave(rr.vreinterpret_lanes(sign01, dtypes.uint8), rr.vreinterpret_lanes(sign23, dtypes.uint8))
                    signs = rr.vreinterpret_lanes(rr.vbitwise_and(signbytes, sign_mask8, mask=full8), dtypes.uint32)
                block_scales, reciprocal = self._compute_scale(maxima, full, False)
                if const_expr(self.bf16):
                    # Pack selects the destination register half, not the
                    # source halfword: first move BF16 bits into the low16.
                    bits16 = rr.vshr(rr.vreinterpret(reciprocal, dtypes.uint32), 16, mask=full)
                    r16 = rr.vpack(bits16, dtypes.uint16, part="lower")
                    index16 = rr.vshr(rr.varange(0, dtypes.uint16), 4, mask=full16)
                    broadcast = rr.vreinterpret(rr.vgather_reg(r16, index16), dtypes.bfloat16)
                    a = rr.vmul(rr.vreinterpret(aa, dtypes.bfloat16), broadcast, mask=full16)
                    b = rr.vmul(rr.vreinterpret(ab, dtypes.bfloat16), broadcast, mask=full16)
                    v0 = rr.vcast(a, dtypes.float32, mask=full16, reg_layout=rr.RegLayout.ZERO)
                    v2 = rr.vcast(a, dtypes.float32, mask=full16, reg_layout=rr.RegLayout.ONE)
                    v1 = rr.vcast(b, dtypes.float32, mask=full16, reg_layout=rr.RegLayout.ZERO)
                    v3 = rr.vcast(b, dtypes.float32, mask=full16, reg_layout=rr.RegLayout.ONE)
                    # Expand each invalid scale to all 32-bit packed output words.
                    invalid_reciprocal = rr.vgather_reg(reciprocal, scale_index)
                else:
                    reciprocal = rr.vgather_reg(reciprocal, scale_index)
                    v0 = rr.vmul(rr.vreinterpret(a0, dtypes.float32), reciprocal, mask=full)
                    v1 = rr.vmul(rr.vreinterpret(a1, dtypes.float32), reciprocal, mask=full)
                    v2 = rr.vmul(rr.vreinterpret(a2, dtypes.float32), reciprocal, mask=full)
                    v3 = rr.vmul(rr.vreinterpret(a3, dtypes.float32), reciprocal, mask=full)
                    invalid_reciprocal = reciprocal
                word = rr.vreinterpret(rr.vload(self.random_words, part * 64), dtypes.uint32)
                reversed_word = rr.vreinterpret(rr.vload(self.reversed_words, part * 64), dtypes.uint32)
                r0 = rr.vbitwise_and(reversed_word, random_mask32, mask=full)
                r1 = rr.vshr(word, 16, mask=full)
                r2 = rr.vshr(reversed_word, 16, mask=full)
                r3 = rr.vbitwise_and(word, random_mask32, mask=full)
                c0 = _sr_magnitude_q16_unbiased_sum(
                    rr.vreinterpret(v0, dtypes.uint32), r0,
                    self.mantissa, self.bias, full)
                c1 = _sr_magnitude_q16_unbiased_sum(
                    rr.vreinterpret(v1, dtypes.uint32), r1,
                    self.mantissa, self.bias, full)
                c2 = _sr_magnitude_q16_unbiased_sum(
                    rr.vreinterpret(v2, dtypes.uint32), r2,
                    self.mantissa, self.bias, full)
                c3 = _sr_magnitude_q16_unbiased_sum(
                    rr.vreinterpret(v3, dtypes.uint32), r3,
                    self.mantissa, self.bias, full)
                # Select bits 23:16 directly; avoid four right shifts
                # and the following per-word shifts/ORs.
                unused_code01, code01 = rr.vdeinterleave(rr.vreinterpret_lanes(c0, dtypes.uint16), rr.vreinterpret_lanes(c1, dtypes.uint16))
                unused_code23, code23 = rr.vdeinterleave(rr.vreinterpret_lanes(c2, dtypes.uint16), rr.vreinterpret_lanes(c3, dtypes.uint16))
                code_bytes, unused_code_bytes = rr.vdeinterleave(rr.vreinterpret_lanes(code01, dtypes.uint8), rr.vreinterpret_lanes(code23, dtypes.uint8))
                packed = rr.vreinterpret_lanes(rr.vgather_reg(code_bytes, pack_index8), dtypes.uint32)
                # Every packed word belongs to one block32. Its scale
                # marks all four bytes invalid together for Inf/NaN input.
                # Q16 bias contains whole code units; remove it once
                # per packed byte after all four carry additions.
                magnitude8 = rr.vadds(rr.vreinterpret_lanes(packed, dtypes.uint8),
                                      (-((128 - self.bias) << self.mantissa)) & 255, mask=full8)
                # Masked min zeroes inactive lanes, including the -1
                # Q16 sentinel (255 after packing), and saturates others.
                saturated = rr.vmins(magnitude8, self.max_code,
                                     mask=rr.vnes(magnitude8, 255, mask=full8))
                packed = rr.vbitwise_or(rr.vreinterpret_lanes(saturated, dtypes.uint32), signs, mask=full)
                # The fast scale path leaves code 255's multiplier as
                # -Inf; override all four output bytes explicitly here.
                invalid = rr.veqs(rr.vreinterpret(invalid_reciprocal, dtypes.uint32), 0xFF800000, mask=full)
                packed = rr.vdups(0x7F7F7F7F, dtypes.uint32, mask=invalid, mode="merging", merge=packed)
                rr.vstore(y, offset, rr.vreinterpret_lanes(packed, dtypes.uint8), full8)
                scale_bytes = rr.vsqueeze_and_storeunalign_init(
                    rr.vreinterpret_lanes(block_scales, dtypes.uint8), mask=scale_byte_mask)
                rr.vsqueeze_and_storeunalign(scales, 0, scale_bytes, scale_cursor)
            rr.vsqueeze_and_storeunalign_finalize(scales, 0, scale_cursor)
            rr.vmem_bar("vst_vld")

    @jit
    def _compute_short_rows(self, x, y, scales, start, rows, columns, pitch_shift):
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
        with vf(mode=_SIMD_VF_MODE):
            for batch in range(batches):
                _generate_random_words_aos(self.random_words, start // 16 + batch * 64, batch * 256)
            rr.vmem_bar("vst_vld")
        # 随机字按同一核 SIMD → SIMT → SIMD 传递，仅 SIMT 产生反转后的副本。
        self._reverse_random_words(batches)
        # The single VF owns the complete output/scale Channel transactions.
        with vf(mode=_SIMD_VF_MODE):
            full = rr.full_mask()
            ids = rr.varange(0, dtypes.uint32)
            eight = rr.update_mask(8, elem_bits=32)[0]
            scale_index = rr.vshr(ids, 3, mask=full)
            zero = _u32(0, full)
            wide_pitch = rr.veqs(_u32(pitch_bits, full), 7, mask=full)
            for part in range(parts):
                virtual4 = rr.vadds(ids, dtypes.uint32(part * 64), mask=full)
                row = rr.vselect(rr.vshr(virtual4, 5, mask=full),
                                 rr.vshr(virtual4, 4, mask=full), cond_mask=wide_pitch)
                column4 = rr.vbitwise_and(virtual4, _u32(column_mask, full), mask=full)
                logical4 = rr.vadd(rr.vmuls(row, columns4, mask=full), column4, mask=full)
                valid = rr.mask_and(rr.vlts(row, row_count, mask=full),
                                    rr.vlts(column4, columns4, mask=full), exec_mask=full)
                if const_expr(self.bf16):
                    index = rr.vshl(logical4, 1, mask=full)
                    a = rr.vgather(packed_x, index, mask=valid)
                    b = rr.vgather(packed_x, rr.vadds(index, 1, mask=full), mask=valid)
                    a = rr.vselect(a, zero, cond_mask=valid)
                    b = rr.vselect(b, zero, cond_mask=valid)
                    v0 = rr.vreinterpret(rr.vshl(a, 16, mask=full), dtypes.float32)
                    v1 = rr.vreinterpret(_and(a, 0xFFFF0000, full), dtypes.float32)
                    v2 = rr.vreinterpret(rr.vshl(b, 16, mask=full), dtypes.float32)
                    v3 = rr.vreinterpret(_and(b, 0xFFFF0000, full), dtypes.float32)
                else:
                    index = rr.vshl(logical4, 2, mask=full)
                    v0 = rr.vgather(x, index, mask=valid)
                    v1 = rr.vgather(x, rr.vadds(index, 1, mask=full), mask=valid)
                    v2 = rr.vgather(x, rr.vadds(index, 2, mask=full), mask=valid)
                    v3 = rr.vgather(x, rr.vadds(index, 3, mask=full), mask=valid)
                    zeros = rr.vdups(0.0, dtypes.float32, mask=full)
                    v0 = rr.vselect(v0, zeros, cond_mask=valid)
                    v1 = rr.vselect(v1, zeros, cond_mask=valid)
                    v2 = rr.vselect(v2, zeros, cond_mask=valid)
                    v3 = rr.vselect(v3, zeros, cond_mask=valid)
                a0 = _and(rr.vreinterpret(v0, dtypes.uint32), 0x7FFFFFFF, full)
                a1 = _and(rr.vreinterpret(v1, dtypes.uint32), 0x7FFFFFFF, full)
                a2 = _and(rr.vreinterpret(v2, dtypes.uint32), 0x7FFFFFFF, full)
                a3 = _and(rr.vreinterpret(v3, dtypes.uint32), 0x7FFFFFFF, full)
                maxima = rr.vreduce_max_datablock(
                    rr.vmax(rr.vmax(a0, a1, mask=full), rr.vmax(a2, a3, mask=full), mask=full), mask=full)
                block_scales, reciprocal = self._compute_scale(maxima, full)
                reciprocal = rr.vgather_reg(reciprocal, scale_index)
                v0 = rr.vmul(v0, reciprocal, mask=full)
                v1 = rr.vmul(v1, reciprocal, mask=full)
                v2 = rr.vmul(v2, reciprocal, mask=full)
                v3 = rr.vmul(v3, reciprocal, mask=full)
                if const_expr(self.bf16):
                    v0 = rr.vcast(rr.vcast(v0, dtypes.bfloat16, mask=full, rounding=rr.RoundingMode.RN), dtypes.float32, mask=full)
                    v1 = rr.vcast(rr.vcast(v1, dtypes.bfloat16, mask=full, rounding=rr.RoundingMode.RN), dtypes.float32, mask=full)
                    v2 = rr.vcast(rr.vcast(v2, dtypes.bfloat16, mask=full, rounding=rr.RoundingMode.RN), dtypes.float32, mask=full)
                    v3 = rr.vcast(rr.vcast(v3, dtypes.bfloat16, mask=full, rounding=rr.RoundingMode.RN), dtypes.float32, mask=full)
                # RNG index follows compact input, never the padded row pitch.
                word = rr.vreinterpret(rr.vgather(self.random_words, logical4, mask=valid), dtypes.uint32)
                reversed_word = rr.vreinterpret(rr.vgather(self.reversed_words, logical4, mask=valid), dtypes.uint32)
                rab = rr.vshr(word, 16, mask=full)
                ref = _and(word, 0xFFFF, full)
                c0 = _sr_fp32_to_fp8(v0, _and(reversed_word, 0xFFFF, full), self.mantissa, self.bias, self.max_code, full)
                c1 = _sr_fp32_to_fp8(v1, rab, self.mantissa, self.bias, self.max_code, full)
                c2 = _sr_fp32_to_fp8(v2, rr.vshr(reversed_word, 16, mask=full), self.mantissa, self.bias, self.max_code, full)
                c3 = _sr_fp32_to_fp8(v3, ref, self.mantissa, self.bias, self.max_code, full)
                lo = rr.vbitwise_or(c0, rr.vshl(c1, 8, mask=full), mask=full)
                hi = rr.vbitwise_or(rr.vshl(c2, 16, mask=full), rr.vshl(c3, 24, mask=full), mask=full)
                rr.vscatter(packed_y, rr.vbitwise_or(lo, hi, mask=full), logical4, mask=valid)
                rr.vstore(self.scale32, part * 8, block_scales, eight)
            rr.vmem_bar("vst_vld")
            for offset in range(0, scale_count, 64):
                scale_codes = rr.vload(self.scale32, offset)
                scale_mask = rr.update_mask(scale_count - offset, elem_bits=32)[0]
                rr.vstore_pack(scales, offset, scale_codes, scale_mask, pack_mode=rr.PackMode.B32_TO_B8)
            rr.vmem_bar("vst_vld")

    @jit
    def _compute_wide64(self, x, y, scales, row, group, quant_length, width, d_offset, tile_rows):
        """Quantize one S tile by 64 physical D lanes without GM transposes."""
        base = ((row * width + d_offset) * quant_length + group * tile_rows) // 16
        scale_pairs = reinterpret(scales, dtypes.uint16, shape=(1, self.tile_elements // 32))
        with vf(mode=_SIMD_VF_MODE):
            full = rr.full_mask()
            lane = rr.varange(0, dtypes.uint32)
            lane_delta = rr.vmuls(lane, dtypes.uint32(quant_length // 16), mask=full)
            for counter in range(tile_rows // 16):
                scalar = base + counter
                scalar_low = _u32(dtypes.uint32(scalar), full)
                low = rr.vadd(scalar_low, lane_delta, mask=full)
                carry = rr.vselect(_u32(1, full), _u32(0, full),
                                   cond_mask=rr.vlt(low, scalar_low, mask=full))
                high = rr.vadd(_u32(dtypes.uint32(scalar // 4294967296), full), carry, mask=full)
                w0, w1, w2, w3 = _philox_counter10(low, high, full)
                rr.vstore(self.random_words, counter * 256, rr.vreinterpret(w0, dtypes.int32), full)
                rr.vstore(self.random_words, counter * 256 + 64, rr.vreinterpret(w1, dtypes.int32), full)
                rr.vstore(self.random_words, counter * 256 + 128, rr.vreinterpret(w2, dtypes.int32), full)
                rr.vstore(self.random_words, counter * 256 + 192, rr.vreinterpret(w3, dtypes.int32), full)
            rr.vmem_bar("vst_vld")
        self._reverse_random_words(tile_rows // 16)
        with vf(mode=_SIMD_VF_MODE):
            full = rr.full_mask()
            for half in range(tile_rows // 32):
                maxima = _u32(0, full)
                for inner in range(32):
                    offset = (half * 32 + inner) * width + d_offset
                    if const_expr(self.bf16):
                        loaded = rr.vload_unpack(x, offset, unpack_mode=rr.UnpackMode.B16_TO_B32)
                        values = rr.vcast(loaded, dtypes.float32, mask=full)
                    else:
                        values = rr.vload(x, offset)
                    absolute = _and(rr.vreinterpret(values, dtypes.uint32), 0x7FFFFFFF, full)
                    maxima = rr.vmax(maxima, absolute, mask=full)
                scale_codes, reciprocal = self._compute_scale(maxima, full)
                rr.vstore(self.scale32, half * 64, scale_codes, full)
                for word in range(8):
                    slot = half * 512 + word * 64
                    original = rr.vreinterpret(rr.vload(self.random_words, slot), dtypes.uint32)
                    reversed_word = rr.vreinterpret(rr.vload(self.reversed_words, slot), dtypes.uint32)
                    first = half * 32 + word * 4
                    self._wide_store_row(x, y, first * width + d_offset, reciprocal,
                                         _and(reversed_word, 0xFFFF, full), full)
                    self._wide_store_row(x, y, (first + 1) * width + d_offset, reciprocal,
                                         rr.vshr(original, 16, mask=full), full)
                    self._wide_store_row(x, y, (first + 2) * width + d_offset, reciprocal,
                                         rr.vshr(reversed_word, 16, mask=full), full)
                    self._wide_store_row(x, y, (first + 3) * width + d_offset, reciprocal,
                                         _and(original, 0xFFFF, full), full)
            rr.vmem_bar("vst_vld")
            pair_mask = rr.update_mask(64, elem_bits=16)[0]
            for pair in range(tile_rows // 64):
                first_scale = rr.vload(self.scale32, pair * 128)
                second_scale = rr.vload(self.scale32, pair * 128 + 64)
                code_pairs = rr.vbitwise_or(first_scale,
                                           rr.vshl(second_scale, 8, mask=full), mask=full)
                packed = rr.vpack(code_pairs, dtypes.uint16, part="lower")
                # A uint8 interleave would write 512 B, overlapping the next
                # D chunk. Store exactly 64 uint16 scale pairs (128 B).
                rr.vstore(scale_pairs, pair * width + d_offset, packed, pair_mask)
            rr.vmem_bar("vst_vld")

    @jit
    def _compute_wide32(self, x, y, scales, row, group, quant_length, width, d_offset, tile_rows):
        """Quantize one S tile by 32 physical D lanes without GM transposes."""
        lanes = 32
        base = ((row * width + d_offset) * quant_length + group * tile_rows) // 16
        scale_pairs = reinterpret(scales, dtypes.uint16, shape=(1, self.tile_elements // 32))
        with vf(mode=_SIMD_VF_MODE):
            full = rr.update_mask(lanes, elem_bits=32)[0]
            lane = rr.varange(0, dtypes.uint32)
            lane_delta = rr.vmuls(lane, dtypes.uint32(quant_length // 16), mask=full)
            for counter in range(tile_rows // 16):
                scalar = base + counter
                scalar_low = _u32(dtypes.uint32(scalar), full)
                low = rr.vadd(scalar_low, lane_delta, mask=full)
                carry = rr.vselect(_u32(1, full), _u32(0, full),
                                   cond_mask=rr.vlt(low, scalar_low, mask=full))
                high = rr.vadd(_u32(dtypes.uint32(scalar // 4294967296), full), carry, mask=full)
                w0, w1, w2, w3 = _philox_counter10(low, high, full)
                rr.vstore(self.random_words, counter * 4 * lanes, rr.vreinterpret(w0, dtypes.int32), full)
                rr.vstore(self.random_words, counter * 4 * lanes + lanes, rr.vreinterpret(w1, dtypes.int32), full)
                rr.vstore(self.random_words, counter * 4 * lanes + 2 * lanes, rr.vreinterpret(w2, dtypes.int32), full)
                rr.vstore(self.random_words, counter * 4 * lanes + 3 * lanes, rr.vreinterpret(w3, dtypes.int32), full)
            rr.vmem_bar("vst_vld")
        self._reverse_random_words(tile_rows * lanes // 1024)
        with vf(mode=_SIMD_VF_MODE):
            full = rr.update_mask(lanes, elem_bits=32)[0]
            for half in range(tile_rows // 32):
                if const_expr(not self.bf16):
                    full64 = rr.full_mask()
                    lane64 = rr.varange(0, dtypes.uint32)
                    maxima64 = _u32(0, full64)
                    for inner in range(16):
                        offset = (half * 32 + inner * 2) * width + d_offset
                        values64 = rr.vload(x, offset)
                        absolute64 = _and(rr.vreinterpret(values64, dtypes.uint32), 0x7FFFFFFF, full64)
                        maxima64 = rr.vmax(maxima64, absolute64, mask=full64)
                    swap32 = rr.vbitwise_xor(lane64, _u32(32, full64), mask=full64)
                    maxima = rr.vmax(maxima64, rr.vgather_reg(maxima64, swap32), mask=full64)
                else:
                    maxima = _u32(0, full)
                    for inner in range(32):
                        offset = (half * 32 + inner) * width + d_offset
                        loaded = rr.vload_unpack(x, offset, unpack_mode=rr.UnpackMode.B16_TO_B32)
                        values = rr.vcast(loaded, dtypes.float32, mask=full)
                        absolute = _and(rr.vreinterpret(values, dtypes.uint32), 0x7FFFFFFF, full)
                        maxima = rr.vmax(maxima, absolute, mask=full)
                scale_codes, reciprocal = self._compute_scale(maxima, full)
                rr.vstore(self.scale32, half * lanes, scale_codes, full)
                for word in range(8):
                    slot = half * 8 * lanes + word * lanes
                    original = rr.vreinterpret(rr.vload(self.random_words, slot), dtypes.uint32)
                    reversed_word = rr.vreinterpret(rr.vload(self.reversed_words, slot), dtypes.uint32)
                    first = half * 32 + word * 4
                    self._wide_store_row(x, y, first * width + d_offset, reciprocal,
                                         _and(reversed_word, 0xFFFF, full), full)
                    self._wide_store_row(x, y, (first + 1) * width + d_offset, reciprocal,
                                         rr.vshr(original, 16, mask=full), full)
                    self._wide_store_row(x, y, (first + 2) * width + d_offset, reciprocal,
                                         rr.vshr(reversed_word, 16, mask=full), full)
                    self._wide_store_row(x, y, (first + 3) * width + d_offset, reciprocal,
                                         _and(original, 0xFFFF, full), full)
            rr.vmem_bar("vst_vld")
            pair_mask = rr.update_mask(lanes, elem_bits=16)[0]
            for pair in range(tile_rows // 64):
                first_scale = rr.vload(self.scale32, pair * 2 * lanes)
                second_scale = rr.vload(self.scale32, pair * 2 * lanes + lanes)
                code_pairs = rr.vbitwise_or(first_scale,
                                           rr.vshl(second_scale, 8, mask=full), mask=full)
                packed = rr.vpack(code_pairs, dtypes.uint16, part="lower")
                rr.vstore(scale_pairs, pair * width + d_offset, packed, pair_mask)
            rr.vmem_bar("vst_vld")

    @jit
    def _compute_wide8(self, x, y, scales, row, group, quant_length, width, d_offset, tile_rows):
        """Quantize one S tile by compile-time physical D lanes without GM transposes."""
        lanes = self.wide_lanes
        base = ((row * width + d_offset) * quant_length + group * tile_rows) // 16
        scale_pairs = reinterpret(scales, dtypes.uint16, shape=(1, self.tile_elements // 32))
        with vf(mode=_SIMD_VF_MODE):
            full = rr.update_mask(lanes, elem_bits=32)[0]
            lane = rr.varange(0, dtypes.uint32)
            lane_delta = rr.vmuls(lane, dtypes.uint32(quant_length // 16), mask=full)
            for counter in range(tile_rows // 16):
                scalar = base + counter
                scalar_low = _u32(dtypes.uint32(scalar), full)
                low = rr.vadd(scalar_low, lane_delta, mask=full)
                carry = rr.vselect(_u32(1, full), _u32(0, full),
                                   cond_mask=rr.vlt(low, scalar_low, mask=full))
                high = rr.vadd(_u32(dtypes.uint32(scalar // 4294967296), full), carry, mask=full)
                w0, w1, w2, w3 = _philox_counter10(low, high, full)
                rr.vstore(self.random_words, counter * 4 * lanes, rr.vreinterpret(w0, dtypes.int32), full)
                rr.vstore(self.random_words, counter * 4 * lanes + lanes, rr.vreinterpret(w1, dtypes.int32), full)
                rr.vstore(self.random_words, counter * 4 * lanes + 2 * lanes, rr.vreinterpret(w2, dtypes.int32), full)
                rr.vstore(self.random_words, counter * 4 * lanes + 3 * lanes, rr.vreinterpret(w3, dtypes.int32), full)
            rr.vmem_bar("vst_vld")
        self._reverse_random_words(tile_rows * lanes // 1024)
        with vf(mode=_SIMD_VF_MODE):
            full = rr.update_mask(lanes, elem_bits=32)[0]
            y_cursor = rr.vstore_unalign_begin(y)
            for half in range(tile_rows // 32):
                if const_expr(lanes == 16 and not self.bf16):
                    full64 = rr.full_mask()
                    lane64 = rr.varange(0, dtypes.uint32)
                    maxima64 = _u32(0, full64)
                    for inner in range(8):
                        offset = (half * 32 + inner * 4) * width + d_offset
                        values64 = rr.vload(x, offset)
                        absolute64 = _and(rr.vreinterpret(values64, dtypes.uint32), 0x7FFFFFFF, full64)
                        maxima64 = rr.vmax(maxima64, absolute64, mask=full64)
                    swap16 = rr.vbitwise_xor(lane64, _u32(16, full64), mask=full64)
                    maxima64 = rr.vmax(maxima64, rr.vgather_reg(maxima64, swap16), mask=full64)
                    swap32 = rr.vbitwise_xor(lane64, _u32(32, full64), mask=full64)
                    maxima = rr.vmax(maxima64, rr.vgather_reg(maxima64, swap32), mask=full64)
                else:
                    maxima = _u32(0, full)
                    for inner in range(32):
                        offset = (half * 32 + inner) * width + d_offset
                        if const_expr(self.bf16):
                            loaded = rr.vload_unpack(x, offset, unpack_mode=rr.UnpackMode.B16_TO_B32)
                            values = rr.vcast(loaded, dtypes.float32, mask=full)
                        else:
                            values = rr.vload(x, offset)
                        absolute = _and(rr.vreinterpret(values, dtypes.uint32), 0x7FFFFFFF, full)
                        maxima = rr.vmax(maxima, absolute, mask=full)
                scale_codes, reciprocal = self._compute_scale(maxima, full)
                rr.vstore(self.scale32, half * lanes, scale_codes, full)
                for word in range(8):
                    slot = half * 8 * lanes + word * lanes
                    original = rr.vreinterpret(rr.vload(self.random_words, slot), dtypes.uint32)
                    reversed_word = rr.vreinterpret(rr.vload(self.reversed_words, slot), dtypes.uint32)
                    first = half * 32 + word * 4
                    self._wide8_store_row_unalign(x, y, first * width + d_offset, reciprocal,
                                         _and(reversed_word, 0xFFFF, full), full, y_cursor)
                    self._wide8_store_row_unalign(x, y, (first + 1) * width + d_offset, reciprocal,
                                         rr.vshr(original, 16, mask=full), full, y_cursor)
                    self._wide8_store_row_unalign(x, y, (first + 2) * width + d_offset, reciprocal,
                                         rr.vshr(reversed_word, 16, mask=full), full, y_cursor)
                    self._wide8_store_row_unalign(x, y, (first + 3) * width + d_offset, reciprocal,
                                         _and(original, 0xFFFF, full), full, y_cursor)
            rr.vsqueeze_and_storeunalign_finalize(y, 0, y_cursor)
            rr.vmem_bar("vst_vld")
            pair_mask = rr.update_mask(lanes * 2, elem_bits=8)[0]
            scale_cursor = rr.vstore_unalign_begin(scales)
            for pair in range(tile_rows // 64):
                first_scale = rr.vload(self.scale32, pair * 2 * lanes)
                second_scale = rr.vload(self.scale32, pair * 2 * lanes + lanes)
                code_pairs = rr.vbitwise_or(first_scale,
                                           rr.vshl(second_scale, 8, mask=full), mask=full)
                packed = rr.vpack(code_pairs, dtypes.uint16, part="lower")
                packed_bytes = rr.vreinterpret_lanes(packed, dtypes.uint8)
                squeezed = rr.vsqueeze_and_storeunalign_init(packed_bytes, mask=pair_mask)
                rr.vsqueeze_and_storeunalign(scales, 0, squeezed, scale_cursor)
            rr.vsqueeze_and_storeunalign_finalize(scales, 0, scale_cursor)
            rr.vmem_bar("vst_vld")

    @jit
    def _wide_store_row(self, x, y, offset, reciprocal, random16, mask):
        if const_expr(self.bf16):
            loaded = rr.vload_unpack(x, offset, unpack_mode=rr.UnpackMode.B16_TO_B32)
            value = rr.vcast(loaded, dtypes.float32, mask=mask)
        else:
            value = rr.vload(x, offset)
        normalized = rr.vmul(value, reciprocal, mask=mask)
        if const_expr(self.bf16):
            rounded = rr.vcast(normalized, dtypes.bfloat16, mask=mask, rounding=rr.RoundingMode.RN)
            normalized = rr.vcast(rounded, dtypes.float32, mask=mask)
        codes = _sr_fp32_to_fp8(normalized, random16, self.mantissa, self.bias, self.max_code, mask)
        rr.vstore_pack(y, offset, codes, mask, pack_mode=rr.PackMode.B32_TO_B8)

    def _wide8_store_row_unalign(self, x, y, offset, reciprocal, random16, mask, y_cursor):
        if const_expr(self.bf16):
            loaded = rr.vload_unpack(x, offset, unpack_mode=rr.UnpackMode.B16_TO_B32)
            value = rr.vcast(loaded, dtypes.float32, mask=mask)
        else:
            value = rr.vload(x, offset)
        normalized = rr.vmul(value, reciprocal, mask=mask)
        if const_expr(self.bf16):
            rounded = rr.vcast(normalized, dtypes.bfloat16, mask=mask, rounding=rr.RoundingMode.RN)
            normalized = rr.vcast(rounded, dtypes.float32, mask=mask)
        codes = _sr_fp32_to_fp8(normalized, random16, self.mantissa, self.bias, self.max_code, mask)
        packed16 = rr.vpack(codes, dtypes.uint16, part="lower")
        packed8 = rr.vpack(packed16, dtypes.uint8, part="lower")
        byte_mask = rr.update_mask(self.wide_lanes, elem_bits=8)[0]
        squeezed = rr.vsqueeze_and_storeunalign_init(packed8, mask=byte_mask)
        rr.vsqueeze_and_storeunalign(y, 0, squeezed, y_cursor)

    def run_wide_non_tail(self, x: Tensor, y: Tensor, scales: Tensor,
                          quant_length: int, width: int):
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
            mem_copy(self.input.produce(), tile_slice(x, (1, tile), (row, group), alignment=(1, 64)))
            x_tile = self.input.consume()
            y_tile = self.output.produce()
            scale_tile = self.scale.produce()
            for d_chunk in range(width // 64):
                self._compute_wide64(x_tile, y_tile, scale_tile,
                                     row, group, quant_length, width, d_chunk * 64, tile_rows)
            mem_copy(tile_slice(y, (1, tile), (row, group), alignment=(1, 64)),
                     self.output.consume())
            mem_copy(tile_slice(scales, (1, tile_rows * width // 32), (row, group), alignment=(1, 128)),
                     self.scale.consume())

    def run_wide32_non_tail(self, x: Tensor, y: Tensor, scales: Tensor,
                          quant_length: int, width: int):
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
            mem_copy(self.input.produce(), tile_slice(x, (1, tile), (row, group), alignment=(1, 64)))
            x_tile = self.input.consume()
            y_tile = self.output.produce()
            scale_tile = self.scale.produce()
            self._compute_wide32(x_tile, y_tile, scale_tile,
                                 row, group, quant_length, width, 0, tile_rows)
            mem_copy(tile_slice(y, (1, tile), (row, group), alignment=(1, 64)),
                     self.output.consume())
            mem_copy(tile_slice(scales, (1, tile_rows * width // 32), (row, group), alignment=(1, 64)),
                     self.scale.consume())

    def run_wide8_non_tail(self, x: Tensor, y: Tensor, scales: Tensor,
                          quant_length: int, width: int):
        # 256 rows provide a 64-byte scale DMA tile for width 8.
        tile_rows = 256
        groups = quant_length // tile_rows
        tasks = x.shape[0] * groups
        for task in range(get_block_idx(), tasks, get_block_num()):
            row = task // groups
            group = task % groups
            tile = tile_rows * width
            mem_copy(self.input.produce(), tile_slice(x, (1, tile), (row, group), alignment=(1, 64)))
            x_tile = self.input.consume()
            y_tile = self.output.produce()
            scale_tile = self.scale.produce()
            self._compute_wide8(x_tile, y_tile, scale_tile,
                                 row, group, quant_length, width, 0, tile_rows)
            mem_copy(tile_slice(y, (1, tile), (row, group), alignment=(1, 64)),
                     self.output.consume())
            mem_copy(tile_slice(scales, (1, tile_rows * width // 32), (row, group), alignment=(1, 64)),
                     self.scale.consume())

    @jit
    def _run_short_rows(self, x, y, scales):
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
            mem_copy(self.input.produce(), tile_slice(flat_x, (1, batch_rows * columns), (0, task), alignment=(1, 16)))
            input_tile = self.input.consume()
            output_tile = self.output.produce()
            scale_tile = self.scale.produce()
            self._compute_short_rows(input_tile, output_tile, scale_tile,
                                     task * batch_rows * columns, count, columns, pitch_shift)
            mem_copy(tile_slice(flat_y, (1, batch_rows * columns), (0, task), alignment=(1, 16)), self.output.consume())
            mem_copy(tile_slice(flat_scales, (1, batch_rows * scale_columns), (0, task), alignment=(1, 2)), self.scale.consume())

    @jit
    def _run_original(self, x, y, scales):
        columns = x.shape[1]
        pairs = (columns + 63) // 64
        # The public carrier planner only emits these two scale layouts.
        # K32 coalescing has two bytes per32 inputs; normal layout per64.
        scale_factor = 2
        if scales.shape[1] == columns // 16:
            scale_factor = 4
        # Runtime geometry keeps small shapes distributed across the requested
        # cores without specializing the binary or changing the logical RNG.
        # 按每核工作量均分到若干 tile，避免 UB 最大 tile 让少数核多做一整块。
        pairs_per_core = (x.shape[0] * pairs + get_block_num() - 1) // get_block_num()
        tiles_per_core = (pairs_per_core + self.max_pairs - 1) // self.max_pairs
        pair_batch = (pairs_per_core + tiles_per_core - 1) // tiles_per_core
        # Keep full tiles on the 256-element fast path; max_pairs is a
        # multiple of 16, so rounding up to four pairs stays within UB.
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
            mem_copy(self.input.produce(), tile_slice(x, (1, tile_columns), (row, tile), alignment=(1, 64)))
            input_tile = self.input.consume()
            output_tile = self.output.produce()
            scale_tile = self.scale.produce()
            if count >= 256 and count % 256 == 0:
                self._compute_256(input_tile, output_tile, scale_tile, row * columns + tile * tile_columns, count, scale_factor)
            else:
                self._compute(input_tile, output_tile, scale_tile, row * columns + tile * tile_columns, count, scale_factor)
            mem_copy(tile_slice(y, (1, tile_columns), (row, tile), alignment=(1, 64)),
                     self.output.consume())
            mem_copy(tile_slice(scales, (1, pair_batch * scale_factor), (row, tile), alignment=(1, 2)),
                     self.scale.consume())

    @jit
    def _run_coalesced(self, x, y, scales):
        # Give each core a contiguous, balanced range. Fill UB tiles within
        # that range, instead of shrinking every tile to the average tail.
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
            core_x = tile_slice(x, (1, core_columns), (0, get_block_idx()), alignment=(1, 64))
            core_y = tile_slice(y, (1, core_columns), (0, get_block_idx()), alignment=(1, 64))
            core_s = tile_slice(scales, (1, core_pairs * scale_factor), (0, get_block_idx()), alignment=(1, 2))
            core_count = columns - core_start
            if core_count > core_columns:
                core_count = core_columns
            tiles = (core_count + self.tile_elements - 1) // self.tile_elements
            for tile in range(tiles):
                count = core_count - tile * self.tile_elements
                if count > self.tile_elements:
                    count = self.tile_elements
                mem_copy(self.input.produce(), tile_slice(core_x, (1, self.tile_elements), (0, tile), alignment=(1, 64)))
                input_tile = self.input.consume()
                output_tile = self.output.produce()
                scale_tile = self.scale.produce()
                start = core_start + tile * self.tile_elements
                if count >= 256 and count % 256 == 0:
                    self._compute_256(input_tile, output_tile, scale_tile, start, count, scale_factor)
                else:
                    self._compute(input_tile, output_tile, scale_tile, start, count, scale_factor)
                mem_copy(tile_slice(core_y, (1, self.tile_elements), (0, tile), alignment=(1, 64)), self.output.consume())
                mem_copy(tile_slice(core_s, (1, self.max_pairs * scale_factor), (0, tile), alignment=(1, 2)), self.scale.consume())

    def run_non_tail(self, x0: Tensor, x1: Tensor, y: Tensor, scales: Tensor):
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
                mem_copy(self.input.produce(), tile_slice(x0, (1, tile_columns), (row, tile), alignment=(1, 64)))
            else:
                mem_copy(self.input.produce(), tile_slice(x1, (1, tile_columns), (row, tile), alignment=(1, 64)))
            input_tile = self.input.consume()
            output_tile = self.output.produce()
            scale_tile = self.scale.produce()
            start = logical_row * columns + tile * tile_columns
            if count >= 256 and count % 256 == 0:
                self._compute_256(input_tile, output_tile, scale_tile, start, count, 2)
            else:
                self._compute(input_tile, output_tile, scale_tile, start, count, 2)
            mem_copy(tile_slice(y, (1, tile_columns), (logical_row, tile), alignment=(1, 64)),
                     self.output.consume())
            mem_copy(tile_slice(scales, (1, pair_batch * 2), (logical_row, tile), alignment=(1, 2)),
                     self.scale.consume())

    def run_non_tail_fused(self, x0: Tensor, x1: Tensor, y: Tensor, scales: Tensor):
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
            mem_copy(self.input.produce(), tile_slice(x0, (1, tile_columns), (row, tile), alignment=(1, 64)))
            x_tile = self.input.consume()
            y_tile = self.output.produce()
            s_tile = self.scale.produce()
            self._compute_256(x_tile, y_tile, s_tile, row * 2 * columns + tile * tile_columns, count, 2)
            mem_copy(self.saved_output, self.output.consume())
            mem_copy(self.saved_scale, self.scale.consume())
            mem_copy(self.input.produce(), tile_slice(x1, (1, tile_columns), (row, tile), alignment=(1, 64)))
            x_tile = self.input.consume()
            y_tile = self.output.produce()
            s_tile = self.scale.produce()
            self._compute_256(x_tile, y_tile, s_tile, (row * 2 + 1) * columns + tile * tile_columns, count, 2)
            second_y = self.output.consume()
            second_s = self.scale.consume()
            first_pairs = reinterpret(self.saved_scale, dtypes.uint16, shape=(1, self.tile_elements // 32))
            second_pairs = reinterpret(second_s, dtypes.uint16, shape=(1, self.tile_elements // 32))
            with vf(mode=_SIMD_VF_MODE):
                for offset in range(0, count, 256):
                    a = rr.vload(self.saved_output, offset)
                    b = rr.vload(second_y, offset)
                    rr.vstore_interleave(self.interleaved_output, offset * 2, a, b)
                for offset in range(0, count // 64, 128):
                    a = rr.vload(first_pairs, offset)
                    b = rr.vload(second_pairs, offset)
                    rr.vstore_interleave(self.interleaved_scale, offset * 2, a, b)
            mem_copy(tile_slice(y, (1, tile_columns * 2), (row, tile), alignment=(1, 512)),
                     self.interleaved_output)
            scale_bytes = reinterpret(self.interleaved_scale, dtypes.uint8,
                                      shape=(1, self.tile_elements // 8))
            mem_copy(tile_slice(scales, (1, pair_batch * 4), (row, tile), alignment=(1, 16)),
                     scale_bytes)

    def __call__(self, x: Tensor, y: Tensor, scales: Tensor):
        if (x.shape[0] == 1 and x.shape[1] % 64 == 0
                and x.shape[1] > get_block_num() * self.tile_elements
                and x.shape[1] % (get_block_num() * self.tile_elements) != 0):
            self._run_coalesced(x, y, scales)
        elif x.shape[0] > 1 and x.shape[1] < 128:
            self._run_short_rows(x, y, scales)
        else:
            self._run_original(x, y, scales)


class _Launch:
    def __init__(self, input_dtype, e5m2, tiling, scale_alg=1, max_low_bound_bits=0, wide_lanes=8):
        self.wide_lanes = wide_lanes
        self.scale_alg = scale_alg
        self.max_low_bound_bits = max_low_bound_bits
        self.input_dtype, self.e5m2, self.tiling = input_dtype, e5m2, tiling

    @host
    def run(self, x, y, scales, blocks: int):
        op = _BitReverseKernel(self.input_dtype, self.e5m2, self.tiling.tile_elements, self.tiling.scale_capacity, self.scale_alg, self.max_low_bound_bits)
        op[blocks](x, y, scales)


    @host
    def run_non_tail(self, x0, x1, y, scales, blocks: int):
        op = _BitReverseKernel(self.input_dtype, self.e5m2, self.tiling.tile_elements, self.tiling.scale_capacity, self.scale_alg, self.max_low_bound_bits)
        op[blocks].run_non_tail(x0, x1, y, scales)


    @host
    def run_wide_non_tail(self, x, y, scales, quant_length: int, width: int, blocks: int):
        op = _BitReverseKernel(self.input_dtype, self.e5m2, self.tiling.tile_elements,
                               self.tiling.scale_capacity, self.scale_alg,
                               self.max_low_bound_bits)
        op[blocks].run_wide_non_tail(x, y, scales, quant_length, width)

    @host
    def run_wide32_non_tail(self, x, y, scales, quant_length: int, width: int, blocks: int):
        op = _BitReverseKernel(self.input_dtype, self.e5m2, self.tiling.tile_elements,
                               self.tiling.scale_capacity, self.scale_alg,
                               self.max_low_bound_bits)
        op[blocks].run_wide32_non_tail(x, y, scales, quant_length, width)


    @host
    def run_wide8_non_tail(self, x, y, scales, quant_length: int, width: int, blocks: int):
        op = _BitReverseKernel(self.input_dtype, self.e5m2, self.tiling.tile_elements,
                               self.tiling.scale_capacity, self.scale_alg,
                               self.max_low_bound_bits, wide_lanes=self.wide_lanes)
        op[blocks].run_wide8_non_tail(x, y, scales, quant_length, width)


    @host
    def run_non_tail_fused(self, x0, x1, y, scales, blocks: int):
        op = _BitReverseKernel(self.input_dtype, self.e5m2, self.tiling.tile_elements,
                               self.tiling.scale_capacity, self.scale_alg,
                               self.max_low_bound_bits, True)
        op[blocks].run_non_tail_fused(x0, x1, y, scales)


def _compile_dsl_kernel(input_dtype, e5m2, ub_bytes, scale_alg=1, max_low_bound_bits=0):
    tiling = _make_tiling(input_dtype, ub_bytes)
    return cannbotdsl.compile(_Launch(input_dtype, e5m2, tiling, scale_alg, max_low_bound_bits).run,
        TensorSpec((Dim("rows", min=1), Dim("columns", min=16 if input_dtype == dtypes.bfloat16 else 32, multiple_of=16)), input_dtype),
        TensorSpec((Dim("rows", min=1), Dim("columns", min=16 if input_dtype == dtypes.bfloat16 else 32, multiple_of=16)), dtypes.uint8),
        TensorSpec((Dim("rows", min=1), Dim("scale_columns", min=2, multiple_of=2)), dtypes.uint8),
        dtypes.int64,
    )


def _compile_dsl_non_tail_kernel(input_dtype, e5m2, ub_bytes, scale_alg=1, max_low_bound_bits=0):
    tiling = _make_tiling(input_dtype, ub_bytes)
    rows = Dim("rows", min=1)
    columns = Dim("columns", min=64, multiple_of=64)
    row_stride = Dim("source_row_stride", min=128, multiple_of=128)
    logical_rows = Dim("logical_rows", min=2, multiple_of=2)
    scale_columns = Dim("scale_columns", min=2, multiple_of=2)
    source = TensorSpec((rows, columns), input_dtype, stride=(row_stride, 2))
    return cannbotdsl.compile(_Launch(input_dtype, e5m2, tiling, scale_alg, max_low_bound_bits).run_non_tail,
        source, source,
        TensorSpec((logical_rows, columns), dtypes.uint8),
        TensorSpec((logical_rows, scale_columns), dtypes.uint8),
        dtypes.int64,
    )


def _compile_dsl_non_tail_fused_kernel(input_dtype, e5m2, ub_bytes, scale_alg=1, max_low_bound_bits=0):
    tiling = _make_tiling(input_dtype, ub_bytes, True)
    rows = Dim("rows", min=1)
    columns = Dim("columns", min=256, multiple_of=256)
    row_stride = Dim("source_row_stride", min=512, multiple_of=512)
    source = TensorSpec((rows, columns), input_dtype, stride=(row_stride, 2))
    return cannbotdsl.compile(_Launch(input_dtype, e5m2, tiling, scale_alg, max_low_bound_bits).run_non_tail_fused,
        source, source,
        TensorSpec((rows, Dim("packed_columns", min=512, multiple_of=512)), dtypes.uint8),
        TensorSpec((rows, Dim("packed_scale_columns", min=16, multiple_of=16)), dtypes.uint8),
        dtypes.int64,
    )


def _compile_dsl_wide_non_tail_kernel(input_dtype, e5m2, ub_bytes, scale_alg=1, max_low_bound_bits=0):
    tiling = _make_tiling(input_dtype, ub_bytes)
    rows = Dim("rows", min=1)
    physical_columns = Dim("physical_columns", min=4096, multiple_of=4096)
    scale_columns = Dim("physical_scale_columns", min=128, multiple_of=128)
    return cannbotdsl.compile(_Launch(input_dtype, e5m2, tiling, scale_alg, max_low_bound_bits).run_wide_non_tail,
        TensorSpec((rows, physical_columns), input_dtype),
        TensorSpec((rows, physical_columns), dtypes.uint8),
        TensorSpec((rows, scale_columns), dtypes.uint8),
        dtypes.int64, dtypes.int64, dtypes.int64,
    )


def _compile_dsl_wide32_non_tail_kernel(input_dtype, e5m2, ub_bytes, scale_alg=1, max_low_bound_bits=0):
    tiling = _make_tiling(input_dtype, ub_bytes)
    rows = Dim("rows", min=1)
    physical_columns = Dim("physical_columns", min=2048, multiple_of=2048)
    scale_columns = Dim("physical_scale_columns", min=64, multiple_of=64)
    return cannbotdsl.compile(_Launch(input_dtype, e5m2, tiling, scale_alg, max_low_bound_bits).run_wide32_non_tail,
        TensorSpec((rows, physical_columns), input_dtype),
        TensorSpec((rows, physical_columns), dtypes.uint8),
        TensorSpec((rows, scale_columns), dtypes.uint8),
        dtypes.int64, dtypes.int64, dtypes.int64,
    )


def _compile_dsl_wide8_non_tail_kernel(input_dtype, e5m2, ub_bytes, scale_alg=1, max_low_bound_bits=0):
    tiling = _make_tiling(input_dtype, ub_bytes)
    rows = Dim("rows", min=1)
    physical_columns = Dim("physical_columns", min=2048, multiple_of=2048)
    scale_columns = Dim("physical_scale_columns", min=64, multiple_of=64)
    return cannbotdsl.compile(_Launch(input_dtype, e5m2, tiling, scale_alg, max_low_bound_bits).run_wide8_non_tail,
        TensorSpec((rows, physical_columns), input_dtype),
        TensorSpec((rows, physical_columns), dtypes.uint8),
        TensorSpec((rows, scale_columns), dtypes.uint8),
        dtypes.int64, dtypes.int64, dtypes.int64,
    )


def _compile_dsl_wide16_non_tail_kernel(input_dtype, e5m2, ub_bytes, scale_alg=1, max_low_bound_bits=0):
    tiling = _make_tiling(input_dtype, ub_bytes)
    rows = Dim("rows", min=1)
    physical_columns = Dim("physical_columns", min=4096, multiple_of=4096)
    scale_columns = Dim("physical_scale_columns", min=128, multiple_of=128)
    return cannbotdsl.compile(_Launch(input_dtype, e5m2, tiling, scale_alg, max_low_bound_bits, wide_lanes=16).run_wide8_non_tail,
        TensorSpec((rows, physical_columns), input_dtype),
        TensorSpec((rows, physical_columns), dtypes.uint8),
        TensorSpec((rows, scale_columns), dtypes.uint8),
        dtypes.int64, dtypes.int64, dtypes.int64,
    )


@lru_cache(maxsize=1)
def _load_static_compiler():
    """Load the sibling compiler once for package and standalone imports."""
    path = Path(__file__).with_name("bitreverse_static_compile.py")
    spec = importlib.util.spec_from_file_location("_mx_quant_sr_static_compile", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load mixed SR compiler from {path}")
    compiler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler)
    return compiler


@lru_cache(maxsize=64)
def _get_compiled_kernel(
    input_dtype, e5m2, ub_bytes, scale_alg=1, max_low_bound_bits=0, *, layout="tail"
):
    """Cache layout/dtype/attribute specializations; shapes stay dynamic."""
    name = "bf16" if input_dtype == dtypes.bfloat16 else "fp32"
    return _load_static_compiler().build_static_program(
        Path(__file__).resolve(), name, e5m2, ub_bytes, scale_alg,
        max_low_bound_bits, layout=layout,
    )


def _resolve_block_dim(x, block_dim):
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
    budget = min(count, 64)
    columns = x.shape[-1]
    if columns % 64 == 0 or columns == 32:
        budget = min(budget, (x.numel() + 1023) // 1024)
    return budget


def _kernel_views(x, encoded, scales):
    """Coalesce aligned rows without copies or changing logical RNG indices.

    K divisible by64 has no per-row padding. K32 can also coalesce, but keeps
    its interleaved scale padding; the dynamic kernel handles that layout.
    Other K values retain their row boundaries.
    """
    columns = x.shape[-1]
    rows = math.prod(x.shape[:-1])
    if columns % 64 == 0 or columns == 32:
        return x.view(1, -1), encoded.view(1, -1), scales.view(1, -1)
    return (x.view(rows, columns), encoded.view(rows, columns),
            scales.view(rows, 2 * ((columns + 63) // 64)))


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


def _program_for(x, dst_dtype, scale_alg, bound_bits, layout="tail"):
    """Resolve shared compile arguments inside the input device context."""
    dtype = dtypes.bfloat16 if x.dtype == torch.bfloat16 else dtypes.float32
    return _get_compiled_kernel(
        dtype, dst_dtype == torch.float8_e5m2, get_mem_size("ub"),
        scale_alg, bound_bits, layout=layout,
    )


def _launch_tail(x, dst_dtype, scale_alg, bound_bits, block_dim=None):
    rows = math.prod(x.shape[:-1])
    pairs = (x.shape[-1] + 63) // 64
    encoded = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    scales = torch.empty((*x.shape[:-1], pairs, 2), dtype=torch.uint8, device=x.device)
    with torch.npu.device(x.device):
        blocks = min(_resolve_block_dim(x, block_dim), rows * pairs)
        program = _program_for(x, dst_dtype, scale_alg, bound_bits)
        program(*_kernel_views(x, encoded, scales), blocks)
    return encoded.view(dst_dtype), scales.view(torch.float8_e8m0fnu)


def _dynamic_mx_quant_sr_impl(
    x, dst_dtype=torch.float8_e4m3fn, *, block_dim=None, scale_alg=1, max_low_bound=0.0
):
    """Tail-axis entry with an optional AIV budget for tuning/whitebox tests.

    The fixed Philox key (0,0) replays the same random stream at any block count.
    Explicit budgets are integers in [1,64]; None uses the current stream quota.
    Shapes and block counts reuse a binary for each dtype pair, scale algorithm,
    max_low_bound and UB capacity. Non-tail layouts have separate specializations.
    """
    _validate_input(x)
    if dst_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise TypeError("dst_dtype must be float8_e4m3fn or float8_e5m2")
    if block_dim is not None and (type(block_dim) is not int or not 1 <= block_dim <= 64):
        raise ValueError("block_dim must be None or an integer in [1,64]")
    return _launch_tail(x, dst_dtype, scale_alg, _encode_bound(max_low_bound), block_dim)


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


def _select_layout(x, axis):
    """Select a kernel using geometry and dtype only; preserve tuned thresholds."""
    if axis == x.ndim - 1:
        return "tail"
    if x.ndim == 3 and axis == 1 and x.shape[2] == 2 and x.shape[1] >= 512:
        if x.shape[1] % 256 == 0:
            return "non_tail_fused"
        if x.shape[1] % 64 == 0:
            return "non_tail"
    if axis != x.ndim - 2:
        return "transpose"
    quant_length, width = x.shape[-2:]
    if width == 8 and x.dtype == torch.float32 and quant_length >= 256 and quant_length % 256 == 0:
        return "wide8_non_tail"
    if width == 16 and quant_length >= 256 and quant_length % 256 == 0:
        # BF16 direct stores win for small inputs, long-axis 1M inputs, and
        # the 2M/4096-axis case. Other shapes use the general transpose path.
        elements = x.numel()
        if elements <= 2 * 1024 * 1024 and (
            x.dtype == torch.float32 or elements <= 256 * 1024
            or (quant_length >= 4096 and elements <= 1024 * 1024)
            or quant_length == 4096
        ):
            return "wide16_non_tail"
    if width in (32, 64, 128) and quant_length >= 64 and quant_length % 64 == 0:
        return "wide32_non_tail" if width == 32 else "wide_non_tail"
    return "transpose"


def _launch_stride2(x, dst_dtype, scale_alg, bound_bits, *, fused):
    """Quantize the two source channels; fused kernels also restore GM layout."""
    rows, columns, _ = x.shape
    pairs = columns // 64
    output_shape = (rows, columns, 2) if fused else (rows, 2, columns)
    scale_shape = (rows, pairs, 2, 2) if fused else (rows, 2, pairs, 2)
    encoded = torch.empty(output_shape, dtype=torch.uint8, device=x.device)
    scales = torch.empty(scale_shape, dtype=torch.uint8, device=x.device)
    logical_rows = rows if fused else rows * 2
    logical_columns = columns * 2 if fused else columns
    scale_columns = pairs * (4 if fused else 2)
    layout = "non_tail_fused" if fused else "non_tail"
    with torch.npu.device(x.device):
        blocks = min(_resolve_block_dim(x, None), (x.numel() + 1023) // 1024,
                     logical_rows * pairs)
        program = _program_for(x, dst_dtype, scale_alg, bound_bits, layout)
        program(x[:, :, 0], x[:, :, 1], encoded.view(logical_rows, logical_columns),
                scales.view(logical_rows, scale_columns), blocks)
    y = encoded.view(dst_dtype)
    if not fused:
        y = y.movedim(1, 2).contiguous()
        scales = scales.movedim(1, 2).contiguous()
    return y, scales.view(torch.float8_e8m0fnu)


def _launch_wide_axis(x, dst_dtype, scale_alg, bound_bits, layout):
    """Launch contiguous D-lane kernels with a common output/scale layout."""
    quant_length, width = x.shape[-2:]
    rows = x.numel() // (quant_length * width)
    encoded = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    scale_shape = (*x.shape[:-2], quant_length // 64, width, 2)
    scales = torch.empty(scale_shape, dtype=torch.uint8, device=x.device)
    rows_per_task = 256 if layout in ("wide8_non_tail", "wide16_non_tail") else 64
    with torch.npu.device(x.device):
        blocks = min(_resolve_block_dim(x, None), rows * (quant_length // rows_per_task))
        program = _program_for(x, dst_dtype, scale_alg, bound_bits, layout)
        program(x.view(rows, quant_length * width), encoded.view(rows, quant_length * width),
                scales.view(rows, quant_length // 64 * width * 2), quant_length, width, blocks)
    return encoded.view(dst_dtype), scales.view(torch.float8_e8m0fnu)


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
    """Stochastic MXFP8 with the argument names/order of npu_dynamic_mx_quant.

    Unlike the original Torch API's rint/FP4/scale_alg=0 defaults, this SR
    entry defaults to stochastic/E4M3FN/scale_alg=1. ``dst_type`` uses Torch
    integer codes: 23=E5M2, 24=E4M3FN (not CANN's 35/36 codes).

    FP32/BF16, any axis, block_size=32, stochastic rounding,
    scale_alg=0/1 and dst_type_max=0 are supported. Unsupported attribute
    values are rejected, never ignored. max_low_bound is accepted for
    scale_alg=1; it extends the original Torch API.

    Returns (y, mx_scale) on the input device. y has the input shape and the
    selected FP8 dtype; mx_scale is float8_e8m0fnu with shape
    input shape with the quantization axis replaced by ceil(K/64), plus a
    trailing dimension of 2. Input must be contiguous/nonempty, with the
    quantization axis a multiple of 16 (at least 32 for FP32). The fixed Philox key
    (0,0) replays the same random stream on each call. Scheduling uses the
    current stream's effective AIV budget automatically.
    """
    axis, dst_dtype, bound_bits = _validate_attributes(
        input, axis, round_mode, dst_type, block_size, scale_alg, dst_type_max, max_low_bound
    )
    _validate_input(input, axis)
    layout = _select_layout(input, axis)
    if layout == "tail":
        return _launch_tail(input, dst_dtype, scale_alg, bound_bits)
    if layout in ("non_tail", "non_tail_fused"):
        return _launch_stride2(
            input, dst_dtype, scale_alg, bound_bits, fused=layout == "non_tail_fused"
        )
    if layout != "transpose":
        return _launch_wide_axis(input, dst_dtype, scale_alg, bound_bits, layout)
    # The generic extension follows the RNG order of a contiguous moved-axis view.
    moved = input.movedim(axis, -1).contiguous()
    y, scale = _launch_tail(moved, dst_dtype, scale_alg, bound_bits)
    scale_bytes = scale.view(torch.uint8).movedim(-2, axis).contiguous()
    return y.movedim(-1, axis).contiguous(), scale_bytes.view(torch.float8_e8m0fnu)


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
