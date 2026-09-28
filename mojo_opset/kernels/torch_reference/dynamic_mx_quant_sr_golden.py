# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU byte reference for the DynamicMxQuantV3 stochastic DSL sample.

Verified against ops-nn commit 22c0528222199af3a276237c1291080329cd86a8,
quant/dynamic_mx_quant/tests/assets/golden.py. This reference deliberately
implements FP32/BF16 -> E4M3FN/E5M2, any axis, blocksize=32,
round_mode="stochastic", scale_alg=0/1 and scale_alg=1 max_low_bound. It has no DSL or
device dependencies. Both outputs contain raw uint8 encoding bytes.
"""

import numpy as np
from ml_dtypes import bfloat16


_FP8_SR_PARAMS = {
    "float8_e4m3fn": (7, 3, 0x7E),
    "float8_e5m2": (15, 2, 0x7B),
}


def _philox4x32_10(c0, c1, c2, c3, k0=0, k1=0):
    """Philox4x32-10 words, with uint32 wrap after every key update."""
    c0, c1, c2, c3, k0, k1 = (
        np.asarray(value, dtype=np.uint32) for value in (c0, c1, c2, c3, k0, k1)
    )
    for _ in range(10):
        p0 = c0.astype(np.uint64) * np.uint64(0xD2511F53)
        p1 = c2.astype(np.uint64) * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = (
            (p1 >> np.uint64(32)).astype(np.uint32) ^ c1 ^ k0,
            p1.astype(np.uint32),
            (p0 >> np.uint64(32)).astype(np.uint32) ^ c3 ^ k1,
            p0.astype(np.uint32),
        )
        k0 = (k0.astype(np.uint64) + np.uint64(0x9E3779B9)).astype(np.uint32)
        k1 = (k1.astype(np.uint64) + np.uint64(0xBB67AE85)).astype(np.uint32)
    return c0, c1, c2, c3


def _philox_word_stream(num_words):
    """stream[p] = Philox10(counter=p//4, key=0)[p%4]."""
    counters = np.arange((num_words + 3) // 4, dtype=np.uint64)
    words = _philox4x32_10(
        counters.astype(np.uint32), (counters >> np.uint64(32)).astype(np.uint32), 0, 0
    )
    return np.stack(words, axis=-1).reshape(-1)[:num_words]


def _reverse16(value):
    value = np.asarray(value, dtype=np.uint32) & np.uint32(0xFFFF)
    for shift, mask in ((1, 0x5555), (2, 0x3333), (4, 0x0F0F), (8, 0x00FF)):
        value = ((value & mask) << shift) | ((value >> shift) & mask)
    return value


def _random_bits(numel):
    """The four element lanes share a word, including across row boundaries."""
    words = _philox_word_stream((numel + 3) // 4)
    high, low = words >> np.uint32(16), words & np.uint32(0xFFFF)
    return np.stack((_reverse16(high), high, _reverse16(low), low), axis=-1).reshape(-1)[:numel]


def _sr_scalar(bits, r16, dst_dtype):
    """Integer scalar Algorithm 19; no host FP8 cast or floating rounding."""
    bias, mantissa_bits, max_code = _FP8_SR_PARAMS[dst_dtype]
    sign = (bits >> 24) & 0x80
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    if exponent == 0xFF and fraction:
        return 0x7F
    source_exp = exponent - 127 if exponent else -126
    significand = fraction | 0x800000 if exponent else fraction
    emin = 1 - bias
    normal = source_exp >= emin
    if not normal:
        significand >>= min(emin - source_exp, 24)
    cut = 23 - mantissa_bits
    base = significand >> cut
    discarded = (significand >> (cut - 16)) & 0xFFFF
    carry = (discarded + (r16 & 0xFFFF)) >> 16
    if normal:
        base = ((source_exp + bias) << mantissa_bits) | (base & ((1 << mantissa_bits) - 1))
    return sign | min(base + carry, max_code)


def _sr_cast_fp32_to_fp8(values, r16, dst_dtype):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    random = np.broadcast_to(np.asarray(r16, dtype=np.uint32), bits.shape)
    result = np.fromiter(
        (_sr_scalar(int(value), int(rand), dst_dtype) for value, rand in zip(bits.flat, random.flat)),
        dtype=np.uint8,
        count=bits.size,
    )
    return result.reshape(bits.shape)


def _mx_scale_alg0_kernel(abs_max_f32, dst_dtype):
    """ASC ComputeScaleOcp, including the reciprocal-zero underflow rule."""
    shift = {"float8_e4m3fn": 8, "float8_e5m2": 15}[dst_dtype]
    exponent = np.ascontiguousarray(abs_max_f32, dtype=np.float32).view(np.uint32) >> 23
    shared = np.maximum(exponent, shift) - shift
    reciprocal = ((254 - shared) << 7).astype(np.uint16)
    reciprocal = np.where(shared == 0, 0, reciprocal)
    reciprocal = np.where(shared == 254, 0x0040, reciprocal)
    reciprocal = np.where(exponent == 255, 0x7f81, reciprocal).astype(np.uint16)
    code = np.where(exponent == 255, 255, shared).astype(np.uint8)
    return code, reciprocal


def _mx_scale_alg1_kernel(abs_max_f32, dst_dtype):
    """ComputeScaleCeilAlg: return E8M0 bytes and BF16 reciprocal bits."""
    inv_bits = {"float8_e4m3fn": 0x3B124925, "float8_e5m2": 0x37924925}[dst_dtype]
    inv_max = np.uint32(inv_bits).view(np.float32)
    maxima = np.ascontiguousarray(abs_max_f32, dtype=np.float32)
    bits = maxima.view(np.uint32)
    finite, zero = bits < np.uint32(0x7F800000), bits == 0
    with np.errstate(invalid="ignore", under="ignore"):
        product = (maxima * inv_max).astype(np.float32).view(np.uint32)
    exponent, fraction = product >> 23, product & np.uint32(0x007FFFFF)
    ceil_up = ((exponent > 0) & (exponent < 254) & (fraction > 0)) | (
        (exponent == 0) & (fraction > np.uint32(0x00400000))
    )
    scale = np.where(ceil_up, exponent + 1, exponent)
    scale = np.where(finite, scale, np.uint32(0xFF))
    scale = np.where(zero, np.uint32(0), scale)
    reciprocal = (np.uint32(254) - scale) << np.uint32(7)
    reciprocal = np.where(finite, reciprocal, np.uint32(0x7F81))
    reciprocal = np.where(zero, np.uint32(0), reciprocal)
    return scale.astype(np.uint8), (reciprocal & np.uint32(0xFFFF)).astype(np.uint16)


def dynamic_mx_quant_golden(x, dst_dtype="float8_e4m3fn", scale_alg=1, max_low_bound=0.0, axis=-1):
    """Return raw FP8 and interleaved E8M0 bytes for the selected axis."""
    if not isinstance(x, np.ndarray) or x.dtype.name not in ("float32", "bfloat16"):
        raise TypeError("x must be a NumPy FP32 or ml_dtypes BF16 array")
    if dst_dtype not in _FP8_SR_PARAMS:
        raise ValueError("only float8_e4m3fn and float8_e5m2 are supported")
    if x.ndim == 0 or x.size == 0:
        raise ValueError("x must be nonempty")
    if type(axis) is not int or not -x.ndim <= axis < x.ndim:
        raise ValueError("axis is out of range")
    axis %= x.ndim
    if axis != x.ndim - 1:
        moved = np.ascontiguousarray(np.moveaxis(x, axis, -1))
        y, scale = dynamic_mx_quant_golden(moved, dst_dtype, scale_alg,
                                           max_low_bound, axis=-1)
        return np.ascontiguousarray(np.moveaxis(y, -1, axis)), \
               np.ascontiguousarray(np.moveaxis(scale, -2, axis))
    if x.shape[-1] % 16:
        raise ValueError("K must be a positive multiple of 16")
    k = x.shape[-1]
    groups = (k + 31) // 32
    rows = x.reshape(-1, k)
    blocks = np.pad(rows, ((0, 0), (0, groups * 32 - k))).reshape(-1, groups, 32)
    maxima = np.max(np.abs(blocks.astype(np.float32)), axis=-1)
    if scale_alg not in (0, 1):
        raise ValueError("scale_alg must be 0 or 1")
    if not isinstance(max_low_bound, (int, float)) or not np.isfinite(max_low_bound) or max_low_bound < 0:
        raise ValueError("max_low_bound must be a finite non-negative number")
    if scale_alg != 1 and max_low_bound != 0:
        raise ValueError("max_low_bound must be zero when scale_alg != 1")
    if max_low_bound != 0:
        # ASC remembers the zero-group mask before clamping. A zero group
        # retains scale code 0 even with a positive lower bound.
        maxima = np.where(maxima == 0, maxima, np.maximum(maxima, np.float32(max_low_bound)))
    scale_fn = _mx_scale_alg0_kernel if scale_alg == 0 else _mx_scale_alg1_kernel
    scale, reciprocal_bits = scale_fn(maxima, dst_dtype)
    reciprocal = np.repeat(reciprocal_bits.view(bfloat16), 32, axis=-1)[:, :k]
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        normalized = rows.astype(np.float32) * reciprocal.astype(np.float32)
        if x.dtype.name == "bfloat16":
            # Device BF16 multiplication rounds to BF16 before the FP32 SR core.
            normalized = normalized.astype(bfloat16).astype(np.float32)
    codes = _sr_cast_fp32_to_fp8(normalized.reshape(x.shape), _random_bits(x.size).reshape(x.shape), dst_dtype)
    # E8M0 byte 0 denotes 2^-127; it is also the required absent-block pad byte.
    scale = np.pad(scale, ((0, 0), (0, groups % 2)), constant_values=0)
    return codes, scale.reshape(*x.shape[:-1], (k + 63) // 64, 2)
