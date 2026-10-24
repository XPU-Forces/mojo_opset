"""CPU byte reference for the DynamicMxQuantV3 stochastic DSL sample.

Quantization semantics originate from ops-nn commit
22c0528222199af3a276237c1291080329cd86a8, quant/dynamic_mx_quant/tests/assets/golden.py.
RNG uses16384 fixed virtual streams and the CANN32-bit-word offset convention.
Logical group g=i//16 maps to stream g%16384 and draw g//16384; the low64
counter is offset//4+draw. FP32 selects the eight-field map for supported
tail/transpose layouts with N>=2097152; otherwise four fields. BF16 tail and eligible width16 layouts use
the adjacent-byte map. Selection is independent of UB, core count and eager/graph execution.
Generic fields use H/rotate8(W).low16/L/rotate8(W).high16. Optimized
tails use dtype-specific eight-field mappings; full fields are correlated.
This reference deliberately
implements FP32/BF16 -> E4M3FN/E5M2, any axis, blocksize=32,
round_mode="stochastic", scale_alg=0/1 and scale_alg=1 max_low_bound. It has no DSL or
device dependencies. Both outputs contain raw uint8 encoding bytes.
"""

import numpy as np

from ml_dtypes import bfloat16

_FP8_SR_PARAMS = {"float8_e4m3fn": (7, 3, 126), "float8_e5m2": (15, 2, 123)}


def _philox4x32_10(c0, c1, c2, c3, k0=0, k1=0):
    """Philox4x32-10 words, with uint32 wrap after every key update."""
    c0, c1, c2, c3, k0, k1 = (np.asarray(value, dtype=np.uint32) for value in (c0, c1, c2, c3, k0, k1))
    for _ in range(10):
        p0 = c0.astype(np.uint64) * np.uint64(3528531795)
        p1 = c2.astype(np.uint64) * np.uint64(3449720151)
        c0, c1, c2, c3 = (
            (p1 >> np.uint64(32)).astype(np.uint32) ^ c1 ^ k0,
            p1.astype(np.uint32),
            (p0 >> np.uint64(32)).astype(np.uint32) ^ c3 ^ k1,
            p0.astype(np.uint32),
        )
        k0 = (k0.astype(np.uint64) + np.uint64(2654435769)).astype(np.uint32)
        k1 = (k1.astype(np.uint64) + np.uint64(3144134277)).astype(np.uint32)
    return (c0, c1, c2, c3)


def _philox_blocks(groups, *, quant_length, seed=0, block_offset=0):
    """Independent NPU linear counter map, word-order (3,2,0,1), returning (..., 4) uint32 words."""
    uint64_max = 2**64 - 1
    for name, value in (("seed", seed), ("block_offset", block_offset), ("quant_length", quant_length)):
        if type(value) is not int:
            raise TypeError(f"{name} must be a Python int")
        if not 0 <= value <= uint64_max:
            raise ValueError(f"{name} must be uint64")
    if quant_length == 0 or quant_length % 16:
        raise ValueError("quant_length must be a positive multiple of 16")
    if block_offset % 4 or block_offset > uint64_max - 4:
        raise ValueError("block_offset must be four-aligned with a nonwrapping reservation")
    if isinstance(groups, np.ndarray) and groups.dtype.kind in "iu":
        if groups.dtype.kind == "i" and np.any(groups < 0):
            raise ValueError("groups must be uint64")
    else:
        groups = np.asarray(groups, dtype=object)
        for group in groups.flat:
            if isinstance(group, (bool, np.bool_)) or not isinstance(group, (int, np.integer)):
                raise TypeError("groups must contain integers, not bool or converted scalars")
            if not 0 <= int(group) <= uint64_max:
                raise ValueError("groups must be uint64")
    groups = np.asarray(groups, dtype=np.uint64)
    subsequence = groups % np.uint64(16384)
    draws = groups // np.uint64(16384)
    counter = np.uint64(block_offset // 4) + draws
    words = _philox4x32_10(
        (subsequence >> np.uint64(32)).astype(np.uint32),
        subsequence.astype(np.uint32),
        counter.astype(np.uint32),
        (counter >> np.uint64(32)).astype(np.uint32),
        seed & 4294967295,
        seed >> 32,
    )
    return np.stack(words, axis=-1)


def _philox_word_stream(num_words, *, quant_length, seed=0, block_offset=0):
    """Four consecutive words per mapped logical group, with runtime state."""
    groups = np.arange((num_words + 3) // 4, dtype=np.uint64)
    return _philox_blocks(groups, quant_length=quant_length, seed=seed, block_offset=block_offset).reshape(-1)[
        :num_words
    ]


def _random_bits(numel, *, quant_length, seed=0, block_offset=0):
    """ABI5 fields H, rotate8(W).low16, L, rotate8(W).high16.

    Each field is uniform16 for a uniform Philox32 word. The four top8
    fields use disjoint bytes; the full16 fields are related, not independent.
    Counter mapping and word consumption are unchanged from ABI4.
    """
    words = _philox_word_stream((numel + 3) // 4, quant_length=quant_length, seed=seed, block_offset=block_offset)
    rotate = words << np.uint32(8) | words >> np.uint32(24)
    return np.stack(
        (words >> np.uint32(16), rotate & np.uint32(65535), words & np.uint32(65535), rotate >> np.uint32(16)), axis=-1
    ).reshape(-1)[:numel]


def _uses_fp32_eight_mapping(shape, axis):
    if int(np.prod(shape)) < 2 * 1024 * 1024:
        return False
    axis %= len(shape)
    k = shape[axis]
    if k % 256:
        return False
    if axis == len(shape) - 1:
        return True
    width = int(np.prod(shape[axis + 1 :]))
    n = int(np.prod(shape))
    if width == 2 and k >= 512 and (k % 64 == 0):
        return False
    if width == 8 and k >= 256:
        return False
    if width == 16 and k >= 256 and (n <= 2 * 1024 * 1024):
        return False
    if width in (32, 64, 128) and k >= 64 and (k % 64 == 0):
        return False
    return True


def _random_bits_fp32_eight(numel, *, quant_length, seed=0, block_offset=0, dst_dtype=None):
    words = _philox_blocks(
        np.arange((numel + 31) // 32, dtype=np.uint64), quant_length=quant_length, seed=seed, block_offset=block_offset
    ).reshape(-1)
    ids = np.arange(numel, dtype=np.uint64)
    pairs = np.asarray([(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 0), (0, 3)], dtype=np.uint32)
    word = words[(ids // 8).astype(np.int64)]
    pair = pairs[(ids % 8).astype(np.int64)]
    return word >> pair[:, 0] * np.uint32(8) & np.uint32(255) | (
        word >> pair[:, 1] * np.uint32(8) & np.uint32(255)
    ) << np.uint32(8)


def _uses_nibble_mapping(shape, axis, dtype_name, dst_dtype):
    if dtype_name == "float32":
        return "fp32_eight" if _uses_fp32_eight_mapping(shape, axis) else False
    if dtype_name != "bfloat16" or dst_dtype not in ("float8_e4m3fn", "float8_e5m2"):
        return False
    axis %= len(shape)
    k = shape[axis]
    if k % 256:
        return False
    if axis == len(shape) - 1:
        return True
    width = int(np.prod(shape[axis + 1 :]))
    n = int(np.prod(shape))
    if width == 2 and k >= 512 and (k % 64 == 0):
        return False
    if width == 8 and k >= 256 and (k % 256 == 0) and (n <= 128 * 1024):
        return False
    # BF16 width16 uses adjacent-byte fields for direct and transpose layouts.
    if width in (32, 64, 128) and k >= 64 and (k % 64 == 0):
        return False
    return True


def _random_bits_nibble(numel, *, quant_length, seed=0, block_offset=0, dst_dtype="float8_e4m3fn"):
    words = _philox_blocks(
        np.arange((numel + 31) // 32, dtype=np.uint64), quant_length=quant_length, seed=seed, block_offset=block_offset
    ).reshape(-1)
    ids = np.arange(numel, dtype=np.uint64)
    f = (ids % 8).astype(np.uint32)
    mantissa = np.uint32(_FP8_SR_PARAMS[dst_dtype][1])
    normal_start = f // np.uint32(2) * np.uint32(8) + (np.uint32(1) + mantissa) * (f % np.uint32(2))
    shift = (normal_start + np.uint32(32) - np.uint32(9) - mantissa) % np.uint32(32)
    w = words[(ids // 8).astype(np.int64)]
    return (w >> shift | w << (np.uint32(32) - shift) % np.uint32(32)) & np.uint32(65535)


def _sr_scalar(bits, r16, dst_dtype):
    """Integer scalar Algorithm 19; no host FP8 cast or floating rounding."""
    bias, mantissa_bits, max_code = _FP8_SR_PARAMS[dst_dtype]
    sign = bits >> 24 & 128
    exponent = bits >> 23 & 255
    fraction = bits & 8388607
    if exponent == 255 and fraction:
        return 127
    source_exp = exponent - 127 if exponent else -126
    significand = fraction | 8388608 if exponent else fraction
    emin = 1 - bias
    normal = source_exp >= emin
    if not normal:
        significand >>= min(emin - source_exp, 24)
    cut = 23 - mantissa_bits
    base = significand >> cut
    discarded = significand >> cut - 16 & 65535
    carry = discarded + (r16 & 65535) >> 16
    if normal:
        base = source_exp + bias << mantissa_bits | base & (1 << mantissa_bits) - 1
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


def _sr_cast_fp32_to_fp8_vectorized(values, r16, dst_dtype):
    """Chunked integer Algorithm 19, checked against the scalar oracle.

    The independent scalar routine above remains available to tests. This
    follows its exponent/significand/discarded-bit algorithm, not the device's
    Q16 implementation. Temporaries are bounded to 262144 input elements.
    """
    bias, mantissa_bits, max_code = _FP8_SR_PARAMS[dst_dtype]
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    random = np.broadcast_to(np.asarray(r16, dtype=np.uint32), bits.shape)
    result = np.empty(bits.size, dtype=np.uint8)
    flat_bits, flat_random = (bits.reshape(-1), random.reshape(-1))
    for start in range(0, bits.size, 1 << 18):
        stop = min(start + (1 << 18), bits.size)
        word, field = (flat_bits[start:stop], flat_random[start:stop])
        exponent = (word >> np.uint32(23) & np.uint32(255)).astype(np.int32)
        fraction = word & np.uint32(8388607)
        source_exp = np.where(exponent != 0, exponent - 127, -126)
        significand = fraction | np.where(exponent != 0, np.uint32(8388608), np.uint32(0))
        emin = 1 - bias
        normal = source_exp >= emin
        significand >>= np.clip(emin - source_exp, 0, 24).astype(np.uint32)
        cut = 23 - mantissa_bits
        base = significand >> np.uint32(cut)
        discarded = significand >> np.uint32(cut - 16) & np.uint32(65535)
        carry = discarded + (field & np.uint32(65535)) >> np.uint32(16)
        normal_base = (source_exp + bias << mantissa_bits).astype(np.uint32)
        normal_base |= base & np.uint32((1 << mantissa_bits) - 1)
        base = np.where(normal, normal_base, base)
        packed = np.minimum(base + carry, np.uint32(max_code))
        packed |= word >> np.uint32(24) & np.uint32(128)
        result[start:stop] = np.where((exponent == 255) & (fraction != 0), np.uint32(127), packed).astype(np.uint8)
    return result.reshape(bits.shape)


def _mx_scale_alg0_kernel(abs_max_f32, dst_dtype):
    """ASC ComputeScaleOcp, including the reciprocal-zero underflow rule."""
    shift = {"float8_e4m3fn": 8, "float8_e5m2": 15}[dst_dtype]
    exponent = np.ascontiguousarray(abs_max_f32, dtype=np.float32).view(np.uint32) >> 23
    shared = np.maximum(exponent, shift) - shift
    reciprocal = (254 - shared << 7).astype(np.uint16)
    reciprocal = np.where(shared == 0, 0, reciprocal)
    reciprocal = np.where(shared == 254, 64, reciprocal)
    reciprocal = np.where(exponent == 255, 32641, reciprocal).astype(np.uint16)
    code = np.where(exponent == 255, 255, shared).astype(np.uint8)
    return (code, reciprocal)


def _mx_scale_alg1_kernel(abs_max_f32, dst_dtype):
    """ComputeScaleCeilAlg: return E8M0 bytes and BF16 reciprocal bits."""
    inv_bits = {"float8_e4m3fn": 991054117, "float8_e5m2": 932333861}[dst_dtype]
    inv_max = np.uint32(inv_bits).view(np.float32)
    maxima = np.ascontiguousarray(abs_max_f32, dtype=np.float32)
    bits = maxima.view(np.uint32)
    finite, zero = (bits < np.uint32(2139095040), bits == 0)
    with np.errstate(invalid="ignore", under="ignore"):
        product = (maxima * inv_max).astype(np.float32).view(np.uint32)
    exponent, fraction = (product >> 23, product & np.uint32(8388607))
    ceil_up = (exponent > 0) & (exponent < 254) & (fraction > 0) | (exponent == 0) & (fraction > np.uint32(4194304))
    scale = np.where(ceil_up, exponent + 1, exponent)
    scale = np.where(finite, scale, np.uint32(255))
    scale = np.where(zero, np.uint32(0), scale)
    reciprocal = np.uint32(254) - scale << np.uint32(7)
    reciprocal = np.where(finite, reciprocal, np.uint32(32641))
    reciprocal = np.where(zero, np.uint32(0), reciprocal)
    return (scale.astype(np.uint8), (reciprocal & np.uint32(65535)).astype(np.uint16))


def dynamic_mx_quant_golden(
    x, dst_dtype="float8_e4m3fn", scale_alg=1, max_low_bound=0.0, axis=-1, *, seed=0, block_offset=0, _use_nibble=None
):
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
    if _use_nibble is None:
        _use_nibble = _uses_nibble_mapping(x.shape, axis, x.dtype.name, dst_dtype)
    if axis != x.ndim - 1:
        moved = np.ascontiguousarray(np.moveaxis(x, axis, -1))
        y, scale = dynamic_mx_quant_golden(
            moved,
            dst_dtype,
            scale_alg,
            max_low_bound,
            axis=-1,
            seed=seed,
            block_offset=block_offset,
            _use_nibble=_use_nibble,
        )
        return (np.ascontiguousarray(np.moveaxis(y, -1, axis)), np.ascontiguousarray(np.moveaxis(scale, -2, axis)))
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
        maxima = np.where(maxima == 0, maxima, np.maximum(maxima, np.float32(max_low_bound)))
    scale_fn = _mx_scale_alg0_kernel if scale_alg == 0 else _mx_scale_alg1_kernel
    scale, reciprocal_bits = scale_fn(maxima, dst_dtype)
    reciprocal = np.repeat(reciprocal_bits.view(bfloat16), 32, axis=-1)[:, :k]
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        normalized = rows.astype(np.float32) * reciprocal.astype(np.float32)
        if x.dtype.name == "bfloat16":
            normalized = normalized.astype(bfloat16).astype(np.float32)
    random_fn = (
        (_random_bits_fp32_eight if _use_nibble == "fp32_eight" else _random_bits_nibble)
        if _use_nibble
        else _random_bits
    )
    fields = random_fn(
        x.size, quant_length=k, seed=seed, block_offset=block_offset, **{"dst_dtype": dst_dtype} if _use_nibble else {}
    ).reshape(x.shape)
    codes = _sr_cast_fp32_to_fp8_vectorized(normalized.reshape(x.shape), fields, dst_dtype)
    scale = np.pad(scale, ((0, 0), (0, groups % 2)), constant_values=0)
    return (codes, scale.reshape(*x.shape[:-1], (k + 63) // 64, 2))
