"""Ascend A5 (950) Triton latent RoPE + quantised paged-cache insert.

Only group-tail tokens (``(position + 1) % compress_ratio == 0``) with a valid
slot write anything:

* **584-byte record** (``uint8 [pages, block, 584]``): the first 448 dims are
  UE8M0 FP8 in 8x64 groups (only the first 7 groups stored), then 128 bytes of
  BF16 RoPE tail; the 7 scale bytes (plus one zero pad byte) live outside the
  row at ``page + block*576 + row*8``.
* **528-byte record** (``uint8 [pages, block, 528]``): RoPE first, then MXFP8 on
  all 512 dims in 16x32 groups; the 16 scale bytes follow the values in the
  record.
* **plain** (``bf16`` or ``float8_e4m3fn`` ``[pages, block, 512]``): row is
  ``[448 NoPE | 64 RoPE]``; the FP8 variant scales by ``1 / fp8_scale`` after
  the tail is taken as BF16 (order matters bit-wise).

The rotation is GPT-J half-rotation on the last 64 dims at the group start
position; the NoPE pairs read ``(cos, sin) = (1, 0)`` and are identity.

Two write structures per record variant:

* **2D group path** - when ``slot_contig`` marks that a whole ``RPP`` group is a
  contiguous run of group-tail slots (production prefill fills one compressed
  block at a time and walks positions in order, so this is the common prefill
  shape), the group is processed as one ``[RPP, 512]`` tile: wide loads, the
  rotation and the whole UE8M0 quantisation chain vectorised across tokens, and
  wide strided stores. This is 1.7-3.2x faster than per-token writes and makes
  the quantisation chain nearly free. The group must also lie inside one page:
  consecutive slots are consecutive rows only within a page. A group whose slots
  straddle a page boundary stays on the per-token path, and where such groups
  can occur (always for the 584-byte record, for the other records when the page
  stride exceeds their row area) the token groups are visited in column-major
  order so that the slower groups are spread over all cores.
* **Per-token path** - the fallback for scattered slots (decode, block
  boundaries), with ``RPP`` tokens unrolled per program and three row-assembly
  optimisations when the triton-ascend extension provides them.

NVFP4 (288-byte) records are not implemented: the e2m1 conversion is inline PTX,
which has no Ascend equivalent.
"""

import torch
import triton
import triton.language as tl

try:
    # triton-ascend extension: insert_slice places a sub-block into a larger block
    # (a local UB move) and extract_slice pulls a strided sub-block out of a
    # loaded row.
    from triton.language.extra.cann import extension as _cann_ext

    _HAS_INSERT_SLICE = True
    _HAS_EXTRACT_SLICE = hasattr(_cann_ext, "extract_slice")
except Exception:  # pragma: no cover - non triton-ascend backends
    _cann_ext = None
    _HAS_INSERT_SLICE = False
    _HAS_EXTRACT_SLICE = False

_MERGE_ROW = _HAS_INSERT_SLICE
_TAIL_FROM_ROW = _HAS_EXTRACT_SLICE
_CS_MERGE = _HAS_EXTRACT_SLICE
_FAST_GROUP = _HAS_INSERT_SLICE

_RPP = 8
_GROUP_RPP = 8  # 2D tile width for contiguous runs (serial cap does not apply)
_ROW_THRESHOLD = 32
_ACTIVE_PER_PROGRAM = 4
_MXFP8_RPP = 4


@triton.jit
def _rotated_tail(cos_sin, base, position, COS_STRIDE: tl.constexpr, COMPRESS_RATIO: tl.constexpr):
    """GPT-J half-rotation of the last 64 dims (32 pairs read from GM)."""
    j = tl.arange(0, 32)
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + j).to(tl.float32)
    s = tl.load(cs + 32 + j).to(tl.float32)
    even = tl.load(base + 448 + 2 * j).to(tl.float32)
    odd = tl.load(base + 449 + 2 * j).to(tl.float32)
    return tl.interleave(even * c - odd * s, odd * c + even * s)


@triton.jit
def _rotated_tail_from_row(
    cos_sin,
    row,
    position,
    COS_STRIDE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    CS_MERGE: tl.constexpr = False,
):
    """Same rotation, but the even/odd lanes are sliced out of the loaded row."""
    if CS_MERGE:
        j64 = tl.arange(0, 64)
        cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
        cs64 = tl.load(cs + j64).to(tl.float32)
        c = _cann_ext.extract_slice(cs64, offsets=(0,), sizes=(32,), strides=(1,)).to(tl.float32)
        s = _cann_ext.extract_slice(cs64, offsets=(32,), sizes=(32,), strides=(1,)).to(
            tl.float32
        )
    else:
        j = tl.arange(0, 32)
        cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
        c = tl.load(cs + j).to(tl.float32)
        s = tl.load(cs + 32 + j).to(tl.float32)
    even = _cann_ext.extract_slice(row, offsets=(448,), sizes=(32,), strides=(2,)).to(tl.float32)
    odd = _cann_ext.extract_slice(row, offsets=(449,), sizes=(32,), strides=(2,)).to(tl.float32)
    return tl.interleave(even * c - odd * s, odd * c + even * s)


@triton.jit
def _ue8m0_sf(amax):
    """Group amax -> UE8M0 scale byte (exponent with mantissa carry)."""
    normalized = amax * (1.0 / 448.0)
    exponent = tl.ceil(tl.log2(normalized))
    return exponent


@triton.jit
def _group_is_contig(slot_contig, base_tok, num_tokens, RPP: tl.constexpr):
    """Vector gate: every token of the group continues its predecessor.

    ``slot_contig[t] > 0`` means token ``t`` has a valid slot that continues
    the previous token's slot at a consecutive position (and is a group tail).
    The group takes the 2D path only when ALL its bytes are set: a run shorter
    than ``RPP`` leaves a zero byte inside the group and safely falls back.
    One 8-byte vector load plus a min reduction.
    """
    if base_tok + RPP > num_tokens:
        return tl.full((), 0, tl.int1)
    r = tl.arange(0, RPP)
    bytes_ = tl.load(slot_contig + base_tok + r)
    return tl.min(bytes_) > 0


@triton.jit
def _insert_fp8_tok(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    t,
    num_tokens,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TAIL_FROM_ROW: tl.constexpr,
):
    if t < num_tokens:
        slot = tl.load(cache_slots + t)
        if slot >= 0:
            position = tl.load(positions + t)
            if (position + 1) % COMPRESS_RATIO == 0:
                d = tl.arange(0, 512)
                base = latent + t.to(tl.int64) * 512
                normed = tl.load(base + d).to(tl.float32)
                page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
                values = page + (slot % CACHE_BLOCK) * 576

                quant = tl.reshape(normed, (8, 64))
                amax = tl.maximum(tl.max(tl.abs(quant), 1), 1e-4)
                exponent = tl.ceil(tl.log2(amax * (1.0 / 448.0)))
                scaled = quant * tl.reshape(tl.exp2(-exponent), (8, 1))
                fp8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)
                tl.store(values + d, tl.reshape(fp8.to(tl.uint8, bitcast=True), (512,)), d < 448)
                s = tl.arange(0, 8)
                encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 255.0)
                # 7 valid scale bytes plus one zero pad byte in a single store.
                tl.store(
                    page + CACHE_BLOCK * 576 + (slot % CACHE_BLOCK) * 8 + s,
                    tl.where(s < 7, encoded.to(tl.uint8), tl.zeros((8,), tl.uint8)),
                )

                if TAIL_FROM_ROW:
                    rot = _rotated_tail_from_row(cos_sin, normed, position, COS_STRIDE, COMPRESS_RATIO)
                else:
                    rot = _rotated_tail(cos_sin, base, position, COS_STRIDE, COMPRESS_RATIO)
                rope_dst = (values + 448).to(tl.pointer_type(tl.bfloat16))
                tl.store(rope_dst + tl.arange(0, 64), rot.to(tl.bfloat16))


@triton.jit
def _insert_mxfp8_tok(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    t,
    num_tokens,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    MERGE_ROW: tl.constexpr,
    TAIL_FROM_ROW: tl.constexpr,
):
    """528-byte record: 14 NoPE groups and 2 rotated groups, or one merged pass."""
    if t < num_tokens:
        slot = tl.load(cache_slots + t)
        if slot >= 0:
            position = tl.load(positions + t)
            if (position + 1) % COMPRESS_RATIO == 0:
                d = tl.arange(0, 512)
                base = latent + t.to(tl.int64) * 512
                normed = tl.load(base + d).to(tl.float32)
                page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
                values = page + (slot % CACHE_BLOCK) * 528
                if TAIL_FROM_ROW:
                    rot = _rotated_tail_from_row(cos_sin, normed, position, COS_STRIDE, COMPRESS_RATIO)
                else:
                    rot = _rotated_tail(cos_sin, base, position, COS_STRIDE, COMPRESS_RATIO)

                if MERGE_ROW:
                    # Insert the tail into the row first, then one 16x32 pass over
                    # the whole 512 dims: one value store and one scale store.
                    row = _cann_ext.insert_slice(
                        normed, rot, offsets=(448,), sizes=(64,), strides=(1,)
                    )
                    quant = tl.reshape(row, (16, 32))
                    amax = tl.maximum(tl.max(tl.abs(quant), 1), 1e-4)
                    exponent = tl.ceil(tl.log2(amax * (1.0 / 448.0)))
                    scaled = quant * tl.reshape(tl.exp2(-exponent), (16, 1))
                    fp8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)
                    tl.store(
                        values + d, tl.reshape(fp8.to(tl.uint8, bitcast=True), (512,))
                    )
                    encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 255.0)
                    tl.store(values + 512 + tl.arange(0, 16), encoded.to(tl.uint8))
                else:
                    quant = tl.reshape(normed, (16, 32))
                    amax = tl.maximum(tl.max(tl.abs(quant), 1), 1e-4)
                    exponent = tl.ceil(tl.log2(amax * (1.0 / 448.0)))
                    scaled = quant * tl.reshape(tl.exp2(-exponent), (16, 1))
                    fp8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)
                    tl.store(
                        values + d,
                        tl.reshape(fp8.to(tl.uint8, bitcast=True), (512,)),
                        d < 448,
                    )
                    encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 255.0)
                    sc = tl.arange(0, 16)
                    tl.store(values + 512 + sc, encoded.to(tl.uint8), sc < 14)

                    # The rotated tail is exactly groups 14 and 15.
                    qt = tl.reshape(rot, (2, 32))
                    at = tl.maximum(tl.max(tl.abs(qt), 1), 1e-4)
                    et = tl.ceil(tl.log2(at * (1.0 / 448.0)))
                    st = qt * tl.reshape(tl.exp2(-et), (2, 1))
                    f8t = tl.clamp(st, -448.0, 448.0).to(tl.float8e4nv)
                    tl.store(
                        values + 448 + tl.arange(0, 64),
                        tl.reshape(f8t.to(tl.uint8, bitcast=True), (64,)),
                    )
                    ent = tl.minimum(tl.maximum(et + 127.0, 0.0), 255.0)
                    tl.store(values + 512 + 14 + tl.arange(0, 2), ent.to(tl.uint8))


@triton.jit
def _insert_plain_tok(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    fp8_scale,
    t,
    num_tokens,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    STORE_FP8: tl.constexpr,
    MERGE_ROW: tl.constexpr,
    TAIL_FROM_ROW: tl.constexpr,
    CS_MERGE: tl.constexpr,
):
    """Plain record: write the NoPE run as loaded and the rotated tail separately."""
    if t < num_tokens:
        slot = tl.load(cache_slots + t)
        if slot >= 0:
            position = tl.load(positions + t)
            if (position + 1) % COMPRESS_RATIO == 0:
                d = tl.arange(0, 512)
                base = latent + t.to(tl.int64) * 512
                row = tl.load(base + d)
                dst = (
                    cache
                    + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
                    + (slot % CACHE_BLOCK) * ROW_STRIDE
                )
                if TAIL_FROM_ROW:
                    rot = _rotated_tail_from_row(
                        cos_sin, row, position, COS_STRIDE, COMPRESS_RATIO, CS_MERGE
                    )
                else:
                    rot = _rotated_tail(cos_sin, base, position, COS_STRIDE, COMPRESS_RATIO)
                if MERGE_ROW:
                    # One full-width store instead of a masked 448-lane store plus
                    # a 64-lane tail store.
                    merged = _cann_ext.insert_slice(
                        row, rot.to(tl.bfloat16), offsets=(448,), sizes=(64,), strides=(1,)
                    )
                    if STORE_FP8:
                        inv = 1.0 / tl.load(fp8_scale)
                        tl.store(
                            dst + d,
                            tl.clamp(merged.to(tl.float32) * inv, -448.0, 448.0).to(tl.float8e4nv),
                        )
                    else:
                        tl.store(dst + d, merged)
                elif STORE_FP8:
                    # The e4m3 record covers the RoPE tail too: take the tail as
                    # BF16 first, then scale (the order is bit-wise load-bearing).
                    inv = 1.0 / tl.load(fp8_scale)
                    tl.store(
                        dst + d,
                        tl.clamp(row.to(tl.float32) * inv, -448.0, 448.0).to(tl.float8e4nv),
                        d < 448,
                    )
                    rt = rot.to(tl.bfloat16).to(tl.float32) * inv
                    tl.store(
                        dst + 448 + tl.arange(0, 64),
                        tl.clamp(rt, -448.0, 448.0).to(tl.float8e4nv),
                    )
                else:
                    tl.store(dst + d, row, d < 448)
                    rope_dst = (dst + 448).to(tl.pointer_type(tl.bfloat16))
                    tl.store(rope_dst + tl.arange(0, 64), rot.to(tl.bfloat16))


# ---------------------------------------------------------------------------
# 2D group fast path: one [RPP, 512] tile per contiguous run of group tails.
# The rotation and the whole UE8M0 quantisation chain are vectorised across
# tokens; stores are wide and strided by the record pitch.
# ---------------------------------------------------------------------------


@triton.jit
def _run_tile(latent, cos_sin, positions, base, RPP: tl.constexpr,
              COS_STRIDE: tl.constexpr, COMPRESS_RATIO: tl.constexpr):
    """Load a [RPP, 512] tile and return (nope_rows_bf16, rotated_tail_f32)."""
    r = tl.arange(0, RPP)
    rows = (base + r).to(tl.int64)
    d = tl.arange(0, 448)
    k = tl.arange(0, 64)
    j = tl.arange(0, 32)
    nope = tl.load(latent + rows[:, None] * 512 + d[None, :])
    tail = tl.load(latent + rows[:, None] * 512 + 448 + k[None, :])
    even, odd = tl.split(tl.reshape(tail, (RPP, 32, 2)))
    pos0 = tl.load(positions + base)
    cs = cos_sin + (pos0 + r)[:, None] * COS_STRIDE
    c = tl.load(cs + j[None, :]).to(tl.float32)
    s = tl.load(cs + 32 + j[None, :]).to(tl.float32)
    rot = tl.reshape(tl.join(even * c - odd * s, odd * c + even * s), (RPP, 64))
    return nope, rot


@triton.jit
def _insert_plain_group_fast(latent, cos_sin, positions, cache, base_tok, slot0,
                             fp8_scale, COS_STRIDE: tl.constexpr,
                             CACHE_STRIDE: tl.constexpr, ROW_STRIDE: tl.constexpr,
                             CACHE_BLOCK: tl.constexpr, STORE_FP8: tl.constexpr,
                             RPP: tl.constexpr):
    r = tl.arange(0, RPP)
    d = tl.arange(0, 512)
    k = tl.arange(0, 64)
    j = tl.arange(0, 32)
    rows = (base_tok + r).to(tl.int64)
    row = tl.load(latent + rows[:, None] * 512 + d[None, :])
    tail = tl.load(latent + rows[:, None] * 512 + 448 + k[None, :])
    even, odd = tl.split(tl.reshape(tail, (RPP, 32, 2)))
    pos0 = tl.load(positions + base_tok)
    cs = cos_sin + (pos0 + r)[:, None] * COS_STRIDE
    c = tl.load(cs + j[None, :]).to(tl.float32)
    s = tl.load(cs + 32 + j[None, :]).to(tl.float32)
    rot = tl.reshape(tl.join(even * c - odd * s, odd * c + even * s), (RPP, 64))
    merged = _cann_ext.insert_slice(
        row, rot.to(tl.bfloat16), offsets=(0, 448), sizes=(RPP, 64), strides=(1, 1)
    )
    dst = (
        cache
        + (slot0 // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
        + (slot0 % CACHE_BLOCK) * ROW_STRIDE
    )
    if STORE_FP8:
        inv = 1.0 / tl.load(fp8_scale)
        tl.store(
            dst + r[:, None] * ROW_STRIDE + d[None, :],
            tl.clamp(merged.to(tl.float32) * inv, -448.0, 448.0).to(tl.float8e4nv),
        )
    else:
        tl.store(dst + r[:, None] * ROW_STRIDE + d[None, :], merged)


@triton.jit
def _insert_fp8_group_fast(latent, cos_sin, positions, cache, base_tok, slot0,
                           COS_STRIDE: tl.constexpr, CACHE_STRIDE: tl.constexpr,
                           CACHE_BLOCK: tl.constexpr, RPP: tl.constexpr):
    """[RPP, 7, 64] NoPE quant + [RPP, 64] bf16 tail + [RPP, 8] scale, wide stores."""
    r = tl.arange(0, RPP)
    d7 = tl.arange(0, 448)
    k = tl.arange(0, 64)
    j = tl.arange(0, 32)
    rows = (base_tok + r).to(tl.int64)
    nope = tl.load(latent + rows[:, None] * 512 + d7[None, :])
    tail = tl.load(latent + rows[:, None] * 512 + 448 + k[None, :])
    even, odd = tl.split(tl.reshape(tail, (RPP, 32, 2)))
    pos0 = tl.load(positions + base_tok)
    cs = cos_sin + (pos0 + r)[:, None] * COS_STRIDE
    c = tl.load(cs + j[None, :]).to(tl.float32)
    s = tl.load(cs + 32 + j[None, :]).to(tl.float32)
    rot = tl.reshape(tl.join(even * c - odd * s, odd * c + even * s), (RPP, 64))

    g = tl.reshape(nope.to(tl.float32), (RPP, 7, 64))
    amax = tl.maximum(tl.max(tl.abs(g), 2), 1e-4)
    exponent = tl.ceil(tl.log2(amax * (1.0 / 448.0)))
    scaled = g * tl.reshape(tl.exp2(-exponent), (RPP, 7, 1))
    fp8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)

    page = cache + (slot0 // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    values = page + (slot0 % CACHE_BLOCK) * 576
    tl.store(
        values.to(tl.pointer_type(tl.float8e4nv)) + r[:, None] * 576 + d7[None, :],
        tl.reshape(fp8, (RPP, 448)),
    )
    rope_dst = (values + 448).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_dst + r[:, None] * 288 + k[None, :], rot.to(tl.bfloat16))
    sbase = page + CACHE_BLOCK * 576 + (slot0 % CACHE_BLOCK + r)[:, None] * 8
    encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 255.0)
    s7 = tl.arange(0, 7)
    tl.store(sbase + s7[None, :], encoded.to(tl.uint8))
    tl.store(sbase + 7 + tl.arange(0, 1)[None, :], tl.zeros((RPP, 1), tl.uint8))


@triton.jit
def _insert_mxfp8_group_fast(latent, cos_sin, positions, cache, base_tok, slot0,
                             COS_STRIDE: tl.constexpr, CACHE_STRIDE: tl.constexpr,
                             CACHE_BLOCK: tl.constexpr, RPP: tl.constexpr):
    """14 NoPE groups from the loaded rows, 2 rotated groups from the fp32 tail."""
    r = tl.arange(0, RPP)
    d4 = tl.arange(0, 448)
    k = tl.arange(0, 64)
    j = tl.arange(0, 32)
    rows = (base_tok + r).to(tl.int64)
    nope = tl.load(latent + rows[:, None] * 512 + d4[None, :])
    tail = tl.load(latent + rows[:, None] * 512 + 448 + k[None, :])
    even, odd = tl.split(tl.reshape(tail, (RPP, 32, 2)))
    pos0 = tl.load(positions + base_tok)
    cs = cos_sin + (pos0 + r)[:, None] * COS_STRIDE
    c = tl.load(cs + j[None, :]).to(tl.float32)
    s = tl.load(cs + 32 + j[None, :]).to(tl.float32)
    rot = tl.reshape(tl.join(even * c - odd * s, odd * c + even * s), (RPP, 64))

    page = cache + (slot0 // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    values = page + (slot0 % CACHE_BLOCK) * 528
    g1 = tl.reshape(nope.to(tl.float32), (RPP, 14, 32))
    g2 = tl.reshape(rot, (RPP, 2, 32))
    amax1 = tl.maximum(tl.max(tl.abs(g1), 2), 1e-4)
    amax2 = tl.maximum(tl.max(tl.abs(g2), 2), 1e-4)
    e1 = tl.ceil(tl.log2(amax1 * (1.0 / 448.0)))
    e2 = tl.ceil(tl.log2(amax2 * (1.0 / 448.0)))
    q1 = tl.clamp(g1 * tl.reshape(tl.exp2(-e1), (RPP, 14, 1)), -448.0, 448.0).to(tl.float8e4nv)
    q2 = tl.clamp(g2 * tl.reshape(tl.exp2(-e2), (RPP, 2, 1)), -448.0, 448.0).to(tl.float8e4nv)

    dst = values.to(tl.pointer_type(tl.float8e4nv))
    tl.store(dst + r[:, None] * 528 + d4[None, :], tl.reshape(q1, (RPP, 448)))
    tl.store(dst + r[:, None] * 528 + 448 + k[None, :], tl.reshape(q2, (RPP, 64)))
    spage = values + 512 + r[:, None] * 528
    enc1 = tl.minimum(tl.maximum(e1 + 127.0, 0.0), 255.0)
    enc2 = tl.minimum(tl.maximum(e2 + 127.0, 0.0), 255.0)
    tl.store(spage + tl.arange(0, 14)[None, :], enc1.to(tl.uint8))
    tl.store(spage + 14 + tl.arange(0, 2)[None, :], enc2.to(tl.uint8))


@triton.jit
def _rope_quant_insert_fp8(
    latent, positions, cos_sin, cache, cache_slots, slot_contig, num_tokens,
    COS_STRIDE: tl.constexpr, CACHE_STRIDE: tl.constexpr, CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr, RPP: tl.constexpr, TAIL_FROM_ROW: tl.constexpr,
    FAST_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    # Groups that straddle a page boundary take the slower per-token path (see
    # the guard below). Blocks reach the vector cores by index in a fixed
    # rotation, and such groups recur with a period that is a multiple of RPP,
    # so in row-major order they keep landing on the same few cores. Visiting
    # the groups in column-major order makes them consecutive in block order,
    # which spreads them over every core.
    cols = (num_tokens + RPP * RPP - 1) // (RPP * RPP)
    base_tok = ((pid % cols) * RPP + pid // cols) * RPP
    if base_tok < num_tokens:
        fast = tl.full((), 0, tl.int1)
        if FAST_GROUP and RPP > 1:
            fast = _group_is_contig(slot_contig, base_tok, num_tokens, RPP)
            # The scales of the 584-byte record sit after the row area of the page, so
            # consecutive slots are consecutive rows only inside one page.
            slot0 = tl.load(cache_slots + base_tok)
            fast = fast & ((slot0 % CACHE_BLOCK) + RPP <= CACHE_BLOCK)
        if fast:
            _insert_fp8_group_fast(
                latent, cos_sin, positions, cache, base_tok,
                tl.load(cache_slots + base_tok),
                COS_STRIDE, CACHE_STRIDE, CACHE_BLOCK, RPP,
            )
        else:
            for r in tl.static_range(RPP):
                _insert_fp8_tok(
                    latent, positions, cos_sin, cache, cache_slots, base_tok + r,
                    num_tokens, COS_STRIDE, CACHE_STRIDE, CACHE_BLOCK,
                    COMPRESS_RATIO, TAIL_FROM_ROW,
                )


@triton.jit
def _rope_quant_insert_mxfp8(
    latent, positions, cos_sin, cache, cache_slots, slot_contig, num_tokens,
    COS_STRIDE: tl.constexpr, CACHE_STRIDE: tl.constexpr, CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr, RPP: tl.constexpr, MERGE_ROW: tl.constexpr,
    TAIL_FROM_ROW: tl.constexpr, FAST_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    PAGE_BREAK: tl.constexpr = CACHE_STRIDE != CACHE_BLOCK * 528
    if PAGE_BREAK:
        # Groups that straddle a page boundary take the slower per-token path (see
        # the guard below). Blocks reach the vector cores by index in a fixed
        # rotation, and such groups recur with a period that is a multiple of RPP,
        # so in row-major order they keep landing on the same few cores. Visiting
        # the groups in column-major order makes them consecutive in block order,
        # which spreads them over every core.
        cols = (num_tokens + RPP * RPP - 1) // (RPP * RPP)
        base_tok = ((pid % cols) * RPP + pid // cols) * RPP
    else:
        base_tok = pid * RPP
    if base_tok < num_tokens:
        fast = tl.full((), 0, tl.int1)
        if FAST_GROUP and RPP > 1:
            fast = _group_is_contig(slot_contig, base_tok, num_tokens, RPP)
            if PAGE_BREAK:
                # A page stride larger than the row area breaks the run of consecutive
                # slots at the page boundary: such a group keeps the per-token path.
                slot0 = tl.load(cache_slots + base_tok)
                fast = fast & ((slot0 % CACHE_BLOCK) + RPP <= CACHE_BLOCK)
        if fast:
            _insert_mxfp8_group_fast(
                latent, cos_sin, positions, cache, base_tok,
                tl.load(cache_slots + base_tok),
                COS_STRIDE, CACHE_STRIDE, CACHE_BLOCK, RPP,
            )
        else:
            for r in tl.static_range(RPP):
                _insert_mxfp8_tok(
                    latent, positions, cos_sin, cache, cache_slots, base_tok + r,
                    num_tokens, COS_STRIDE, CACHE_STRIDE, CACHE_BLOCK,
                    COMPRESS_RATIO, MERGE_ROW, TAIL_FROM_ROW,
                )


@triton.jit
def _rope_plain_insert(
    latent, positions, cos_sin, cache, cache_slots, slot_contig, fp8_scale, num_tokens,
    COS_STRIDE: tl.constexpr, CACHE_STRIDE: tl.constexpr, ROW_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr, COMPRESS_RATIO: tl.constexpr, STORE_FP8: tl.constexpr,
    RPP: tl.constexpr, MERGE_ROW: tl.constexpr, TAIL_FROM_ROW: tl.constexpr,
    CS_MERGE: tl.constexpr, FAST_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    PAGE_BREAK: tl.constexpr = CACHE_STRIDE != CACHE_BLOCK * ROW_STRIDE
    if PAGE_BREAK:
        # Groups that straddle a page boundary take the slower per-token path (see
        # the guard below). Blocks reach the vector cores by index in a fixed
        # rotation, and such groups recur with a period that is a multiple of RPP,
        # so in row-major order they keep landing on the same few cores. Visiting
        # the groups in column-major order makes them consecutive in block order,
        # which spreads them over every core.
        cols = (num_tokens + RPP * RPP - 1) // (RPP * RPP)
        base_tok = ((pid % cols) * RPP + pid // cols) * RPP
    else:
        base_tok = pid * RPP
    if base_tok < num_tokens:
        fast = tl.full((), 0, tl.int1)
        if FAST_GROUP and RPP > 1:
            fast = _group_is_contig(slot_contig, base_tok, num_tokens, RPP)
            if PAGE_BREAK:
                # A page stride larger than the row area breaks the run of consecutive
                # slots at the page boundary: such a group keeps the per-token path.
                slot0 = tl.load(cache_slots + base_tok)
                fast = fast & ((slot0 % CACHE_BLOCK) + RPP <= CACHE_BLOCK)
        if fast:
            _insert_plain_group_fast(
                latent, cos_sin, positions, cache, base_tok,
                tl.load(cache_slots + base_tok), fp8_scale,
                COS_STRIDE, CACHE_STRIDE, ROW_STRIDE, CACHE_BLOCK, STORE_FP8, RPP,
            )
        else:
            for r in tl.static_range(RPP):
                _insert_plain_tok(
                    latent, positions, cos_sin, cache, cache_slots, fp8_scale, base_tok + r,
                    num_tokens, COS_STRIDE, CACHE_STRIDE, ROW_STRIDE, CACHE_BLOCK,
                    COMPRESS_RATIO, STORE_FP8, MERGE_ROW, TAIL_FROM_ROW, CS_MERGE,
                )


def _pick_rpp(num_tokens, compress_ratio, cap):
    if num_tokens < _ROW_THRESHOLD:
        return 1
    return max(1, min(cap, _ACTIVE_PER_PROGRAM * compress_ratio))


def _slot_contig_for(slot_mapping, num_tokens, compress_ratio, positions):
    """Host fallback: derive per-token slot continuity on device.

    ``slot_contig[t] = 1`` iff token ``t`` is a group tail whose slot AND
    position continue the previous token's. Used only when the caller did not
    supply flags produced alongside the slot mapping
    (``get_compressed_slot_mapping``); costs one cheap elementwise pass.
    """
    slots = slot_mapping[:num_tokens]
    prev = torch.cat((torch.full((1,), -2, dtype=slots.dtype, device=slots.device), slots[:-1]))
    pos = positions[:num_tokens]
    prev_pos = torch.cat((torch.tensor([-2], device=pos.device), pos[:-1]))
    continues = (slots >= 0) & (slots == prev + 1) & (pos == prev_pos + 1)
    tail = (pos + 1) % compress_ratio == 0
    return (continues & tail).to(torch.uint8)


def _rows_continue_across_pages(kv_cache, row_stride):
    """Whether the row area of a page is followed directly by the next page's."""
    return kv_cache.stride(0) == kv_cache.shape[1] * row_stride


def _programs(num_tokens, rpp, kv_cache, row_stride=None):
    """Grid of one launch (see the kernels for the column-major group order).

    Groups that straddle a page boundary take the slower per-token path. The
    584-byte record breaks a run of consecutive slots at every page boundary;
    the other records do so only when the page stride exceeds their row area
    (``row_stride`` is their per-row pitch). Only then are the groups visited
    in column-major order; the layout then has ``cdiv(groups, rpp) * rpp``
    slots, and the few slots past the last group return without work.
    """
    if row_stride is not None and _rows_continue_across_pages(kv_cache, row_stride):
        return triton.cdiv(num_tokens, rpp)
    return triton.cdiv(num_tokens, rpp * rpp) * rpp


def _rope_quant_insert_impl(latent, positions, cos_sin_cache, kv_cache, slot_mapping,
                            compress_ratio, fp8_scale, slot_contig=None):
    num_tokens = slot_mapping.numel()
    if num_tokens == 0:
        return
    if slot_contig is None:
        slot_contig = _slot_contig_for(slot_mapping, num_tokens, compress_ratio, positions)
    group_rpp = _GROUP_RPP if _FAST_GROUP else 1
    if kv_cache.dtype == torch.uint8:
        width = kv_cache.shape[-1]
        if width == 288:
            raise NotImplementedError(
                "rope_quant_insert: 288-byte NVFP4 records use NVIDIA inline PTX "
                "(cvt.rn.satfinite.e2m1x2.f32) and have no Ascend equivalent"
            )
        if width not in (584, 528):
            raise ValueError(f"unsupported paged KV record width {width}")
        if width == 584:
            rpp = group_rpp if _FAST_GROUP else _pick_rpp(num_tokens, compress_ratio, _RPP)
            _rope_quant_insert_fp8[(_programs(num_tokens, rpp, kv_cache),)](
                latent,
                positions,
                cos_sin_cache,
                kv_cache,
                slot_mapping,
                slot_contig,
                num_tokens,
                COS_STRIDE=cos_sin_cache.stride(0),
                CACHE_STRIDE=kv_cache.stride(0),
                CACHE_BLOCK=kv_cache.shape[1],
                COMPRESS_RATIO=compress_ratio,
                RPP=rpp,
                TAIL_FROM_ROW=_TAIL_FROM_ROW,
                FAST_GROUP=_FAST_GROUP,
            )
        else:
            rpp = group_rpp if _FAST_GROUP else _pick_rpp(num_tokens, compress_ratio, _MXFP8_RPP if _MXFP8_RPP <= 4 else 8)
            _rope_quant_insert_mxfp8[(_programs(num_tokens, rpp, kv_cache, 528),)](
                latent,
                positions,
                cos_sin_cache,
                kv_cache,
                slot_mapping,
                slot_contig,
                num_tokens,
                COS_STRIDE=cos_sin_cache.stride(0),
                CACHE_STRIDE=kv_cache.stride(0),
                CACHE_BLOCK=kv_cache.shape[1],
                COMPRESS_RATIO=compress_ratio,
                RPP=rpp,
                MERGE_ROW=_MERGE_ROW,
                TAIL_FROM_ROW=_TAIL_FROM_ROW,
                FAST_GROUP=_FAST_GROUP,
            )
        return

    store_fp8 = kv_cache.dtype == torch.float8_e4m3fn
    rpp = group_rpp if _FAST_GROUP else _pick_rpp(num_tokens, compress_ratio, _RPP)
    _rope_plain_insert[(_programs(num_tokens, rpp, kv_cache, kv_cache.stride(1)),)](
        latent,
        positions,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        slot_contig,
        fp8_scale if store_fp8 else None,
        num_tokens,
        COS_STRIDE=cos_sin_cache.stride(0),
        CACHE_STRIDE=kv_cache.stride(0),
        ROW_STRIDE=kv_cache.stride(1),
        CACHE_BLOCK=kv_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        STORE_FP8=store_fp8,
        RPP=rpp,
        MERGE_ROW=_MERGE_ROW,
        TAIL_FROM_ROW=_TAIL_FROM_ROW,
        CS_MERGE=_CS_MERGE,
        FAST_GROUP=_FAST_GROUP,
    )


@torch.library.custom_op(
    "mojo_npu_triton_a5::rope_quant_insert_fwd",
    mutates_args=("kv_cache",),
)
def rope_quant_insert_fwd(
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    fp8_scale: torch.Tensor | None,
    slot_contig: torch.Tensor | None,
) -> None:
    _rope_quant_insert_impl(
        latent.contiguous(),
        positions.contiguous(),
        cos_sin_cache,
        kv_cache,
        slot_mapping.contiguous(),
        compress_ratio,
        fp8_scale,
        None if slot_contig is None else slot_contig.contiguous(),
    )


@rope_quant_insert_fwd.register_fake
def _rope_quant_insert_fake(
    latent, positions, cos_sin_cache, kv_cache, slot_mapping, compress_ratio, fp8_scale, slot_contig
) -> None:
    return None
