"""Ascend 910B3 GLM indexer pool compression and paged K-cache write."""

from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from triton.extension.buffer import language as bl


@triton.jit
def _encode_e4m3fn(x):
    # x is finite and clamped to the largest finite E4M3FN magnitude.
    bits = x.to(tl.int32, bitcast=True)
    magnitude = bits & 0x7FFFFFFF
    sign = (bits >> 24) & 0x80

    # In a normal value the FP32 mantissa loses 20 bits. Bias by half
    # minus one and by the retained parity to round midpoint ties to even.
    parity = (magnitude >> 20) & 1
    normal = ((magnitude + 0x7FFFF + parity) >> 20) - 960

    # E4M3FN subnormals are multiples of 2**-9. At 2**23 FP32 has a
    # unit ULP, so addition rounds the scaled subnormal to an even integer.
    absolute = magnitude.to(tl.float32, bitcast=True)
    rounded = absolute * 512.0 + 8388608.0
    subnormal = rounded.to(tl.int32, bitcast=True) - 0x4B000000
    payload = tl.where(absolute < 0.015625, subnormal, normal)
    return (sign | payload).to(tl.uint8)


@triton.jit
def _aligned_butterfly(value, ROWS: tl.constexpr, DIM: tl.constexpr, STAGE: tl.constexpr):
    STEP: tl.constexpr = 1 << STAGE
    GROUPS: tl.constexpr = DIM // (2 * STEP)
    halves = tl.reshape(value, (ROWS, GROUPS, 2 * STEP))
    left = al.extract_slice(halves, (0, 0, 0), (ROWS, GROUPS, STEP), (1, 1, 1))
    right = al.extract_slice(halves, (0, 0, STEP), (ROWS, GROUPS, STEP), (1, 1, 1))
    combined = al.insert_slice(halves, left + right, (0, 0, 0), (ROWS, GROUPS, STEP), (1, 1, 1))
    combined = al.insert_slice(combined, left - right, (0, 0, STEP), (ROWS, GROUPS, STEP), (1, 1, 1))
    return tl.reshape(combined, (ROWS, DIM))


@triton.jit
def _store_compressed_tile(
    cache_k,
    cache_scale,
    locations,
    result_k,
    result_scale,
    selected,
    encoded,
    scale,
    valid_rows,
    lane,
    ROW_TILE: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    SMALL_ADDRESS: tl.constexpr,
    WIDE_INPUT: tl.constexpr,
    WRITE: tl.constexpr,
    RETURN: tl.constexpr,
):
    SCALE_BYTES: tl.constexpr = ROW_TILE * 4
    group_scale = False
    first_page = tl.full((), 0, tl.int32 if SMALL_ADDRESS else tl.int64)
    first_slot = tl.full((), 0, tl.int32 if SMALL_ADDRESS else tl.int64)
    if WRITE and not RETURN and PAGE >= ROW_TILE and PAGE_STRIDE >= PAGE * (DIM + 4):
        if valid_rows == ROW_TILE:
            row0 = tl.sum(al.extract_slice(selected, (0,), (1,), (1,)), axis=0)
            first = tl.load(locations + row0)
            if SMALL_ADDRESS:
                first = first.to(tl.int32)
            first_page = first // PAGE
            first_slot = first - first_page * PAGE
            scale_word_offset = first_page * (PAGE_STRIDE // 4) + PAGE * DIM // 4 + first_slot
            scale_start = cache_scale + scale_word_offset
            scale_aligned = (scale_start.to(tl.uint64) & (SCALE_BYTES - 1)) == 0
            if (first >= 0) & (first_slot <= PAGE - ROW_TILE) & scale_aligned:
                consecutive = True
                for j in tl.static_range(1, ROW_TILE):
                    row_j = tl.sum(al.extract_slice(selected, (j,), (1,), (1,)), axis=0)
                    actual = tl.load(locations + row_j)
                    consecutive = consecutive & (actual == first + j)
                group_scale = consecutive
    if group_scale:
        for j in tl.static_range(ROW_TILE):
            payload = al.extract_slice(encoded, (j, 0), (1, DIM), (1, 1)).reshape((DIM,))
            payload_dst = cache_k + first_page * PAGE_STRIDE + (first_slot + j) * DIM + lane
            tl.store(payload_dst, payload)
        scale_words = bl.to_buffer(scale, al.ascend_address_space.UB)
        scale_view = bl.subview(scale_words, (0,), (ROW_TILE,), (1,))
        scale_values = bl.to_tensor(scale_view, writable=False)
        scale_word_offset = first_page * (PAGE_STRIDE // 4) + PAGE * DIM // 4 + first_slot
        tl.store(cache_scale + scale_word_offset + tl.arange(0, ROW_TILE), scale_values)
    else:
        for j in tl.static_range(ROW_TILE):
            if j < valid_rows:
                row_slice = al.extract_slice(selected, (j,), (1,), (1,))
                row = tl.sum(row_slice, axis=0)
                payload = al.extract_slice(encoded, (j, 0), (1, DIM), (1, 1)).reshape((DIM,))
                scale_slice = al.extract_slice(scale, (j,), (1,), (1,))
                row_scale = tl.sum(scale_slice, axis=0)
                if WRITE:
                    location = tl.load(locations + row)
                    if SMALL_ADDRESS:
                        location = location.to(tl.int32)
                    else:
                        location = location.to(tl.int64)
                    page = location // PAGE
                    page_row = location - page * PAGE
                    tl.store(cache_k + page * PAGE_STRIDE + page_row * DIM + lane, payload)
                    tl.store(cache_scale + page * (PAGE_STRIDE // 4) + PAGE * DIM // 4 + page_row, row_scale)
                if RETURN:
                    # The input-width gate also covers the narrower output rows.
                    result_row = row.to(tl.int64) if WIDE_INPUT else row
                    tl.store(result_k + result_row * DIM + lane, payload)
                    tl.store(result_scale + row, row_scale)


@triton.jit(do_not_specialize=["n_rows"])
def _compress_rows(
    cache_k,
    cache_scale,
    keys,
    scores,
    ape,
    locations,
    write_mask,
    result_k,
    result_scale,
    n_rows,
    POOL: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    SMALL_ADDRESS: tl.constexpr,
    WIDE_INPUT: tl.constexpr,
    ROUND: tl.constexpr,
    WRITE: tl.constexpr,
    RETURN: tl.constexpr,
    ROW_TILE: tl.constexpr,
    CHUNK: tl.constexpr,
    SHARDS: tl.constexpr,
    MASK: tl.constexpr,
):
    lane = tl.arange(0, DIM)
    slot = tl.arange(0, POOL)
    pool_tile = slot[:, None] * DIM + lane[None, :]
    bias = tl.load(ape + pool_tile)
    flat_lane = tl.arange(0, ROW_TILE * DIM)
    # Carry the prior tile result while the next tile issues its input reads.
    pending_rows = 0
    pending_selected = tl.full((ROW_TILE,), 0, tl.int32)
    pending_encoded = tl.full((ROW_TILE, DIM), 0, tl.uint8)
    pending_scale = tl.full((ROW_TILE,), 0.0, tl.float32)
    for work in range(tl.program_id(0), tl.cdiv(n_rows, CHUNK) * SHARDS, tl.num_programs(0)):
        chunk = work // SHARDS
        shard = work % SHARDS
        begin = chunk * CHUNK
        offsets = tl.arange(0, CHUNK)
        rows = begin + offsets
        live = rows < n_rows
        if MASK:
            live = live & (tl.load(write_mask + rows, live, other=0) != 0)
        count = tl.sum(live.to(tl.int32), axis=0)
        if MASK:
            # Sort offsets once per chunk. Every masked lane sorts after the valid rows.
            packed_offsets = al.sort(tl.where(live, offsets, CHUNK).to(tl.float32)).to(tl.int32)
        else:
            packed_offsets = offsets
        for block in range(shard, tl.cdiv(count, ROW_TILE), SHARDS):
            # Each selected rank has exactly one live row. Tail lanes reuse
            # the last live row, then the write mask discards their outputs.
            ranks = tl.minimum(block * ROW_TILE + tl.arange(0, ROW_TILE), count - 1)
            selected = tl.gather(packed_offsets, ranks, axis=0) + begin
            if WIDE_INPUT:
                input_rows = selected.to(tl.int64)
                input_bases = input_rows * (POOL * DIM)
                key_buffer = tl.full((ROW_TILE, POOL, DIM), 0, tl.bfloat16)
                score_buffer = tl.full((ROW_TILE, POOL, DIM), 0, tl.bfloat16)
                for j in tl.static_range(ROW_TILE):
                    base_slice = al.extract_slice(input_bases, (j,), (1,), (1,))
                    offset = tl.sum(base_slice, axis=0) + pool_tile
                    raw_key = tl.load(keys + offset)
                    raw_score = tl.load(scores + offset)
                    key_buffer = al.insert_slice(key_buffer, raw_key[None, :, :], (j, 0, 0), (1, POOL, DIM), (1, 1, 1))
                    score_buffer = al.insert_slice(
                        score_buffer, raw_score[None, :, :], (j, 0, 0), (1, POOL, DIM), (1, 1, 1)
                    )
            else:
                # Read four complete selected rows directly into the computation tile.
                row_width: tl.constexpr = POOL * DIM
                key_buffer = al.index_select_simd(
                    keys, 0, selected, (n_rows, row_width), (-1, 0), (-1, row_width)
                ).reshape((ROW_TILE, POOL, DIM))
                score_buffer = al.index_select_simd(
                    scores, 0, selected, (n_rows, row_width), (-1, 0), (-1, row_width)
                ).reshape((ROW_TILE, POOL, DIM))
            if pending_rows > 0:
                _store_compressed_tile(
                    cache_k,
                    cache_scale,
                    locations,
                    result_k,
                    result_scale,
                    pending_selected,
                    pending_encoded,
                    pending_scale,
                    pending_rows,
                    lane,
                    ROW_TILE,
                    DIM,
                    PAGE,
                    PAGE_STRIDE,
                    SMALL_ADDRESS,
                    WIDE_INPUT,
                    WRITE,
                    RETURN,
                )
            logits = score_buffer.to(tl.float32) + bias[None, :, :]
            weights = tl.exp(logits - tl.max(logits, axis=1)[:, None, :])
            key = key_buffer.to(tl.float32)
            pooled = tl.sum(key * weights, axis=1) / tl.sum(weights, axis=1)
            rotated = pooled.to(tl.bfloat16).to(tl.float32)
            for stage in tl.static_range(7):
                if stage < 3:
                    stride = 1 << stage
                    flat = tl.reshape(rotated, (ROW_TILE * DIM,))
                    partner = tl.reshape(tl.gather(flat, flat_lane ^ stride, axis=0), (ROW_TILE, DIM))
                    rotated = tl.where((lane & stride)[None, :] == 0, rotated + partner, partner - rotated)
                else:
                    rotated = _aligned_butterfly(rotated, ROW_TILE, DIM, stage)
            rotated = (rotated * 0.08838834764831845).to(tl.bfloat16).to(tl.float32)
            scale = tl.maximum(tl.max(tl.abs(rotated), axis=1), 1e-4) * (1.0 / 448.0)
            if ROUND:
                bits = scale.to(tl.int32, bitcast=True)
                rounded_scale_bits = (bits + 0x7FFFFF) & 0x7F800000
                scale = rounded_scale_bits.to(tl.float32, bitcast=True)
                # A rounded scale is a power of two. Adjust the exponent directly.
                value_bits = rotated.to(tl.int32, bitcast=True)
                shift = 0x3F800000 - rounded_scale_bits
                magnitude = (value_bits & 0x7FFFFFFF) + shift[:, None]
                magnitude = tl.minimum(tl.maximum(magnitude, 0), 0x43E00000)
                quantized = (magnitude | (value_bits & -2147483648)).to(tl.float32, bitcast=True)
            else:
                quantized = tl.minimum(tl.maximum(rotated / scale[:, None], -448.0), 448.0)
            encoded = _encode_e4m3fn(quantized)
            pending_selected = selected
            pending_encoded = encoded
            pending_scale = scale
            pending_rows = tl.minimum(count - block * ROW_TILE, ROW_TILE)
    if pending_rows > 0:
        _store_compressed_tile(
            cache_k,
            cache_scale,
            locations,
            result_k,
            result_scale,
            pending_selected,
            pending_encoded,
            pending_scale,
            pending_rows,
            lane,
            ROW_TILE,
            DIM,
            PAGE,
            PAGE_STRIDE,
            SMALL_ADDRESS,
            WIDE_INPUT,
            WRITE,
            RETURN,
        )


def _kpool_compress_impl(
    kv_cache,
    slot_k,
    slot_score,
    ape,
    loc,
    pool_size,
    head_dim=128,
    write_mask=None,
    round_scale=True,
    return_compressed=False,
    write_cache=True,
):
    """Compress each pool into E4M3FN K bytes and one FP32 scale."""
    if kv_cache.storage_offset() % 4:
        raise ValueError("Cache storage offset must be 4-byte aligned")
    assert head_dim == 128 and pool_size in (4, 16)
    assert slot_k.ndim == 3 and slot_k.shape[1:] == (pool_size, head_dim)
    assert slot_score.shape == slot_k.shape and ape.shape == slot_k.shape[1:]
    assert slot_k.dtype == torch.bfloat16
    assert slot_score.dtype == slot_k.dtype and ape.dtype == torch.float32
    assert kv_cache.ndim == 3 and kv_cache.shape[2] == head_dim + 4
    assert kv_cache.dtype == torch.uint8 and kv_cache.stride(-1) == 1
    assert kv_cache.stride(1) == head_dim + 4 and kv_cache.stride(0) % 4 == 0
    assert loc.shape == (slot_k.shape[0],) and loc.dtype in (torch.int32, torch.int64)
    assert write_cache or return_compressed
    assert write_mask is None or (write_mask.shape == loc.shape and not return_compressed)

    n_rows = slot_k.shape[0]
    cache_scale = kv_cache.view(torch.float32)
    result_k, result_scale = kv_cache, cache_scale
    if return_compressed:
        result_k = torch.empty((n_rows, head_dim), dtype=torch.uint8, device=slot_k.device)
        result_scale = torch.empty((n_rows,), dtype=torch.float32, device=slot_k.device)

    if n_rows:
        keys = slot_k.contiguous().view(-1, head_dim)
        scores = slot_score.contiguous().view(-1, head_dim)
        bias = ape.contiguous().view(-1, head_dim)
        mask = loc if write_mask is None else write_mask.contiguous()
        core_count = triton.runtime.driver.active.utils.get_device_properties(slot_k.device.index)["num_vectorcore"]
        chunk_size = 128
        # The eight-row tile fits the four-slot pool in Ascend 910B3 UB.
        row_tile = 8 if pool_size == 4 else 4
        chunks = triton.cdiv(n_rows, chunk_size)
        shards = min(chunk_size // row_tile, max(1, core_count // chunks))
        with torch.npu.device(slot_k.device):
            _compress_rows[(min(chunks * shards, core_count),)](
                kv_cache,
                cache_scale,
                keys,
                scores,
                bias,
                loc.contiguous(),
                mask,
                result_k,
                result_scale,
                n_rows,
                POOL=pool_size,
                DIM=triton.next_power_of_2(head_dim),
                PAGE=kv_cache.shape[1],
                PAGE_STRIDE=kv_cache.stride(0),
                SMALL_ADDRESS=kv_cache.shape[0] * kv_cache.stride(0) < 2**31,
                WIDE_INPUT=n_rows * pool_size * head_dim >= 2**31,
                ROUND=round_scale,
                WRITE=write_cache,
                RETURN=return_compressed,
                ROW_TILE=row_tile,
                CHUNK=chunk_size,
                SHARDS=shards,
                MASK=write_mask is not None,
                enable_fp_fusion=False,
            )
    if return_compressed:
        return result_k.view(torch.float8_e4m3fn), result_scale


@torch.library.custom_op("mojo_npu_triton_a2::kpool_compress_write", mutates_args=("kv_cache",))
def _kpool_write_leaf(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    pool_size: int,
    write_mask: Optional[torch.Tensor],
    round_scale: bool,
) -> None:
    _kpool_compress_impl(
        kv_cache,
        slot_k,
        slot_score,
        ape,
        loc,
        pool_size,
        write_mask=write_mask,
        round_scale=round_scale,
    )


@_kpool_write_leaf.register_fake
def _kpool_write_fake(kv_cache, slot_k, slot_score, ape, loc, pool_size, write_mask, round_scale):
    return None


@torch.library.custom_op("mojo_npu_triton_a2::kpool_compress_write_return", mutates_args=("kv_cache",))
def _kpool_write_return_leaf(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    pool_size: int,
    round_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _kpool_compress_impl(
        kv_cache,
        slot_k,
        slot_score,
        ape,
        loc,
        pool_size,
        round_scale=round_scale,
        return_compressed=True,
    )


def _kpool_output_fake(kv_cache, slot_k, slot_score, ape, loc, pool_size, round_scale):
    return (
        torch.empty((slot_k.shape[0], 128), dtype=torch.float8_e4m3fn, device=slot_k.device),
        torch.empty((slot_k.shape[0],), dtype=torch.float32, device=slot_k.device),
    )


_kpool_write_return_leaf.register_fake(_kpool_output_fake)


@torch.library.custom_op("mojo_npu_triton_a2::kpool_compress_return", mutates_args=())
def _kpool_return_leaf(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    pool_size: int,
    round_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _kpool_compress_impl(
        kv_cache,
        slot_k,
        slot_score,
        ape,
        loc,
        pool_size,
        round_scale=round_scale,
        return_compressed=True,
        write_cache=False,
    )


_kpool_return_leaf.register_fake(_kpool_output_fake)


def kpool_compress_and_write_cache_fwd(
    kv_cache,
    slot_k,
    slot_score,
    ape,
    loc,
    pool_size,
    head_dim=128,
    write_mask=None,
    round_scale=True,
    return_compressed=False,
    write_cache=True,
):
    if not return_compressed:
        return _kpool_write_leaf(kv_cache, slot_k, slot_score, ape, loc, pool_size, write_mask, round_scale)
    if write_cache:
        return _kpool_write_return_leaf(kv_cache, slot_k, slot_score, ape, loc, pool_size, round_scale)
    return _kpool_return_leaf(kv_cache, slot_k, slot_score, ape, loc, pool_size, round_scale)
