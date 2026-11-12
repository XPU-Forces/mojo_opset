"""Ascend 910B3 GLM indexer pool compression and paged K-cache write."""

from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def _hadamard(x, lane):
    for stage in tl.static_range(7):
        stride = 1 << stage
        partner = tl.gather(x, lane ^ stride, axis=0)
        x = tl.where((lane & stride) == 0, x + partner, partner - x)
    return x * 0.08838834764831845


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
    payload = tl.where(magnitude < 0x3C800000, subnormal, normal)
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
def _pool_feature_half(
    keys,
    scores,
    ape,
    rows,
    active,
    POOL: tl.constexpr,
    DIM: tl.constexpr,
    START: tl.constexpr,
):
    lane = START + tl.arange(0, DIM // 2)
    slot = tl.arange(0, POOL)
    pool_tile = slot[:, None] * DIM + lane[None, :]
    bias = tl.load(ape + pool_tile)
    offsets = rows[:, None, None] * (POOL * DIM) + pool_tile[None, :, :]
    mask = active[:, None, None]
    logits = tl.load(scores + offsets, mask=mask, other=0).to(tl.float32)
    key_input = tl.load(keys + offsets, mask=mask, other=0)
    logits += bias[None, :, :]
    weights = tl.exp(logits - tl.max(logits, axis=1)[:, None, :])
    key = key_input.to(tl.float32)
    return (tl.sum(key * weights, axis=1) / tl.sum(weights, axis=1)).to(tl.bfloat16)


@triton.jit
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
    n_rows: tl.constexpr,
    POOL: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    SMALL_ADDRESS: tl.constexpr,
    WIDE_INPUT: tl.constexpr,
    ROUND: tl.constexpr,
    MASK: tl.constexpr,
    WRITE: tl.constexpr,
    RETURN: tl.constexpr,
    ROW_TILE: tl.constexpr,
    LARGE_BUTTERFLY: tl.constexpr,
):
    if ROW_TILE == 1:
        lane = tl.arange(0, DIM).to(tl.int32)
        slot = tl.arange(0, POOL).to(tl.int32)
        tile = slot[:, None] * DIM + lane[None, :]
        bias = tl.load(ape + tile)
        for row in range(tl.program_id(0).to(tl.int32), n_rows, tl.num_programs(0)):
            active = True
            if MASK:
                active = tl.load(write_mask + row) != 0

            row_address = row
            if WIDE_INPUT:
                row_address = row.to(tl.int64)
            offset = row_address * (POOL * DIM) + tile
            logits = tl.load(scores + offset, mask=active, other=0).to(tl.float32)
            logits += bias
            weights = tl.exp(logits - tl.max(logits, axis=0)[None, :])
            key = tl.load(keys + offset, mask=active, other=0).to(tl.float32)
            pooled = tl.sum(key * weights, axis=0) / tl.sum(weights, axis=0)
            pooled = pooled.to(tl.bfloat16).to(tl.float32)
            rotated = _hadamard(pooled, lane).to(tl.bfloat16).to(tl.float32)

            scale = tl.maximum(tl.max(tl.abs(rotated), axis=0), 1e-4) * (1.0 / 448.0)
            if ROUND:
                scale_bits = scale.to(tl.int32, bitcast=True)
                scale = ((scale_bits + 0x7FFFFF) & 0x7F800000).to(tl.float32, bitcast=True)
            quantized = tl.minimum(tl.maximum(rotated / scale, -448.0), 448.0)
            encoded = _encode_e4m3fn(quantized)

            if WRITE:
                location = tl.load(locations + row, mask=active, other=0)
                if SMALL_ADDRESS:
                    location = location.to(tl.int32)
                else:
                    location = location.to(tl.int64)
                page = location // PAGE
                page_row = location - page * PAGE
                tl.store(cache_k + page * PAGE_STRIDE + page_row * DIM + lane, encoded, mask=active)
                tl.store(cache_scale + page * (PAGE_STRIDE // 4) + PAGE * DIM // 4 + page_row, scale, mask=active)
            if RETURN:
                tl.store(result_k + row * DIM + lane, encoded)
                tl.store(result_scale + row, scale)

    else:
        lane = tl.arange(0, DIM).to(tl.int32)
        slot = tl.arange(0, POOL).to(tl.int32)
        row_in_tile = tl.arange(0, ROW_TILE).to(tl.int32)
        pool_tile = slot[:, None] * DIM + lane[None, :]
        if ROW_TILE != 8:
            bias = tl.load(ape + pool_tile)
        pid = tl.program_id(0).to(tl.int32)
        for block in range(pid, tl.cdiv(n_rows, ROW_TILE), tl.num_programs(0)):
            rows = block * ROW_TILE + row_in_tile
            if ROW_TILE >= 4 and not MASK and n_rows % ROW_TILE == 0:
                active = tl.full((ROW_TILE,), True, tl.int1)
            else:
                active = rows < n_rows
            if MASK:
                active &= tl.load(write_mask + rows, mask=active, other=0) != 0

            row_address = rows
            if WIDE_INPUT:
                row_address = rows.to(tl.int64)
            if ROW_TILE == 8:
                left = _pool_feature_half(keys, scores, ape, row_address, active, POOL, DIM, 0)
                right = _pool_feature_half(keys, scores, ape, row_address, active, POOL, DIM, DIM // 2)
                halves = tl.full((ROW_TILE, DIM), 0, tl.bfloat16)
                halves = al.insert_slice(halves, left, (0, 0), (ROW_TILE, DIM // 2), (1, 1))
                halves = al.insert_slice(halves, right, (0, DIM // 2), (ROW_TILE, DIM // 2), (1, 1))
                rotated = halves.to(tl.float32)
            else:
                offsets = row_address[:, None, None] * (POOL * DIM) + pool_tile[None, :, :]
                tile_mask = active[:, None, None]
                logits = tl.load(scores + offsets, mask=tile_mask, other=0).to(tl.float32)
                logits += bias[None, :, :]
                weights = tl.exp(logits - tl.max(logits, axis=1)[:, None, :])
                key = tl.load(keys + offsets, mask=tile_mask, other=0).to(tl.float32)
                pooled = tl.sum(key * weights, axis=1) / tl.sum(weights, axis=1)
                rotated = pooled.to(tl.bfloat16).to(tl.float32)
            if ROW_TILE == 8:
                # Group each contiguous set of eight lanes, then put the
                # first three butterfly bits on the major dimension.
                GROUPS: tl.constexpr = ROW_TILE * DIM // 8
                transposed = tl.trans(tl.reshape(rotated, (GROUPS, 8)))
                flat = tl.reshape(transposed, (1, ROW_TILE * DIM))
                for stage in tl.static_range(7, 10):
                    flat = _aligned_butterfly(flat, 1, ROW_TILE * DIM, stage)
                rotated = tl.reshape(tl.trans(tl.reshape(flat, (8, GROUPS))), (ROW_TILE, DIM))
                for stage in tl.static_range(3, 7):
                    rotated = _aligned_butterfly(rotated, ROW_TILE, DIM, stage)
            else:
                for stage in tl.static_range(7):
                    if LARGE_BUTTERFLY and stage >= 3:
                        rotated = _aligned_butterfly(rotated, ROW_TILE, DIM, stage)
                    else:
                        stride = 1 << stage
                        if LARGE_BUTTERFLY:
                            flat = tl.reshape(rotated, (ROW_TILE * DIM,))
                            flat_lane = tl.arange(0, ROW_TILE * DIM).to(tl.int32)
                            partner = tl.reshape(tl.gather(flat, flat_lane ^ stride, axis=0), (ROW_TILE, DIM))
                        else:
                            indexes = tl.broadcast_to((lane ^ stride)[None, :], (ROW_TILE, DIM))
                            partner = tl.gather(rotated, indexes, axis=1)
                        low = (lane & stride) == 0
                        if LARGE_BUTTERFLY:
                            rotated = tl.fma(rotated, tl.where(low, 1.0, -1.0)[None, :], partner)
                        else:
                            rotated = tl.where(low[None, :], rotated + partner, partner - rotated)
            rotated = (rotated * 0.08838834764831845).to(tl.bfloat16).to(tl.float32)

            scale = tl.maximum(tl.max(tl.abs(rotated), axis=1), 1e-4) * (1.0 / 448.0)
            if ROUND:
                scale_bits = scale.to(tl.int32, bitcast=True)
                scale = ((scale_bits + 0x7FFFFF) & 0x7F800000).to(tl.float32, bitcast=True)
            if ROW_TILE == 8 and ROUND:
                # Rounded scale is an exact power of two. At the minimum
                # scale, even the largest BF16 subnormal still rounds to FP8 zero.
                bits = rotated.to(tl.int32, bitcast=True)
                exponent_shift = 0x3F800000 - scale.to(tl.int32, bitcast=True)
                magnitude = (bits & 0x7FFFFFFF) + exponent_shift[:, None]
                magnitude = tl.minimum(tl.maximum(magnitude, 0), 0x43E00000)
                quantized = (magnitude | (bits & -2147483648)).to(tl.float32, bitcast=True)
            else:
                quantized = tl.minimum(tl.maximum(rotated / scale[:, None], -448.0), 448.0)
            encoded = _encode_e4m3fn(quantized)

            if WRITE:
                location = tl.load(locations + rows, mask=active, other=0)
                if SMALL_ADDRESS:
                    location = location.to(tl.int32)
                else:
                    location = location.to(tl.int64)
                page = location // PAGE
                page_row = location - page * PAGE
                k_base = page * PAGE_STRIDE + page_row * DIM
                tl.store(cache_k + k_base[:, None] + lane[None, :], encoded, mask=active[:, None])
                tl.store(cache_scale + page * (PAGE_STRIDE // 4) + PAGE * DIM // 4 + page_row, scale, mask=active)
            if RETURN:
                tl.store(result_k + rows[:, None] * DIM + lane[None, :], encoded, mask=active[:, None])
                tl.store(result_scale + rows, scale, mask=active)


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
        rows_per_tile = 4 if n_rows >= 8 * core_count else (2 if n_rows >= core_count else 1)
        if rows_per_tile == 4 and pool_size == 16 and write_mask is None and n_rows % 8 == 0:
            rows_per_tile = 8
        tiles = triton.cdiv(n_rows, rows_per_tile)
        _compress_rows[(min(tiles, core_count),)](
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
            MASK=write_mask is not None,
            WRITE=write_cache,
            RETURN=return_compressed,
            ROW_TILE=rows_per_tile,
            LARGE_BUTTERFLY=n_rows >= 8 * core_count,
            enable_fp_fusion=False,
            multibuffer=False if rows_per_tile == 8 else None,
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
