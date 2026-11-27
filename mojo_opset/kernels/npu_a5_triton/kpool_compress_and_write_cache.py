# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 950PR GLM indexer pool compression and paged K-cache write."""

from __future__ import annotations

from typing import Optional
from typing import Tuple

import torch
import triton
import triton.language as tl

from triton.language.extra.cann import extension as ext

# The GLM-5.3-Flash indexer head dimension is fixed at 128.
INDEX_HEAD_DIM = 128


# Hadamard-128 rotation


@triton.jit
def _fold_pool_numerator(products, POOL: tl.constexpr, WIDTH: tl.constexpr):
    HALF: tl.constexpr = POOL // 2
    left = ext.extract_slice(products, (0, 0), (HALF, WIDTH), (1, 1))
    right = ext.extract_slice(products, (HALF, 0), (HALF, WIDTH), (1, 1))
    return left + right


@triton.jit
def _kpool_uniform(
    cache,
    scales,
    keys,
    scores,
    ape,
    locations,
    write_mask,
    result_k,
    result_s,
    n_rows,
    POOL: tl.constexpr,
    LOG_POOL: tl.constexpr,
    ROWS: tl.constexpr,
    CHUNK: tl.constexpr,
    SHARDS: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    ROUND: tl.constexpr,
    WRITE: tl.constexpr,
    RETURN: tl.constexpr,
):
    WIDTH: tl.constexpr = POOL * 128
    slots = tl.arange(0, POOL)
    cols = tl.arange(0, 128)
    row_lanes = tl.arange(0, ROWS)
    flat = tl.arange(0, ROWS * 128)
    offsets = tl.arange(0, CHUNK)
    bias = tl.load(ape + slots[:, None] * 128 + (flat % 128)[None, :])
    for work in range(tl.program_id(0), tl.cdiv(n_rows, CHUNK) * SHARDS, tl.num_programs(0)):
        begin = (work // SHARDS) * CHUNK
        absolute = begin + offsets
        count = tl.minimum(n_rows - begin, CHUNK)
        if HAS_MASK:
            live = absolute < n_rows
            live = tl.load(write_mask + absolute, live, other=0) != 0
            count = tl.sum(live.to(tl.int32), axis=0)
            packed = ext.sort(tl.where(live, offsets, CHUNK).to(tl.float32)).to(tl.int32)
        for tile in range(work % SHARDS, tl.cdiv(count, ROWS), SHARDS):
            rank = tile * ROWS + row_lanes
            valid = rank < count
            local = rank
            if HAS_MASK:
                local = tl.gather(packed, tl.minimum(rank, count - 1), axis=0)
            selected = (begin + local).to(tl.int64)
            source_offsets = selected[None, :, None] * WIDTH + slots[:, None, None] * 128 + cols[None, None, :]
            gate = (
                tl.load(scores + source_offsets, valid[None, :, None], other=0)
                .reshape((POOL, ROWS * 128))
                .to(tl.float32)
            )
            gate += bias
            weight = tl.exp(gate - tl.max(gate, axis=0)[None, :])
            key = (
                tl.load(keys + source_offsets, valid[None, :, None], other=0).reshape((POOL, ROWS * 128)).to(tl.float32)
            )
            products = key * weight
            for stage in tl.static_range(0, LOG_POOL):
                products = _fold_pool_numerator(products, POOL >> stage, ROWS * 128)
            pooled = products.reshape((ROWS * 128,)) / tl.sum(weight, axis=0)
            rotated = pooled.to(tl.bfloat16).to(tl.float32).reshape((ROWS * 128,))
            for stage in tl.static_range(0, 7):
                distance = 1 << stage
                partner = tl.gather(rotated, flat ^ distance, axis=0)
                rotated = tl.where((flat & distance) == 0, rotated + partner, partner - rotated)
            rotated = (rotated * 0.08838834764831845).to(tl.bfloat16).to(tl.float32).reshape((ROWS, 128))
            scale = tl.maximum(tl.max(tl.abs(rotated), axis=1), 1e-4) * (1.0 / 448.0)
            if ROUND:
                scale_bits = (scale.to(tl.int32, bitcast=True) + 0x007FFFFF) & 0x7F800000
                scale = scale_bits.to(tl.float32, bitcast=True)
                inverse = (0x7F000000 - scale_bits).to(tl.float32, bitcast=True)
                quantized = rotated * inverse[:, None]
            else:
                quantized = rotated / scale[:, None]
            quantized = tl.minimum(tl.maximum(quantized, -448.0), 448.0)
            encoded = quantized.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
            if WRITE:
                loc = tl.load(locations + selected, valid, other=0).to(tl.int64)
                page = loc // PAGE
                slot = loc - page * PAGE
                key_offsets = page * PAGE_STRIDE + slot * 128
                scale_offsets = page * (PAGE_STRIDE // 4) + PAGE * 32 + slot
                tl.store(cache + key_offsets[:, None] + cols[None, :], encoded, valid[:, None])
                tl.store(scales + scale_offsets, scale, valid)
            if RETURN:
                tl.store(result_k + selected[:, None] * 128 + cols[None, :], encoded, valid[:, None])
                tl.store(result_s + selected, scale, valid)


def _kpool_compress_impl(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    write_mask: torch.Tensor | None = None,
    round_scale: bool = True,
    return_compressed: bool = False,
    write_cache: bool = True,
):
    """Compress ``pool_size`` tokens into one fp8 K and write at ``loc``.

    Args:
        kv_cache: indexer K cache ``[num_blocks, block_size, head_dim+4]`` uint8.
        slot_k: ``[n_pools, pool_size, head_dim]`` bf16 — raw per-token K.
        slot_score: ``[n_pools, pool_size, head_dim]`` bf16 — per-token gate score.
        ape: ``[pool_size, head_dim]`` fp32 — per-slot position bias.
        loc: ``[n_pools]`` int64 — flat physical slot per pool.
        pool_size: Number of tokens compressed into one cache entry.
        head_dim: Indexer head dimension.
        write_mask: ``[n_pools]`` bool — pools to write, or None for all.
        round_scale: Round each FP8 scale up to a power of two.
        return_compressed: Also return the compressed K and scales.
        write_cache: Write the compressed result into ``kv_cache``.

    """
    if kv_cache.storage_offset() % 4:
        raise ValueError("Cache storage offset must be 4-byte aligned")
    assert head_dim == INDEX_HEAD_DIM and pool_size in (4, 16)
    assert slot_k.ndim == 3 and slot_k.shape[1] == pool_size
    assert slot_score.shape == slot_k.shape
    assert ape.shape == slot_k.shape[1:]
    assert slot_k.shape[2] == head_dim
    assert slot_k.dtype == torch.bfloat16, "slot_k must use torch.bfloat16"
    assert slot_score.dtype == slot_k.dtype
    assert ape.dtype == torch.float32
    assert kv_cache.dtype == torch.uint8
    assert loc.dtype in (torch.int32, torch.int64)
    assert write_cache or return_compressed

    page_size = kv_cache.shape[1]
    buf = kv_cache
    slot_k = slot_k.contiguous()
    slot_score = slot_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()
    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=slot_k.device)
        has_write_mask = False
    else:
        assert write_mask.shape == (slot_k.shape[0],)
        write_mask = write_mask.contiguous()
        has_write_mask = True
        assert not return_compressed

    if slot_k.shape[0] == 0:
        if return_compressed:
            return (
                torch.empty(
                    (0, head_dim),
                    dtype=torch.float8_e4m3fn,
                    device=slot_k.device,
                ),
                torch.empty((0,), dtype=torch.float32, device=slot_k.device),
            )
        return None

    buf_fp8 = buf
    buf_fp32 = buf.view(torch.float32)
    # bytes per page (last dim of kv_cache) viewed as uint8
    buf_numel_per_page = buf.stride(0)

    if return_compressed:
        compressed_k = torch.empty(
            (slot_k.shape[0], head_dim),
            dtype=torch.uint8,
            device=slot_k.device,
        )
        compressed_scale = torch.empty((slot_k.shape[0],), dtype=torch.float32, device=slot_k.device)
    else:
        compressed_k = buf_fp8
        compressed_scale = buf_fp32

    n_rows = slot_k.shape[0]
    vector_cores = triton.runtime.driver.active.utils.get_device_properties(slot_k.device.index)["num_vectorcore"]
    row_budget = min(16, 64 // pool_size)
    ideal_rows = triton.next_power_of_2(triton.cdiv(n_rows, vector_cores * 2))
    rows_per_tile = min(row_budget, max(1 + has_write_mask, ideal_rows))
    chunk_rows = min(128, max(32 * has_write_mask, triton.next_power_of_2(n_rows)))
    chunks = triton.cdiv(n_rows, chunk_rows)
    shards = min(chunk_rows // rows_per_tile, max(1, vector_cores // chunks))
    _kpool_uniform[(min(chunks * shards, vector_cores),)](
        buf_fp8,
        buf_fp32,
        slot_k,
        slot_score,
        ape,
        loc,
        write_mask,
        compressed_k,
        compressed_scale,
        n_rows,
        POOL=pool_size,
        LOG_POOL=pool_size.bit_length() - 1,
        ROWS=rows_per_tile,
        CHUNK=chunk_rows,
        SHARDS=shards,
        PAGE=page_size,
        PAGE_STRIDE=buf_numel_per_page,
        HAS_MASK=has_write_mask,
        ROUND=round_scale,
        WRITE=write_cache,
        RETURN=return_compressed,
        enable_fp_fusion=False,
    )

    if return_compressed:
        return compressed_k.view(torch.float8_e4m3fn), compressed_scale
    return None


@torch.library.custom_op("mojo_npu_triton_a5::kpool_compress_write", mutates_args=("kv_cache",))
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


@torch.library.custom_op("mojo_npu_triton_a5::kpool_compress_write_return", mutates_args=("kv_cache",))
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


@torch.library.custom_op("mojo_npu_triton_a5::kpool_compress_return", mutates_args=())
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
