# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""kpool (key-pooling) Triton kernels for the sparse-attention indexer.

The cache stores POOLS (1 entry per ``pool_size`` consecutive tokens) rather
than individual tokens. ``compress_ratio == pool_size`` on the kv_cache_spec
makes the metadata builder emit pool-granular slot_mapping / seq_lens /
cu_seq_lens / page_table for free; this file supplies the compress-write
kernel (replacing ``indexer_k_quant_and_cache``) and the pool-level topk
helpers (select pools -> expand to tokens -> append tail).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton.language.extra.cann import extension as ext

# The GLM-5.3-Flash indexer head dimension is fixed at 128.
INDEX_HEAD_DIM = 128


# Hadamard-128 rotation


@triton.jit
def _hadamard128(x):
    lanes = tl.arange(0, 128)
    for group in tl.static_range(0, 3):
        low = 1 << (group * 2)
        high = low << 1
        low_partner = tl.gather(x, lanes ^ low, axis=0)
        high_partner = tl.gather(x, lanes ^ high, axis=0)
        diagonal = tl.gather(x, lanes ^ (low | high), axis=0)
        lower_half = (lanes & low) == 0
        row = tl.where(lower_half, x + low_partner, low_partner - x)
        paired_row = tl.where(lower_half, high_partner + diagonal, diagonal - high_partner)
        x = tl.where((lanes & high) == 0, row + paired_row, paired_row - row)
    partner = tl.gather(x, lanes ^ 64, axis=0)
    x = tl.where((lanes & 64) == 0, x + partner, partner - x)
    return x * 0.08838834764831845


# Fused pool compression and cache write.


@triton.jit
def _compress_one_pool(
    row,
    buf_fp8_ptr,
    buf_fp32_ptr,
    slot_k_ptr,
    slot_score_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    compressed_k_ptr,
    compressed_scale_ptr,
    slot_k_stride_0: tl.constexpr,
    slot_k_stride_1: tl.constexpr,
    slot_score_stride_0: tl.constexpr,
    slot_score_stride_1: tl.constexpr,
    ape_stride_0: tl.constexpr,
    SMALL_LOC: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    RETURN_COMPRESSED: tl.constexpr,
    WRITE_CACHE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per pool. softmax(slot_score+ape)-weighted sum of slot_k ->
    Hadamard-128 -> per-vector fp8 absmax quant -> write to cache at ``loc``."""
    do_write = True
    if HAS_WRITE_MASK:
        do_write = tl.load(write_mask_ptr + row) != 0

    offs = tl.arange(0, BLOCK_D)
    mask = (offs < HEAD_DIM) & do_write

    if POOL_SIZE == 4:
        # --- Pass 1: per-dim max over the pool (softmax numerical stability) ---
        max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
        for slot in tl.static_range(0, POOL_SIZE):
            score = tl.load(
                slot_score_ptr + row * slot_score_stride_0 + slot * slot_score_stride_1 + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(tl.float32)
            max_score = tl.maximum(max_score, score)

        # --- Pass 2: softmax-weighted sum of K ---
        acc = tl.full((BLOCK_D,), 0.0, tl.float32)
        denom = tl.full((BLOCK_D,), 0.0, tl.float32)
        for slot in tl.static_range(0, POOL_SIZE):
            score = tl.load(
                slot_score_ptr + row * slot_score_stride_0 + slot * slot_score_stride_1 + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(tl.float32)
            prob = tl.exp(score - max_score)
            denom += prob
            k = tl.load(
                slot_k_ptr + row * slot_k_stride_0 + slot * slot_k_stride_1 + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += k * prob

        x = acc / denom
    else:
        slots = tl.arange(0, POOL_SIZE)
        columns = offs[None, :]
        score = tl.load(
            slot_score_ptr + row * slot_score_stride_0 + slots[:, None] * slot_score_stride_1 + columns,
            mask=mask[None, :],
            other=0.0,
        ).to(tl.float32)
        score += tl.load(
            ape_ptr + slots[:, None] * ape_stride_0 + columns,
            mask=mask[None, :],
            other=0.0,
        )
        weights = tl.exp(score - tl.max(score, axis=0)[None, :])
        keys = tl.load(
            slot_k_ptr + row * slot_k_stride_0 + slots[:, None] * slot_k_stride_1 + columns,
            mask=mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x = tl.sum(keys * weights, axis=0) / tl.sum(weights, axis=0)
    x = x.to(tl.bfloat16).to(tl.float32)

    # Match the unfused pooled-K path's bf16 precision before quantization.
    x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

    # --- per-vector absmax fp8 quant ---
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max
    absmax = tl.max(tl.abs(x), axis=0)
    absmax = tl.maximum(absmax, 1e-4)
    if ROUND_SCALE:
        # The clamped ratio is positive and normal for finite inputs.
        ratio_bits = (absmax * fp8_max_inv).to(tl.int32, bitcast=True)
        exponent = (ratio_bits + 0x007FFFFF) & 0x7F800000
        scale = exponent.to(tl.float32, bitcast=True)
        inverse = (0x7F000000 - exponent).to(tl.float32, bitcast=True)
        quantized = x * inverse
    else:
        scale = absmax * fp8_max_inv
        quantized = x / scale
    quantized = tl.minimum(tl.maximum(quantized, -fp8_max), fp8_max)

    encoded = quantized.to(tl.float8e4nv).to(tl.uint8, bitcast=True)

    if WRITE_CACHE:
        loc = tl.load(loc_ptr + row, mask=do_write, other=0)
        if SMALL_LOC:
            loc = loc.to(tl.int32)
        loc_page_index = loc // PAGE_SIZE
        loc_token_offset_in_page = loc - loc_page_index * PAGE_SIZE
        out_k_offsets = loc_page_index * BUF_NUMEL_PER_PAGE + loc_token_offset_in_page * HEAD_DIM + offs
        out_s_offset = (
            loc_page_index * BUF_NUMEL_PER_PAGE // 4 + S_OFFSET_NBYTES_IN_PAGE // 4 + loc_token_offset_in_page
        )
        tl.store(buf_fp8_ptr + out_k_offsets, encoded, mask=mask)
        tl.store(buf_fp32_ptr + out_s_offset, scale, mask=do_write)

    if RETURN_COMPRESSED:
        tl.store(
            compressed_k_ptr + row * HEAD_DIM + offs,
            encoded,
            mask=offs < HEAD_DIM,
        )
        tl.store(compressed_scale_ptr + row, scale)


@triton.jit
def _kpool_softmax_rotate_write_cache_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    slot_k_ptr,
    slot_score_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    compressed_k_ptr,
    compressed_scale_ptr,
    slot_k_stride_0: tl.constexpr,
    slot_k_stride_1: tl.constexpr,
    slot_score_stride_0: tl.constexpr,
    slot_score_stride_1: tl.constexpr,
    ape_stride_0: tl.constexpr,
    n_pools,
    SINGLE_ROUND: tl.constexpr,
    SMALL_LOC: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    RETURN_COMPRESSED: tl.constexpr,
    WRITE_CACHE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per pool. softmax(slot_score+ape)-weighted sum of slot_k ->
    Hadamard-128 -> per-vector fp8 absmax quant -> write to cache at ``loc``."""
    pid = tl.program_id(0).to(tl.int32)
    if SINGLE_ROUND:
        row = pid
        _compress_one_pool(
            row,
            buf_fp8_ptr,
            buf_fp32_ptr,
            slot_k_ptr,
            slot_score_ptr,
            ape_ptr,
            loc_ptr,
            write_mask_ptr,
            compressed_k_ptr,
            compressed_scale_ptr,
            slot_k_stride_0,
            slot_k_stride_1,
            slot_score_stride_0,
            slot_score_stride_1,
            ape_stride_0,
            SMALL_LOC,
            PAGE_SIZE,
            BUF_NUMEL_PER_PAGE,
            POOL_SIZE,
            HEAD_DIM,
            S_OFFSET_NBYTES_IN_PAGE,
            ROUND_SCALE,
            HAS_WRITE_MASK,
            RETURN_COMPRESSED,
            WRITE_CACHE,
            BLOCK_D,
        )
    else:
        for row in range(pid, n_pools, tl.num_programs(0)):
            _compress_one_pool(
                row,
                buf_fp8_ptr,
                buf_fp32_ptr,
                slot_k_ptr,
                slot_score_ptr,
                ape_ptr,
                loc_ptr,
                write_mask_ptr,
                compressed_k_ptr,
                compressed_scale_ptr,
                slot_k_stride_0,
                slot_k_stride_1,
                slot_score_stride_0,
                slot_score_stride_1,
                ape_stride_0,
                SMALL_LOC,
                PAGE_SIZE,
                BUF_NUMEL_PER_PAGE,
                POOL_SIZE,
                HEAD_DIM,
                S_OFFSET_NBYTES_IN_PAGE,
                ROUND_SCALE,
                HAS_WRITE_MASK,
                RETURN_COMPRESSED,
                WRITE_CACHE,
                BLOCK_D,
            )


@triton.jit
def _kpool_dense_row_tile(
    cache,
    scales,
    k,
    score,
    ape,
    loc,
    n_pools,
    POOL: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    FULL_TILES: tl.constexpr,
    EXPLICIT_P16_NUM: tl.constexpr = False,
    CONTIG_TILE_INPUTS: tl.constexpr = False,
):
    lanes = tl.arange(0, ROWS * 128)
    columns = lanes % 128
    slots = tl.arange(0, POOL)
    row_ids = tl.arange(0, ROWS)
    bias = tl.load(ape + slots[:, None] * 128 + columns[None, :])
    for tile in range(tl.program_id(0), tl.cdiv(n_pools, ROWS), tl.num_programs(0)):
        rows = tile * ROWS + row_ids
        live = tl.full((ROWS,), True, tl.int1) if FULL_TILES else rows < n_pools
        lane_rows = tile * ROWS + lanes // 128
        offsets = lane_rows[None, :] * (POOL * 128) + slots[:, None] * 128 + columns[None, :]
        lane_live = tl.full((ROWS * 128,), True, tl.int1) if FULL_TILES else lane_rows < n_pools
        if CONTIG_TILE_INPUTS:
            physical_lanes = tl.arange(0, ROWS * POOL * 128)
            physical_rows = tile * ROWS + physical_lanes // (POOL * 128)
            physical_live = tl.full((ROWS * POOL * 128,), True, tl.int1) if FULL_TILES else physical_rows < n_pools
            physical_offsets = tile * ROWS * POOL * 128 + physical_lanes
            score_tile = tl.load(score + physical_offsets, physical_live, other=0).reshape((ROWS, POOL, 128))
            gate = tl.trans(score_tile, 1, 0, 2).reshape((POOL, ROWS * 128)).to(tl.float32)
        else:
            gate = tl.load(score + offsets, lane_live[None, :], other=0).to(tl.float32)
        gate += bias
        weights = tl.exp(gate - tl.max(gate, axis=0)[None, :])
        if CONTIG_TILE_INPUTS:
            key_tile = tl.load(k + physical_offsets, physical_live, other=0).reshape((ROWS, POOL, 128))
            keys = tl.trans(key_tile, 1, 0, 2).reshape((POOL, ROWS * 128)).to(tl.float32)
        else:
            keys = tl.load(k + offsets, lane_live[None, :], other=0).to(tl.float32)
        if POOL == 4:
            # Retain bulk GM loads, max and denominator. Only numerator lowering changes.
            products = keys * weights
            product0 = ext.extract_slice(products, (0, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            product1 = ext.extract_slice(products, (1, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            product2 = ext.extract_slice(products, (2, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            product3 = ext.extract_slice(products, (3, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            partial0 = (0.0 + product0) + product2
            partial1 = (0.0 + product1) + product3
            numerator = (0.0 + partial0) + partial1
            x = (numerator / tl.sum(weights, axis=0)).to(tl.bfloat16).to(tl.float32)
        elif EXPLICIT_P16_NUM:
            # Bulk GM loads and generic max/denominator keep their original layout.
            products = keys * weights
            left0 = ext.extract_slice(products, (0, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            right0 = ext.extract_slice(products, (8, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            a0 = (0.0 + left0) + right0
            left1 = ext.extract_slice(products, (1, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            right1 = ext.extract_slice(products, (9, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            a1 = (0.0 + left1) + right1
            left2 = ext.extract_slice(products, (2, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            right2 = ext.extract_slice(products, (10, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            a2 = (0.0 + left2) + right2
            left3 = ext.extract_slice(products, (3, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            right3 = ext.extract_slice(products, (11, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            a3 = (0.0 + left3) + right3
            left4 = ext.extract_slice(products, (4, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            right4 = ext.extract_slice(products, (12, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            a4 = (0.0 + left4) + right4
            left5 = ext.extract_slice(products, (5, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            right5 = ext.extract_slice(products, (13, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            a5 = (0.0 + left5) + right5
            left6 = ext.extract_slice(products, (6, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            right6 = ext.extract_slice(products, (14, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            a6 = (0.0 + left6) + right6
            left7 = ext.extract_slice(products, (7, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            right7 = ext.extract_slice(products, (15, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            a7 = (0.0 + left7) + right7
            b0 = (0.0 + a0) + a4
            b1 = (0.0 + a1) + a5
            b2 = (0.0 + a2) + a6
            b3 = (0.0 + a3) + a7
            c0 = (0.0 + b0) + b2
            c1 = (0.0 + b1) + b3
            numerator = (0.0 + c0) + c1
            denominator = tl.full((ROWS * 128,), 0.0, tl.float32)
            weight0 = ext.extract_slice(weights, (0, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            denominator = denominator + weight0
            weight1 = ext.extract_slice(weights, (1, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            denominator = denominator + weight1
            weight2 = ext.extract_slice(weights, (2, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            denominator = denominator + weight2
            weight3 = ext.extract_slice(weights, (3, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            denominator = denominator + weight3
            weight4 = ext.extract_slice(weights, (4, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            denominator = denominator + weight4
            weight5 = ext.extract_slice(weights, (5, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            denominator = denominator + weight5
            weight6 = ext.extract_slice(weights, (6, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            denominator = denominator + weight6
            weight7 = ext.extract_slice(weights, (7, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            denominator = denominator + weight7
            weight8 = ext.extract_slice(weights, (8, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            denominator = denominator + weight8
            weight9 = ext.extract_slice(weights, (9, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            denominator = denominator + weight9
            weight10 = ext.extract_slice(weights, (10, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            denominator = denominator + weight10
            weight11 = ext.extract_slice(weights, (11, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            denominator = denominator + weight11
            weight12 = ext.extract_slice(weights, (12, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            denominator = denominator + weight12
            weight13 = ext.extract_slice(weights, (13, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            denominator = denominator + weight13
            weight14 = ext.extract_slice(weights, (14, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            denominator = denominator + weight14
            weight15 = ext.extract_slice(weights, (15, 0), (1, ROWS * 128), (1, 1)).reshape((ROWS * 128,))
            denominator = denominator + weight15
            x = (numerator / denominator).to(tl.bfloat16).to(tl.float32)
        else:
            x = (tl.sum(keys * weights, axis=0) / tl.sum(weights, axis=0)).to(tl.bfloat16).to(tl.float32)
        for stage in tl.static_range(0, 7):
            stride = 1 << stage
            partner = tl.gather(x, lanes ^ stride, axis=0)
            x = tl.where((lanes & stride) == 0, x + partner, partner - x)
        x = tl.reshape((x * 0.08838834764831845).to(tl.bfloat16).to(tl.float32), (ROWS, 128))
        peak = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
        bits = (peak * (1.0 / 448.0)).to(tl.int32, bitcast=True)
        exponent = (bits + 0x007FFFFF) & 0x7F800000
        scale = exponent.to(tl.float32, bitcast=True)
        inverse = (0x7F000000 - exponent).to(tl.float32, bitcast=True)
        quant = tl.minimum(tl.maximum(x * inverse[:, None], -448.0), 448.0)
        encoded = quant.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        loc_rows = tl.load(loc + rows, live, other=0).to(tl.int32)
        first_page = 0
        first_slot = 0
        contiguous = False
        if FULL_TILES:
            first = tl.min(loc_rows - row_ids, axis=0)
            first_page = first // PAGE
            first_slot = first - first_page * PAGE
            contiguous = (tl.max(loc_rows - row_ids, axis=0) == first) & (first_slot + ROWS <= PAGE)
        if contiguous:
            k_base = first_page * PAGE_STRIDE + first_slot * 128
            s_base = first_page * (PAGE_STRIDE // 4) + PAGE * 32 + first_slot
            tl.store(cache + k_base + lanes, tl.reshape(encoded, (ROWS * 128,)))
            tl.store(scales + s_base + row_ids, scale)
        else:
            page = loc_rows // PAGE
            in_page = loc_rows - page * PAGE
            k_off = page * PAGE_STRIDE + in_page * 128
            cols = tl.arange(0, 128)
            tl.store(cache + k_off[:, None] + cols[None, :], encoded, live[:, None])
            tl.store(scales + page * (PAGE_STRIDE // 4) + PAGE * 32 + in_page, scale, live)


@triton.jit
def _kpool_split_pool(
    workspace,
    k,
    score,
    ape,
    n_pools,
    POOL: tl.constexpr,
    ROWS: tl.constexpr,
    FULL_TILES: tl.constexpr,
    EVEN_GRID: tl.constexpr,
):
    lanes = tl.arange(0, ROWS * 64)
    lane_columns = lanes % 64
    slots = tl.arange(0, POOL)
    pid = tl.program_id(0)
    if EVEN_GRID:
        columns = pid % 2 * 64 + lane_columns
        bias = tl.load(ape + slots[:, None] * 128 + columns[None, :])
    for tile in range(pid, tl.cdiv(n_pools, ROWS) * 2, tl.num_programs(0)):
        if not EVEN_GRID:
            columns = tile % 2 * 64 + lane_columns
            bias = tl.load(ape + slots[:, None] * 128 + columns[None, :])
        lane_rows = (tile // 2) * ROWS + lanes // 64
        offsets = lane_rows[None, :] * (POOL * 128) + slots[:, None] * 128 + columns[None, :]
        lane_live = tl.full((ROWS * 64,), True, tl.int1) if FULL_TILES else lane_rows < n_pools
        gate = tl.load(score + offsets, lane_live[None, :], other=0).to(tl.float32)
        gate += bias
        weights = tl.exp(gate - tl.max(gate, axis=0)[None, :])
        keys = tl.load(k + offsets, lane_live[None, :], other=0).to(tl.float32)
        products = keys * weights
        left0 = ext.extract_slice(products, (0, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        right0 = ext.extract_slice(products, (8, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        a0 = 0.0 + left0 + right0
        left1 = ext.extract_slice(products, (1, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        right1 = ext.extract_slice(products, (9, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        a1 = 0.0 + left1 + right1
        left2 = ext.extract_slice(products, (2, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        right2 = ext.extract_slice(products, (10, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        a2 = 0.0 + left2 + right2
        left3 = ext.extract_slice(products, (3, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        right3 = ext.extract_slice(products, (11, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        a3 = 0.0 + left3 + right3
        left4 = ext.extract_slice(products, (4, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        right4 = ext.extract_slice(products, (12, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        a4 = 0.0 + left4 + right4
        left5 = ext.extract_slice(products, (5, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        right5 = ext.extract_slice(products, (13, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        a5 = 0.0 + left5 + right5
        left6 = ext.extract_slice(products, (6, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        right6 = ext.extract_slice(products, (14, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        a6 = 0.0 + left6 + right6
        left7 = ext.extract_slice(products, (7, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        right7 = ext.extract_slice(products, (15, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        a7 = 0.0 + left7 + right7
        b0 = 0.0 + a0 + a4
        b1 = 0.0 + a1 + a5
        b2 = 0.0 + a2 + a6
        b3 = 0.0 + a3 + a7
        c0 = 0.0 + b0 + b2
        c1 = 0.0 + b1 + b3
        numerator = 0.0 + c0 + c1
        denominator = tl.full((ROWS * 64,), 0.0, tl.float32)
        weight0 = ext.extract_slice(weights, (0, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        denominator = denominator + weight0
        weight1 = ext.extract_slice(weights, (1, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        denominator = denominator + weight1
        weight2 = ext.extract_slice(weights, (2, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        denominator = denominator + weight2
        weight3 = ext.extract_slice(weights, (3, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        denominator = denominator + weight3
        weight4 = ext.extract_slice(weights, (4, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        denominator = denominator + weight4
        weight5 = ext.extract_slice(weights, (5, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        denominator = denominator + weight5
        weight6 = ext.extract_slice(weights, (6, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        denominator = denominator + weight6
        weight7 = ext.extract_slice(weights, (7, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        denominator = denominator + weight7
        weight8 = ext.extract_slice(weights, (8, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        denominator = denominator + weight8
        weight9 = ext.extract_slice(weights, (9, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        denominator = denominator + weight9
        weight10 = ext.extract_slice(weights, (10, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        denominator = denominator + weight10
        weight11 = ext.extract_slice(weights, (11, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        denominator = denominator + weight11
        weight12 = ext.extract_slice(weights, (12, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        denominator = denominator + weight12
        weight13 = ext.extract_slice(weights, (13, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        denominator = denominator + weight13
        weight14 = ext.extract_slice(weights, (14, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        denominator = denominator + weight14
        weight15 = ext.extract_slice(weights, (15, 0), (1, ROWS * 64), (1, 1)).reshape((ROWS * 64,))
        denominator = denominator + weight15
        x = (numerator / denominator).to(tl.bfloat16).to(tl.float32)
        tl.store(workspace + lane_rows * 128 + columns, x.to(tl.bfloat16), lane_live)


@triton.jit
def _kpool_split_rotate(
    cache,
    scales,
    workspace,
    loc,
    n_pools,
    PAGE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    FULL_TILES: tl.constexpr,
):
    lanes = tl.arange(0, ROWS * 128)
    row_ids = tl.arange(0, ROWS)
    for tile in range(tl.program_id(0), tl.cdiv(n_pools, ROWS), tl.num_programs(0)):
        rows = tile * ROWS + row_ids
        live = tl.full((ROWS,), True, tl.int1) if FULL_TILES else rows < n_pools
        lane_rows = tile * ROWS + lanes // 128
        lane_live = tl.full((ROWS * 128,), True, tl.int1) if FULL_TILES else lane_rows < n_pools
        x = tl.load(workspace + lane_rows * 128 + lanes % 128, lane_live, other=0).to(tl.float32)
        for stage in tl.static_range(0, 7):
            stride = 1 << stage
            partner = tl.gather(x, lanes ^ stride, axis=0)
            x = tl.where((lanes & stride) == 0, x + partner, partner - x)
        x = tl.reshape((x * 0.08838834764831845).to(tl.bfloat16).to(tl.float32), (ROWS, 128))
        peak = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
        bits = (peak * (1.0 / 448.0)).to(tl.int32, bitcast=True)
        exponent = (bits + 0x007FFFFF) & 0x7F800000
        scale = exponent.to(tl.float32, bitcast=True)
        inverse = (0x7F000000 - exponent).to(tl.float32, bitcast=True)
        quant = tl.minimum(tl.maximum(x * inverse[:, None], -448.0), 448.0)
        encoded = quant.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        loc_rows = tl.load(loc + rows, live, other=0).to(tl.int32)
        first_page = 0
        first_slot = 0
        contiguous = False
        if FULL_TILES:
            first = tl.min(loc_rows - row_ids, axis=0)
            first_page = first // PAGE
            first_slot = first - first_page * PAGE
            contiguous = (tl.max(loc_rows - row_ids, axis=0) == first) & (first_slot + ROWS <= PAGE)
        if contiguous:
            k_base = first_page * PAGE_STRIDE + first_slot * 128
            s_base = first_page * (PAGE_STRIDE // 4) + PAGE * 32 + first_slot
            tl.store(cache + k_base + lanes, tl.reshape(encoded, (ROWS * 128,)))
            tl.store(scales + s_base + row_ids, scale)
        else:
            page = loc_rows // PAGE
            in_page = loc_rows - page * PAGE
            k_off = page * PAGE_STRIDE + in_page * 128
            cols = tl.arange(0, 128)
            tl.store(cache + k_off[:, None] + cols[None, :], encoded, live[:, None])
            tl.store(scales + page * (PAGE_STRIDE // 4) + PAGE * 32 + in_page, scale, live)


def kpool_compress_and_write_cache_fwd(
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
    s_offset_nbytes_in_page = page_size * head_dim

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

    properties = triton.runtime.driver.active.utils.get_device_properties(slot_k.device.index)
    grid = min(slot_k.shape[0], properties["num_vectorcore"])
    if (
        slot_k.shape[0] >= 64
        and not has_write_mask
        and write_cache
        and not return_compressed
        and round_scale
        and kv_cache.shape[0] * kv_cache.stride(0) < 2**31
    ):
        rows = 16 if pool_size == 4 else 8
        if slot_k.shape[0] < 512:
            rows = 2
        tiled_grid = min(triton.cdiv(slot_k.shape[0], rows), grid)
        if pool_size == 16 and slot_k.shape[0] >= 2048:
            n_pools = slot_k.shape[0]
            split_rows = 16
            split_grid = min(triton.cdiv(n_pools, split_rows), grid)
            workspace = torch.empty((n_pools, 128), dtype=torch.bfloat16, device=slot_k.device)
            pool_rows = 16
            pool_grid = min(triton.cdiv(n_pools, pool_rows) * 2, grid)
            _kpool_split_pool[(pool_grid,)](
                workspace,
                slot_k,
                slot_score,
                ape,
                n_pools,
                POOL=pool_size,
                ROWS=pool_rows,
                FULL_TILES=n_pools % pool_rows == 0,
                EVEN_GRID=pool_grid % 2 == 0,
            )
            _kpool_split_rotate[(split_grid,)](
                buf_fp8,
                buf_fp32,
                workspace,
                loc,
                n_pools,
                PAGE=page_size,
                PAGE_STRIDE=buf_numel_per_page,
                ROWS=split_rows,
                FULL_TILES=n_pools % split_rows == 0,
            )
            return None
        _kpool_dense_row_tile[(tiled_grid,)](
            buf_fp8,
            buf_fp32,
            slot_k,
            slot_score,
            ape,
            loc,
            slot_k.shape[0],
            POOL=pool_size,
            PAGE=page_size,
            PAGE_STRIDE=buf_numel_per_page,
            ROWS=rows,
            FULL_TILES=slot_k.shape[0] % rows == 0,
            EXPLICIT_P16_NUM=pool_size == 16 and slot_k.shape[0] == 512,
            CONTIG_TILE_INPUTS=pool_size == 4 and slot_k.shape[0] == 4096,
        )
        return None

    _kpool_softmax_rotate_write_cache_kernel[(grid,)](
        buf_fp8,
        buf_fp32,
        slot_k,
        slot_score,
        ape,
        loc,
        write_mask,
        compressed_k,
        compressed_scale,
        slot_k.stride(0),
        slot_k.stride(1),
        slot_score.stride(0),
        slot_score.stride(1),
        ape.stride(0),
        slot_k.shape[0],
        SINGLE_ROUND=grid == slot_k.shape[0],
        SMALL_LOC=kv_cache.shape[0] * kv_cache.stride(0) < 2**31,
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        POOL_SIZE=slot_k.shape[1],
        HEAD_DIM=head_dim,
        S_OFFSET_NBYTES_IN_PAGE=s_offset_nbytes_in_page,
        ROUND_SCALE=round_scale,
        HAS_WRITE_MASK=has_write_mask,
        RETURN_COMPRESSED=return_compressed,
        WRITE_CACHE=write_cache,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )

    if return_compressed:
        return compressed_k.view(torch.float8_e4m3fn), compressed_scale
    return None
