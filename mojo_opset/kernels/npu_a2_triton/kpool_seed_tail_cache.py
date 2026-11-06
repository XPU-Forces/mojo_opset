"""Triton-Ascend KPool tail seeding for A2 (D=128, KPOOL=4)."""

from functools import lru_cache

import torch
import triton
import triton.backends.ascend.runtime  # noqa: F401
import triton.language as tl
import triton.runtime.driver as driver


@triton.jit
def _first_selected_lane(bitmap):
    # The lowest set bit is an exact positive power of two in FP32.
    low = (bitmap & -bitmap).to(tl.float32)
    return ((low.to(tl.int32, bitcast=True) >> 23) & 255) - 127


@triton.jit
def _copy_short_tile(
    KEY,
    GATE,
    SLOT,
    BACKING,
    tile,
    N: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    CHANNEL_STRIDE: tl.constexpr,
    OFFSET: tl.constexpr,
    SLOT_BITS: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    lanes = tl.arange(0, ROW_TILE)
    bits = 1 << lanes
    cols = tl.arange(0, 128)
    slot_stride: tl.constexpr = 2 if SLOT_BITS == 64 else 1
    row_base = tile * ROW_TILE
    rows = row_base + lanes
    if SLOT_BITS == 64:
        words = tl.arange(0, 2 * ROW_TILE)
        word_rows = 2 * row_base + words
        current_words = tl.load(SLOT + word_rows, mask=word_rows < 2 * N, other=-1)
        ahead_words = tl.load(SLOT + word_rows + 8, mask=word_rows + 8 < 2 * N, other=-1)
        raw, high = tl.split(tl.reshape(current_words, (ROW_TILE, 2)))
        ahead, ahead_high = tl.split(tl.reshape(ahead_words, (ROW_TILE, 2)))
        valid = high >= 0
        ahead_valid = ahead_high >= 0
    else:
        raw = tl.load(SLOT + rows, mask=rows < N, other=-1)
        ahead = tl.load(SLOT + rows + 4, mask=rows + 4 < N, other=-1)
        valid = (raw >> (SLOT_BITS - 1)).to(tl.int32) == 0
        ahead_valid = (ahead >> (SLOT_BITS - 1)).to(tl.int32) == 0
    blocks = tl.maximum(raw.to(tl.int32), 0) // RING
    ahead_blocks = tl.maximum(ahead.to(tl.int32), 0) // RING
    selected = valid & ~(ahead_valid & (blocks == ahead_blocks))
    bitmap = tl.sum(tl.where(selected, bits, 0), axis=0)

    if RING >= 4:
        if bitmap != 0:
            first = _first_selected_lane(bitmap)
            if bitmap == (15 << first):
                row = row_base + first
                slot = tl.load(SLOT + row * slot_stride).to(tl.int32)
                block = slot // RING
                pos = slot - block * RING
                contiguous = True
                for j in tl.static_range(1, 4):
                    expected = block * RING + (pos + j) % RING
                    contiguous = contiguous & (tl.load(SLOT + (row + j) * slot_stride).to(tl.int32) == expected)
                if contiguous:
                    span = tl.arange(0, 512)
                    src = row.to(tl.int64) * 128
                    dst = OFFSET + block.to(tl.int64) * BLOCK_STRIDE + pos * 128
                    width = (RING - pos) * 128
                    key = tl.load(KEY + src + span)
                    gate = tl.load(GATE + src + span)
                    tl.store(BACKING + dst + span, key, mask=span < width)
                    tl.store(BACKING + dst + CHANNEL_STRIDE + span, gate, mask=span < width)
                    tl.store(BACKING + dst - RING * 128 + span, key, mask=span >= width)
                    tl.store(BACKING + dst - RING * 128 + CHANNEL_STRIDE + span, gate, mask=span >= width)
                    bitmap = 0

    while bitmap != 0:
        row = row_base + _first_selected_lane(bitmap)
        slot = tl.load(SLOT + row * slot_stride).to(tl.int32)
        block = slot // RING
        pos = slot - block * RING
        src = row.to(tl.int64) * 128 + cols
        dst = OFFSET + block.to(tl.int64) * BLOCK_STRIDE + pos * 128 + cols
        tl.store(BACKING + dst, tl.load(KEY + src))
        tl.store(BACKING + dst + CHANNEL_STRIDE, tl.load(GATE + src))
        bitmap = bitmap & (bitmap - 1)


@triton.jit
def _copy_short_four(
    KEY,
    GATE,
    BACKING,
    row,
    slot,
    RING: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    CHANNEL_STRIDE: tl.constexpr,
    OFFSET: tl.constexpr,
):
    # Same masked four-row movement used by the original short version.
    block = slot // RING
    pos = slot - block * RING
    span = tl.arange(0, 512)
    src = row.to(tl.int64) * 128
    dst = OFFSET + block.to(tl.int64) * BLOCK_STRIDE + pos * 128
    width = (RING - pos) * 128
    key = tl.load(KEY + src + span)
    gate = tl.load(GATE + src + span)
    tl.store(BACKING + dst + span, key, mask=span < width)
    tl.store(BACKING + dst + CHANNEL_STRIDE + span, gate, mask=span < width)
    tl.store(BACKING + dst - RING * 128 + span, key, mask=span >= width)
    tl.store(BACKING + dst - RING * 128 + CHANNEL_STRIDE + span, gate, mask=span >= width)


@triton.jit
def _seed_tail_a2_tile(
    KEY,
    GATE,
    SLOT,
    BACKING,
    N: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    CHANNEL_STRIDE: tl.constexpr,
    OFFSET: tl.constexpr,
    SLOT_BITS: tl.constexpr,
    ROW_TILE: tl.constexpr,
    STEP: tl.constexpr,
):
    GROUP_ROWS: tl.constexpr = 256
    groups: tl.constexpr = (N + GROUP_ROWS - 1) // GROUP_ROWS
    chunk: tl.constexpr = (groups + STEP - 1) // STEP
    first_group = tl.program_id(0) * chunk
    lanes = tl.arange(0, GROUP_ROWS)
    slot_stride: tl.constexpr = 2 if SLOT_BITS == 64 else 1
    for group in range(first_group, tl.minimum(first_group + chunk, groups)):
        base = group * GROUP_ROWS
        rows = base + lanes
        if SLOT_BITS == 64:
            words = tl.arange(0, 2 * GROUP_ROWS)
            word_rows = 2 * base + words
            current_words = tl.load(SLOT + word_rows, mask=word_rows < 2 * N, other=-1)
            ahead_words = tl.load(SLOT + word_rows + 8, mask=word_rows + 8 < 2 * N, other=-1)
            raw, high = tl.split(tl.reshape(current_words, (GROUP_ROWS, 2)))
            ahead, ahead_high = tl.split(tl.reshape(ahead_words, (GROUP_ROWS, 2)))
            valid = high >= 0
            ahead_valid = ahead_high >= 0
        else:
            raw = tl.load(SLOT + rows, mask=rows < N, other=-1)
            ahead = tl.load(SLOT + rows + 4, mask=rows + 4 < N, other=-1)
            valid = (raw >> (SLOT_BITS - 1)).to(tl.int32) == 0
            ahead_valid = (ahead >> (SLOT_BITS - 1)).to(tl.int32) == 0
        blocks = tl.maximum(raw.to(tl.int32), 0) // RING
        ahead_blocks = tl.maximum(ahead.to(tl.int32), 0) // RING
        selected = valid & ~(ahead_valid & (blocks == ahead_blocks))
        first = tl.min(tl.where(selected, lanes, GROUP_ROWS), axis=0)
        last = tl.max(tl.where(selected, lanes, -1), axis=0)
        second = last - 3
        expected = ((lanes >= first) & (lanes < first + 4)) | ((lanes >= second) & (lanes <= last))
        matches = tl.max((selected ^ expected).to(tl.int32), axis=0) == 0
        done = False
        if (second >= first + 4) & matches:
            row_a = base + first
            row_b = base + second
            slot_a = tl.load(SLOT + row_a * slot_stride).to(tl.int32)
            slot_b = tl.load(SLOT + row_b * slot_stride).to(tl.int32)
            block_a = slot_a // RING
            block_b = slot_b // RING
            pos_a = slot_a - block_a * RING
            pos_b = slot_b - block_b * RING
            contiguous = True
            for j in tl.static_range(1, 4):
                expected_a = block_a * RING + (pos_a + j) % RING
                expected_b = block_b * RING + (pos_b + j) % RING
                contiguous = contiguous & (tl.load(SLOT + (row_a + j) * slot_stride).to(tl.int32) == expected_a)
                contiguous = contiguous & (tl.load(SLOT + (row_b + j) * slot_stride).to(tl.int32) == expected_b)
            if contiguous & (block_a != block_b):
                _copy_short_four(KEY, GATE, BACKING, row_a, slot_a, RING, BLOCK_STRIDE, CHANNEL_STRIDE, OFFSET)
                _copy_short_four(KEY, GATE, BACKING, row_b, slot_b, RING, BLOCK_STRIDE, CHANNEL_STRIDE, OFFSET)
                done = True
        if first < GROUP_ROWS and not done:
            for tile in range(group * 16, tl.minimum(group * 16 + 16, (N + ROW_TILE - 1) // ROW_TILE)):
                _copy_short_tile(
                    KEY, GATE, SLOT, BACKING, tile, N, RING, BLOCK_STRIDE, CHANNEL_STRIDE, OFFSET, SLOT_BITS, ROW_TILE
                )


@lru_cache(maxsize=None)
def _vector_core_count(device_index):
    return int(driver.active.utils.get_device_properties(device_index)["num_vectorcore"])


def kpool_seed_tail_cache_fwd(tail_kv_cache, key, gate_score, tslot, kpool=4, head_dim=128):
    """Update cache in place; valid slots must fit the signed-int32 slot range."""
    if kpool != 4 or head_dim != 128:
        raise ValueError("This GLM-5.3-Flash kernel fixes kpool=4 and head_dim=128")
    if key.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("K, gate and cache must share BF16 or FP16 dtype")
    if gate_score.dtype != key.dtype or tail_kv_cache.dtype != key.dtype:
        raise ValueError("K, gate and cache must have identical dtype")
    if not (key.device == gate_score.device == tail_kv_cache.device == tslot.device):
        raise ValueError("All tensors must be on the same NPU")
    if key.device.type != "npu":
        raise ValueError("This implementation requires an Ascend NPU")
    if key.ndim != 2 or key.shape[1] != 128 or gate_score.shape != key.shape:
        raise ValueError("K and gate must have shape [N, 128]")
    if tail_kv_cache.ndim != 4 or tail_kv_cache.shape[1] != 2 or tail_kv_cache.shape[3] != 128:
        raise ValueError("Cache must have shape [num_blocks, 2, RING, 128]")
    if tail_kv_cache.shape[2] < 4 or tail_kv_cache.stride(3) != 1 or tail_kv_cache.stride(2) != 128:
        raise ValueError("RING >= 4 and contiguous cache rows are required")
    if tail_kv_cache.shape[0] * tail_kv_cache.shape[2] >= 2**31:
        raise ValueError("The addressable slot range must fit signed int32")
    if tslot.ndim != 1 or tslot.shape[0] != key.shape[0] or tslot.dtype not in (torch.int32, torch.int64):
        raise ValueError("tslot must be signed int32/int64 with shape [N]")
    if not key.is_contiguous() or not gate_score.is_contiguous() or not tslot.is_contiguous():
        raise ValueError("K, gate and tslot must be contiguous")
    n = key.shape[0]
    if n == 0:
        return None
    groups = triton.cdiv(n, 256)
    vector_cores = _vector_core_count(key.device.index)
    chunk_groups = triton.cdiv(groups, min(vector_cores, groups))
    programs = triton.cdiv(groups, chunk_groups)
    # A tensor view already points at its storage offset: do not add it twice.
    _seed_tail_a2_tile[(programs,)](
        key,
        gate_score,
        tslot.view(torch.int32) if tslot.element_size() == 8 else tslot,
        tail_kv_cache,
        n,
        tail_kv_cache.shape[2],
        tail_kv_cache.stride(0),
        tail_kv_cache.stride(1),
        0,
        tslot.element_size() * 8,
        ROW_TILE=16,
        STEP=programs,
    )
    return None
