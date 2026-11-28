"""Triton-Ascend KPool tail seeding for A2 (D=128, KPOOL=4)."""

from functools import lru_cache

import torch
import triton
import triton.backends.ascend.runtime  # noqa: F401
import triton.language as tl
import triton.runtime.driver as driver

from triton.language.extra.cann.extension import extract_slice
from triton.language.extra.cann.extension import get_element


@triton.jit
def _first_selected_lane(bitmap):
    # The lowest set bit is an exact positive power of two in FP32.
    low = (bitmap & -bitmap).to(tl.float32)
    return ((low.to(tl.int32, bitcast=True) >> 23) & 255) - 127


@triton.jit
def _or_predicates(lhs, rhs):
    return lhs | rhs


@triton.jit
def _exact_pow2_bits(exponents):
    # We only build one-hot values with exponents 0..15. FP32 is exact here.
    ieee = (exponents + 127) << 23
    return ieee.to(tl.float32, bitcast=True).to(tl.int32)


@triton.jit
def _copy_selected_windows(
    KEY, GATE, BACKING, base, raw, blocks, selected,
    RING: tl.constexpr, BLOCK_STRIDE: tl.constexpr, CHANNEL_STRIDE: tl.constexpr,
    OFFSET: tl.constexpr, ROW_TILE: tl.constexpr, GROUP_ROWS: tl.constexpr,
):
    lanes = tl.arange(0, GROUP_ROWS)
    span = tl.arange(0, 4 * 128)
    lane_bits = _exact_pow2_bits(lanes % ROW_TILE)
    next_slot = tl.gather(raw, tl.minimum(lanes + 1, GROUP_ROWS - 1), axis=0)
    ring_end = raw - blocks * RING == RING - 1
    span_end = (next_slot != raw + 1) | ring_end
    # A 32-bit word carries 16 hit bits and 16 end-of-span bits.
    packed = tl.where(selected, lane_bits, 0).to(tl.uint32)
    packed |= tl.where(span_end, lane_bits, 0).to(tl.uint32) << 16
    descriptors = tl.sum(tl.reshape(packed, (GROUP_ROWS // ROW_TILE, ROW_TILE)), axis=1)
    # The four UB words each describe 16 rows. Form a single window cursor
    # after the Vector reduction, using only fixed descriptor indices.
    d0 = get_element(descriptors, (0,)).to(tl.uint32).to(tl.uint64)
    d1 = get_element(descriptors, (1,)).to(tl.uint32).to(tl.uint64)
    d2 = get_element(descriptors, (2,)).to(tl.uint32).to(tl.uint64)
    d3 = get_element(descriptors, (3,)).to(tl.uint32).to(tl.uint64)
    remaining = (d0 & 0xFFFF) | ((d1 & 0xFFFF) << 16) | ((d2 & 0xFFFF) << 32) | ((d3 & 0xFFFF) << 48)
    ends = (d0 >> 16) | ((d1 >> 16) << 16) | ((d2 >> 16) << 32) | ((d3 >> 16) << 48)
    ends |= 0x8000800080008000
    one = tl.full((), 1, tl.uint64)
    while remaining != 0:
        lowest = remaining & -remaining
        high_bit = (lowest >> 32).to(tl.uint32)
        word = lowest.to(tl.uint32) | high_bit
        exponent = (word.to(tl.float32).to(tl.int32, bitcast=True) >> 23) & 255
        lane = exponent - 127 + 32 * (high_bit != 0).to(tl.int32)
        # Only the next three potential stops affect a length capped at four.
        stop_code = ((ends | ~(remaining >> 1)) >> lane).to(tl.uint32) & 7
        length = (((0x1213 >> (2 * stop_code)) & 3) + 1).to(tl.int32)
        lane_index = tl.full((1,), lane, tl.int32)
        slot = tl.sum(tl.gather(raw, lane_index, axis=0), axis=0)
        block = slot // RING
        pos = slot - block * RING
        src = (base + lane).to(tl.int64) * 128 + span
        dst = OFFSET + block.to(tl.int64) * BLOCK_STRIDE + pos * 128 + span
        mask = span < length * 128
        tl.store(BACKING + dst, tl.load(KEY + src, mask, other=0), mask)
        tl.store(BACKING + dst + CHANNEL_STRIDE, tl.load(GATE + src, mask, other=0), mask)
        remaining &= ~(((one << length) - 1) << lane)


@triton.jit
def _load_slot_window(
    SLOT, tile, owned, N, WORDS: tl.constexpr,
    TILE_WORDS: tl.constexpr,
):
    lanes = tl.arange(0, 2 * TILE_WORDS)
    offsets = tile * TILE_WORDS + lanes
    return tl.load(SLOT + offsets, owned & (lanes < TILE_WORDS + 4 * WORDS)
                   & (offsets < N * WORDS), other=-1)


@triton.jit
def _kpool_tail_seed_kernel(
    KEY,
    GATE,
    SLOT,
    BACKING,
    N,
    RING: tl.constexpr,
    RING_SHIFT: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    CHANNEL_STRIDE: tl.constexpr,
    OFFSET: tl.constexpr,
    SLOT_BITS: tl.constexpr,
    ROW_TILE: tl.constexpr,
    GROUPS_PER_SCAN: tl.constexpr,
    INDEX64: tl.constexpr,
):
    GROUP_ROWS: tl.constexpr = 256
    TILE_ROWS: tl.constexpr = 64
    WORDS: tl.constexpr = 2 if SLOT_BITS == 64 else 1
    HALO_WORDS: tl.constexpr = 4 * WORDS
    SCAN_ROWS: tl.constexpr = GROUP_ROWS * GROUPS_PER_SCAN
    SCAN_WORDS: tl.constexpr = SCAN_ROWS * WORDS
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    if INDEX64:
        pid = pid.to(tl.int64)
        N = N.to(tl.int64)
    if GROUPS_PER_SCAN == 1:
        groups = tl.cdiv(N, GROUP_ROWS)
        chunk = tl.cdiv(groups, programs)
        first_group = pid * chunk
        last_group = tl.minimum(first_group + chunk, groups)
    else:
        # Boundaries partition complete scan tiles. The masked loads still own
        # the physical tail and expose the next four valid slots as lookahead.
        scans = tl.cdiv(N, SCAN_ROWS)
        first_group = (scans * pid // programs) * GROUPS_PER_SCAN
        last_group = (scans * (pid + 1) // programs) * GROUPS_PER_SCAN
    NUM_WINDOWS: tl.constexpr = SCAN_ROWS // TILE_ROWS
    window_ids = tl.arange(0, NUM_WINDOWS)
    if GROUPS_PER_SCAN == 1:
        tile_bits = 1 << window_ids
    else:
        tile_bits = tl.full((NUM_WINDOWS,), 1, tl.uint32) << window_ids.to(tl.uint32)
    word_lanes = tl.arange(0, 2 * SCAN_WORDS)
    for group in range(first_group, last_group, GROUPS_PER_SCAN):
        if GROUPS_PER_SCAN == 1:
            wide = _load_slot_window(SLOT, group, True, N, WORDS, SCAN_WORDS)
        else:
            offsets = group * GROUP_ROWS * WORDS + word_lanes
            wide = tl.load(SLOT + offsets, (word_lanes < SCAN_WORDS + HALO_WORDS)
                           & (offsets < N * WORDS), other=-1)
        now = extract_slice(wide, (0,), (SCAN_WORDS,), (1,))
        ahead = extract_slice(wide, (HALO_WORDS,), (SCAN_WORDS,), (1,))
        if SLOT_BITS == 64:
            raw, high = tl.split(tl.reshape(now, (SCAN_ROWS, 2)))
            after, after_high = tl.split(tl.reshape(ahead, (SCAN_ROWS, 2)))
            valid = high >= 0
            after_valid = after_high >= 0
        else:
            raw = now
            after = ahead
            valid = raw >= 0
            after_valid = after >= 0
        ODD_RING: tl.constexpr = RING >> RING_SHIFT
        blocks = (tl.maximum(raw, 0) >> RING_SHIFT) // ODD_RING
        after_blocks = (tl.maximum(after, 0) >> RING_SHIFT) // ODD_RING
        # Every valid current row in this scan belongs to this program.
        selected = valid & ~(after_valid & (blocks == after_blocks))
        hit_windows = tl.reduce(tl.reshape(selected, (NUM_WINDOWS, TILE_ROWS)), 1, _or_predicates)
        MASK_DTYPE: tl.constexpr = tl.int32 if GROUPS_PER_SCAN == 1 else tl.uint32
        active_tiles = tl.sum(tl.where(hit_windows, tile_bits, 0), axis=0, dtype=MASK_DTYPE)
        hit_words = selected.to(tl.int16)
        while active_tiles != 0:
            lane_offset = _first_selected_lane(active_tiles) * TILE_ROWS
            narrow_raw = extract_slice(raw, (lane_offset,), (TILE_ROWS,), (1,))
            narrow_blocks = extract_slice(blocks, (lane_offset,), (TILE_ROWS,), (1,))
            narrow_hits = extract_slice(hit_words, (lane_offset,), (TILE_ROWS,), (1,)) != 0
            _copy_selected_windows(
                KEY, GATE, BACKING, group * GROUP_ROWS + lane_offset, narrow_raw, narrow_blocks,
                narrow_hits, RING, BLOCK_STRIDE, CHANNEL_STRIDE, OFFSET, ROW_TILE, TILE_ROWS,
            )
            active_tiles &= active_tiles - 1


@lru_cache(maxsize=None)
def _vector_core_count(device_index):
    return int(driver.active.utils.get_device_properties(device_index)["num_vectorcore"])


def _kpool_seed_tail_cache_impl(tail_kv_cache, key, gate_score, tslot, kpool=4, head_dim=128):
    """Update cache in place; valid slots must fit the signed-int32 slot range."""
    if kpool != 4 or head_dim != 128:
        raise ValueError("This GLM-5.3-Flash kernel fixes kpool=4 and head_dim=128")
    if key.dtype != torch.bfloat16:
        raise ValueError("K, gate and cache must use BF16")
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
    groups_per_scan = 8 if chunk_groups >= 8 else (4 if chunk_groups >= 4 else 1)
    if groups_per_scan == 1:
        programs = triton.cdiv(groups, chunk_groups)
    else:
        programs = min(vector_cores, triton.cdiv(groups, groups_per_scan))
    ring = tail_kv_cache.shape[2]
    ring_shift = (ring & -ring).bit_length() - 1
    slot_words = tslot.element_size() // 4
    scan_rows = 256 * groups_per_scan
    index64 = triton.cdiv(n, scan_rows) * programs >= 2**31 or (n + 3 * scan_rows) * slot_words >= 2**31
    # A tensor view already points at its storage offset: do not add it twice.
    _kpool_tail_seed_kernel[(programs,)](
        key,
        gate_score,
        tslot.view(torch.int32) if tslot.element_size() == 8 else tslot,
        tail_kv_cache,
        n,
        ring,
        ring_shift,
        tail_kv_cache.stride(0),
        tail_kv_cache.stride(1),
        0,
        tslot.element_size() * 8,
        ROW_TILE=16,
        GROUPS_PER_SCAN=groups_per_scan,
        INDEX64=index64,
    )
    return None


@torch.library.custom_op("mojo_npu_triton_a2::kpool_seed_tail_cache", mutates_args=("tail_kv_cache",))
def _kpool_seed_tail_cache_leaf(
    tail_kv_cache: torch.Tensor, key: torch.Tensor, gate_score: torch.Tensor, tslot: torch.Tensor
) -> None:
    # Triton launches on the active device's current stream.
    with torch.npu.device(key.device):
        _kpool_seed_tail_cache_impl(tail_kv_cache, key, gate_score, tslot)


@_kpool_seed_tail_cache_leaf.register_fake
def _kpool_seed_tail_cache_fake(tail_kv_cache, key, gate_score, tslot):
    return None


def kpool_seed_tail_cache_fwd(tail_kv_cache, key, gate_score, tslot, kpool=4, head_dim=128):
    if kpool != 4 or head_dim != 128:
        raise ValueError("This GLM-5.3-Flash kernel fixes kpool=4 and head_dim=128")
    # Keep pointer checks, hardware queries and Triton launch outside graph capture.
    return _kpool_seed_tail_cache_leaf(tail_kv_cache, key, gate_score, tslot)
