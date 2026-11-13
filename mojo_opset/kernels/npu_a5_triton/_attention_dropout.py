"""A5 RNG and sparse-dropout metadata helpers for the existing FlexAttention v2 provider."""

import torch
import triton
import triton.language as tl

from packaging.version import Version

from mojo_opset.kernels._npu_triton_utils import get_ascend_compiler_version
from mojo_opset.kernels._npu_triton_utils import get_num_cores
from mojo_opset.utils.attention_dropout import SparseDropoutMask
from mojo_opset.utils.attention_dropout import _dropout_parameters
from mojo_opset.utils.attention_dropout import _DropoutLayout
from mojo_opset.utils.attention_dropout import _pack_random_bytes_torch


def _use_native_philox():
    kind, version = get_ascend_compiler_version()
    minimum = Version("3.6.0.post1" if kind == "internal" else "3.6.0")
    return version is not None and version >= minimum


_NATIVE_PHILOX = tl.constexpr(_use_native_philox())


def _build_dkv_split_schedule(backward_rows, groups, cores, chunk_size=None, *, balance_moderate_rows=False):
    """Split long reverse-mask rows, returning task ranges and fixed reduction order.

    ``chunk_size=0`` disables splitting; a positive value forces it for testing.
    Auto mode only changes schedules with substantial static load imbalance.
    ``balance_moderate_rows`` also splits moderately imbalanced rows into
    32-Q-block tasks, useful for the D=128 dropout kernel on 950PR.
    Each task stores [kv_block, partial_begin, partial_end, full_begin, full_end].
    """
    if groups <= 0 or cores <= 0:
        raise ValueError("groups and cores must be positive")
    if chunk_size is not None and chunk_size < 0:
        raise ValueError("chunk_size must be nonnegative")
    counts = [len(row) for row in backward_rows]
    if not counts or not sum(counts) or chunk_size == 0:
        return None
    grid = min(cores, len(counts) * groups)
    loads = [0] * grid
    for task, count in enumerate(counts * groups):
        loads[task % grid] += count
    imbalance = max(loads) / (sum(loads) / grid)
    if chunk_size is None:
        if max(counts) < 32:
            return None
        if imbalance < 1.8:
            if not balance_moderate_rows or imbalance < 1.6 or max(counts) <= 32:
                return None
            # Bound temporary storage while shortening moderately long tasks.
            chunk_size = 32
        else:
            # Roughly four waves give long prefix rows enough independent tasks.
            target = max(1, sum(counts) * groups // (4 * cores))
            chunk_size = max(8, 1 << (target.bit_length() - 1))
    tasks, ranges = [], []
    for kb, row in enumerate(backward_rows):
        partial = sum(not full for _, full in row)
        total = len(row)
        begin = len(tasks)
        for start in range(0, max(total, 1), chunk_size):
            end = min(start + chunk_size, total)
            tasks.append([kb, min(start, partial), min(end, partial), max(start - partial, 0), max(end - partial, 0)])
        ranges.append([begin, len(tasks)])
    return {"tasks": tasks, "ranges": ranges}


def allocate_dropout_mask(q, k, block_mask, dropout_p, *, bitpacked=False, dkv_chunk_size=None):
    """Prepare reusable metadata and an uninitialized random buffer, outside timing/capture.

    Imbalanced reverse-mask rows use private split-Q dKV tiles and a fixed-order
    reduction. ``dkv_chunk_size=0`` keeps the original schedule; positive values
    force a maximum number of reverse Q blocks per task (mainly for validation).
    """
    threshold, scale = _dropout_parameters(dropout_p)
    layout, backward_rows = _DropoutLayout.from_block_mask(block_mask, q.shape[2], k.shape[2])
    plan = _build_dkv_split_schedule(
        backward_rows,
        q.shape[0] * k.shape[1],
        get_num_cores("cube"),
        dkv_chunk_size,
        balance_moderate_rows=dropout_p > 0 and q.shape[-1] == 128,
    )
    layout.dkv_schedule = None if plan is None else torch.tensor(plan["tasks"], dtype=torch.int32, device=q.device)
    layout.dkv_ranges = None if plan is None else torch.tensor(plan["ranges"], dtype=torch.int32, device=q.device)
    layout.dkv_num_tasks = 0 if plan is None else len(plan["tasks"])
    bm, bn = block_mask.BLOCK_SIZE
    if (bm, bn) != (128, 128):
        raise ValueError("A5 dropout currently requires 128x128 sparse blocks")
    tiles = (
        None
        if dropout_p == 0
        else torch.empty(
            (q.shape[0], q.shape[1], layout.num_active, bm // 8 if bitpacked else bm, bn),
            dtype=torch.uint8,
            device=q.device,
        )
    )
    return SparseDropoutMask(
        layout,
        q.shape[0],
        q.shape[1],
        dropout_p,
        bitpacked,
        tiles,
        threshold,
        scale,
    )


@triton.jit
def _philox4x32(seed, counters):
    # Older Triton randint4x selects Philox4x64 for uint64 offsets. Split the
    # counter explicitly to keep Philox4x32 and its high counter bits on both versions.
    c0 = counters.to(tl.uint32)
    c1 = (counters >> 32).to(tl.uint32)
    zero = tl.full(counters.shape, 0, tl.uint32)
    if _NATIVE_PHILOX:
        return tl.philox(seed, c0, c1, zero, zero)
    else:
        # Legacy Ascend lowers uint32 umulhi as signed mulhi. Widen only the
        # high-product calculation; the state and low products remain uint32.
        c2, c3 = zero, zero
        seed64 = seed.to(tl.uint64)
        k0, k1 = seed64.to(tl.uint32), (seed64 >> 32).to(tl.uint32)
        for _ in tl.static_range(10):
            old_c0, old_c2 = c0, c2
            # Mask explicitly: legacy codegen can elide a uint64->uint32->uint64 cast.
            a = (old_c0.to(tl.uint64) & 0xFFFFFFFF) * 0xD2511F53
            b = (old_c2.to(tl.uint64) & 0xFFFFFFFF) * 0xCD9E8D57
            c0 = (b >> 32).to(tl.uint32) ^ c1 ^ k0
            c2 = (a >> 32).to(tl.uint32) ^ c3 ^ k1
            c1 = tl.core.mul(old_c2, 0xCD9E8D57, sanitize_overflow=False)
            c3 = tl.core.mul(old_c0, 0xD2511F53, sanitize_overflow=False)
            k0 = tl.core.add(k0, 0x9E3779B9, sanitize_overflow=False)
            k1 = tl.core.add(k1, 0xBB67AE85, sanitize_overflow=False)
        return c0, c1, c2, c3


@triton.jit(do_not_specialize=["SEED", "OFFSET", "WORDS"])
def _fill_random_bytes(OUT, SEED, OFFSET, WORDS, BLOCK: tl.constexpr):
    # One Philox4x32 result supplies all 16 raw bytes, without dropping 3/4 of its bits.
    for block in range(tl.program_id(0), tl.cdiv(WORDS, BLOCK * 4), tl.num_programs(0)):
        counters = OFFSET.to(tl.uint64) + block.to(tl.uint64) * BLOCK + tl.arange(0, BLOCK)
        a, b, c, d = _philox4x32(SEED, counters)
        words = tl.interleave(tl.interleave(a, c), tl.interleave(b, d))
        indices = block.to(tl.int64) * (BLOCK * 4) + tl.arange(0, BLOCK * 4)
        tl.store(OUT + indices, words.to(tl.int32, bitcast=True), mask=indices < WORDS)


@triton.jit
def _drop_byte_flags(words, THRESHOLD: tl.constexpr):
    # Four independent byte comparisons in one uint32. Each low-7-bit sum is
    # <=254, so no carry can cross a byte boundary. Bit 7 compares the low 7 bits.
    overflow = (words & 0x7F7F7F7F) + (127 - (THRESHOLD & 127)) * 0x01010101
    if THRESHOLD >= 128:
        flags = overflow & words
    else:
        flags = overflow | words
    return (flags & 0x80808080) >> 7


@triton.jit(do_not_specialize=["SEED", "OFFSET", "TILES"])
def _fill_q_planes(OUT, SEED, OFFSET, TILES, THRESHOLD: tl.constexpr):
    # Generate two sparse tiles at a time, preserving the raw Philox10 stream.
    # Reduce each Philox output across Q planes BEFORE interleaving: the
    # interleave then moves 1/8 as many elements as the raw-word arrangement.
    lanes = tl.arange(0, 2048)
    planes = (lanes % 1024) // 128
    for block in range(tl.program_id(0), tl.cdiv(TILES, 2), tl.num_programs(0)):
        counters = OFFSET.to(tl.uint64) + block.to(tl.uint64) * 2048 + lanes
        a, b, c, d = _philox4x32(SEED, counters)
        pa = tl.sum((_drop_byte_flags(a, THRESHOLD) << planes).reshape(2, 8, 128), 1).reshape(256)
        pb = tl.sum((_drop_byte_flags(b, THRESHOLD) << planes).reshape(2, 8, 128), 1).reshape(256)
        pc = tl.sum((_drop_byte_flags(c, THRESHOLD) << planes).reshape(2, 8, 128), 1).reshape(256)
        pd = tl.sum((_drop_byte_flags(d, THRESHOLD) << planes).reshape(2, 8, 128), 1).reshape(256)
        drop_bits = tl.interleave(tl.interleave(pa, pc), tl.interleave(pb, pd))
        indices = block.to(tl.int64) * 1024 + tl.arange(0, 1024)
        tl.store(OUT + indices, (~drop_bits).to(tl.int32, bitcast=True), mask=indices < TILES.to(tl.int64) * 512)


def refill_dropout_mask(dropout, *, generator=None, seed=None, offset=0):
    """Fill raw bytes; an explicit seed selects Triton Philox instead of Torch RNG.

    ``offset`` is measured in Philox4x32 counters (16 bytes). A repeated seed and
    offset intentionally reproduce the same mask; callers must advance one for
    each new training forward. Triton uses Philox4x32-10 here; this does not reproduce CUDA FA2's
    Philox7 sequence or coordinate mapping. Only byte-threshold/scale semantics match.
    """
    if seed is not None and generator is not None:
        raise ValueError("Choose an explicit Triton seed or a Torch generator")
    if seed is not None and (not 0 <= seed < 2**64 or not 0 <= offset < 2**64):
        raise ValueError("Philox seed and counter offset must be unsigned 64-bit integers")
    if dropout.tiles is not None and dropout.tiles.numel():
        if seed is None:
            if dropout.bitpacked:
                raw = torch.empty(dropout.tiles.numel() * 8, dtype=torch.uint8, device=dropout.tiles.device)
                raw.random_(0, 256, generator=generator)
                _pack_random_bytes_torch(raw, dropout.tiles, dropout.threshold)
            else:
                dropout.tiles.random_(0, 256, generator=generator)
        else:
            raw_bytes = dropout.tiles.numel() * (8 if dropout.bitpacked else 1)
            words = raw_bytes // 4
            if offset + triton.cdiv(words, 4) > 2**64:
                raise ValueError("Philox counter range overflows uint64")
            words_per_program = 8192
            grid = (min(get_num_cores("vector"), triton.cdiv(words, words_per_program)),)
            if dropout.bitpacked:
                _fill_q_planes[grid](
                    dropout.tiles.view(torch.int32), seed, offset, raw_bytes // 16384, dropout.threshold
                )
            else:
                _fill_random_bytes[grid](
                    dropout.tiles.view(torch.int32),
                    seed,
                    offset,
                    words,
                    BLOCK=2048,
                )
            torch.autograd.graph.increment_version(dropout.tiles)
    return dropout
