"""A5 RNG and sparse-dropout metadata helpers for the existing FlexAttention v2 provider."""

import numpy as np
import torch
import triton
import triton.language as tl

from typing import Optional

from packaging.version import Version

from mojo_opset.kernels._npu_triton_utils import get_ascend_compiler_version
from mojo_opset.kernels._npu_triton_utils import get_num_cores
from mojo_opset.utils.attention_dropout import SparseDropoutMask
from mojo_opset.utils.attention_dropout import _CachedLayout
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
    if isinstance(backward_rows, tuple) and len(backward_rows) == 2:
        partials, counts = backward_rows
    else:
        partials = [sum(not full for _, full in row) for row in backward_rows]
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
    for kb, (partial, total) in enumerate(zip(partials, counts)):
        begin = len(tasks)
        for start in range(0, max(total, 1), chunk_size):
            end = min(start + chunk_size, total)
            tasks.append([kb, min(start, partial), min(end, partial), max(start - partial, 0), max(end - partial, 0)])
        ranges.append([begin, len(tasks)])
    return {"tasks": tasks, "ranges": ranges}


@triton.jit(do_not_specialize=["NQ", "NK", "KV_STRIDE", "FULL_STRIDE"])
def _fill_block_table(
    TABLE,
    KV_NUM,
    KV_IDX,
    FULL_NUM,
    FULL_IDX,
    NQ,
    NK,
    KV_STRIDE,
    FULL_STRIDE,
    HAS_FULL: tl.constexpr,
    PREFIX_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    zero64 = tl.full((), 0, tl.int64)
    prefix_offs = tl.arange(0, PREFIX_BLOCK)
    col_offs = tl.arange(0, COL_BLOCK)
    for qb in range(tl.program_id(0), NQ, tl.num_programs(0)):
        prefix_acc = tl.full((PREFIX_BLOCK,), 0, tl.int32)
        for block in range(0, tl.cdiv(qb, PREFIX_BLOCK)):
            rows = block * PREFIX_BLOCK + prefix_offs
            mask = rows < qb
            counts = tl.load(KV_NUM + rows, mask=mask, other=0)
            if HAS_FULL:
                counts += tl.load(FULL_NUM + rows, mask=mask, other=0)
            prefix_acc += counts
        row_start = tl.sum(prefix_acc, axis=0)

        qb64 = qb + zero64
        row_table = TABLE + qb64 * NK
        p_cnt = tl.load(KV_NUM + qb)
        p_idx_base = KV_IDX + qb64 * KV_STRIDE
        for block in range(0, tl.cdiv(p_cnt, COL_BLOCK)):
            item = block * COL_BLOCK + col_offs
            valid = item < p_cnt
            cols = tl.load(p_idx_base + item, mask=valid, other=0).to(tl.int64)
            tl.store(row_table + cols, row_start + item, mask=valid)

        if HAS_FULL:
            f_cnt = tl.load(FULL_NUM + qb)
            f_base = row_start + p_cnt
            f_idx_base = FULL_IDX + qb64 * FULL_STRIDE
            for block in range(0, tl.cdiv(f_cnt, COL_BLOCK)):
                item = block * COL_BLOCK + col_offs
                valid = item < f_cnt
                cols = tl.load(f_idx_base + item, mask=valid, other=0).to(tl.int64)
                tl.store(row_table + cols, f_base + item, mask=valid)


def _contiguous_i32(tensor):
    return tensor if tensor.dtype == torch.int32 and tensor.is_contiguous() else tensor.to(torch.int32).contiguous()


def _row_contiguous_i32(tensor):
    if tensor.dtype == torch.int32 and (tensor.is_contiguous() or tensor.stride(-1) == 1):
        return tensor
    return tensor.to(torch.int32).contiguous()


def build_dropout_table(block_mask, nq, nk, *, num_active=None, prefix_block=1024, col_block=256):
    """Materialize the int32 sparse dropout block_table directly on device via Triton."""
    device = block_mask.kv_indices.device
    table = torch.full((nq, nk), -1, dtype=torch.int32, device=device)
    if nq > 0 and nk > 0 and (num_active is None or num_active > 0):
        kv_num = _contiguous_i32(block_mask.kv_num_blocks)
        kv_idx = _row_contiguous_i32(block_mask.kv_indices)
        has_full = block_mask.full_kv_num_blocks is not None
        full_num = _contiguous_i32(block_mask.full_kv_num_blocks) if has_full else kv_num
        full_idx = _row_contiguous_i32(block_mask.full_kv_indices) if has_full else kv_idx
        cores = get_num_cores("vector") if device.type == "npu" else 4
        grid = (min(max(cores, 1), nq),)
        _fill_block_table[grid](
            table,
            kv_num,
            kv_idx,
            full_num,
            full_idx,
            nq,
            nk,
            kv_idx.stride(2),
            full_idx.stride(2),
            HAS_FULL=has_full,
            PREFIX_BLOCK=prefix_block,
            COL_BLOCK=col_block,
        )
    return table


def build_dropout_layout_entry(block_mask, nq, nk, *, prefix_block=1024, col_block=256):
    """Build a cached A5 dropout layout with one small count transfer and zero index copies."""
    q_num = block_mask.q_num_blocks.reshape(nk)
    fq_num = block_mask.full_q_num_blocks
    if fq_num is None:
        rp_cnt = np.asarray(q_num.detach().cpu(), dtype=np.int32)
        rt_cnt = rp_cnt
    else:
        pair = np.asarray(torch.stack((q_num, fq_num.reshape(nk))).detach().cpu(), dtype=np.int32)
        rp_cnt, rt_cnt = pair[0], pair[0] + pair[1]
    num_active = int(rt_cnt.sum())
    table = build_dropout_table(
        block_mask, nq, nk, num_active=num_active, prefix_block=prefix_block, col_block=col_block
    )
    return _CachedLayout(table, num_active, (rp_cnt.tolist(), rt_cnt.tolist()))


def allocate_dropout_mask(q, k, block_mask, dropout_p, *, bitpacked=False, dkv_chunk_size=None):
    """Prepare reusable metadata and an uninitialized random buffer, outside timing/capture.

    Imbalanced reverse-mask rows use private split-Q dKV tiles and a fixed-order
    reduction. ``dkv_chunk_size=0`` keeps the original schedule; positive values
    force a maximum number of reverse Q blocks per task (mainly for validation).
    """
    bm, bn = block_mask.BLOCK_SIZE
    if (bm, bn) != (128, 128):
        raise ValueError("A5 dropout currently requires 128x128 sparse blocks")
    threshold, scale = _dropout_parameters(dropout_p)
    layout, cached = _DropoutLayout.resolve(block_mask, q.shape[2], k.shape[2])
    groups = q.shape[0] * k.shape[1]
    cores = get_num_cores("cube")
    balance_moderate = dropout_p > 0 and q.shape[-1] == 128
    plan_key = (groups, cores, dkv_chunk_size, balance_moderate, q.device)
    if plan_key not in cached.dkv_plans:
        plan = _build_dkv_split_schedule(
            cached.reverse_counts, groups, cores, dkv_chunk_size, balance_moderate_rows=balance_moderate
        )
        if len(cached.dkv_plans) >= 8:
            cached.dkv_plans.clear()
        cached.dkv_plans[plan_key] = (
            (None, None, 0)
            if plan is None
            else (
                torch.tensor(plan["tasks"], dtype=torch.int32, device=q.device),
                torch.tensor(plan["ranges"], dtype=torch.int32, device=q.device),
                len(plan["tasks"]),
            )
        )
    layout.dkv_schedule, layout.dkv_ranges, layout.dkv_num_tasks = cached.dkv_plans[plan_key]
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


@triton.jit(do_not_specialize=["WORDS"])
def _fill_packed_mask(OUT, SEED_PTR, WORDS, THRESHOLD: tl.constexpr):
    k0 = tl.load(SEED_PTR).to(tl.uint32)
    k1 = tl.load(SEED_PTR + 1).to(tl.uint32)
    seed = k0.to(tl.uint64) | (k1.to(tl.uint64) << 32)
    lanes = tl.arange(0, 2048)
    planes = (lanes % 1024) // 128
    for block in range(tl.program_id(0), tl.cdiv(WORDS, 1024), tl.num_programs(0)):
        counters = block.to(tl.uint64) * 2048 + lanes
        a, b, c, d = _philox4x32(seed, counters)
        pa = tl.sum((_drop_byte_flags(a, THRESHOLD) << planes).reshape(2, 8, 128), 1).reshape(256)
        pb = tl.sum((_drop_byte_flags(b, THRESHOLD) << planes).reshape(2, 8, 128), 1).reshape(256)
        pc = tl.sum((_drop_byte_flags(c, THRESHOLD) << planes).reshape(2, 8, 128), 1).reshape(256)
        pd = tl.sum((_drop_byte_flags(d, THRESHOLD) << planes).reshape(2, 8, 128), 1).reshape(256)
        drop_bits = tl.interleave(tl.interleave(pa, pc), tl.interleave(pb, pd))
        indices = block.to(tl.int64) * 1024 + tl.arange(0, 1024)
        tl.store(OUT + indices, (~drop_bits).to(tl.int32, bitcast=True), mask=indices < WORDS.to(tl.int64))


def mask_row_bytes(total_k: int, causal: bool, local_window: Optional[int], global_window: Optional[int]) -> int:
    """Return the aligned storage per query/head row, including ignored padding."""
    if causal and local_window is not None:
        prefix = ((global_window or 0) + 127) // 128 * 128
        # L+1 keys, at most 127 extra for the query block and 127 for alignment;
        # the final 127 rounds this bound up to a multiple of 128.
        width = prefix + (local_window + 382) // 128 * 128
    else:
        width = (total_k + 63) // 64 * 64
    return width // 8


@torch.library.custom_op("mojo_npu_triton_a5::swa_dropout_mask", mutates_args=())
def generate_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    p: float,
    causal: bool,
    local_window: Optional[int],
    global_window: Optional[int],
    seed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Draw keep bits from the device RNG using Triton Philox; explicit masks bypass this operation."""
    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in [0, 1)")
    threshold = int((1.0 - p) * 255)
    row_words = mask_row_bytes(k.shape[0], causal, local_window, global_window) // 4
    num_words = q.shape[0] * q.shape[1] * row_words
    mask = torch.empty((q.shape[0], q.shape[1], row_words), device=q.device, dtype=torch.int32)
    if num_words:
        if seed is None:
            seed = torch.randint(0, 2**31, (2,), device=q.device, dtype=torch.int32)
        grid = (min(get_num_cores("vector"), triton.cdiv(num_words, 1024)),)
        _fill_packed_mask[grid](mask.view(-1), seed, num_words, threshold)
    return mask


@generate_mask.register_fake
def _generate_mask_fake(q, k, p, causal, local_window, global_window, seed=None):
    row_words = mask_row_bytes(k.shape[0], causal, local_window, global_window) // 4
    return q.new_empty((q.shape[0], q.shape[1], row_words), dtype=torch.int32)
