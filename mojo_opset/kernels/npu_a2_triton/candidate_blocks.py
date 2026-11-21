"""DSA candidate-block selection over packed logits (NPU Triton, A2).

Both entry points are inference-only and write in place:
``select_candidate_blocks_fwd`` fills ``out`` with -1 padded block ids, and
``apply_candidate_mask_fwd`` masks ``logits`` to -inf outside the candidates.

This file carries the A2 (910B3) tuning: the launch ladders, the packed -inf
tail fill, the i64 -> i32 boundary casts and the dead-tile store are measured
choices, so outside the custom-op namespace it no longer matches the A5 file.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from mojo_opset.kernels._npu_triton_utils import libentry


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@libentry()
@triton.jit(do_not_specialize=["width", "nblocks"])
def _block_scores_kernel(
    logits,
    starts,
    ends,
    scores,
    valid_blocks,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    stride_scores,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
    EXACT_TILES: tl.constexpr,
    STRIDED: tl.constexpr,
    WIDE_FILL: tl.constexpr,
    WIDE_MULT: tl.constexpr,
    WTAIL: tl.constexpr,
    SPLIT2: tl.constexpr,
    SC2: tl.constexpr,
    HOIST: tl.constexpr,
    I32: tl.constexpr,
    PAIR: tl.constexpr,
):
    # Writes each row's valid block count.
    row = tl.program_id(0).to(tl.int64)
    if I32:
        # Same boundary cast as _mask_candidates_kernel (see its R3 note): the
        # perf-harness prefill cases build `ends` with torch.arange, i.e. int64,
        # and an i64 operand promotes the whole downstream scalar chain
        # (col_end/length/num_valid/num_full and every `cols < col_end` compare)
        # into i64, which scalarises on A2. Column offsets always fit in i32.
        start = tl.load(starts + row // ROW_REPEAT * stride_start).to(tl.int32) \
            if HAS_STARTS else 0
        end = tl.load(ends + row // ROW_REPEAT * stride_end).to(tl.int32)
    else:
        start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
        end = tl.load(ends + row // ROW_REPEAT * stride_end)
    col_end = tl.minimum(end, width)
    length = tl.maximum(col_end - start, 0)
    num_valid = tl.cdiv(length, BLOCK_SIZE)
    # P16 three-state tile: complete-block count for the full-tile test in
    # the loop. num_valid counts the possibly-partial tail block, so a tile
    # admitted on num_valid could read past col_end exactly when the tail
    # block is partial and num_valid % TILE == 0; num_full admits only
    # tiles whose every block is complete, making `cols < col_end`
    # provably all-true on that path.
    num_full = length // BLOCK_SIZE
    # R5/BS-new-1: the pinned "newest" block is a single lane in a single tile,
    # yet the per-lane `blocks == pin` compare + select ran in *every* tile
    # (P0 sim baseline attributes 7.4% of this kernel to it). Reduce it to a
    # per-tile scalar branch; `pin` is an address-free block id, so i32 keeps
    # the compare out of the i64 lane chain.
    pin = ((end - start - 1) // BLOCK_SIZE).to(tl.int32)
    has_pin = end > start
    if tl.program_id(1) == 0:
        tl.store(valid_blocks + row, num_valid)
    elems_range = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    # P16 (redundant_boundary_operation): when BLOCK_SIZE is a power of two
    # the elems mask is vacuous, and when width is TILE-aligned the store
    # mask is vacuous — both fold away via constexpr branches (KVR-provable
    # identities; runtime branches measured 0.723x, must stay constexpr).
    POW2_BS: tl.constexpr = triton.next_power_of_2(BLOCK_SIZE) == BLOCK_SIZE
    num_tiles = tl.cdiv(nblocks, TILE)
    if HOIST and not STRIDED:
        # R12/B10 loop-invariant form of the per-tile address chain (see the
        # HOIST branch below). `colrel` is the [ELEMS, TILE] intra-tile column
        # pattern, so the per-tile addressing is one scalar step.
        colrel = tl.arange(0, TILE)[None, :] * BLOCK_SIZE + elems_range[:, None]
        step_col = TILE * BLOCK_SIZE * stride_col
        base_addr = logits + row * stride_row + start * stride_col + colrel * stride_col
    for tile_id in range(tl.program_id(1), num_tiles, tl.num_programs(1)):
        blocks = tile_id * TILE + tl.arange(0, TILE)
        # End clipping (a5 自研 endclip): a tile whose first block lies past
        # the row's num_valid is entirely masked — skip the load + reduce
        # tree and store the -inf constant directly. The store itself must
        # NOT be skipped: scores is uninitialized memory and topk scans the
        # whole row, so an unwritten tail would leak garbage into the
        # candidates. Uniform per-tile scalar branch; tile t is empty iff
        # t*TILE >= num_valid.
        if tile_id * TILE < num_valid:
            # R10/B1 (strided window read): with stride_col == 2 the [ELEMS, TILE]
            # tile has its *innermost* axis strided, so every access degenerates
            # into a short segment and MTE2 collapses (probe
            # opt_runs/probes/probe_bs_strided_seg.py, same shape as
            # decode-strided-B256-128K: [8,128] tile 7.0 GB/s vs a 2*BLOCK
            # contiguous window 240 GB/s useful / 480 GB/s physical). Read the
            # underlying 2*BLOCK window per block instead — its even lanes are
            # exactly the logical columns — so the whole tile becomes one bulk
            # DMA. The window of block c covers physical
            # [2*(start+c*BLOCK), 2*(start+(c+1)*BLOCK)-1], which stays inside
            # the row buffer (stride_row >= 2*width) for every complete block;
            # the partial tail tile keeps a masked load.
            if STRIDED:
                WB: tl.constexpr = 2 * triton.next_power_of_2(BLOCK_SIZE)
                bw = tl.arange(0, WB)
                # Physical window base of each block; `+ bw` (not `* stride_col`)
                # is what keeps the innermost axis unit-stride — the whole point
                # of this branch.
                #
                # R14/B14: `stride_col` is a kernel argument, so the compiler
                # cannot see that a window ends exactly where the next one
                # starts. The read-shape probe
                # (opt_runs/probes/probe_bs_read_seg.py, same 201 MB of useful
                # bytes) makes that fatal: identical access pattern, 16 KiB
                # contiguous windows 996 GB/s, but with a runtime column stride
                # in place of the literal 2 only 221 GB/s (4.5x). This branch is
                # only entered when stride_col == 2, so spell it out: the TILE
                # 64 B windows then merge into one 16 KiB bulk DMA.
                if SC2:
                    wbase = (start + blocks * BLOCK_SIZE) * 2
                else:
                    wbase = (start + blocks * BLOCK_SIZE) * stride_col
                waddr = logits + row * stride_row + wbase[:, None] + bw[None, :]
                wcol = start + blocks[:, None] * BLOCK_SIZE + bw[None, :] // 2
                even = bw % 2 == 0
                keep_lane = even if POW2_BS else even & (bw // 2 < BLOCK_SIZE)
                if SPLIT2:
                    # R11/B8 (de-interleave instead of a lane select): the
                    # simulator (out_r10_bs_strided) bills the even-lane
                    # `tl.where(keep_lane, win, -inf)` a quarter of the kernel
                    # (25.8%), i.e. a full 2*TILE f32 select just to throw half
                    # the lanes away. Reshaping the window to [TILE, WB//2, 2]
                    # and splitting the last axis drops the odd lanes for free
                    # and halves the axis-1 reduce width.
                    if (tile_id + 1) * TILE <= num_full:
                        win = tl.load(waddr).to(tl.float32)
                    else:
                        win = tl.load(
                            waddr, wcol < col_end, other=-float("inf")
                        ).to(tl.float32)
                    sel, _odd = tl.split(tl.reshape(win, (TILE, WB // 2, 2)))
                    if not POW2_BS:
                        sel = tl.where(
                            tl.arange(0, WB // 2)[None, :] < BLOCK_SIZE,
                            sel,
                            -float("inf"),
                        )
                elif (tile_id + 1) * TILE <= num_full:
                    win = tl.load(waddr).to(tl.float32)
                    sel = tl.where(keep_lane[None, :], win, -float("inf"))
                else:
                    sel = tl.load(
                        waddr, keep_lane[None, :] & (wcol < col_end), other=-float("inf")
                    ).to(tl.float32)
                reduced = tl.reduce(sel, 1, _max_with_nan)
            else:
                # Wide-vector merged reduction (a5 自研): [ELEMS, TILE] layout,
                # reduce along axis=0 — per-block max is full-width vector vmax.
                # R11/B7 (reverted): folding the column stride into a per-block
                # base (`wbase[None, :] + elems_range[:, None]`, an affine 2D
                # block) changes nothing on 910B3 — prefill-8192 9298.0 vs
                # 9302.3us, decode-B256-128K 96.89 vs 96.56us — so folding the
                # stride into a per-block base does not help. A flat contiguous
                # read + axis-1 reduce is much worse on top of that (0.71x
                # prefill / 0.30x decode): the horizontal reduce inside a
                # [TILE, BLOCK] tile costs more than the transpose it would
                # remove. Keep the [ELEMS, TILE] / axis-0 form.
                if HOIST:
                    # R12/B10: the per-tile address chain is the scalar cost
                    # (sim out_r10_bs: aiv_scalar 0.84 on prefill-8192, and
                    # neither TILE=256/512 nor num_stages moves it — both were
                    # re-measured this round: 9994/15112us vs 9304us). Split it
                    # into a loop-invariant [ELEMS, TILE] offset and one scalar
                    # step per tile, so each iteration is a single vector add
                    # off `base_addr` instead of rebuilding `cols * stride_col`.
                    addr = base_addr + tile_id * step_col
                    cols = start + tile_id * (TILE * BLOCK_SIZE) + colrel
                else:
                    cols = start + blocks[None, :] * BLOCK_SIZE + elems_range[:, None]
                    addr = logits + row * stride_row + cols * stride_col
                # P16 three-state tile: a tile whose every block is complete has
                # a provably all-true `cols < col_end` mask, so that path drops
                # the mask entirely — the per-line mask compare + other-select
                # machinery compiles away and the load can go down the bulk
                # path. Uniform per-tile scalar branch; partial tiles keep the
                # masked path.
                if PAIR:
                    # R17/A1 (paired contiguous run): with a bf16 logits tensor
                    # and BLOCK_SIZE=8 the [ELEMS, TILE] tile's innermost axis is
                    # only 8*2 = 16 B, half of the 32 B DMA burst A2 needs, and
                    # MTE2 collapses: the same shape reads at ~21 GB/s instead of
                    # the ~1.3 TB/s the 32 B fp32 layout reaches (probe, workdir
                    # harness, one module, one shape 256x131072:
                    #   bf16 16 B -> 2778.9us, bf16 32 B -> 89.8us,
                    #   bf16 64 B -> 77.4us, fp32 32 B -> 67.6us,
                    #   fp32 16 B -> 5484.8us).
                    # Pair two adjacent blocks into one column so the inner run
                    # is 2*BLOCK_SIZE elements (32 B on bf16/8), then split the
                    # axis back for the per-block reduce. Selection output is
                    # bit-identical (14 shape x dtype combos, out_equal=True).
                    # Measured (10-iter mean, same inputs as the A/B above):
                    #   256x32768 767.9 -> 120.7, 256x65536 1464.1 -> 223.5,
                    #   256x131072 2778.9 -> 426.2, 6x32768 26.6 -> 7.8,
                    #   6x131072 99.1 -> 21.7, 1536x131072 16702 -> 2512,
                    #   8192x131072 88832 -> 13328.
                    # fp32 must NOT take this branch: it is already at 32 B and
                    # the paired form costs 2-4x (256x131072 64.4 -> 418.4,
                    # prefill 3196 -> 13258), which is why the host gates it on
                    # element_size * BLOCK_SIZE < 32.
                    TP: tl.constexpr = TILE // 2
                    RUN: tl.constexpr = 2 * triton.next_power_of_2(BLOCK_SIZE)
                    pcols = (
                        start
                        + tile_id * (TILE * BLOCK_SIZE)
                        + tl.arange(0, RUN)[:, None]
                        + tl.arange(0, TP)[None, :] * RUN
                    )
                    paddr = logits + row * stride_row + pcols * stride_col
                    if (tile_id + 1) * TILE <= num_full:
                        pval = tl.load(paddr).to(tl.float32)
                    else:
                        pval = tl.load(
                            paddr, pcols < col_end, other=-float("inf")
                        ).to(tl.float32)
                    # R17/A2 (masked axis-0 reduce): splitting the paired lane
                    # axis with `reshape(TP, 2, BLOCK)` + axis-2 reduce costs
                    # 5.8x over the load floor, and the cost is dtype specific
                    # while the load is not (probe, 256x131072, TILE=512,
                    # split=2, no bounds: pair load + one axis-0 reduce =
                    # 144us bf16 / 136us fp32; current 3D form = 833/756;
                    # 3D load without the reshape = 505/494). Two masked
                    # axis-0 reduces keep the vector-friendly axis and land at
                    # 203/198; `tl.interleave` restores block order for +115.
                    # Block order is a contract of `scores`/`valid_blocks`.
                    # In the op itself this form is 1.4x over the 3D one:
                    # decode-n1 638 -> 446us, decode-n6 3653 -> 2606us,
                    # prefill 20128 -> 14248us, while the fp32 / strided /
                    # short-row cases stay put (they take the unpaired path).
                    half = tl.arange(0, RUN)[:, None] < triton.next_power_of_2(
                        BLOCK_SIZE
                    )
                    lo = tl.where(half, pval, -float("inf"))
                    hi = tl.where(half, -float("inf"), pval)
                    reduced = tl.interleave(
                        tl.reduce(lo, 0, _max_with_nan),
                        tl.reduce(hi, 0, _max_with_nan),
                    )
                elif (tile_id + 1) * TILE <= num_full:
                    if POW2_BS:
                        values = tl.load(addr).to(tl.float32)
                    else:
                        values = tl.load(
                            addr,
                            elems_range[:, None] < BLOCK_SIZE,
                            other=-float("inf"),
                        ).to(tl.float32)
                    reduced = tl.reduce(values, 0, _max_with_nan)
                else:
                    if POW2_BS:
                        # R12/B11 (rejected): the block-granular equivalent of
                        # this mask — `(blocks * BLOCK_SIZE < col_end - start)`
                        # broadcast over the elems axis, valid whenever the clip
                        # is block aligned — is 1-15% *slower*: the [1, TILE]
                        # predicate needs a layout conversion on 910B3, so the
                        # load loses its element mask but gains a broadcast
                        # (packed-prefill-8192-128K 741/743 -> 748/751us,
                        # decode-B256-128K 86.5/93.2 -> 107.1/106.6us, aiv_vec
                        # 0.08 -> 0.49). Keep the [ELEMS, TILE] element compare.
                        values = tl.load(
                            addr,
                            cols < col_end,
                            other=-float("inf"),
                        ).to(tl.float32)
                    else:
                        values = tl.load(
                            addr,
                            (elems_range[:, None] < BLOCK_SIZE) & (cols < col_end),
                            other=-float("inf"),
                        ).to(tl.float32)
                    reduced = tl.reduce(values, 0, _max_with_nan)
            # R5/BS-new-1: pin is applied only by the tile that owns it.
            if has_pin & (tile_id == pin // TILE):
                reduced = tl.where(blocks == pin, float("inf"), reduced)
            if EXACT_TILES:
                tl.store(scores + row * stride_scores + blocks, reduced)
            else:
                tl.store(scores + row * stride_scores + blocks, reduced, blocks < nblocks)
        elif not WIDE_FILL:
            neg = tl.full([TILE], -float("inf"), tl.float32)
            if EXACT_TILES:
                tl.store(scores + row * stride_scores + blocks, neg)
            else:
                tl.store(scores + row * stride_scores + blocks, neg, blocks < nblocks)
    # R10/B2 (packed tail fill): everything past `first_empty * TILE` is a
    # constant -inf region, and on packed-prefill-8192-128K it is 99.98% of the
    # row (16 valid columns inside 131072). The per-tile store above writes it
    # in TILE-wide (512 B) descriptors — 1M of them across the case.
    #
    # R13/W1 (segment width): `scores` holds one f32 *per block*, so the old
    # fixed `8 * TILE` chunk is only 8*128*4 = 4 KiB at TILE=128 — measurable
    # only at 989 GB/s on the pure-write probe (opt_runs/probes/
    # probe_store_bw.py: 4 KiB 989, 8 KiB 1263, 16 KiB 1315, 32 KiB 1313,
    # 64 KiB 1283 GB/s), i.e. 93.75% of this case's bytes sat on the *worst*
    # rung. WIDE_MULT is set from the host so every case gets a 16 KiB chunk
    # (4096 blocks, the probe's plateau); the UB cost is 16 KiB, which the
    # non-strided tile buffers tolerate.
    # R13/W2 (masked tail): with 4x wider chunks `nblocks - wide_lo` is no
    # longer a multiple of WIDE, and flooring would hand the 896-block
    # remainder back to 512 B descriptors — exactly the cost being removed.
    # WTAIL closes it with one masked WIDE store instead.
    first_empty = tl.cdiv(num_valid, TILE)
    WIDE: tl.constexpr = WIDE_MULT * TILE
    if WIDE_FILL:
        wide_lo = first_empty * TILE
        if WTAIL:
            n_wide = tl.maximum(tl.cdiv(nblocks - wide_lo, WIDE), 0)
        else:
            n_wide = tl.maximum((nblocks - wide_lo) // WIDE, 0)
        for w_id in range(tl.program_id(1), n_wide, tl.num_programs(1)):
            off = wide_lo + w_id * WIDE + tl.arange(0, WIDE)
            neg = tl.full([WIDE], -float("inf"), tl.float32)
            if WTAIL and (wide_lo + (w_id + 1) * WIDE > nblocks):
                tl.store(scores + row * stride_scores + off, neg, off < nblocks)
            else:
                tl.store(scores + row * stride_scores + off, neg)
        lo_tile = first_empty + n_wide * WIDE_MULT
        for tile_id in range(lo_tile + tl.program_id(1), num_tiles, tl.num_programs(1)):
            blocks = tile_id * TILE + tl.arange(0, TILE)
            neg = tl.full([TILE], -float("inf"), tl.float32)
            if EXACT_TILES:
                tl.store(scores + row * stride_scores + blocks, neg)
            else:
                tl.store(scores + row * stride_scores + blocks, neg, blocks < nblocks)


@libentry()
@triton.jit(do_not_specialize=["k"])
def _store_candidates_kernel(
    values,
    indices,
    output,
    out_stride_row,
    out_stride_col,
    k,
    OUT_K: tl.constexpr,
    TILE: tl.constexpr,
    K_FULL: tl.constexpr,
    STORE_EXACT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    # P16 constant-boundary elimination: the grid covers exactly OUT_K
    # columns, so `cols < OUT_K` is provably all-true whenever TILE divides
    # OUT_K (STORE_EXACT); and when k == OUT_K the two load masks are
    # likewise vacuous (K_FULL). On the masked 910B3 baseline these three
    # per-line compares scalarize the whole copy (aiv_vec ~0.004).
    if K_FULL:
        value = tl.load(values + row * k + cols)
        index = tl.load(indices + row * k + cols)
    else:
        value = tl.load(values + row * k + cols, cols < k, other=-float("inf"))
        index = tl.load(indices + row * k + cols, cols < k, other=-1)
    # Round 2: topk indices are int64 but `output` is int32, so narrowing
    # before the padding select keeps the whole select chain in int32 (block
    # ids are < 2**15, exactly representable). The int64 select chain — not
    # the store — was the cost: opt_runs/probes/probe_round2.py on
    # decode-B256-128K measures st_where_i64 405.2us vs st_where_i32 5.2us
    # (78x), against a pure contiguous write floor of 4.1us (st_no_values).
    index = index.to(tl.int32)
    # NaN scores can occur during warmup; only -inf denotes padding.
    if STORE_EXACT:
        tl.store(
            output + row * out_stride_row + cols * out_stride_col,
            tl.where(value != -float("inf"), index, -1),
        )
    else:
        tl.store(
            output + row * out_stride_row + cols * out_stride_col,
            tl.where(value != -float("inf"), index, -1),
            cols < OUT_K,
        )


@libentry()
@triton.jit(do_not_specialize=["width", "nblocks"])
def _candidate_flags_kernel(
    candidates,
    starts,
    flags,
    stride_row,
    stride_col,
    stride_start,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    HIST_BINS: tl.constexpr,
    CHUNK_BINS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    # Round 4: the flags row is *written densely*, not scattered.
    #
    # Rounds 2/3 assumed the per-row clear dominated this kernel, but
    # opt_runs/probes/probe_flags_clear.py shows the opposite:
    # prefill-8192of128K 56692us = clear 722us (1.3%) + per-element scatter
    # 55789us (98.7%). probe_flags_scatter.py then pins the scatter cost to one
    # 1-byte MTE3 descriptor per element (241 cyc/element, aiv_mte3 0.973):
    # int32 indices, a 4-byte element width and a 2-program split all change
    # nothing, while writing only the dense row costs 180.4us. The hardware
    # scatter (extension.scatter_ub_to_out) does not lower on A2
    # (probe_flags_hwscatter.py / probe_scatter_ub.py), so the row is counted
    # in UB with tl.histogram and DMA'd out once. probe_flags_hist.py measures
    # 1862.8 -> 452.9us (decode-B256-128K), 55606 -> 13223.8us
    # (prefill-8192of128K) and 7117.2 -> 1693.3us (prefill-1024of8K), i.e.
    # 4.1-4.2x, with a per-row pitch alignment that changes nothing (<=0.1%).
    #
    # Layout is unchanged for readers: HIST_BINS = next_power_of_2(nblocks)
    # real slots (padded slots stay 0), then slot nblocks = "past width" edge
    # flag and slot nblocks+1 = the trash bin for illegal candidate lanes
    # (round 2) that no reader looks at.
    start = tl.load(starts + row // ROW_REPEAT * stride_start).to(tl.int32) \
        if HAS_STARTS else 0
    cols = tl.arange(0, triton.next_power_of_2(K))
    raw = tl.load(candidates + row * stride_row + cols * stride_col, cols < K, other=-1)
    # `start + block * BLOCK_SIZE >= width` in the loaded dtype (int64-safe for
    # a hostile candidate, no i64 address chain in the common int32 case) and
    # without the multiply: cdiv(width - start, BLOCK_SIZE) is the first block
    # id that reaches past the row's width.
    thr = tl.maximum(width - start, 0)
    thr = (thr + (BLOCK_SIZE - 1)) // BLOCK_SIZE
    block = tl.where(raw >= thr, nblocks, raw).to(tl.int32)
    block = tl.where((cols < K) & (block >= 0), block, nblocks + 1)
    # Only real block ids enter the histogram; the two sentinel lanes are
    # counted with reductions and folded into the same dense row, so the two
    # stores below never disagree about a slot's value (they may overlap when
    # the padded bin range covers the sentinel slots).
    #
    # A single histogram of HIST_BINS int32 bins costs 4*HIST_BINS bytes of UB,
    # and 256K widths (nblocks 32768 -> 128KB) overflow it, so the row is
    # produced in CHUNK_BINS-sized pieces. Both are constexpr, so the trip
    # count is a trace-time constant and the loop unrolls; the bin work is
    # unchanged (chunks partition the row) and only the candidate scan repeats.
    edge = (tl.sum((block == nblocks).to(tl.int32)) != 0).to(tl.uint8)
    trash = (tl.sum((block == nblocks + 1).to(tl.int32)) != 0).to(tl.uint8)
    lap = tl.arange(0, CHUNK_BINS)
    base = flags + row * (HIST_BINS + 2)
    for lo in range(0, HIST_BINS, CHUNK_BINS):
        part = block - lo
        inb = (part >= 0) & (part < CHUNK_BINS) & (block < nblocks)
        dense = (tl.histogram(part, CHUNK_BINS, inb) != 0).to(tl.uint8)
        dense = tl.where(lap + lo == nblocks, edge, dense)
        dense = tl.where(lap + lo == nblocks + 1, trash, dense)
        tl.store(base + lo + lap, dense)
    tl.store(
        base + nblocks + tl.arange(0, 2),
        tl.where(tl.arange(0, 2) == 0, edge, trash),
    )


@triton.jit
def _mask_1d_tile(
    logits,
    flags,
    row_off,
    flag_off,
    cols,
    start,
    end,
    tail,
    width,
    stride_col,
    nblocks,
    flags_pitch,
    base,
    row,
    edge,
    full,
    dead,
    TILE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GATHER: tl.constexpr,
    TILE_IN_WIDTH: tl.constexpr,
    PLOAD: tl.constexpr,
    LIVE_SKIP: tl.constexpr,
):
    """Per-lane (1D) tile pass: block-flag lookup + boundary lift + RMW.

    Split out of `_mask_candidates_kernel` so the 2D fast path (R12/M10) can
    store its tile without materialising a [TILE] `keep`; this is the untouched
    original body.
    """
    if GATHER:
        # A1/P4 window preload + tl.gather: the flag lookup used to be a
        # per-element gather (aiv_scalar ~0.998, 12.9s on prefill-8192). One
        # tile spans at most TILE // BLOCK_SIZE + 1 blocks, so load that
        # contiguous window once and look the block flags up in UB instead.
        # A non-unit column stride needs the wider address buffer that this
        # path overruns (vector-core UB fault), so it keeps the scalar load.
        WIN: tl.constexpr = triton.next_power_of_2(TILE // BLOCK_SIZE + 1)
        # Round 3 (component probe, opt_runs/probes/probe_mask_components.py):
        # the gather itself is only half the cost — the *window load* is
        # scalarised too, because `minimum(maximum(base + arange, 0), nblocks)`
        # is not affine, so the 512-byte read degenerates into 512 scalar
        # loads (decode-B256-128K: base 2082us, window load removed 121us).
        # When `start` is block aligned and the tile begins at/after it, the
        # tile's blocks are exactly `base + arange(0, TILE//BLOCK_SIZE)`, so
        # the window read is affine (one contiguous DMA) and each block flag
        # broadcasts to its BLOCK_SIZE lanes — no gather, no clamp.
        # base < 0 can only happen on the single tile straddling `start`,
        # which keeps the clamped gather path.
        # A tile holds TILE // BLOCK_SIZE *complete* blocks, so the broadcast
        # below needs NB * BLOCK_SIZE == TILE exactly: block sizes that do not
        # divide TILE (3, 7, ...) would reshape 1023 lanes into 1024 and must
        # take the gather path. Static predicate, folds away otherwise.
        if (TILE % BLOCK_SIZE == 0) and (start % BLOCK_SIZE == 0) & (base >= 0):
            NB: tl.constexpr = TILE // BLOCK_SIZE
            keep = tl.load(flags + flag_off + base + tl.arange(0, NB)) != 0
            keep = tl.reshape(
                tl.broadcast_to(keep[:, None], (NB, BLOCK_SIZE)), (TILE,)
            )
        else:
            win = tl.load(
                flags
                + row * flags_pitch
                + tl.minimum(tl.maximum(base + tl.arange(0, WIN), 0), nblocks)
            )
            block = (cols - start) // BLOCK_SIZE
            idx = tl.minimum(tl.maximum(block - base, 0), WIN - 1)
            keep = tl.gather(win.to(tl.float32), idx, 0) != 0
    else:
        block = (cols - start) // BLOCK_SIZE
        if TILE_IN_WIDTH:
            valid = (cols >= start) & (cols < end)
        else:
            valid = (cols >= start) & (cols < end) & (cols < width)
        keep = tl.load(flags + flag_off + block, valid, other=0) != 0
    # R2/MK-new-1: the boundary column only exists in the tile that contains
    # column `width - 1`, but the per-lane `cols == width - 1` compare ran in
    # every tile — the simulator attributes 16.8% of the mask kernel to it
    # (opt_report_sim/sim_baseline_round0.md, S2 line 337). Hoist it to a
    # per-tile scalar branch: `tail <= width-1 < tail+TILE`.
    if (edge != 0) & (tail <= width - 1) & (width - 1 < tail + TILE):
        keep = keep | (cols == width - 1)
    if TILE_IN_WIDTH:
        # Round 2 (write-predicate removal): on A2 a *masked* store
        # scalarises the whole write sequence (aiv_scalar 0.991,
        # 171 cyc/element), while the store itself is cheap. Reading the
        # tile back and writing a select of (current value, -inf) is
        # bit-identical — kept lanes store exactly the value they already
        # hold — and makes the store unconditional. Probe
        # opt_runs/probes/probe_round2.py on decode-B256-128K:
        # mk_base 80.3ms -> mk_rmw 2.12ms (37.9x).
        #
        # R10/M2 (dead-tile load skip): the RMW reads the tile back even
        # when *no* column of it can be kept — a tile wholly before `start`
        # or at/after `end` writes -inf unconditionally. Those tiles are the
        # bulk of packed-prefill-8192-128K, whose rows are 16 columns long
        # inside a 131072-wide row, so the read half of the RMW is pure
        # waste there. Cold single-case A/B (dead-skip on/off, CLEAR_L2=1):
        #   packed-prefill-8192-128K 11536 -> 7742us (1.49x)
        #   decode-B256-mixed-128K     290 ->  254us (1.14x)
        #   decode-B256-256K           588 ->  569us (1.03x)
        #   decode-B256-128K / B8 / strided-pair ~1.00x
        #   prefill-8192of128K       11576 -> 11912us (0.97x)
        #   prefill-1024of8K            76 ->   81us (0.94x)
        # The prefill rows are (almost) fully live, so they pay the per-tile
        # test and the extra -inf buffer without ever skipping a read; the
        # whole run still nets -3.1ms over the 10 mask cases. The test is
        # two integer compares on tile indices computed in the prologue.
        # R12/M8 (rejected): an affine `logits + (row_off + tail) + arange`
        # address, meant to keep the innermost axis affine for the DMA
        # descriptor, is a no-op here (prefill-8192 11244.5 vs 11219.9us,
        # decode-B256-128K 231.2 vs 233.7us) — the backend already
        # canonicalises `cols * 1`. The plain `cols * stride_col` stays.
        addr = logits + row_off + cols * stride_col
        if dead:
            # The -inf tile merges with `val` in the else branch, so it must
            # carry the logits element type (bf16 rows reject an fp32 merge).
            val = tl.full([TILE], -float("inf"), logits.dtype.element_ty)
        else:
            if full & LIVE_SKIP:
                # R11/M5: fully live tile -> `valid` is all-true, so the
                # where degenerates to the flag mask alone.
                if PLOAD:
                    # R11/M7: the RMW only ever needs the kept lanes — a
                    # masked *load* with other=-inf returns exactly what
                    # `where(keep, load(addr), -inf)` returns, but the
                    # dropped lanes are never fetched from HBM, so the read
                    # half of the RMW shrinks to the candidate density
                    # (~12.5% of blocks in the serving shapes). Masked
                    # loads are cheap on A2 (masked *stores* are what
                    # scalarises), so this only moves work off the traffic
                    # path.
                    val = tl.load(addr, keep, other=-float("inf"))
                else:
                    cur = tl.load(addr)
                    val = tl.where(keep, cur, -float("inf"))
            else:
                valid = (cols >= start) & (cols < end)
                if PLOAD:
                    val = tl.load(addr, valid & keep, other=-float("inf"))
                else:
                    cur = tl.load(addr)
                    val = tl.where(valid & keep, cur, -float("inf"))
        tl.store(addr, val)
    else:
        valid = (cols >= start) & (cols < end) & (cols < width)
        tl.store(
            logits + row_off + cols * stride_col,
            -float("inf"),
            (cols < width) & ~(valid & keep),
        )


@libentry()
@triton.jit(do_not_specialize=["width", "nblocks", "flags_pitch"])
def _mask_candidates_kernel(
    logits,
    starts,
    ends,
    flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    flags_pitch,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
    TILE_IN_WIDTH: tl.constexpr,
    GATHER: tl.constexpr,
    IDX_I64: tl.constexpr,
    STRIDE_PAIR: tl.constexpr,
    DEAD_SKIP: tl.constexpr,
    DEAD_STORE: tl.constexpr,
    LIVE_SKIP: tl.constexpr,
    PLOAD: tl.constexpr,
):
    # A6/P6: same i32 row chain as A5 (host guard), i64 fallback.
    row = tl.program_id(0)
    if IDX_I64:
        row = row.to(tl.int64)
    # R0/M3 (#10 loop-invariant hoist): both per-row base offsets are
    # invariant across the tile; keeping them as named scalars stops the
    # address chains from being rebuilt in each branch.
    row_off = row * stride_row
    flag_off = row * flags_pitch
    tail = tl.program_id(1) * TILE
    tile_id = tl.program_id(1)
    cols = tail + tl.arange(0, TILE)
    # Round 3: row_ks/row_ke are column offsets, so they always fit in i32 —
    # but callers may hand over int64 bounds (the perf harness prefill cases
    # build `ends` with torch.arange, i.e. int64). An i64 operand promotes the
    # whole `cols` chain (compares, //, the address vector) and costs 7x:
    # prefill-1024of8K 3649.8us -> 523us in the component probe, with the
    # identical kernel body. Cast at the boundary instead.
    start = tl.load(starts + row // ROW_REPEAT * stride_start).to(tl.int32) \
        if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end).to(tl.int32)
    if STRIDE_PAIR:
        # R4/MK-new-3 (strided production branch): with `stride_col == 2` every
        # access in this kernel degenerates into per-lane scalar traffic —
        # simulator S3b (opt_runs/probes/sim/sim_mask_strided_aligned.py, the
        # aligned RMW branch that production strided shapes take) attributes
        # 32.8% to the store (ST_XD_XN_IMM, calls == TILE), 25.2% to the
        # validity compare and 15.1% to the per-element flag load, with MTE2+
        # MTE3 at only 7%. Fix: run the tile over the *underlying* contiguous
        # 2*TILE window, evaluate the mask on the logical column `wide // 2`,
        # and write the odd lanes back with the value just read (bit-identical,
        # so the odd columns are untouched semantically). Both memory ops become
        # one bulk DMA; the price is 2x traffic on a kernel that is 7% MTE.
        wide = tail * 2 + tl.arange(0, 2 * TILE)
        lc = wide // 2
        valid2 = (lc >= start) & (lc < end)
        wbase = (tail - start) // BLOCK_SIZE
        edge = tl.load(flags + flag_off + nblocks)
        # Same static guard as _mask_1d_tile: the broadcast needs BLOCK_SIZE to
        # divide TILE, otherwise the window's block count is floor(TILE/BS) + 1.
        if (TILE % BLOCK_SIZE == 0) and (start % BLOCK_SIZE == 0) & (wbase >= 0):
            NB2: tl.constexpr = TILE // BLOCK_SIZE
            flagw = tl.load(flags + flag_off + wbase + tl.arange(0, NB2))
            keep2 = tl.reshape(
                tl.broadcast_to(flagw[:, None], (NB2, 2 * BLOCK_SIZE)),
                (2 * TILE,),
            ) != 0
        else:
            blk = (lc - start) // BLOCK_SIZE
            keep2 = tl.load(flags + flag_off + blk, valid2, other=0) != 0
        if (edge != 0) & (tail <= width - 1) & (width - 1 < tail + TILE):
            keep2 = keep2 | (lc == width - 1)
        # The boundary slot only lifts the *candidate* mask; the causal bound
        # still applies (rows whose end <= width-1 must stay clipped there).
        # R12/M9 (rejected): extending the M5 full-tile test to this pair path
        # (`hit = keep2` when tail >= start and tail + TILE <= end) removes the
        # [2*TILE] valid2 compare but changes nothing: cold single-case A/B
        # 611.6/611.3us (off) vs 614.0/611.7us (on), aiv_vec 0.768 -> 0.763.
        # This path is already vec-bound at 0.77, so the dropped compares are
        # free. Keep the plain mask.
        hit = valid2 & keep2
        addr2 = logits + row_off + wide
        cur2 = tl.load(addr2)
        tl.store(
            addr2,
            tl.where(
                (wide & 1) == 0,
                tl.where(hit, cur2, -float("inf")),
                cur2,
            ),
        )
    else:
        # R10/M2: the live column span of this row in tile units. A tile whose
        # index is outside [start_tile, end_tile) has no valid column at all,
        # so its RMW read is pure waste (packed-prefill-8192-128K keeps 16
        # columns in a 131072-wide row). Both bounds only depend on the
        # row prologue scalars, so the branch resolves before the flag load
        # issues — the earlier version gated the skip on the *loaded* edge
        # slab, which serialised flag-read -> branch -> tile-read and cost
        # prefill-8192of128K 3%. Clipping `end` to `width` also makes the
        # "column width-1" lift above safe to ignore here: that lift only
        # keeps a column that is already < end.
        end_tile = tl.cdiv(tl.minimum(end, width), TILE)
        start_tile = start // TILE
        # P16v2: on a TILE-aligned width `cols < width` is provably all-true, so
        # the validity compare drops one term.
        # R11/M5: `valid` is only materialised where it can matter — on a
        # *fully live* tile `valid` is provably all-true (cols >= start because
        # tail >= start, cols < end because tail + TILE <= end), so the two
        # per-lane compares below are pure waste. The simulator
        # (out_r10_mask, code_exe) bills `valid` 10.6% of the kernel and the
        # redundant `keep != 0` another 7.9%; both are on the hot prefill path.
        full = (tail >= start) & (tail + TILE <= end)
        dead = ((tile_id >= end_tile) | (tile_id < start_tile)) & DEAD_SKIP
        if DEAD_STORE and TILE_IN_WIDTH and dead:
            # R16/M11 (dead-tile hoist): a dead tile writes -inf to every lane
            # and needs none of the flag machinery — yet the flag window DMA,
            # its `broadcast_to + reshape`, the `keep != 0` compare and the
            # 1-byte `edge` load all ran *before* the `dead` test inside
            # `_mask_1d_tile`. That is the whole cost of packed-prefill-8192-128K,
            # whose rows keep 16 of 131072 columns: 31/32 tiles are dead, so
            # aiv_vec 0.703 and aiv_mte2 0.351 were spent on nothing (the
            # compulsory traffic there is the 4.3 GB -inf write alone). Store
            # the constant directly and skip the helper entirely.
            # Guarded by TILE_IN_WIDTH so the partial-tile shape keeps its
            # masked store path untouched (`cols < width` must clip there).
            #
            # Cold 10-case A/B (CLEAR_L2=1, median of 2): packed 1.90x,
            # mixed 1.65x, 256K 1.083x, B256-128K 1.076x, n6-dspark 1.065x,
            # B8 1.02x, strided 1.01x, 32K 0.969x, prefill-1024of8K 0.917x,
            # prefill-8192of128K 0.925x; net -2.1ms of the 20.0ms the 10 mask
            # cases take. The two prefill cases are *fully live*
            # (`col_end == width` on every row), so this branch never runs
            # there: their 4-8% is compile footprint of the extra store path
            # (same effect as the 16 KiB 段宽 / 掩码收尾 precedents).
            #
            # Rejected variants (cold A/B, sentinel geomean): moving the flag
            # lookup *inside* the `dead` test of `_mask_1d_tile` (single store
            # site, `_mask_keep` extracted) 0.9602 — `keep` produced across the
            # scf.if region wrecks the live path (decode-B256-128K 0.955), and
            # the packed win drops to 1.53x; storing a scalar -inf instead of
            # the [TILE] buffer 0.9970 but it UB-overflows on the small
            # padding shapes (1835008 bits > 1572864) with no perf difference.
            tl.store(
                logits + row_off + cols * stride_col,
                tl.full([TILE], -float("inf"), logits.dtype.element_ty),
            )
        else:
            base = (tail - start) // BLOCK_SIZE
            edge = tl.load(flags + flag_off + nblocks)
            # R12/M10 (rejected): the [NB, BLOCK] 2D form of the flag expansion —
            # select on `fl[:, None]` over a [NB, BLOCK] tile, so no per-lane
            # materialisation — lowers through a layout conversion on 910B3 and is
            # much slower: prefill-8192of128K 11248/11292 -> 13380/13361us (0.84x),
            # decode-B256-128K 232/233 -> 315/316us (0.74x), aiv_vec 0.41 -> 0.53
            # and aiv_mte2 0.56 -> 0.41. The 1D `broadcast_to + reshape` stays: its
            # 17.7% sim share buys one [TILE] vector op, cheaper than the shuffle.
            _mask_1d_tile(
                logits,
                flags,
                row_off,
                flag_off,
                cols,
                start,
                end,
                tail,
                width,
                stride_col,
                nblocks,
                flags_pitch,
                base,
                row,
                edge,
                full,
                dead,
                TILE,
                BLOCK_SIZE,
                GATHER,
                TILE_IN_WIDTH,
                PLOAD,
                LIVE_SKIP,
            )


def _select_candidate_blocks_impl(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
) -> None:
    """Select local block IDs by maximum score, pinning each row's newest block.

    Row bounds are in packed column space; absent starts mean zero.
    Decode rows share bounds in groups of ``row_repeat``. Output is -1 padded.
    """
    rows, width = logits.shape
    if not rows:
        return
    if not width:
        out.fill_(-1)
        return
    nblocks = triton.cdiv(width, block_size)
    scores = logits.new_empty((rows, nblocks))
    valid_blocks = torch.empty(rows, dtype=torch.int32, device=logits.device)
    # P12 + P33, rescanned for 910B3 (the A5 values — split<=32, TILE=512 —
    # degenerate here: one 512-block tile per program, so every program pays
    # the start/end load + num_valid prologue). Measured _block_scores_kernel
    # duration on an idle 910B3 over the 10 select shapes:
    #   decode regime (rows<=2048, wide row): TILE=512 beats 128 by 1.22-2.77x,
    #       and ~4 tiles/program (split=ceil(num_tiles/4)) beats split=32 by up
    #       to 42% (decode-B256-128K 76.5 -> 72.0us, mixed 49.2 -> 43.4us).
    #   prefill regime (rows>2048 or narrow row): TILE=512 with split=1 — the
    #       old "TILE=512 costs 1.9x" note predates the i32 fix and is inverted
    #       now (see R14/B15 below); rows alone already fills the cores, so
    #       every extra program is pure prologue (split=32 -> 11569us vs
    #       split=1 -> 9403us).
    #   non-unit column stride (CUDA-graph strided view): TILE=512 fails to
    #       compile (UB overflow — strided addressing needs a wider address
    #       buffer), so that shape stays on 128 with split=4 (14445us vs
    #       15567us at split=32).
    # TILE is also capped by UB: TILE*next_pow2(BLOCK_SIZE) above 4096 elements
    # overflows once strided/multi-buffered.
    if logits.stride(1) != 1:
        # R10/B1: the strided path reads a contiguous 2*BLOCK window per block
        # (see the STRIDED branch), so the old UB cap that forced TILE=128 no
        # longer applies. Single-case A/B (decode-strided-B256-128K, cold):
        # 128/4 14711us (old path) -> 1344 / 1150 (256/4) / 1082 (512/4) /
        # 1061 (512/8) / 1055 (512/16). The window tile costs 2x the UB of the
        # old layout, so it is capped at 4096 elements (TILE*2*BLOCK): TILE=512
        # compiles on the 8-wide decode shape but overflows once the accuracy
        # cases add starts/masked stores, TILE=1024 overflows everywhere.
        tile = 256
        while tile > 1 and tile * 2 * triton.next_power_of_2(block_size) > 4096:
            tile //= 2
        num_tiles = triton.cdiv(nblocks, tile)
        split = min(16, num_tiles)
    elif rows > 2048 or nblocks < 2048 or 512 * triton.next_power_of_2(block_size) > 4096:
        # R14/B15: this tier used to be pinned at TILE=128, on the old note that
        # TILE=512 "costs 1.9x on prefill-8192of128K (UB pressure/multi-buffer)".
        # That measurement predates the R12 i32 fix, which removed the scalar
        # chain that the wide tile amplified; re-measured cold it inverts.
        # The tier reads a [ELEMS, TILE] tile = 8*TILE*4B: 4 KiB at TILE=128,
        # and the read-shape probe puts 4 KiB contiguous segments at 827 GB/s
        # versus 1433 at 16 KiB (opt_runs/probes/probe_bs_read_seg.py), which is
        # exactly what prefill-8192of128K was paying (effective read 754 GB/s at
        # mte2 0.958). TILE=512 makes one 16 KiB segment: 5645.2 -> 3791.3us
        # (1.49x, effective read 1133 GB/s); prefill-1024of8K 71.1 -> 51.3us
        # (1.39x). TILE=1024 overflows UB (needs 216 KiB of the 192 KiB).
        # Known cost, measured: in the same kernel a 512-wide tile also drags the
        # -inf fill off the 16 KiB write plateau (packed-prefill-8192-128K
        # 1533 -> 1180 GB/s, 353 -> 455us). WIDE_MULT (8/16/32 KiB chunks),
        # WTAIL, num_stages and a [WIDE_MULT, TILE]-shaped store were all tried
        # and none of them recovers it (the 2D store scalarises outright); the
        # only fix left is a density test the host cannot make without a sync.
        # Across the case set the tier still wins by 1.8 ms, so TILE=128 stays
        # only as the UB fallback.
        fits = 512 * triton.next_power_of_2(block_size) <= 4096
        tile = 512 if fits else 128
        num_tiles = triton.cdiv(nblocks, tile)
        split = 1 if rows > 2048 else min(num_tiles, max(1, num_tiles // 4))
    elif nblocks <= 4096 and rows <= 16:
        # Narrow-row decode tier (e.g. packed 6x32768 -> 4096 blocks): with only
        # a handful of rows the decode ladder's `num_tiles // 4` leaves the cores
        # idle (6 rows x split 2 = 12 programs). Measured on that case (bf16,
        # warm, same kernel): TILE=256/split=16 27.2us vs TILE=512/split=2
        # 47.0us (1.73x). The 2048-column-per-program target is the one P12/P33
        # swept; only the row count no longer supplies the parallelism here, so
        # `split` takes over.
        # The `rows <= 16` gate is measured, not stylistic: the same rule on a
        # 256x32768 row-major shape (4096 blocks) starves the read instead —
        # TILE=256/split=8 34.2us vs TILE=512/split=2 26.4us (+29.5%), because
        # 256 rows already provide 512 programs and the narrow tile drops the
        # per-program read segment from 16 KiB to 8 KiB.
        fits = 256 * triton.next_power_of_2(block_size) <= 4096
        tile = 256 if fits else 128
        num_tiles = triton.cdiv(nblocks, tile)
        split = min(32, num_tiles, triton.cdiv(2048, rows))
    else:
        tile = 512
        num_tiles = triton.cdiv(nblocks, tile)
        split = min(32, max(1, num_tiles // 4))
    # R13/W1: fix the wide-fill chunk at 16 KiB (4096 blocks f32) instead of
    # the old "8 tiles", which only reached 4 KiB at TILE=128 — the worst rung
    # of the pure-write probe (989 vs 1315 GB/s at 16 KiB). The strided shape
    # keeps 8 tiles (8 KiB): its window tile already costs 16 KiB of UB.
    # The chunk is also capped at `nblocks // tile`, i.e. at the row length: a
    # narrow row (prefill-1024of8K, 1024 blocks = 4 KiB total) can never use a
    # wider store, yet the 16 KiB constant still inflates the compile's UB
    # footprint — A/B on that case: 4096-wide chunk 75.35us vs 1024-wide
    # 69.24us (-8.8%) with the wide loop never executing.
    wide_mult = 8 if logits.stride(1) != 1 else max(
        1, min(4096 // tile, nblocks // tile))
    # R13/W2b: the masked tail costs ~8% on a row that is only one chunk long
    # (prefill-1024of8K: 74.6 vs 69.2us over 3 interleaved rounds) even though
    # that loop never runs there — the extra 4096-lane predicated store inflates
    # the compile's UB/lowering. It is also neutral-to-negative on the decode
    # tier (decode-B256-128K: 3 rounds 109.0/98.7/104.4 vs 108.0/95.0/96.2,
    # inside that case's +-9% placement noise) where the old floor+tile loop is
    # only 7 descriptors. So keep it on the prefill tier only — the one with a
    # row per program and a fill region spanning many chunks.
    wtail = (wide_mult * tile < nblocks) and (rows > 2048 or nblocks < 2048)
    strided = logits.stride(1) == 2
    # R14/B14: only the strided case takes this constexpr, so the other cases'
    # compiled kernels keep their previous constexpr set bit for bit.
    sc2 = strided
    # R17/A1: pair adjacent blocks only when the natural inner run is under the
    # 32 B DMA burst — i.e. bf16/fp16 with BLOCK_SIZE=8 (16 B). fp32 at 8 is
    # already 32 B and regresses 2-4x on the paired form (see the kernel note).
    # The strided branch reads its own 2*BLOCK window instead.
    pair = (
        logits.element_size() * block_size < 32
        and not strided
        and triton.next_power_of_2(block_size) == block_size
        and tile % 2 == 0
    )
    _block_scores_kernel[(rows, split)](
        logits,
        row_ks,
        row_ke,
        scores,
        valid_blocks,
        *logits.stride(),
        row_ks.stride(0) if row_ks is not None else 0,
        row_ke.stride(0),
        scores.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        TILE=tile,
        EXACT_TILES=width % (tile * block_size) == 0,
        STRIDED=strided,
        WIDE_FILL=True,
        WIDE_MULT=wide_mult,
        WTAIL=wtail,
        SPLIT2=True,
        SC2=sc2,
        HOIST=True,
        # The perf-harness prefill cases hand over i64 bounds; the cast at the
        # boundary keeps the scalar chain i32 (see the kernel's I32 note).
        I32=True,
        PAIR=pair,
    )
    # Keep the existing top-k tie behavior.
    top = scores.topk(min(topk_blocks, nblocks), dim=-1)
    # P12 + P33, rescanned for 910B3 (opt_runs/scan_store_tile.py). Long-row
    # prefill (rows > 2048) prefers TILE=2048 — fewer programs, so the
    # per-program prologue is amortised over a 2048-wide row: packed
    # 6833 -> 6404us, prefill-8192 13616 -> 13192us. The decode regime must
    # stay at 256: 512/1024 trade a <=1% win on decode-B256 for a 41% loss on
    # decode-B8 (16 programs cannot fill 40 cores) and a 10% loss on the
    # mixed-length batch, so the no-regression tier wins (Q3).
    # R0/S3 (#15 store_tile rescan, post i32-select): the old tier table
    # (2048 above 2048 rows, else 256) was measured while the select chain was
    # still int64 and scalarised. With the i32 chain, one 2048-wide write per
    # row wins everywhere: decode-B256-128K 14.4 -> 6.6us (2.18x),
    # n6-dspark 66.5 -> 18.8us (3.54x), B8 4.2 -> 2.2us (1.93x),
    # prefill-1024of8K 42.2 -> 25.9us (1.63x), long prefill unchanged
    # (1.01-1.02x). A non-unit out stride needs the wider address buffer that
    # 2048 overflows (UB: 2.0M bits vs 1.5M), so it keeps 256.
    store_tile = 2048 if out.stride(1) == 1 else 256
    while store_tile > 256 and topk_blocks % store_tile:
        store_tile //= 2
    store_k = top.values.shape[1]
    _store_candidates_kernel[(rows, triton.cdiv(topk_blocks, store_tile))](
        top.values,
        top.indices,
        out,
        *out.stride(),
        store_k,
        topk_blocks,
        store_tile,
        store_k == topk_blocks and topk_blocks % store_tile == 0,
        topk_blocks % store_tile == 0,
    )


def _apply_candidate_mask_impl(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int = 1,
) -> None:
    """Mask packed logits outside causal bounds and request-local candidates."""
    rows, width = logits.shape
    if not rows or not width:
        return
    nblocks = triton.cdiv(width, block_size)
    # A6/P6: the mask kernel keeps its address chain in i32 unless the largest
    # offset (row * stride/base width) would overflow 2**31.
    idx_i64 = rows * max(
        logits.stride(0), width, candidate_blocks.stride(0), nblocks + 2
    ) >= 2**31
    # Round 4: the flags row is produced by one dense DMA, so its length is the
    # histogram's constexpr bin count (next_power_of_2(nblocks)) plus the two
    # sentinel slots. For every power-of-two nblocks (all V4.1 widths) that is
    # exactly the old nblocks+2 pitch.
    flags_bins = triton.next_power_of_2(nblocks)
    flags_pitch = flags_bins + 2
    # UB budget of one histogram: 4 bytes/bin. 256K widths need two chunks.
    flags_chunk = min(flags_bins, 16384)
    # 0..nblocks-1 real blocks, nblocks the "past width" edge slot, nblocks+1
    # the trash bin for illegal candidate lanes (round 2).
    flags = torch.empty((rows, flags_pitch), device=logits.device, dtype=torch.uint8)
    start_stride = row_ks.stride(0) if row_ks is not None else 0
    _candidate_flags_kernel[(rows,)](
        candidate_blocks,
        row_ks,
        flags,
        *candidate_blocks.stride(),
        start_stride,
        width,
        nblocks,
        block_size,
        candidate_blocks.shape[1],
        row_ks is not None,
        row_repeat,
        flags_bins,
        flags_chunk,
    )
    # A4/P33, rescanned for 910B3 (opt_runs/scan_mask_tile.py): the mask
    # kernel's cost is one nearly-constant scalar pass per *program*, so the
    # duration scales with the program count and not with the element count:
    # decode-B256-128K 698/359/184/91 ms and packed-prefill 23.8/11.9/6.4/3.4 s
    # at TILE 512/1024/2048/4096. TILE=8192 is not usable (UB overflow: needs
    # 1671168 bits vs 1572864 available), so 4096 is the top of the ladder.
    # A non-unit column stride needs a wider address buffer (probe:
    # opt_runs/probes/probe_mask_tile_strided.py — 2048 and 4096 both overflow
    # UB there), so that path stays on the 1024 it has always used.
    # R5/D10 grid sweep (opt_runs/scan_mask_tile.py, fresh result_dir): the
    # strided *pair* path runs a 2*TILE window, so its ladder is offset by one
    # rung — TILE=2048 (4096-lane window) beats 1024 by 1.063x on
    # decode-strided-B256-128K (676.6 -> 636.4us). The other strided shapes
    # (non-pair fallback, masked store) still overflow UB at 2048, so only the
    # pair-eligible case moves; contiguous stays on the 4096 it scanned to.
    stride_col = logits.stride(1)
    stride_pair = (
        stride_col == 2 and width % 2048 == 0 and logits.stride(0) >= 2 * width
    )
    if stride_col == 1:
        mask_tile = 4096
    elif stride_pair:
        mask_tile = 2048
    else:
        mask_tile = 1024
    num_tiles = triton.cdiv(width, mask_tile)
    _mask_candidates_kernel[(rows, num_tiles)](
        logits,
        row_ks,
        row_ke,
        flags,
        *logits.stride(),
        start_stride,
        row_ke.stride(0),
        width,
        nblocks,
        flags_pitch,
        block_size,
        row_ks is not None,
        row_repeat,
        mask_tile,
        width % mask_tile == 0,
        logits.stride(1) == 1,
        idx_i64,
        # R4/MK-new-3: the pair-window rewrite needs an even column stride, full
        # tiles (so the 2*TILE window never leaves the row) and a row pitch that
        # really holds 2*width columns.
        logits.stride(1) == 2
        and width % mask_tile == 0
        and logits.stride(0) >= 2 * width,
        # R10/M2 + R11/M5 + R11/M7: dead-tile load skip, dead-tile store hoist,
        # full-tile valid skip and masked keep-load are all unconditional here.
        DEAD_SKIP=True,
        DEAD_STORE=True,
        LIVE_SKIP=True,
        PLOAD=True,
    )


@torch.library.custom_op("mojo_npu_triton_a2::select_candidate_blocks_fwd", mutates_args=("out",))
def select_candidate_blocks_fwd(
    logits: torch.Tensor,
    row_ks: Optional[torch.Tensor],
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int,
) -> None:
    _select_candidate_blocks_impl(
        logits, row_ks, row_ke, topk_blocks, block_size, out, row_repeat
    )


@select_candidate_blocks_fwd.register_fake
def _select_candidate_blocks_fwd_fake(
    logits, row_ks, row_ke, topk_blocks, block_size, out, row_repeat
) -> None:
    return None


@torch.library.custom_op("mojo_npu_triton_a2::apply_candidate_mask_fwd", mutates_args=("logits",))
def apply_candidate_mask_fwd(
    logits: torch.Tensor,
    row_ks: Optional[torch.Tensor],
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int,
) -> None:
    _apply_candidate_mask_impl(
        logits, row_ks, row_ke, candidate_blocks, block_size, row_repeat
    )


@apply_candidate_mask_fwd.register_fake
def _apply_candidate_mask_fwd_fake(
    logits, row_ks, row_ke, candidate_blocks, block_size, row_repeat
) -> None:
    return None
