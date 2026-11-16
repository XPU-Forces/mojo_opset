"""DSA candidate-block selection over packed logits (NPU Triton).

Both entry points are inference-only and write in place:
``select_candidate_blocks_fwd`` fills ``out`` with -1 padded block ids, and
``apply_candidate_mask_fwd`` masks ``logits`` to -inf outside the candidates.
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
):
    # Writes each row's valid block count.
    row = tl.program_id(0).to(tl.int64)
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
    if tl.program_id(1) == 0:
        tl.store(valid_blocks + row, num_valid)
    elems_range = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    num_tiles = tl.cdiv(nblocks, TILE)
    # P16 (redundant_boundary_operation): when BLOCK_SIZE is a power of two
    # the elems_range mask is vacuous, and when width is TILE-aligned the
    # store mask is vacuous — both compile away (KVR-provable identities).
    pow2_bs = (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0
    # Wide-vector merged reduction (§9.8 遗留 #1, 写法 A): blocks live on the
    # lane axis — [ELEMS, TILE] with the reduction along axis=0 — so the
    # per-block max compiles to full-width vmax across the element rows:
    # one vector instruction advances 32 blocks at once, instead of a
    # narrow stride-BLOCK_SIZE segmented sequence per block (measured 668
    # cycles/block, vector-issue bound with mte2 idle).
    #
    # End clipping (方向 2, §6 of select_kernels_gpu_vs_npu.md): a tile whose
    # first block lies past the row's num_valid is entirely masked — skip the
    # 16KB load + reduce tree and store the -inf constant directly. The
    # store itself must NOT be skipped: scores is uninitialized memory and
    # topk scans the whole row, so an unwritten tail would leak garbage
    # into the candidates. The criterion reuses the prologue scalar
    # num_valid (already computed for valid_blocks) — zero extra loads.
    # Uniform per-tile scalar branch; tile t is empty iff t*TILE >= num_valid.
    # R1 (coalesce_tail_fill): the empty tiles all sit at the row tail, so they
    # used to be written one TILE (2KB) at a time by whichever program owned
    # the stripe slot — 16 separate 2KB stores per row on the packed shape,
    # i.e. a 417 GB/s fill against a 1.6 TB/s memset wall. The loop now stops
    # at the first empty tile and the remaining [-inf] tail is filled by a
    # second striped loop whose store is FILL_CHUNK (16KB) wide.
    first_empty = tl.cdiv(num_valid, TILE)
    FILL_CHUNK: tl.constexpr = 4096
    for tile_id in range(tl.program_id(1), num_tiles, tl.num_programs(1)):
        blocks = tile_id * TILE + tl.arange(0, TILE)
        if tile_id * TILE < num_valid:
            cols = start + blocks[None, :] * BLOCK_SIZE + elems_range[:, None]
            # P16 three-state tile (simulator evidence, §5.5): a tile whose
            # every block is complete has a provably all-true
            # `cols < col_end` mask, so that path drops the mask entirely —
            # the per-line mask compare + other-select machinery (~34 SIMT
            # instructions per 128B line in the load chain, ISETP compares
            # among them) compile away. Uniform per-tile scalar branch,
            # same class as the endclip fast-path; partial tiles keep the
            # masked path.
            if (tile_id + 1) * TILE <= num_full:
                # P4 (contiguous flat load) is retired: loading a full tile as
                # one contiguous run and transposing it in registers moved the
                # load chain off MTE2 onto the vector pipe (profiler on
                # decode-128K: mte2 0.80 -> 0.55, vec 0.54 -> 0.97) and was
                # slower on every shape measured, so the masked gather below
                # serves full and partial tiles alike.
                values = tl.load(
                    logits + row * stride_row + cols * stride_col,
                    elems_range[:, None] < BLOCK_SIZE,
                    other=-float("inf"),
                ).to(tl.float32)
            else:
                if pow2_bs:
                    values = tl.load(
                        logits + row * stride_row + cols * stride_col,
                        cols < col_end,
                        other=-float("inf"),
                    ).to(tl.float32)
                else:
                    values = tl.load(
                        logits + row * stride_row + cols * stride_col,
                        (elems_range[:, None] < BLOCK_SIZE) & (cols < col_end),
                        other=-float("inf"),
                    ).to(tl.float32)
            reduced = tl.reduce(values, 0, _max_with_nan)
            # Pin the newest block.
            pinned = (end > start) & (blocks == (end - start - 1) // BLOCK_SIZE)
            reduced = tl.where(pinned, float("inf"), reduced)
            if EXACT_TILES:
                tl.store(scores + row * stride_scores + blocks, reduced)
            else:
                tl.store(scores + row * stride_scores + blocks, reduced, blocks < nblocks)
        # R1: empty tiles write nothing here — the whole -inf tail is emitted
        # by the FILL_CHUNK loop below. Keeping this loop's trip count static
        # (num_tiles) is what preserves the compiler's pipelining; a runtime
        # `min(num_tiles, first_empty)` bound cost 1.46x on prefill-8192.
    # R1: the -inf tail is emitted in FILL_CHUNK-wide (16KB) contiguous stores,
    # striped across the same programs. Total bytes are unchanged; the store
    # count drops from nblocks/TILE to nblocks/FILL_CHUNK and each MTE3 burst
    # is 8x wider.
    fill_base = first_empty * TILE
    n_fill = tl.maximum(nblocks - fill_base, 0)
    fill_val = tl.full([FILL_CHUNK], -float("inf"), tl.float32)
    pid = tl.program_id(1)
    nprog = tl.num_programs(1)
    # R3 (unmasked_tail_fill): `idx < nblocks` can only ever be false inside
    # the LAST partial chunk — every earlier chunk lies inside the row. Binding
    # the mask at chunk granularity instead of element granularity peels that
    # chunk off, so the bulk store issues with no per-element compare/select on
    # the vector pipe. The packed shape is 94% fill bytes and was vec-bound
    # (0.88) with mte2 idle (0.00), i.e. the mask compare was critical-pipe
    # work, not the DMA.
    n_full = n_fill // FILL_CHUNK
    for chunk in range(pid, n_full, nprog):
        idx = fill_base + chunk * FILL_CHUNK + tl.arange(0, FILL_CHUNK)
        tl.store(scores + row * stride_scores + idx, fill_val)
    rem_base = fill_base + n_full * FILL_CHUNK
    rem = nblocks - rem_base
    if (pid == 0) & (rem > 0):
        idx = rem_base + tl.arange(0, FILL_CHUNK)
        tl.store(scores + row * stride_scores + idx, fill_val, idx < nblocks)


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
    FULL: tl.constexpr,
):
    # P16 (redundant_boundary_operation): `cols < k` uses the runtime scalar k
    # (do_not_specialize), so the compiler cannot prove it vacuous even when
    # k == OUT_K (9/10 cases) and pays the per-element gather/select
    # machinery on every masked load (measured ~300 cycles of non-data
    # overhead per 1.5KB masked load, mte2-bound). When k == OUT_K and
    # OUT_K % TILE == 0 the load mask is provably all-true AND the store
    # mask `cols < OUT_K` (also not foldable: cols carries the runtime pid)
    # is provably all-true — both compile away on a constexpr path. The
    # k < OUT_K case (1024of8K: k=896) keeps the original masked path.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    if FULL:
        value = tl.load(values + row * k + cols)
        index = tl.load(indices + row * k + cols)
    else:
        value = tl.load(values + row * k + cols, cols < k, other=-float("inf"))
        index = tl.load(indices + row * k + cols, cols < k, other=-1)
    # NaN scores can occur during warmup; only -inf denotes padding.
    if FULL:
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
    I32_INDEX: tl.constexpr,
    BINS: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    # P34 (histogram_dense_row): instead of scattering K random 1-byte stores
    # into the row, bin the candidates in UB with tl.histogram and write the
    # whole row back with contiguous stores. Measured on 950PR (rows=256,
    # K=2048): 52.1us -> 18.5us @128K (2.79x), 1435us -> 467us @prefill-8192
    # (3.08x), 38.2us -> 17.0us @32K (2.25x), 92.0us -> 54.5us @8K (1.69x).
    # The dense write-back also removes the host zero-init (P24).
    #
    # A5 lowering limits (measured, both hard failures): bins < 4096 does not
    # compile (1025/2048 fail), and a single pass tops out at 24576 bins
    # (28672 = 112KB UB fails). A loop body double-buffers, so only 10240 bins
    # per pass compile (12288 fails) — 256K bins in 4 passes. Beyond ~32769
    # buckets the write-out grows with width while the scatter cost does not,
    # so wider rows keep the scatter path (host passes BINS=0).
    #
    # P24 (preinitialized_output_optimization): with BINS=0 the flags buffer
    # is host zero-initialized (torch.zeros), so the kernel only scatters the
    # sparse 1s — the in-kernel default-fill pass and its ordering barrier are
    # gone (each program exclusively owns its row, so no same-address hazard).
    #
    # P6 (avoid_scalar_lowering): the index chain used to be i64 (explicit
    # .to(tl.int64) on row and block). HIVM lowers i64 vadd/vmul to scalar
    # loops (10x-100x), and integer vcmp only survives as i32 EQ/NE — so the
    # i64 index chain AND both ordering compares (`pos >= width`,
    # `block >= 0`) were scalar loops over K=2048 lanes, which is exactly the
    # measured profile (aiv_vec 1.00, mte3 0.00). Now indices stay i32 (host
    # guarantees rows*max(stride_row, nblocks+1) < 2**31) and the two
    # ordering compares are f32 vcmp (exact: |pos| <= 2**24).
    if I32_INDEX:
        row = tl.program_id(0)
    else:
        row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    cols = tl.arange(0, triton.next_power_of_2(K))
    # P16 (redundant_boundary_operation): with power-of-two K the cols < K
    # guards on the gather and scatter are vacuous and compile away.
    if (K & (K - 1)) == 0:
        block = tl.load(candidates + row * stride_row + cols * stride_col)
    else:
        block = tl.load(
            candidates + row * stride_row + cols * stride_col, cols < K, other=-1
        )
    if not I32_INDEX:
        block = block.to(tl.int64)
    blockf = block.to(tl.float32)
    # Preserve the packed-column clamp for candidates beyond the logits width.
    block = tl.where(start + blockf * BLOCK_SIZE >= width, nblocks, block)
    if (K & (K - 1)) == 0:
        keep = blockf >= 0
    else:
        keep = (cols < K) & (blockf >= 0)
    if BINS == 0:
        tl.store(flags + row * (nblocks + 1) + block, 1, keep)
    else:
        # `block` is already clamped into [0, nblocks], i.e. exactly the row's
        # bucket space, so bins == buckets and no index shift is needed.
        buckets = nblocks + 1
        o = tl.arange(0, BINS)
        if NCHUNK == 1:
            h = tl.histogram(tl.where(keep, block, 0).to(tl.int32), BINS, keep)
            tl.store(flags + row * buckets + o, (h != 0).to(tl.uint8), o < buckets)
        else:
            for c in tl.static_range(0, NCHUNK):
                lo = c * BINS
                m = keep & (block >= lo) & (block < lo + BINS)
                h = tl.histogram(
                    tl.where(m, block - lo, 0).to(tl.int32), BINS, m
                )
                tl.store(
                    flags + row * buckets + lo + o,
                    (h != 0).to(tl.uint8),
                    lo + o < buckets,
                )


@libentry()
@triton.jit(do_not_specialize=["width", "nblocks"])
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
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
    EXACT_TILES: tl.constexpr,
    I32_INDEX: tl.constexpr,
):
    # P4 (discrete_memory_access): the per-column keep flag used to be a
    # global-memory gather at computed addresses. The tile's flags are one
    # contiguous window (blocks are consecutive), so preload that window
    # and look it up with tl.gather — the KB preload-then-gather pattern.
    # The window base follows the tile; out-of-window lanes only occur
    # where valid is already false, so a clamped index is harmless. uint8
    # is cast to fp32 because tl.gather only supports floating dtypes.
    #
    # M2 (avoid_scalar_lowering, same recipe as P6 on the flags kernel): the
    # i64 `row` widened both per-lane address chains (logits write and flags
    # window/edge read) to i64 vadd, which HIVM lowers to scalar loops. i32
    # row keeps the whole chain i32; host guarantees the largest byte offset
    # of both tensors fits in 2**31 and falls back to the i64 path otherwise.
    if I32_INDEX:
        row = tl.program_id(0)
    else:
        row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    # P16 (redundant_boundary_operation): with width % TILE == 0 every tile
    # lies in-bounds, so the cols < width guards are vacuous and compile
    # away. Constexpr branch only — no runtime control flow (a runtime
    # last-tile branch measured a 28% regression).
    if EXACT_TILES:
        valid = (cols >= start) & (cols < end)
    else:
        valid = (cols >= start) & (cols < end) & (cols < width)
    block = (cols - start) // BLOCK_SIZE
    BPT: tl.constexpr = triton.next_power_of_2(TILE // BLOCK_SIZE + 2)
    # The window base follows the tile; floor (not ceil) so the first
    # columns of a misaligned tile — whose blocks sit at b_lo - 1 — stay
    # covered (the window carries +2 blocks of slack). Out-of-window lanes
    # only occur where valid is already false, so clamping is harmless.
    b_lo = tl.maximum((tl.program_id(1) * TILE - start) // BLOCK_SIZE, 0)
    offs = tl.arange(0, BPT)
    window = tl.load(
        flags + row * (nblocks + 1) + b_lo + offs,
        b_lo + offs <= nblocks,
        other=0,
    ).to(tl.float32)
    keep = tl.gather(window, tl.minimum(tl.maximum(block - b_lo, 0), BPT - 1), 0)
    edge = tl.load(flags + row * (nblocks + 1) + nblocks)
    keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
    if EXACT_TILES:
        tl.store(
            logits + row * stride_row + cols * stride_col,
            -float("inf"),
            ~(valid & keep),
        )
    else:
        tl.store(
            logits + row * stride_row + cols * stride_col,
            -float("inf"),
            (cols < width) & ~(valid & keep),
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
    # R2 (P0 offline sweep -> static dispatch, opt_runs/sweep_dispatch.md):
    # the previous rule (P12/P33) was a hand-tuned pair of thresholds; a full
    # sweep over TILE x split on all 10 select shapes found two systematic
    # gaps:
    #  * TILE: 256 wins every narrow shape (nblocks <= 4096) — the 8K chunk
    #    went 80.0 -> 70.4 us (1.14x) — while wide rows keep 512 (TILE=1024
    #    was re-confirmed a loss: prefill-8192 5192 -> 5345 us, the 32KB
    #    [8,1024] tile halves resident occupancy).
    #  * split: the load is 16KB/tile, so ~2048 columns per program is the
    #    sweet spot; split = cdiv(2048, rows) beats a flat 32 on every decode
    #    shape (s8/s16 1.03~1.06x) because the prologue (row bounds, three-
    #    state tests) is re-paid per program. Packed-8192 is the extreme:
    #    split=32 1230.8 -> split=1 518.4 us (2.37x).
    # Store TILE stays P33's rule: the sweep's "1024 for rows < 2048" hint
    # regressed every decode shape (store 5.11 -> 5.81 us, 0.88x) — the
    # 2048-wide FULL path is already one program per row there.
    TILE = 256 if nblocks <= 4096 else 512
    num_tiles = triton.cdiv(nblocks, TILE)
    split = max(1, min(32, num_tiles, triton.cdiv(2048, rows)))
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
        TILE=TILE,
        EXACT_TILES=width % (TILE * block_size) == 0,
    )
    # Keep the existing top-k tie behavior.
    top = scores.topk(min(topk_blocks, nblocks), dim=-1)
    k = top.values.shape[1]
    # P33 (block_size_scaling): TILE ladder measured per shape — 2048 for
    # the FULL path (one program per row, 32KB/笔) but 1024 when k < OUT_K
    # (1024of8K k=896: a 2048 tile would half-fill masked loads, measured
    # 0.924x; 1024 is its measured optimum). Host-side shape dispatch per
    # the P12 precedent.
    STORE_TILE = 2048 if k == topk_blocks else 1024
    _store_candidates_kernel[(rows, triton.cdiv(topk_blocks, STORE_TILE))](
        top.values,
        top.indices,
        out,
        *out.stride(),
        k,
        topk_blocks,
        STORE_TILE,
        # P16: FULL only when the load/store masks are provably vacuous.
        FULL=(k == topk_blocks) and (topk_blocks % STORE_TILE == 0),
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
    # P34: pick the flags build path. bins >= 4096 (A5 lowering floor) and one
    # pass tops out at 24576 bins; a loop body double-buffers, so 10240 bins per
    # pass is the widest that compiles (256K -> 4 passes). Past ~32769 buckets
    # the dense write-out grows with width while the scatter does not, so keep
    # the scatter path there.
    buckets = nblocks + 1
    if buckets <= 24576:
        bins, nchunk = max(4096, buckets), 1
    elif buckets <= 40960:
        bins, nchunk = 10240, triton.cdiv(buckets, 10240)
    else:
        bins, nchunk = 0, 1
    flags = torch.empty(
        (rows, buckets), device=logits.device, dtype=torch.uint8
    ) if bins else torch.zeros(
        (rows, buckets), device=logits.device, dtype=torch.uint8
    )
    start_stride = row_ks.stride(0) if row_ks is not None else 0
    # P6: keep the whole index chain in i32 when the largest byte offset of
    # both the candidate read (rows*stride_row) and the flags scatter
    # (rows*(nblocks+1)) fits; otherwise fall back to the i64 path.
    i32_index = rows * max(candidate_blocks.stride(0), nblocks + 1) < 2**31
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
        i32_index,
        bins,
        nchunk,
    )
    # P33 (block_size_scaling): the mask kernel's per-tile scalar prologue
    # (start/end loads + window setup) dominates at small TILE, so the ladder
    # climbs: 512 -> 0.439x, 1024 base, 2048 -> 1.20x, 4096 -> 1.34x cumulated;
    # 8192 rejected (decode-B8 drops below the 0.97 floor — 128 programs
    # cannot fill the AIVs). 4096 is the measured optimum for the 32K..256K
    # width band of the real-workload cases.
    _mask_candidates_kernel[(rows, triton.cdiv(width, 4096))](
        logits,
        row_ks,
        row_ke,
        flags,
        *logits.stride(),
        start_stride,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        4096,
        width % 4096 == 0,
        # M2: the largest offsets the mask kernel touches are the last logits
        # element (rows-1)*stride_row + (width-1)*stride_col and the last
        # flags slot rows*(nblocks+1); both fit i32 in every measured shape,
        # but a caller with a pathological stride falls back to the i64 path.
        max((rows - 1) * logits.stride(0) + (width - 1) * logits.stride(1),
            rows * (nblocks + 1)) < 2**31,
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
