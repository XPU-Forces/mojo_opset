# SPDX-License-Identifier: Apache-2.0
"""vision_pos_embed_interp: fused bilinear position-embedding interpolation
kernel for Ascend NPU, based on triton-ascend >= 3.6.0.

Port of the upstream vLLM Triton kernel ``_bilinear_pos_embed_kernel`` /
``triton_pos_embed_interpolate``
(vllm/model_executor/models/qwen3_vl.py:186-298, main@aa31fa0), with:

1. Ascend affinity: ``tl.math.floor`` replaced by int truncation
   (``h_frac``/``w_frac`` are non-negative by construction, where
   ``floor(x) == (int)x``).
2. Multi-image merged single-launch kernel
   (``triton_pos_embed_interpolate_merged``): replaces the per-image
   launch + ``torch.cat`` loop in ``fast_pos_embed_interpolate``
   (qwen3_vl.py:744-763). Output is bitwise identical to the per-image
   version.
3. Public API of the upstream wrapper is unchanged; the merged wrapper is
   an additive API.

Semantics (must match native benchmark ``pos_embed_interpolate_native``):
- spatial-merge reorder baked into index arithmetic;
- bilinear weights computed in fp32 then cast to output dtype before the
  multiply-accumulate (:236-251 rounding path preserved);
- temporal dim recomputed per program (``pid % (h*w)``), so ``t > 1``
  output is bitwise identical to ``t == 1`` repeated.

Ascend tuning (r6 NPU evidence, 268800-row large-video):
- grid capped at 32768 programs + CHUNKS static unroll (beats 65535:
  FP32 92.5% vs 87.4% MBU);
- hidden split into exact power-of-two lane blocks (1152 -> 1024 + 128)
  so no vector lane is masked padding: BF16 is compute-bound at BLOCK_D
  padding (40.1% MBU vs FP32 92.5%), exact lanes cut the waste.

mojo_opset delivery layout (per task book):
- kernels/npu_a2_triton/ and kernels/npu_a5_triton/: this module,
  registered in each package __init__.py OPS and config/npu_a2.yaml /
  npu_a5.yaml;
- functions/: single-image ``triton_pos_embed_interpolate``;
- modules/: ``fast_pos_embed_interpolate`` integration wrapper;
- tests/: precision suite (tests/ops/test_vit_bilinear_pos_embed.py),
  performance suite (tests/performance/...).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # pragma: no cover - environment without triton
    HAS_TRITON = False

__all__ = [
    "HAS_TRITON",
    "triton_ascend_available",
    "triton_pos_embed_interpolate",
    "triton_pos_embed_interpolate_merged",
    "pos_embed_interpolate_native",
]

_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)

# Ascend rtKernelLaunch caps the TOTAL program count (product of grid
# dims) below 65536 (EE1003, r3 evidence). r6 sweep on a 268800-row
# launch showed the tighter cap is faster for both dtypes (FP32 92.5%
# vs 87.4% MBU, BF16 40.1% vs 37.4%): each program unrolls CHUNKS rows.
_GRID_CAP = 32768

# Bench-only launch override hook (r19 diagnostics). ``None`` keeps the
# triton default num_warps; the bench driver may set e.g. 8/16 to probe
# issue-rate limits on the element-bound BF16 path. Never set by library
# code, so production behaviour is unchanged.
_LAUNCH_NUM_WARPS = None
# r56 P5 bench-only probe: num_stages (never tested; None = backend
# default). See _launch_opts() comment. Production stays None.
_LAUNCH_NUM_STAGES = None

# Bench-only grid-cap override (r24 diagnostics). A lower cap raises the
# per-program CHUNKS unroll, i.e. more independent row chains inside one
# program -- r23 (job 942) op_summary data shows the tail shape (7200
# pair slots, CHUNKS=1) at 32 Gelem/s while every CHUNKS>=2 shape runs
# 87-108 Gelem/s, so per-program chain count is the prime latency-hiding
# suspect. Bitwise-neutral: the same rows are computed with identical
# per-row ops regardless of the cap. ``None`` keeps ``_GRID_CAP``.
_GRID_CAP_OVERRIDE = None

# Bench-only kernel-choice override (r24 diagnostics). Truthy forces the
# quad (4-rows-per-program) kernel in the single-image wrapper whenever
# its launch contract holds (r9-proven bitwise identical); ``None`` keeps
# the production pair/single choice.
_FORCE_QUAD = None

# ---- r53 bench-only 2D-pair force switch -----------------------------------
# Truthy forces the 2D pair kernel in the single-image wrapper whenever the
# pair launch contract holds. r53 evidence base: mid-bf16 41.9% MBU runs the
# plain pair kernel where w_scale == 1.0 makes EVERY pair take the diff
# branch (8 scalar 1D embed loads, zero sharing), and large-bf16 34.7% runs
# mb(BODIES=2) while x-nomb was only ever measured on f32. The 2D pair
# kernel issues the pair's two-column embed rows as [2, WIDTH] strided
# ops: the two columns are ADJACENT grid columns (wf1 == wf0 + 1, wc1 ==
# wc0 + 1), so their embed rows sit exactly HIDDEN_DIM apart in the
# row-major [G*G, hidden] table -- one [2, WIDTH] load covers both, the
# per-element FMA is unchanged, and the two 1D stores merge into one 2D
# store with row stride HIDDEN_DIM. Halves vector-instruction count per
# element on the diff branch. Values and addresses are identical to the
# scalar pair kernel -> bitwise identical by construction (rawAB-pair2d
# gates it every round). Never set by library code, so production
# behaviour is unchanged.
_FORCE_PAIR2D = None

# ---- r54 bench-only vertical-only pair kernel (w_scale == 1.0 shapes) --------
# Truthy forces the vpair kernel in the single-image wrapper whenever the
# pair launch contract holds AND w_scale == 1.0 exactly (host float
# compare). r54 evidence base: tb-mid (1,32,48) has w_scale = 47/47 = 1.0
# exact, so every pair takes the diff branch with dw0 == dw1 == +0.0 --
# the wc-column embed rows carry w01 == w11 == +0.0 AND b0 == a1, d0 == c1
# are loaded twice: 8 scalar 1D embed loads per pair serve 4 weight-bearing
# rows. The vpair body loads only (a0, c0, a1, c1): 8 -> 4 loads and
# 8 mul + 6 add -> 4 mul + 2 add per segment per pair, attacking the r7
# evidence that tl.load occupies the same issue pipeline as the bf16 FMA
# work (mid-bf16 41.6% is the remaining line FAIL). Weights are computed
# with the verbatim production f32 chain; the dropped terms are exactly
# +0.0 additions, so divergence is limited to vanishing +/-0 corner cases
# (rawAB-vpair gates vpair vs pair bitwise on tb-mid both dtypes). Only
# W == G shapes qualify (w_scale == 1.0 iff W == G for W > 1), so wf1 =
# col0 + 1 <= G - 1 stays in bounds. Never set by library code, so
# production behaviour is unchanged.
_FORCE_VPAIR = None

# ---- r55 vpair production promotion (mid-bf16 instruction-issue attack) ----
# r55 promotes the vpair kernel from bench-only to PRODUCTION dispatch for
# every shape that already satisfies the pair launch contract AND has
# w_scale == 1.0 exactly (tb-mid 1x32x48 is the only line shape; W == G
# is the exact-host-compare gate). Evidence chain: mid-bf16 t_dev == mid-f32
# t_dev at half the bytes across r50-r53 (0.030ms both) -> instruction-count
# bound, and vpair cuts the per-element issue count from 11 slots (4 loads +
# 4 mul + 3 add) to 5 (2 loads + 2 mul + 1 add) with stores untouched.
# Correctness: r54 rawAB-vpair PASSED bitwise vs production pair on tb-mid
# BOTH dtypes on-machine; the dropped wc-column terms are exactly +0.0
# additions, so divergence is limited to vanishing +/-0 corners AND inputs
# carrying Inf/NaN (production propagates 0.0*Inf/NaN into the output,
# vpair drops the term) -- adversarial-review note; make_embed data and
# real embedding tables are finite, and the full 79-case precision suite
# re-proves every round (P27-merged-bitwise becomes a cross-kernel bitwise
# comparison for its (1,32,48) leg after promotion). _VPAIR_DISABLE exists
# so run.py can bench the legacy pair kernel inside the same run
# (within-run A/B immune to DVFS machine state, r54 lesson).
# r56 ROLLBACK: 1030 (r54b healthy retest) measured x-vpair-tb-mid at
# 0.030ms == production pair 0.030ms BOTH dtypes -- the 55% instruction
# cut buys ZERO time, so mid is not issue-bound; promotion reverted to
# bench-only (_FORCE_VPAIR) until a lever that actually moves the wall
# (vmb S-amortization) is proven. The flag stays for the vmb chain.
_VPAIR_PROMOTE = False
_VPAIR_DISABLE = False

# r56 bench-only multi-body vpair (mid S-attack). Int BODIES (2/3/4) forces
# _bilinear_pos_embed_kernel_vmb on shapes that satisfy the pair contract,
# w_scale == 1.0 exact AND total_out >= _VMB_MIN_ROWS; None keeps
# production unchanged. 1030 healthy-state evidence: mid is
# program-startup bound (x-vpair instruction cut == zero time change,
# 19-wave S model ~= observed 30us), so BODIES amortization -- not
# instruction count -- is the mid lever.
_FORCE_VMB = None
# minimum output rows for the vmb dispatch (mid 1536
# qualifies; short-video 800 stays plain pair, its line is green).
_VMB_MIN_ROWS = 1024
# r57 PROMOTION (pre-registered 1033 verdict): rawAB-vmb bitwise PASS both
# dtypes on-machine + x-vmb2 device p50 17.8us (f32) / 17.3us (bf16) vs
# production pair 30.6/30.2us -> S halved on BOTH dtypes (no dtype-split,
# unlike mb4-mid). _VMB_PROMOTE = production BODIES (2 -> 384 programs on
# mid, ~9.6 waves); projected mid-bf16 MBU 41.7*30.2/17.3 ~= 73 >= 60.
# _VMB_DISABLE lets run.py bench the legacy pair kernel in the same run
# for machine-state-immune A/B and non-tautological rawAB gates.
_VMB_PROMOTE = 2
_VMB_DISABLE = False

# r58 bench-only ammo (expert panel verdict consult/expert_panel_verdict.md
# R5/R7, dormant until the 1034 probes land and re-seal AFTER job 1033
# returns -- resealing staging while a pending job holds an older sealed
# copy is what killed 1032 with CHEAT_DETECTED, discipline entered
# 2026-11-18). Both follow the r55 lesson: bench-only gate, promote only
# from same-run A/B evidence against a pre-registered threshold.
#
# 1) _FORCE_VMB_LA (int BODIES): vmb with a load-ahead vbody -- the two
#    parameter S/B model says mid's wall is program startup S; vmb2
#    (sequential vbody) reads Bv via x-vmb2-tb-mid-bf16. If Bv lands in
#    (1.11, 1.45] us (panel branch C/D), the la form overlaps each
#    segment's load latency with the previous segment's store round
#    (statement order only, zero arithmetic change -> bitwise identical
#    to vbody/vpair modulo the same vanishing +/-0 corner class,
#    rawAB-vmb-la gates it).
# 2) _FORCE_DIFF6: pair kernel with the diff-branch load reuse -- on
#    0 < w_scale <= 1.0 the diff branch (wf1 != wf0) forces
#    wf1 == wf0 + 1 and wf0+1 <= G-1, so wc0 == wf1 and off_a1 ==
#    off_b0, off_c1 == off_d0 exactly (algebra verified against
#    _bilinear_pair_body; w_scale > 1.0 breaks it and stays excluded).
#    Dropping the two duplicate 8th loads is pure address-equal load
#    reuse: ZERO divergence class (stronger than the vpair +/-0 class);
#    rawAB-diff6 must pass strictly. Large (w_scale ~= 0.595) pays 8
#    loads/pair/segment in the diff branch ~= 60% of its rows.
_FORCE_VMB_LA = None
_FORCE_DIFF6 = None
_FORCE_MB_D6 = None
_FORCE_MB_D6S = None
_FORCE_MB_D6S2 = None
# r59 production promotion: mb_d6 (mb structure x diff6 load-reuse body)
# for t==1 large shapes (0 < w_scale < 1). Local-machine evidence:
# 9.0 vs 16.75 ns/row vs the then-default mb2 production (-46%) on
# (1,60,80) bf16 (4-point slope fit + measurable-shape diff), bitwise
# identical by construction (d6 address-equal reuse, zero divergence).
_MB_D6_PROMOTE = True

# Bench-only band-kernel disable (diagnostics). Truthy keeps the
# production choice at pair/single even when the band grouping qualifies.
# r27 (job 1000): v1 band form 3.1x slower than pair (no CSE of per-row
# embed loads). r28b (job 1002) op_summary truth: band2 big-video-f32
# 16.265ms vs pair 2.768ms (5.9x), tail 1.773ms vs 0.62ms (2.9x) --
# deterministic across all x-nw/x-cap variants. Two strikes: band
# permanently disabled from r29 (job 1003); production stays pair/single.
# run.py still toggles this for the x-noband rows (restores True now).
_BAND_DISABLE = True

# ---- r30 flat multi-body pair kernel knobs ---------------------------------
#
# r29 (job 1003) op_summary truth + x-row evidence: per-program wall time
# fits S + n*B with S ~= 1.7us (program startup, paid once per program)
# and B ~= 3.4us per pair body on the 6-segment tail shape (hidden=1000)
# versus B ~= 0.65us on 2-segment shapes (hidden=1152). Each _seg_* call
# is a serial load->FMA->store round: on the in-order AIV pipeline the
# next segment's loads cannot issue until the previous segment's FMAs
# issue, so a 6-segment body eats ~6 memory latencies. Evidence: quad
# (2 pair bodies/program) 0.761 -> 0.612ms and cap2048 (4 bodies) ->
# 0.654ms on tail, nw8/16 neutral. The flat multi-body kernel attacks
# both terms with zero new addressing (band/band2's row tables are
# proven 3.1x/5.9x slower -- dead end):
# - BODIES pair bodies per program amortizes S (static_range unroll of
#   the proven _bilinear_pair_body mapping, rows 2*(pid*BODIES+b), +1);
# - the FLAT body issues a whole wave of segment loads before any FMA
#   so several memory latencies overlap inside one body.
# Per-element arithmetic (casts, expression order) is untouched ->
# bitwise identical to pair/single by construction (rawAB-mb gates it).
_MB_DISABLE = False
# shapes below this many output rows keep the plain pair kernel (their
# lines are already green; small grids must keep >= ~1 wave of programs)
_MB_MIN_ROWS = 2048
# r31: shapes above this many output rows also keep the plain pair
# kernel. r30 op_summary truth: mb wins when the pair grid would be
# shallow (large 4800 rows 51.7->80.8%, tail 14400 rows +1.6pt) but
# LOSES on deep grids (big-video 268800 rows: mb 49.6% vs pair 64.6% --
# bodies-fold serial chains hurt once the grid already fills the AIVs).
_MB_MAX_ROWS = 100000
# target program count: BODIES = clamp(slots // TARGET, 1, MAX) so the
# grid stays deep enough to fill the AIVs while amortizing startup.
# r55: 1024 -> 512 promotes tb-large (slots 2400) from BODIES=2 to
# BODIES=4. r54 within-run evidence (same degraded-DVFS state, so the
# delta is machine-state-immune): x-mb4-tb-large f32 0.055ms vs mb2
# production 0.092 (-40%, still faster than the r53 HEALTHY mb2 0.058ms)
# and x-mb4-tb-large-bf16 0.067 vs 0.093 (-28%, ties the healthy mb2
# time). mb8 0.091 shows BODIES=4 is the sweet spot. Scope check: mid
# (1536 rows) and short-video (800) sit below _MB_MIN_ROWS and keep the
# pair kernel; big-video (268800 rows) sits above _MB_MAX_ROWS and keeps
# pair; merged/raw paths have their own knobs -- tb-large is the only
# production line shape whose dispatch changes. The trep bodies formula
# is decoupled via _TREP_TARGET_PROGRAMS (adversarial-review fix: it
# previously shared this knob and would have silently moved line-checked
# tb-tail trep BODIES 4->8). mb4-large correctness was
# machine-proven in r54 (precision gates + x-mb4 rows both dtypes).
# r56 ROLLBACK: 1030 (r54b healthy retest) measured x-mb4-tb-large-bf16 at
# 0.070ms vs production mb2 0.067ms (WORSE by 4.5%) while x-mb4-tb-large
# f32 was only -7% (0.054 vs 0.058) -- the r54 degraded-DVFS -28%/-40%
# was an amortization artifact (B inflated by the low clock), not a
# healthy-state win. tb-large returns to BODIES=2; mb4 stays reachable
# via _MB_BODY_OVERRIDE for probes. The trep decoupling knob below is
# kept (harmless, freezes trep at its proven BODIES).
_MB_TARGET_PROGRAMS = 1024
_MB_MAX_BODIES = 8
# bench-only explicit BODIES override (x- rows); None = host formula
_MB_BODY_OVERRIDE = None

# ---- r33 merged multi-body -------------------------------------------------
# Amortise the per-program image-location chain (the short scalar compare
# chain over the <=4 host-known image slots) across BODIES consecutive rows.
# BODIES is the gcd of every image's row count, so an aligned body block can
# never straddle an image boundary and the located (H, W, h_scale, w_scale)
# stay valid for all BODIES rows. Per-row arithmetic is untouched ->
# bitwise identical to the one-row-per-program merged kernel.
_MERGED_MB_DISABLE = False
_MERGED_MB_MIN_ROWS = 512
_MERGED_MB_MAX_BODIES = 8

# ---- r32 temporal-frame replication (trep) ---------------------------------
#
# h_scale/w_scale are SCALAR kernel args of the single-image wrapper, so
# every temporal frame of the same spatial row drives the identical
# bilinear formula on the identical embed rows -> the t frame copies of one
# spatial row are bitwise identical (P17 temporal-exact atol=0 evidence).
# The trep kernel therefore computes each unique spatial row ONCE and
# stores FRAMES copies (tl.static_range over the constexpr frame count),
# cutting embed-load traffic by t while writes stay 1x task-book bytes.
# r31 tail truth: 230MB of embed reads on a 14400-row (t=3) shape -> 77MB
# at t=3 replication; big-video reads drop 16x.
# Bitwise safety: loads, weight arithmetic and FMA order are verbatim
# ``_bilinear_row_body``/``_seg_row``; only the store gains f*FRAME_STRIDE
# (rawAB-trep gates it against the single-row kernel every round).
_TREP_DISABLE = False
# enable only when the frame grid is big enough to matter and the program
# count stays meaningful; small shapes keep pair (lines already green)
_TREP_MIN_ROWS = 4096
# r55: dedicated trep BODIES knob (the pre-r55 _MB_TARGET_PROGRAMS value).
# The trep bodies formula previously reused _MB_TARGET_PROGRAMS, so the
# r55 mb4-large promotion (512) would have silently moved line-checked
# tb-tail trep BODIES 4->8 with zero same-run A/B evidence. Splitting the
# knob freezes every trep dispatch (tail 117.4/57.0 and big-video lines)
# at their proven BODIES while the mb kernel moves independently.
_TREP_TARGET_PROGRAMS = 1024
# r38 bench-only override: flat-load trep body (wave loads hoisted before
# FMAs, r30 pair-flat pattern). Production keeps the r32 body; run.py
# flips this only for the rawAB-trepflat gate and the x-trepflat rows.
_TREP_FLAT = False
# r46 bench-only override: segment-parallel trep -- one program per
# (unique spatial row, exact hidden segment) instead of the in-program
# BODIES x 6-segment serial chain. r45 probes: x-h512/x-h256 collapse the
# chain to one 512 segment on the same t=3 grid and run 5.2x faster
# (0.077 vs 0.404ms), while x-fr2 proved store bytes scale linearly
# (+0.125ms/frame) -- so the serial chain, not store traffic, is the
# tail wall. run.py flips this only for the rawAB-trepseg gate and the
# x-trepseg rows; production keeps the r32 body.
_TREP_SEG = False
# r51 production: masked single-segment trep2d (m1) for the long serial-chain
# tail route. r50 pre-registered gate: x-trep2dm1-tb-tail 0.079ms f32 /
# 0.085ms bf16 (<=0.191ms line, 2.6x over six-segment trep2d 0.204ms),
# rawAB-trep2dm1 bitwise gate passed both dtypes. Enabled only where the
# single masked [next_pow2(hidden)] round wastes few lanes: hidden=1000
# (six exact segments, 2.3% mask waste). hidden=1152 big-video (2 segments,
# 43.8% waste) keeps exact-segment trep2d below -- m1 would lose there.
_TREP_M1 = True

# ---- r42 corner kernel (logical-row w-group load sharing) ------------------
#
# Bench-only force switch. One program per LOGICAL output row (fixed
# hf/hc/dh) walks the distinct w-source groups in-kernel: consecutive
# columns with the same floor(col*w_scale) share the same 4 embed rows, so
# each (group, segment) loads its 4 embed rows ONCE and the FMA/store
# re-issues per column from registers. Tail (w_scale=47/79): 48 groups over
# 80 columns -> 1.67x fewer load ops than the r32 trep body; stores stay 1x
# task-book bytes (trep-style frame replication). GROUPS is passed as
# NUM_GRID (distinct floor values <= NUM_GRID always). Production dispatch
# is UNCHANGED (None); run.py flips this only for the x-corner rows and the
# rawAB-corner gate proves bitwise identity every round.
_FORCE_CORNER = None

# ---- r27 band kernel: source-row-group load sharing ------------------------
#
# r26 (job 945) op_summary evidence: every line-check FAIL shape (tb-tail
# 20.9% FP32, tb-big-video 53.4%, raw 53.7-54.6% FP32, and all large-shape
# BF16 rows) issues ~4-5x the task-book byte count in embed loads because
# every output row loads its 4 embed rows independently. r24 proved the
# cheap levers dead (cap sweep 2048/8192/16384 and the quad kernel are all
# slower or neutral). The band kernel instead groups output rows by their
# source-row tuple (hf, wf) -- hc/wc are derived (hf+1/wf+1 clamped), so
# the tuple fully determines the 4 embed rows -- and loads each embed row
# ONCE per group of rows.
#
# Bitwise safety: each row still recomputes h_frac/w_frac/dh/dw and the
# four weights with the exact expressions of ``_bilinear_row_body``; only
# the embed-row offsets come from the shared table. A wrong grouping
# (host/device float32 divergence) would mis-address embed rows and fail
# the 47-case precision suite, never silently round.
#
# Padding rows repeat the band's last row: same out_row, same offset,
# same weights -> identical store, i.e. idempotent. Every band executes
# exactly BAND_R unmasked rows (no predication; iron rule r18).
_BAND_R = 8

# Activation gate: below this mean rows-per-group the sharing gain does
# not pay for the table overhead, keep the pair kernel.
# r26 shapes: tail 6.0 rows/group, big-video 68.8 -> band; mid/short-video/
# small-A 1-2 -> pair (their lines already pass).
_BAND_MIN_GROUP_ROWS = 3.5

# Column tile width for the 2D band kernel (power of two). Every nonzero
# segment width is a power of two, so WIDTH <= TILE_D takes a single
# unmasked [BAND_R, WIDTH] tile and wider segments tile exactly. 64 keeps
# the [8, 64] accumulator register footprint light for both f32 and bf16.
_BAND_TILE_D = 64

_BAND_CACHE = {}


def _build_band_table(t, h, w, num_grid_per_side, m_size, h_scale, w_scale,
                      hidden_dim, device):
    """Group output rows by source tuple (hf, wf) and build band tables.

    Returns ``(rows_t, off_t, num_bands)`` where ``rows_t`` is int32
    [num_bands * _BAND_R] holding each band's output-row ids (padded with
    repeats of the band's last row) and ``off_t`` is int32 [num_bands * 4]
    holding the band's shared embed-row offsets (off00, off01, off10,
    off11, in elements). Returns ``None`` when the mean rows per group is
    below ``_BAND_MIN_GROUP_ROWS`` (the pair kernel wins there), when
    numpy is unavailable, or when ``w < m_size``.

    Grouping (vectorised, matches the kernel arithmetic bit-exactly):
    spatial-merge inverse map -> row_i/col_i -> h_frac = f32(row_i) *
    f32(h_scale) -> hf = trunc -> same for w. Rows sharing (hf, wf) share
    all four embed rows because hc = min(hf+1, G-1) and wc are derived.
    Temporal repeats (t > 1) collapse into the same group automatically.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return None
    if w < m_size or m_size < 1 or h < 1:
        return None
    cache_key = (t, h, w, num_grid_per_side, m_size, h_scale, w_scale,
                 hidden_dim, str(device))
    hit = _BAND_CACHE.get(cache_key)
    if hit is not None:
        return hit

    total = t * h * w
    rows = np.arange(total, dtype=np.int64)
    spatial = rows % (h * w)
    mb = m_size * m_size
    block_idx = spatial // mb
    local_idx = spatial % mb
    nbw = w // m_size
    row_i = (block_idx // nbw) * m_size + local_idx // m_size
    col_i = (block_idx % nbw) * m_size + local_idx % m_size

    g = num_grid_per_side
    h_frac = row_i.astype(np.float32) * np.float32(h_scale)
    hf = h_frac.astype(np.int32)
    hc = np.minimum(hf + 1, g - 1)
    w_frac = col_i.astype(np.float32) * np.float32(w_scale)
    wf = w_frac.astype(np.int32)
    wc = np.minimum(wf + 1, g - 1)
    key = ((hf.astype(np.int64) * g + hc) * g + wf) * g + wc

    order = np.argsort(key, kind="stable")
    skey = key[order]
    cuts = np.flatnonzero(np.diff(skey)) + 1
    starts = np.concatenate(([0], cuts))
    num_groups = len(starts)
    avg = total / num_groups
    if avg < _BAND_MIN_GROUP_ROWS:
        tbl = None
    else:
        ends = np.concatenate((cuts, [total]))
        n_per = ends - starts
        bands_per = np.ceil(n_per / _BAND_R).astype(np.int64)
        band_base = np.concatenate(([0], np.cumsum(bands_per)))
        num_bands = int(band_base[-1])

        # sorted position -> (group, band, slot); fill rows table
        pos = np.arange(total, dtype=np.int64)
        gid = np.searchsorted(starts, pos, side="right") - 1
        q = pos - starts[gid]
        band_id = band_base[gid] + q // _BAND_R
        flat_rows = np.full(num_bands * _BAND_R, -1, dtype=np.int32)
        flat_rows[(band_id * _BAND_R + q % _BAND_R).astype(np.int64)] = \
            order.astype(np.int32)
        counts = np.bincount(band_id, minlength=num_bands)
        band_rows = flat_rows.reshape(num_bands, _BAND_R)
        for b in range(num_bands):  # pad with the band's last row (idempotent)
            c = counts[b]
            if c < _BAND_R:
                band_rows[b, c:] = band_rows[b, c - 1]

        # per-band offsets from the group's key
        band_gid = np.searchsorted(band_base, np.arange(num_bands),
                                   side="right") - 1
        bkey = skey[starts[band_gid]]
        wc_b = bkey % g
        q2 = bkey // g
        wf_b = q2 % g
        q3 = q2 // g
        hc_b = q3 % g
        hf_b = q3 // g
        off = np.stack([
            (hf_b * g + wf_b) * hidden_dim,
            (hf_b * g + wc_b) * hidden_dim,
            (hc_b * g + wf_b) * hidden_dim,
            (hc_b * g + wc_b) * hidden_dim,
        ], axis=1).astype(np.int32).reshape(-1)

        rows_t = torch.from_numpy(band_rows.reshape(-1)).to(device)
        off_t = torch.from_numpy(off).to(device)
        tbl = (rows_t, off_t, num_bands)

    if len(_BAND_CACHE) > 16:
        _BAND_CACHE.clear()
    _BAND_CACHE[cache_key] = tbl
    return tbl


def _launch_opts():
    opts = {}
    if _LAUNCH_NUM_WARPS:
        opts["num_warps"] = _LAUNCH_NUM_WARPS
    # r56 bench-only probe (P5): num_stages was never tested -- if the
    # triton-ascend backend auto-double-buffers static_range body loops,
    # this is the free version of the manual mb-pipe scheme. Unsupported
    # kwargs raise at launch; run.py P5 rows are try/except-guarded so a
    # rejection only skips the probe rows.
    if _LAUNCH_NUM_STAGES:
        opts["num_stages"] = _LAUNCH_NUM_STAGES
    return opts


def _hidden_split(hidden_dim):
    """Decompose ``hidden_dim`` into exact power-of-two lane blocks.

    Returns ``(block_d, tail_block, tail_start, exact_main)`` for the
    kernel's constexpr lane layout:

    - pow2 hidden (e.g. 2048): one exact block, no mask at all;
    - hidden = 2**a + 2**b with a > b (e.g. 1152 = 1024 + 128): main
      block + exact tail block -- every lane does real work. A padded
      BLOCK_D of 2048 for hidden 1152 wastes 896 of 2048 lanes on
      masked-out FMA work, which is the prime suspect for BF16's
      compute-bound 40.1% MBU (r6);
    - otherwise (e.g. 1000): one masked block of the next power of two
      (previous behaviour; waste <= half a block).
    """
    if hidden_dim > 0 and hidden_dim & (hidden_dim - 1) == 0:
        return (hidden_dim, 0, hidden_dim, True)
    main = 1 << (hidden_dim.bit_length() - 1)
    tail = hidden_dim - main
    if tail > 0 and tail & (tail - 1) == 0:
        return (main, tail, main, True)
    return (1 << hidden_dim.bit_length(), 0, hidden_dim, False)


# Max number of exact column segments the kernels unroll. 6 covers every
# hidden_dim < 8192 whose popcount <= 6 (e.g. 1000 = 512+256+128+64+32+8).
_MAX_SEGS = 6


def _pow2_segments(hidden_dim):
    """Decompose ``hidden_dim`` into <= ``_MAX_SEGS`` exact power-of-two
    column segments (unmasked lanes end to end).

    Returns a list of widths summing to ``hidden_dim`` (padded with zeros
    to ``_MAX_SEGS`` entries), or ``None`` when more than ``_MAX_SEGS``
    segments would be needed -- the caller then falls back to the masked
    single-block path of ``_hidden_split``. Examples: 1000 ->
    [512, 256, 128, 64, 32, 8], 1152 -> [1024, 128], 2048 -> [2048].

    r18 evidence (op_summary device Duration, 926 job): the masked 1024-
    lane block for hidden=1000 runs ~4.3x slower per element than the
    exact unmasked path (predicated memory ops on triton-ascend 3.6.0),
    which is what sank the tb-tail MBU lines (17.2% FP32 / 8.7% BF16 vs
    the >=50% line).
    """
    rem, segs = hidden_dim, []
    while rem:
        segs.append(1 << (rem.bit_length() - 1))
        rem -= segs[-1]
    if len(segs) > _MAX_SEGS:
        return None
    return segs + [0] * (_MAX_SEGS - len(segs))


def _corner_shape_params(w, w_scale):
    """Host estimate of the r42 corner-kernel MAX_RUN static-unroll bound.

    GROUPS is passed as NUM_GRID (distinct floor(col*w_scale) values are
    <= NUM_GRID always, and with a sufficient MAX_RUN the in-kernel chain
    consumes exactly one full equal-floor run per slot). Correctness does
    NOT depend on MAX_RUN either: the chain re-derives every column's floor
    with the exact production expression and advances >= 1 column per slot,
    so an underestimated MAX_RUN only truncates the unroll (caught by the
    rawAB-corner gate), never silently rounds. The f32 and f64 mimicries
    bound the two possible device-side scalar dtypes; their max is taken.
    """
    max_run = 4
    try:
        import numpy as np

        def _runs(scale):
            if scale <= 0.0:
                return 1
            c32 = np.arange(w, dtype=np.float32)
            wf32 = np.floor(c32 * np.float32(scale)).astype(np.int64)
            c64 = np.arange(w, dtype=np.float64)
            wf64 = np.floor(c64 * np.float64(scale)).astype(np.int64)
            best = 1
            for wf in (wf32, wf64):
                run = 1
                for i in range(1, w):
                    if wf[i] == wf[i - 1]:
                        run += 1
                        best = max(best, run)
                    else:
                        run = 1
            return best

        if w_scale:
            max_run = int(min(max(_runs(w_scale), 1), 8))
    except Exception:
        pass
    return max_run


def triton_ascend_available() -> bool:
    """Lazily detect whether the triton-ascend backend can run on this host.

    Cached; used by the vllm-ascend capability dispatch (see
    vllm_ascend_dispatch.py).
    """
    if not HAS_TRITON:
        return False
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


def _validate_inputs(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> None:
    """Host-side legality checks (added by this adaptation; debug asserts).

    Legal-input behaviour is unchanged versus the upstream wrapper.
    """
    assert h % m_size == 0 and w % m_size == 0, (
        f"h={h} and w={w} must be divisible by m_size={m_size}"
    )
    assert embed_weight.dim() == 2, (
        f"embed_weight must be 2D [G*G, hidden_dim], got shape {tuple(embed_weight.shape)}"
    )
    assert embed_weight.shape[0] == num_grid_per_side * num_grid_per_side, (
        f"embed_weight rows ({embed_weight.shape[0]}) must equal "
        f"G*G = {num_grid_per_side * num_grid_per_side}"
    )
    assert dtype in _SUPPORTED_DTYPES, (
        f"unsupported dtype {dtype}, expected one of {_SUPPORTED_DTYPES}"
    )
    assert t >= 1, f"t must be >= 1, got {t}"


if HAS_TRITON:

    @triton.jit
    def _seg_row(embed_ptr, output_ptr, off00, off01, off10, off11, out_off,
                 w00, w01, w10, w11,
                 START: tl.constexpr, WIDTH: tl.constexpr):
        """One exact (unmasked) column segment of one output row.

        WIDTH is a power of two (host-decomposed by ``_pow2_segments``),
        so no lane is masked and no predicated memory op is emitted --
        predicated loads/stores run ~4x slower per element on
        triton-ascend 3.6.0 (r18 op_summary evidence). Zero-width
        segments compile to nothing (trace-time ``WIDTH > 0``).
        """
        if WIDTH > 0:
            cols = START + tl.arange(0, WIDTH)
            e00 = tl.load(embed_ptr + off00 + cols)
            e01 = tl.load(embed_ptr + off01 + cols)
            e10 = tl.load(embed_ptr + off10 + cols)
            e11 = tl.load(embed_ptr + off11 + cols)
            val = w00 * e00 + w01 * e01 + w10 * e10 + w11 * e11
            tl.store(output_ptr + out_off + cols, val)

    @triton.jit
    def _seg_row_trep(embed_ptr, output_ptr, off00, off01, off10, off11,
                      out_off, w00, w01, w10, w11,
                      START: tl.constexpr, WIDTH: tl.constexpr,
                      FRAMES: tl.constexpr, FRAME_STRIDE):
        """``_seg_row`` with temporal-frame replication (r32).

        Loads and the FMA are verbatim ``_seg_row`` (same expression
        order, same casts) -- only the store is issued once per temporal
        frame at ``out_off + f * FRAME_STRIDE``, with ``f`` unrolled by
        ``tl.static_range`` (iron rule: only static_range compiles).
        FRAMES == 1 degenerates to exactly one store, i.e. the single
        frame path bitwise.
        """
        if WIDTH > 0:
            cols = START + tl.arange(0, WIDTH)
            e00 = tl.load(embed_ptr + off00 + cols)
            e01 = tl.load(embed_ptr + off01 + cols)
            e10 = tl.load(embed_ptr + off10 + cols)
            e11 = tl.load(embed_ptr + off11 + cols)
            val = w00 * e00 + w01 * e01 + w10 * e10 + w11 * e11
            for f in tl.static_range(0, FRAMES):
                tl.store(output_ptr + out_off + f * FRAME_STRIDE + cols, val)

    @triton.jit
    def _bilinear_row_body(embed_ptr, output_ptr, out_row, row, H, W, h_scale,
                           w_scale, NUM_GRID: tl.constexpr,
                           M_SIZE: tl.constexpr, HIDDEN_DIM: tl.constexpr,
                           BLOCK_D: tl.constexpr, TAIL_START: tl.constexpr,
                           TAIL_BLOCK: tl.constexpr, EXACT_MAIN: tl.constexpr,
                           SEG0: tl.constexpr, SEG1: tl.constexpr,
                           SEG2: tl.constexpr, SEG3: tl.constexpr,
                           SEG4: tl.constexpr, SEG5: tl.constexpr):
        """Compute one output row (spatial-merge address arithmetic).

        ``row`` is the per-image local row driving the spatial/temporal
        index arithmetic; ``out_row`` is the row of the output tensor this
        program writes (global row in merged mode, identical to ``row``
        for the single-image launch).

        Ascend affinity: ``hf/wf`` via int truncation (non-negative domain
        where truncation == floor).

        D lanes use exact power-of-two blocks (see ``_hidden_split``):
        ``EXACT_MAIN`` blocks need no bounds mask, and a ``TAIL_BLOCK``
        covers the remainder exactly. Per-element op order is unchanged
        versus the masked single-block form, so results stay bitwise
        identical.
        """
        total_spatial = H * W
        spatial_idx = row % total_spatial

        num_blocks_w = W // M_SIZE
        block_idx = spatial_idx // (M_SIZE * M_SIZE)
        local_idx = spatial_idx % (M_SIZE * M_SIZE)
        br = block_idx // num_blocks_w
        bc = block_idx % num_blocks_w
        lr = local_idx // M_SIZE
        lc = local_idx % M_SIZE
        row = br * M_SIZE + lr
        col = bc * M_SIZE + lc

        h_frac = row.to(tl.float32) * h_scale
        w_frac = col.to(tl.float32) * w_scale

        hf = h_frac.to(tl.int32)
        wf = w_frac.to(tl.int32)
        hc = tl.minimum(hf + 1, NUM_GRID - 1)
        wc = tl.minimum(wf + 1, NUM_GRID - 1)

        dh = h_frac - hf.to(tl.float32)
        dw = w_frac - wf.to(tl.float32)
        w11 = dh * dw
        w10 = dh - w11
        w01 = dw - w11
        w00 = 1.0 - dh - w01

        off00 = (hf * NUM_GRID + wf) * HIDDEN_DIM
        off01 = (hf * NUM_GRID + wc) * HIDDEN_DIM
        off10 = (hc * NUM_GRID + wf) * HIDDEN_DIM
        off11 = (hc * NUM_GRID + wc) * HIDDEN_DIM
        out_off = out_row * HIDDEN_DIM

        out_dtype = output_ptr.dtype.element_ty
        w00_c = w00.to(out_dtype)
        w01_c = w01.to(out_dtype)
        w10_c = w10.to(out_dtype)
        w11_c = w11.to(out_dtype)

        # Exact power-of-two column segments (SEG0..SEG5, host-decomposed
        # by ``_pow2_segments``): every lane unmasked, zero predication.
        # Per-element op order is unchanged, so results stay bitwise
        # identical to both previous forms. SEG0 == 0 means hidden_dim
        # needs more than 6 segments -- fall back to the masked
        # single-block loop (previous behaviour, kept for generality).
        if SEG0 > 0:
            _seg_row(embed_ptr, output_ptr, off00, off01, off10, off11,
                     out_off, w00_c, w01_c, w10_c, w11_c, 0, SEG0)
            _seg_row(embed_ptr, output_ptr, off00, off01, off10, off11,
                     out_off, w00_c, w01_c, w10_c, w11_c, SEG0, SEG1)
            _seg_row(embed_ptr, output_ptr, off00, off01, off10, off11,
                     out_off, w00_c, w01_c, w10_c, w11_c,
                     SEG0 + SEG1, SEG2)
            _seg_row(embed_ptr, output_ptr, off00, off01, off10, off11,
                     out_off, w00_c, w01_c, w10_c, w11_c,
                     SEG0 + SEG1 + SEG2, SEG3)
            _seg_row(embed_ptr, output_ptr, off00, off01, off10, off11,
                     out_off, w00_c, w01_c, w10_c, w11_c,
                     SEG0 + SEG1 + SEG2 + SEG3, SEG4)
            _seg_row(embed_ptr, output_ptr, off00, off01, off10, off11,
                     out_off, w00_c, w01_c, w10_c, w11_c,
                     SEG0 + SEG1 + SEG2 + SEG3 + SEG4, SEG5)
        else:
            # Iron rule (triton-ascend 3.6.0): tl.range AND plain range()
            # both downgrade to `import shmem` (r0/r1 real-machine
            # evidence); only tl.static_range (full unroll) compiles on
            # the eval environment.
            for d in tl.static_range(0, TAIL_START, BLOCK_D):
                cols = d + tl.arange(0, BLOCK_D)
                if EXACT_MAIN:
                    e00 = tl.load(embed_ptr + off00 + cols)
                    e01 = tl.load(embed_ptr + off01 + cols)
                    e10 = tl.load(embed_ptr + off10 + cols)
                    e11 = tl.load(embed_ptr + off11 + cols)

                    val = w00_c * e00 + w01_c * e01 + w10_c * e10 + w11_c * e11

                    tl.store(output_ptr + out_off + cols, val)
                else:
                    mask = cols < HIDDEN_DIM

                    e00 = tl.load(embed_ptr + off00 + cols, mask=mask)
                    e01 = tl.load(embed_ptr + off01 + cols, mask=mask)
                    e10 = tl.load(embed_ptr + off10 + cols, mask=mask)
                    e11 = tl.load(embed_ptr + off11 + cols, mask=mask)

                    val = w00_c * e00 + w01_c * e01 + w10_c * e10 + w11_c * e11

                    tl.store(output_ptr + out_off + cols, val, mask=mask)
            if TAIL_BLOCK > 0:
                cols = TAIL_START + tl.arange(0, TAIL_BLOCK)
                e00 = tl.load(embed_ptr + off00 + cols)
                e01 = tl.load(embed_ptr + off01 + cols)
                e10 = tl.load(embed_ptr + off10 + cols)
                e11 = tl.load(embed_ptr + off11 + cols)

                val = w00_c * e00 + w01_c * e01 + w10_c * e10 + w11_c * e11

                tl.store(output_ptr + out_off + cols, val)

    @triton.jit
    def _bilinear_trep_body(embed_ptr, output_ptr, spatial_row, H, W,
                            h_scale, w_scale, NUM_GRID: tl.constexpr,
                            M_SIZE: tl.constexpr, HIDDEN_DIM: tl.constexpr,
                            FRAMES: tl.constexpr, FRAME_STRIDE,
                            SEG0: tl.constexpr, SEG1: tl.constexpr,
                            SEG2: tl.constexpr, SEG3: tl.constexpr,
                            SEG4: tl.constexpr, SEG5: tl.constexpr):
        """One unique spatial row, stored to FRAMES temporal frames (r32).

        Address arithmetic, weight computation, casts and segment layout
        are verbatim ``_bilinear_row_body`` (SEG0 > 0 exact path only --
        the trep dispatch requires the host power-of-two segmentation, so
        the masked fallback never traces). ``spatial_row`` is in
        [0, H*W); frame f writes at spatial_row*HIDDEN_DIM +
        f*FRAME_STRIDE with FRAME_STRIDE = H*W*HIDDEN_DIM. Because
        h_scale/w_scale are scalar kernel args, all FRAMES copies are
        bitwise identical to what the per-frame kernels would write.
        """
        total_spatial = H * W
        spatial_idx = spatial_row % total_spatial

        num_blocks_w = W // M_SIZE
        block_idx = spatial_idx // (M_SIZE * M_SIZE)
        local_idx = spatial_idx % (M_SIZE * M_SIZE)
        br = block_idx // num_blocks_w
        bc = block_idx % num_blocks_w
        lr = local_idx // M_SIZE
        lc = local_idx % M_SIZE
        row_i = br * M_SIZE + lr
        col = bc * M_SIZE + lc

        h_frac = row_i.to(tl.float32) * h_scale
        w_frac = col.to(tl.float32) * w_scale

        hf = h_frac.to(tl.int32)
        wf = w_frac.to(tl.int32)
        hc = tl.minimum(hf + 1, NUM_GRID - 1)
        wc = tl.minimum(wf + 1, NUM_GRID - 1)

        dh = h_frac - hf.to(tl.float32)
        dw = w_frac - wf.to(tl.float32)
        w11 = dh * dw
        w10 = dh - w11
        w01 = dw - w11
        w00 = 1.0 - dh - w01

        off00 = (hf * NUM_GRID + wf) * HIDDEN_DIM
        off01 = (hf * NUM_GRID + wc) * HIDDEN_DIM
        off10 = (hc * NUM_GRID + wf) * HIDDEN_DIM
        off11 = (hc * NUM_GRID + wc) * HIDDEN_DIM
        out_off = spatial_row * HIDDEN_DIM

        out_dtype = output_ptr.dtype.element_ty
        w00_c = w00.to(out_dtype)
        w01_c = w01.to(out_dtype)
        w10_c = w10.to(out_dtype)
        w11_c = w11.to(out_dtype)

        _seg_row_trep(embed_ptr, output_ptr, off00, off01, off10, off11,
                      out_off, w00_c, w01_c, w10_c, w11_c, 0, SEG0,
                      FRAMES, FRAME_STRIDE)
        _seg_row_trep(embed_ptr, output_ptr, off00, off01, off10, off11,
                      out_off, w00_c, w01_c, w10_c, w11_c, SEG0, SEG1,
                      FRAMES, FRAME_STRIDE)
        _seg_row_trep(embed_ptr, output_ptr, off00, off01, off10, off11,
                      out_off, w00_c, w01_c, w10_c, w11_c,
                      SEG0 + SEG1, SEG2, FRAMES, FRAME_STRIDE)
        _seg_row_trep(embed_ptr, output_ptr, off00, off01, off10, off11,
                      out_off, w00_c, w01_c, w10_c, w11_c,
                      SEG0 + SEG1 + SEG2, SEG3, FRAMES, FRAME_STRIDE)
        _seg_row_trep(embed_ptr, output_ptr, off00, off01, off10, off11,
                      out_off, w00_c, w01_c, w10_c, w11_c,
                      SEG0 + SEG1 + SEG2 + SEG3, SEG4, FRAMES, FRAME_STRIDE)
        _seg_row_trep(embed_ptr, output_ptr, off00, off01, off10, off11,
                      out_off, w00_c, w01_c, w10_c, w11_c,
                      SEG0 + SEG1 + SEG2 + SEG3 + SEG4, SEG5,
                      FRAMES, FRAME_STRIDE)

    @triton.jit
    def _bilinear_pos_embed_kernel(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        TOTAL_ROWS,
        CHUNKS: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        TAIL_START: tl.constexpr,
        TAIL_BLOCK: tl.constexpr,
        EXACT_MAIN: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """Fused bilinear pos-embed interpolation with spatial-merge reorder.

        Ascend affinity: rtKernelLaunch caps the TOTAL program count
        (product of grid dims; EE1003, r3 evidence), so the grid is capped
        at _GRID_CAP (= 32768, r6: beats 65535 on 268800-row launches for
        both dtypes) and each program handles CHUNKS = cdiv(rows,
        _GRID_CAP) rows via an unrolled static loop; CHUNKS is constexpr
        so each shape specializes cleanly.
        """
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            pid = tl.program_id(0) + c * grid_size
            if pid < TOTAL_ROWS:
                _bilinear_row_body(
                    embed_ptr, output_ptr, pid, pid, H, W, h_scale, w_scale,
                    NUM_GRID, M_SIZE, HIDDEN_DIM, BLOCK_D,
                    TAIL_START, TAIL_BLOCK, EXACT_MAIN,
                    SEG0, SEG1, SEG2, SEG3, SEG4, SEG5)

    @triton.jit
    def _bilinear_pos_embed_merged_kernel(
        embed_ptr,
        output_ptr,
        cu_rows_ptr,
        img_hw_ptr,
        img_scale_ptr,
        NUM_IMAGES,
        TOTAL_ROWS,
        CHUNKS: tl.constexpr,
        MAX_IMAGES: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        TAIL_START: tl.constexpr,
        TAIL_BLOCK: tl.constexpr,
        EXACT_MAIN: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """Multi-image merged single-launch variant.

        Per-program image location: ``k`` = number of images whose
        cu_rows[i+1] <= pid (vectorised counting over the padded prefix
        sums). The spatial/temporal index arithmetic must use the
        per-image LOCAL row ``pid - cu_rows[k]`` (r4 evidence: feeding the
        global pid broke bitwise equality for every image after the
        first, since the global row offset is generally not a multiple of
        H*W); the global ``pid`` only drives the output row offset, which
        makes the result bitwise identical to the per-image launch +
        torch.cat version.

        Same grid-cap handling as the single-image kernel: grid <=
        _GRID_CAP, CHUNKS = cdiv(rows, _GRID_CAP) unrolled statically.
        """
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            pid = tl.program_id(0) + c * grid_size
            if pid < TOTAL_ROWS:
                img_idx = tl.arange(0, MAX_IMAGES)
                img_mask = img_idx < NUM_IMAGES
                ends = tl.load(
                    cu_rows_ptr + img_idx + 1, mask=img_mask, other=2147483647
                )
                k = tl.sum(tl.where(img_mask & (pid >= ends), 1, 0)).to(tl.int32)

                H = tl.load(img_hw_ptr + 2 * k)
                W = tl.load(img_hw_ptr + 2 * k + 1)
                h_scale = tl.load(img_scale_ptr + 2 * k)
                w_scale = tl.load(img_scale_ptr + 2 * k + 1)
                local_row = pid - tl.load(cu_rows_ptr + k)
                _bilinear_row_body(
                    embed_ptr, output_ptr, pid, local_row, H, W, h_scale,
                    w_scale, NUM_GRID, M_SIZE, HIDDEN_DIM, BLOCK_D,
                    TAIL_START, TAIL_BLOCK, EXACT_MAIN,
                    SEG0, SEG1, SEG2, SEG3, SEG4, SEG5)

    @triton.jit
    def _bilinear_pos_embed_merged_scalar_kernel(
        embed_ptr,
        output_ptr,
        END0,
        END1,
        END2,
        END3,
        IH0,
        IW0,
        IH1,
        IW1,
        IH2,
        IW2,
        IH3,
        IW3,
        HS0,
        WS0,
        HS1,
        WS1,
        HS2,
        WS2,
        HS3,
        WS3,
        TOTAL_ROWS,
        CHUNKS: tl.constexpr,
        BODIES: tl.constexpr,
        NUM_IMAGES: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        TAIL_START: tl.constexpr,
        TAIL_BLOCK: tl.constexpr,
        EXACT_MAIN: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """Merged variant with host-known per-image scalars (<= 4 images).

        Every image boundary (ENDi = cu_rows[i+1]), size (IHi/IWi) and
        scale (HSi/WSi) is a plain scalar kernel argument computed on the
        host, so the per-program image lookup is a short scalar compare
        chain instead of the device-tensor version's dependent chain of
        global scalar loads. r18 op_summary evidence: that latency chain
        dominated the merged kernel (2432 rows in 189 us == 14.8 Gelem/s,
        ~4x below the single-image kernel's element rate).

        Unused image slots (i >= NUM_IMAGES) are guarded by the
        trace-time ``NUM_IMAGES > i`` tests, so their (dummy) arguments
        never participate. Spatial/temporal arithmetic uses the
        per-image LOCAL row exactly like the device-tensor kernel (r4
        evidence), so output stays bitwise identical to per-image launch
        + torch.cat.
        """
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            pid = (tl.program_id(0) + c * grid_size) * BODIES
            if pid < TOTAL_ROWS:
                row_base = 0
                H = IH0
                W = IW0
                h_scale = HS0
                w_scale = WS0
                if NUM_IMAGES > 1:
                    if pid >= END0:
                        row_base = END0
                        H = IH1
                        W = IW1
                        h_scale = HS1
                        w_scale = WS1
                if NUM_IMAGES > 2:
                    if pid >= END1:
                        row_base = END1
                        H = IH2
                        W = IW2
                        h_scale = HS2
                        w_scale = WS2
                if NUM_IMAGES > 3:
                    if pid >= END2:
                        row_base = END2
                        H = IH3
                        W = IW3
                        h_scale = HS3
                        w_scale = WS3
                local_row = pid - row_base
                # One image-location chain serves BODIES consecutive rows.
                # The host guarantees every image's row count is a multiple
                # of BODIES, so an aligned body block never straddles an
                # image boundary and the (H, W, scale) above stay valid.
                # Per-row arithmetic is untouched -> bitwise identical.
                for b in tl.static_range(BODIES):
                    r = pid + b
                    if r < TOTAL_ROWS:
                        _bilinear_row_body(
                            embed_ptr, output_ptr, r, local_row + b, H, W,
                            h_scale, w_scale, NUM_GRID, M_SIZE, HIDDEN_DIM,
                            BLOCK_D, TAIL_START, TAIL_BLOCK, EXACT_MAIN,
                            SEG0, SEG1, SEG2, SEG3, SEG4, SEG5)

    @triton.jit
    def _band_seg2(embed_ptr, output_ptr, off00, off01, off10, off11,
                   out_off, w00, w01, w10, w11,
                   START: tl.constexpr, WIDTH: tl.constexpr,
                   TILE_D: tl.constexpr):
        """One exact column segment for a whole band: the four embed rows
        are loaded ONCE per column tile and broadcast against the per-lane
        weights into a [BAND_R, TILE] store.

        Per element the FMA order (w00*e00 + w01*e01 + w10*e10 + w11*e11)
        is exactly ``_seg_row``'s; weights arrive pre-cast to out_dtype.
        WIDTH and TILE_D are powers of two: WIDTH <= TILE_D runs one
        unmasked [BAND_R, WIDTH] tile, wider segments loop unmasked
        TILE_D-wide tiles (TILE_D divides every power-of-two WIDTH >= it).
        """
        if WIDTH > 0:
            if WIDTH <= TILE_D:
                cols = START + tl.arange(0, WIDTH)
                e00 = tl.load(embed_ptr + off00 + cols)
                e01 = tl.load(embed_ptr + off01 + cols)
                e10 = tl.load(embed_ptr + off10 + cols)
                e11 = tl.load(embed_ptr + off11 + cols)
                val = (w00[:, None] * e00[None, :]
                       + w01[:, None] * e01[None, :]
                       + w10[:, None] * e10[None, :]
                       + w11[:, None] * e11[None, :])
                tl.store(output_ptr + out_off[:, None] + cols[None, :], val)
            else:
                for t in tl.static_range(0, WIDTH, TILE_D):
                    cols = START + t + tl.arange(0, TILE_D)
                    e00 = tl.load(embed_ptr + off00 + cols)
                    e01 = tl.load(embed_ptr + off01 + cols)
                    e10 = tl.load(embed_ptr + off10 + cols)
                    e11 = tl.load(embed_ptr + off11 + cols)
                    val = (w00[:, None] * e00[None, :]
                           + w01[:, None] * e01[None, :]
                           + w10[:, None] * e10[None, :]
                           + w11[:, None] * e11[None, :])
                    tl.store(output_ptr + out_off[:, None] + cols[None, :],
                             val)

    @triton.jit
    def _bilinear_pos_embed_kernel_band2(
        embed_ptr,
        output_ptr,
        band_rows_ptr,
        band_off_ptr,
        H,
        W,
        h_scale,
        w_scale,
        NUM_BANDS,
        TOTAL_ROWS,
        CHUNKS: tl.constexpr,
        BAND_R: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        TILE_D: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """r28: 2D-vectorised source-row-group kernel (replaces r27 band).

        r27 (job 1000) op_summary evidence: the v1 band kernel ran 3x
        slower than the pair kernel on its own target shapes (tb-tail-f32
        849us vs x-noband 650us; tb-big-video-f32 8981us vs 2935us) because
        it re-issued the 4 embed-row loads PER ROW inside the unrolled
        row loop -- the identical addresses were not CSE'd across the 8
        unrolled copies -- and fetched both tables with BAND_R+4 scalar
        loads per band. Same-run comparisons (x-noband rows), so the
        machine's clock oscillation that round does not affect the verdict.

        band2 fixes both: the row-id table is fetched as ONE [BAND_R]
        vector load, per-band weights are computed as [BAND_R] lanes with
        elementwise-identical arithmetic to ``_bilinear_row_body``, and
        the four embed rows are loaded ONCE per (band, column tile) and
        broadcast into a [BAND_R, TILE_D] 2D store. Embed-load traffic per
        band drops 8x versus the pair kernel (the r26 evidence demand).

        Bitwise safety: per element the FMA order, the f32->out_dtype
        weight cast point and the address arithmetic are exactly those of
        ``_seg_row``/``_bilinear_row_body``; padding lanes repeat the
        band's last row id, so their stores rewrite identical values
        (idempotent). No masked lane anywhere: BAND_R is constexpr and
        TILE_D divides every nonzero power-of-two segment width.
        """
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            band = tl.program_id(0) + c * grid_size
            if band < NUM_BANDS:
                rid = tl.load(band_rows_ptr + band * BAND_R
                              + tl.arange(0, BAND_R))
                off00 = tl.load(band_off_ptr + band * 4 + 0)
                off01 = tl.load(band_off_ptr + band * 4 + 1)
                off10 = tl.load(band_off_ptr + band * 4 + 2)
                off11 = tl.load(band_off_ptr + band * 4 + 3)

                total_spatial = H * W
                spatial_idx = rid % total_spatial

                num_blocks_w = W // M_SIZE
                block_idx = spatial_idx // (M_SIZE * M_SIZE)
                local_idx = spatial_idx % (M_SIZE * M_SIZE)
                br = block_idx // num_blocks_w
                bc = block_idx % num_blocks_w
                lr = local_idx // M_SIZE
                lc = local_idx % M_SIZE
                row_i = br * M_SIZE + lr
                col_i = bc * M_SIZE + lc

                h_frac = row_i.to(tl.float32) * h_scale
                hf = h_frac.to(tl.int32)
                dh = h_frac - hf.to(tl.float32)
                w_frac = col_i.to(tl.float32) * w_scale
                wf = w_frac.to(tl.int32)
                dw = w_frac - wf.to(tl.float32)
                w11 = dh * dw
                w10 = dh - w11
                w01 = dw - w11
                w00 = 1.0 - dh - w01

                out_dtype = output_ptr.dtype.element_ty
                w00_c = w00.to(out_dtype)
                w01_c = w01.to(out_dtype)
                w10_c = w10.to(out_dtype)
                w11_c = w11.to(out_dtype)
                out_off = rid * HIDDEN_DIM

                _band_seg2(embed_ptr, output_ptr, off00, off01, off10,
                           off11, out_off, w00_c, w01_c, w10_c, w11_c,
                           0, SEG0, TILE_D)
                _band_seg2(embed_ptr, output_ptr, off00, off01, off10,
                           off11, out_off, w00_c, w01_c, w10_c, w11_c,
                           SEG0, SEG1, TILE_D)
                _band_seg2(embed_ptr, output_ptr, off00, off01, off10,
                           off11, out_off, w00_c, w01_c, w10_c, w11_c,
                           SEG0 + SEG1, SEG2, TILE_D)
                _band_seg2(embed_ptr, output_ptr, off00, off01, off10,
                           off11, out_off, w00_c, w01_c, w10_c, w11_c,
                           SEG0 + SEG1 + SEG2, SEG3, TILE_D)
                _band_seg2(embed_ptr, output_ptr, off00, off01, off10,
                           off11, out_off, w00_c, w01_c, w10_c, w11_c,
                           SEG0 + SEG1 + SEG2 + SEG3, SEG4, TILE_D)
                _band_seg2(embed_ptr, output_ptr, off00, off01, off10,
                           off11, out_off, w00_c, w01_c, w10_c, w11_c,
                           SEG0 + SEG1 + SEG2 + SEG3 + SEG4, SEG5, TILE_D)

    @triton.jit
    def _seg_pair_shared(embed_ptr, output_ptr, off_a, off_b, off_c, off_d,
                         out_off0, out_off1,
                         w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                         START: tl.constexpr, WIDTH: tl.constexpr):
        """One exact column segment shared by both rows of a pair (the
        ``wf1 == wf0`` branch: 4 embed rows serve both output rows)."""
        if WIDTH > 0:
            cols = START + tl.arange(0, WIDTH)
            e_a = tl.load(embed_ptr + off_a + cols)
            e_b = tl.load(embed_ptr + off_b + cols)
            e_c = tl.load(embed_ptr + off_c + cols)
            e_d = tl.load(embed_ptr + off_d + cols)
            v0 = w00a * e_a + w01a * e_b + w10a * e_c + w11a * e_d
            v1 = w00b * e_a + w01b * e_b + w10b * e_c + w11b * e_d
            tl.store(output_ptr + out_off0 + cols, v0)
            tl.store(output_ptr + out_off1 + cols, v1)

    @triton.jit
    def _seg_pair_diff(embed_ptr, output_ptr,
                       off_a0, off_b0, off_c0, off_d0,
                       off_a1, off_b1, off_c1, off_d1,
                       out_off0, out_off1,
                       w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                       START: tl.constexpr, WIDTH: tl.constexpr):
        """One exact column segment, independent embed rows per pair row
        (the ``wf1 != wf0`` branch)."""
        if WIDTH > 0:
            cols = START + tl.arange(0, WIDTH)
            e_a = tl.load(embed_ptr + off_a0 + cols)
            e_b = tl.load(embed_ptr + off_b0 + cols)
            e_c = tl.load(embed_ptr + off_c0 + cols)
            e_d = tl.load(embed_ptr + off_d0 + cols)
            v0 = w00a * e_a + w01a * e_b + w10a * e_c + w11a * e_d
            e_a = tl.load(embed_ptr + off_a1 + cols)
            e_b = tl.load(embed_ptr + off_b1 + cols)
            e_c = tl.load(embed_ptr + off_c1 + cols)
            e_d = tl.load(embed_ptr + off_d1 + cols)
            v1 = w00b * e_a + w01b * e_b + w10b * e_c + w11b * e_d
            tl.store(output_ptr + out_off0 + cols, v0)
            tl.store(output_ptr + out_off1 + cols, v1)

    @triton.jit
    def _seg_pair_diff6(embed_ptr, output_ptr,
                        off_a0, off_b0, off_c0, off_d0,
                        off_a1, off_b1, off_c1, off_d1,
                        out_off0, out_off1,
                        w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                        START: tl.constexpr, WIDTH: tl.constexpr):
        """r58 diff-branch segment with duplicate-load reuse (6 loads).

        On the host-gated window ``0 < w_scale <= 1.0`` the diff branch
        (``wf1 != wf0``) forces ``wf1 == wf0 + 1`` (w_frac1 - w_frac0 ==
        w_scale crosses exactly one integer) and ``wf0 + 1 <= NUM_GRID - 1``
        (wf0 == NUM_GRID-1 pushes wf1 == wf0 into the shared branch), so
        ``wc0 == min(wf0+1, NUM_GRID-1) == wf1`` and therefore
        ``off_a1 == off_b0`` and ``off_c1 == off_d0`` hold EXACTLY. The
        row-1 loads at off_a1/off_c1 are the same addresses as the row-0
        loads at off_b0/off_d0 -- reuse the register values instead of
        re-issuing two loads: 8 -> 6 loads per segment per pair, zero
        arithmetic change, zero divergence class (address-equal loads
        return bitwise-equal values; even the vpair vanishing +/-0 corner
        class is absent). rawAB-diff6 gates it on-machine (expected
        strict equality).
        """
        if WIDTH > 0:
            cols = START + tl.arange(0, WIDTH)
            e_a0 = tl.load(embed_ptr + off_a0 + cols)
            e_b0 = tl.load(embed_ptr + off_b0 + cols)
            e_c0 = tl.load(embed_ptr + off_c0 + cols)
            e_d0 = tl.load(embed_ptr + off_d0 + cols)
            v0 = w00a * e_a0 + w01a * e_b0 + w10a * e_c0 + w11a * e_d0
            e_b1 = tl.load(embed_ptr + off_b1 + cols)
            e_d1 = tl.load(embed_ptr + off_d1 + cols)
            # off_a1 == off_b0 and off_c1 == off_d0 (host-proven window):
            # the reloaded row-1 a/c values equal the register e_b0/e_d0.
            v1 = w00b * e_b0 + w01b * e_b1 + w10b * e_d0 + w11b * e_d1
            tl.store(output_ptr + out_off0 + cols, v0)
            tl.store(output_ptr + out_off1 + cols, v1)

    @triton.jit
    def _bilinear_pair_body_flat(embed_ptr, output_ptr, out_row, row, H, W,
                                 h_scale, w_scale, NUM_GRID: tl.constexpr,
                                 M_SIZE: tl.constexpr, HIDDEN_DIM: tl.constexpr,
                                 SEG0: tl.constexpr, SEG1: tl.constexpr,
                                 SEG2: tl.constexpr, SEG3: tl.constexpr,
                                 SEG4: tl.constexpr, SEG5: tl.constexpr):
        """Flat multi-segment variant of ``_bilinear_pair_body`` (r30).

        Identical scalar math and identical per-element arithmetic --
        only the STATEMENT ORDER changes: instead of six sequential
        load->FMA->store rounds (each round stalls the in-order AIV
        pipeline on its own memory latency, B ~= 3.4us/body on the
        6-segment tail shape), each WAVE issues ALL its segment loads
        back-to-back first so several memory latencies overlap, then
        does the FMAs and stores. Waves of 3 segments keep the live
        register set bounded (<= 24 vectors x 512 lanes fp32 ~ 49KB,
        inside the 256KB UB); for 2-segment shapes (hidden=1152) wave 2
        compiles to nothing. Segments are the same host power-of-two
        decomposition, so every lane stays unmasked (iron rule r18) and
        output remains bitwise identical to pair/single.
        """
        total_spatial = H * W
        spatial0 = row % total_spatial

        num_blocks_w = W // M_SIZE
        block_idx = spatial0 // (M_SIZE * M_SIZE)
        local_idx = spatial0 % (M_SIZE * M_SIZE)
        br = block_idx // num_blocks_w
        bc = block_idx % num_blocks_w
        lr = local_idx // M_SIZE
        lc = local_idx % M_SIZE
        row_i = br * M_SIZE + lr
        col0 = bc * M_SIZE + lc

        h_frac = row_i.to(tl.float32) * h_scale
        hf = h_frac.to(tl.int32)
        hc = tl.minimum(hf + 1, NUM_GRID - 1)
        dh = h_frac - hf.to(tl.float32)

        w_frac0 = col0.to(tl.float32) * w_scale
        w_frac1 = (col0 + 1).to(tl.float32) * w_scale
        wf0 = w_frac0.to(tl.int32)
        wf1 = w_frac1.to(tl.int32)
        wc0 = tl.minimum(wf0 + 1, NUM_GRID - 1)
        wc1 = tl.minimum(wf1 + 1, NUM_GRID - 1)
        dw0 = w_frac0 - wf0.to(tl.float32)
        dw1 = w_frac1 - wf1.to(tl.float32)

        w11_0 = dh * dw0
        w10_0 = dh - w11_0
        w01_0 = dw0 - w11_0
        w00_0 = 1.0 - dh - w01_0
        w11_1 = dh * dw1
        w10_1 = dh - w11_1
        w01_1 = dw1 - w11_1
        w00_1 = 1.0 - dh - w01_1

        out_dtype = output_ptr.dtype.element_ty
        w00a = w00_0.to(out_dtype)
        w01a = w01_0.to(out_dtype)
        w10a = w10_0.to(out_dtype)
        w11a = w11_0.to(out_dtype)
        w00b = w00_1.to(out_dtype)
        w01b = w01_1.to(out_dtype)
        w10b = w10_1.to(out_dtype)
        w11b = w11_1.to(out_dtype)

        out_off0 = out_row * HIDDEN_DIM
        out_off1 = out_off0 + HIDDEN_DIM

        if wf1 == wf0:
            off_a = (hf * NUM_GRID + wf0) * HIDDEN_DIM
            off_b = (hf * NUM_GRID + wc0) * HIDDEN_DIM
            off_c = (hc * NUM_GRID + wf0) * HIDDEN_DIM
            off_d = (hc * NUM_GRID + wc0) * HIDDEN_DIM
            # wave 1: segs 0-2, all loads issued before any FMA
            if SEG0 > 0:
                s0c = tl.arange(0, SEG0)
                s0a = tl.load(embed_ptr + off_a + s0c)
                s0b = tl.load(embed_ptr + off_b + s0c)
                s0c_v = tl.load(embed_ptr + off_c + s0c)
                s0d = tl.load(embed_ptr + off_d + s0c)
            if SEG1 > 0:
                s1c = SEG0 + tl.arange(0, SEG1)
                s1a = tl.load(embed_ptr + off_a + s1c)
                s1b = tl.load(embed_ptr + off_b + s1c)
                s1cv = tl.load(embed_ptr + off_c + s1c)
                s1d = tl.load(embed_ptr + off_d + s1c)
            if SEG2 > 0:
                s2c = SEG0 + SEG1 + tl.arange(0, SEG2)
                s2a = tl.load(embed_ptr + off_a + s2c)
                s2b = tl.load(embed_ptr + off_b + s2c)
                s2cv = tl.load(embed_ptr + off_c + s2c)
                s2d = tl.load(embed_ptr + off_d + s2c)
            if SEG0 > 0:
                tl.store(output_ptr + out_off0 + s0c,
                         w00a * s0a + w01a * s0b + w10a * s0c_v + w11a * s0d)
                tl.store(output_ptr + out_off1 + s0c,
                         w00b * s0a + w01b * s0b + w10b * s0c_v + w11b * s0d)
            if SEG1 > 0:
                tl.store(output_ptr + out_off0 + s1c,
                         w00a * s1a + w01a * s1b + w10a * s1cv + w11a * s1d)
                tl.store(output_ptr + out_off1 + s1c,
                         w00b * s1a + w01b * s1b + w10b * s1cv + w11b * s1d)
            if SEG2 > 0:
                tl.store(output_ptr + out_off0 + s2c,
                         w00a * s2a + w01a * s2b + w10a * s2cv + w11a * s2d)
                tl.store(output_ptr + out_off1 + s2c,
                         w00b * s2a + w01b * s2b + w10b * s2cv + w11b * s2d)
            # wave 2: segs 3-5
            if SEG3 > 0:
                s3c = SEG0 + SEG1 + SEG2 + tl.arange(0, SEG3)
                s3a = tl.load(embed_ptr + off_a + s3c)
                s3b = tl.load(embed_ptr + off_b + s3c)
                s3cv = tl.load(embed_ptr + off_c + s3c)
                s3d = tl.load(embed_ptr + off_d + s3c)
            if SEG4 > 0:
                s4c = SEG0 + SEG1 + SEG2 + SEG3 + tl.arange(0, SEG4)
                s4a = tl.load(embed_ptr + off_a + s4c)
                s4b = tl.load(embed_ptr + off_b + s4c)
                s4cv = tl.load(embed_ptr + off_c + s4c)
                s4d = tl.load(embed_ptr + off_d + s4c)
            if SEG5 > 0:
                s5c = (SEG0 + SEG1 + SEG2 + SEG3 + SEG4
                       + tl.arange(0, SEG5))
                s5a = tl.load(embed_ptr + off_a + s5c)
                s5b = tl.load(embed_ptr + off_b + s5c)
                s5cv = tl.load(embed_ptr + off_c + s5c)
                s5d = tl.load(embed_ptr + off_d + s5c)
            if SEG3 > 0:
                tl.store(output_ptr + out_off0 + s3c,
                         w00a * s3a + w01a * s3b + w10a * s3cv + w11a * s3d)
                tl.store(output_ptr + out_off1 + s3c,
                         w00b * s3a + w01b * s3b + w10b * s3cv + w11b * s3d)
            if SEG4 > 0:
                tl.store(output_ptr + out_off0 + s4c,
                         w00a * s4a + w01a * s4b + w10a * s4cv + w11a * s4d)
                tl.store(output_ptr + out_off1 + s4c,
                         w00b * s4a + w01b * s4b + w10b * s4cv + w11b * s4d)
            if SEG5 > 0:
                tl.store(output_ptr + out_off0 + s5c,
                         w00a * s5a + w01a * s5b + w10a * s5cv + w11a * s5d)
                tl.store(output_ptr + out_off1 + s5c,
                         w00b * s5a + w01b * s5b + w10b * s5cv + w11b * s5d)
        else:
            off_a0 = (hf * NUM_GRID + wf0) * HIDDEN_DIM
            off_b0 = (hf * NUM_GRID + wc0) * HIDDEN_DIM
            off_c0 = (hc * NUM_GRID + wf0) * HIDDEN_DIM
            off_d0 = (hc * NUM_GRID + wc0) * HIDDEN_DIM
            off_a1 = (hf * NUM_GRID + wf1) * HIDDEN_DIM
            off_b1 = (hf * NUM_GRID + wc1) * HIDDEN_DIM
            off_c1 = (hc * NUM_GRID + wf1) * HIDDEN_DIM
            off_d1 = (hc * NUM_GRID + wc1) * HIDDEN_DIM
            # wave 1: segs 0-2, 24 independent loads issued first
            if SEG0 > 0:
                d0c = tl.arange(0, SEG0)
                d0a0 = tl.load(embed_ptr + off_a0 + d0c)
                d0b0 = tl.load(embed_ptr + off_b0 + d0c)
                d0c0 = tl.load(embed_ptr + off_c0 + d0c)
                d0dd0 = tl.load(embed_ptr + off_d0 + d0c)
                d0a1 = tl.load(embed_ptr + off_a1 + d0c)
                d0b1 = tl.load(embed_ptr + off_b1 + d0c)
                d0c1 = tl.load(embed_ptr + off_c1 + d0c)
                d0dd1 = tl.load(embed_ptr + off_d1 + d0c)
            if SEG1 > 0:
                d1c = SEG0 + tl.arange(0, SEG1)
                d1a0 = tl.load(embed_ptr + off_a0 + d1c)
                d1b0 = tl.load(embed_ptr + off_b0 + d1c)
                d1c0 = tl.load(embed_ptr + off_c0 + d1c)
                d1dd0 = tl.load(embed_ptr + off_d0 + d1c)
                d1a1 = tl.load(embed_ptr + off_a1 + d1c)
                d1b1 = tl.load(embed_ptr + off_b1 + d1c)
                d1c1 = tl.load(embed_ptr + off_c1 + d1c)
                d1dd1 = tl.load(embed_ptr + off_d1 + d1c)
            if SEG2 > 0:
                d2c = SEG0 + SEG1 + tl.arange(0, SEG2)
                d2a0 = tl.load(embed_ptr + off_a0 + d2c)
                d2b0 = tl.load(embed_ptr + off_b0 + d2c)
                d2c0 = tl.load(embed_ptr + off_c0 + d2c)
                d2dd0 = tl.load(embed_ptr + off_d0 + d2c)
                d2a1 = tl.load(embed_ptr + off_a1 + d2c)
                d2b1 = tl.load(embed_ptr + off_b1 + d2c)
                d2c1 = tl.load(embed_ptr + off_c1 + d2c)
                d2dd1 = tl.load(embed_ptr + off_d1 + d2c)
            if SEG0 > 0:
                tl.store(output_ptr + out_off0 + d0c,
                         w00a * d0a0 + w01a * d0b0 + w10a * d0c0
                         + w11a * d0dd0)
                tl.store(output_ptr + out_off1 + d0c,
                         w00b * d0a1 + w01b * d0b1 + w10b * d0c1
                         + w11b * d0dd1)
            if SEG1 > 0:
                tl.store(output_ptr + out_off0 + d1c,
                         w00a * d1a0 + w01a * d1b0 + w10a * d1c0
                         + w11a * d1dd0)
                tl.store(output_ptr + out_off1 + d1c,
                         w00b * d1a1 + w01b * d1b1 + w10b * d1c1
                         + w11b * d1dd1)
            if SEG2 > 0:
                tl.store(output_ptr + out_off0 + d2c,
                         w00a * d2a0 + w01a * d2b0 + w10a * d2c0
                         + w11a * d2dd0)
                tl.store(output_ptr + out_off1 + d2c,
                         w00b * d2a1 + w01b * d2b1 + w10b * d2c1
                         + w11b * d2dd1)
            # wave 2: segs 3-5
            if SEG3 > 0:
                d3c = SEG0 + SEG1 + SEG2 + tl.arange(0, SEG3)
                d3a0 = tl.load(embed_ptr + off_a0 + d3c)
                d3b0 = tl.load(embed_ptr + off_b0 + d3c)
                d3c0 = tl.load(embed_ptr + off_c0 + d3c)
                d3dd0 = tl.load(embed_ptr + off_d0 + d3c)
                d3a1 = tl.load(embed_ptr + off_a1 + d3c)
                d3b1 = tl.load(embed_ptr + off_b1 + d3c)
                d3c1 = tl.load(embed_ptr + off_c1 + d3c)
                d3dd1 = tl.load(embed_ptr + off_d1 + d3c)
            if SEG4 > 0:
                d4c = (SEG0 + SEG1 + SEG2 + SEG3 + tl.arange(0, SEG4))
                d4a0 = tl.load(embed_ptr + off_a0 + d4c)
                d4b0 = tl.load(embed_ptr + off_b0 + d4c)
                d4c0 = tl.load(embed_ptr + off_c0 + d4c)
                d4dd0 = tl.load(embed_ptr + off_d0 + d4c)
                d4a1 = tl.load(embed_ptr + off_a1 + d4c)
                d4b1 = tl.load(embed_ptr + off_b1 + d4c)
                d4c1 = tl.load(embed_ptr + off_c1 + d4c)
                d4dd1 = tl.load(embed_ptr + off_d1 + d4c)
            if SEG5 > 0:
                d5c = (SEG0 + SEG1 + SEG2 + SEG3 + SEG4
                       + tl.arange(0, SEG5))
                d5a0 = tl.load(embed_ptr + off_a0 + d5c)
                d5b0 = tl.load(embed_ptr + off_b0 + d5c)
                d5c0 = tl.load(embed_ptr + off_c0 + d5c)
                d5dd0 = tl.load(embed_ptr + off_d0 + d5c)
                d5a1 = tl.load(embed_ptr + off_a1 + d5c)
                d5b1 = tl.load(embed_ptr + off_b1 + d5c)
                d5c1 = tl.load(embed_ptr + off_c1 + d5c)
                d5dd1 = tl.load(embed_ptr + off_d1 + d5c)
            if SEG3 > 0:
                tl.store(output_ptr + out_off0 + d3c,
                         w00a * d3a0 + w01a * d3b0 + w10a * d3c0
                         + w11a * d3dd0)
                tl.store(output_ptr + out_off1 + d3c,
                         w00b * d3a1 + w01b * d3b1 + w10b * d3c1
                         + w11b * d3dd1)
            if SEG4 > 0:
                tl.store(output_ptr + out_off0 + d4c,
                         w00a * d4a0 + w01a * d4b0 + w10a * d4c0
                         + w11a * d4dd0)
                tl.store(output_ptr + out_off1 + d4c,
                         w00b * d4a1 + w01b * d4b1 + w10b * d4c1
                         + w11b * d4dd1)
            if SEG5 > 0:
                tl.store(output_ptr + out_off0 + d5c,
                         w00a * d5a0 + w01a * d5b0 + w10a * d5c0
                         + w11a * d5dd0)
                tl.store(output_ptr + out_off1 + d5c,
                         w00b * d5a1 + w01b * d5b1 + w10b * d5c1
                         + w11b * d5dd1)

    @triton.jit
    def _bilinear_pair_body(embed_ptr, output_ptr, out_row, row, H, W, h_scale,
                            w_scale, NUM_GRID: tl.constexpr,
                            M_SIZE: tl.constexpr, HIDDEN_DIM: tl.constexpr,
                            SEG0: tl.constexpr, SEG1: tl.constexpr,
                            SEG2: tl.constexpr, SEG3: tl.constexpr,
                            SEG4: tl.constexpr, SEG5: tl.constexpr):
        """Two adjacent flattened output rows (out_row, out_row+1), one program.

        Launch contract (host-enforced): the pair belongs to one image,
        TOTAL_ROWS is even and m_size is even (hence W even). Then the
        flattened rows (2k, 2k+1) always sit in the same image row with
        adjacent columns (col, col+1): the per-image spatial row 2k % (H*W)
        is even, and for even m_size the spatial-merge mapping of
        (spatial, spatial+1) with spatial even shares block/lr and advances
        lc by one (local indices (2j, 2j+1)). Both rows therefore share
        hf/hc, and share ALL four embed rows whenever wf does not advance
        between the two columns -- the scalar branch halves embed-load
        instructions for those pairs (r7 evidence: BF16 compute-bound at
        53.0% MBU vs FP32 123.9%; tl.load occupies the same issue pipeline
        as the bf16 FMA work, so cutting loads frees FMA slots).

        Per-row arithmetic (casts, expression order) is identical to
        ``_bilinear_row_body``, so output stays bitwise identical to the
        single-row kernel. Requires exact power-of-two column segments
        (``_pow2_segments`` -- the host must fall back to the single-row
        kernel otherwise, since the pair kernel is fully unmasked).
        """
        total_spatial = H * W
        spatial0 = row % total_spatial

        num_blocks_w = W // M_SIZE
        block_idx = spatial0 // (M_SIZE * M_SIZE)
        local_idx = spatial0 % (M_SIZE * M_SIZE)
        br = block_idx // num_blocks_w
        bc = block_idx % num_blocks_w
        lr = local_idx // M_SIZE
        lc = local_idx % M_SIZE
        row_i = br * M_SIZE + lr
        col0 = bc * M_SIZE + lc
        # spatial0 is even and m_size even => local_idx even, lc even (no
        # carry on lc+1), so (spatial0+1) maps to (row_i, col0+1).

        h_frac = row_i.to(tl.float32) * h_scale
        hf = h_frac.to(tl.int32)
        hc = tl.minimum(hf + 1, NUM_GRID - 1)
        dh = h_frac - hf.to(tl.float32)

        w_frac0 = col0.to(tl.float32) * w_scale
        w_frac1 = (col0 + 1).to(tl.float32) * w_scale
        wf0 = w_frac0.to(tl.int32)
        wf1 = w_frac1.to(tl.int32)
        wc0 = tl.minimum(wf0 + 1, NUM_GRID - 1)
        wc1 = tl.minimum(wf1 + 1, NUM_GRID - 1)
        dw0 = w_frac0 - wf0.to(tl.float32)
        dw1 = w_frac1 - wf1.to(tl.float32)

        w11_0 = dh * dw0
        w10_0 = dh - w11_0
        w01_0 = dw0 - w11_0
        w00_0 = 1.0 - dh - w01_0
        w11_1 = dh * dw1
        w10_1 = dh - w11_1
        w01_1 = dw1 - w11_1
        w00_1 = 1.0 - dh - w01_1

        out_dtype = output_ptr.dtype.element_ty
        w00a = w00_0.to(out_dtype)
        w01a = w01_0.to(out_dtype)
        w10a = w10_0.to(out_dtype)
        w11a = w11_0.to(out_dtype)
        w00b = w00_1.to(out_dtype)
        w01b = w01_1.to(out_dtype)
        w10b = w10_1.to(out_dtype)
        w11b = w11_1.to(out_dtype)

        out_off0 = out_row * HIDDEN_DIM
        out_off1 = out_off0 + HIDDEN_DIM

        if wf1 == wf0:
            # Adjacent columns share (hf, hc) x (wf0, wc0): 4 embed rows
            # serve both output rows (wc1 == wc0 follows from wf1 == wf0).
            off_a = (hf * NUM_GRID + wf0) * HIDDEN_DIM
            off_b = (hf * NUM_GRID + wc0) * HIDDEN_DIM
            off_c = (hc * NUM_GRID + wf0) * HIDDEN_DIM
            off_d = (hc * NUM_GRID + wc0) * HIDDEN_DIM
            _seg_pair_shared(embed_ptr, output_ptr, off_a, off_b, off_c,
                             off_d, out_off0, out_off1,
                             w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                             0, SEG0)
            _seg_pair_shared(embed_ptr, output_ptr, off_a, off_b, off_c,
                             off_d, out_off0, out_off1,
                             w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                             SEG0, SEG1)
            _seg_pair_shared(embed_ptr, output_ptr, off_a, off_b, off_c,
                             off_d, out_off0, out_off1,
                             w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                             SEG0 + SEG1, SEG2)
            _seg_pair_shared(embed_ptr, output_ptr, off_a, off_b, off_c,
                             off_d, out_off0, out_off1,
                             w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                             SEG0 + SEG1 + SEG2, SEG3)
            _seg_pair_shared(embed_ptr, output_ptr, off_a, off_b, off_c,
                             off_d, out_off0, out_off1,
                             w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                             SEG0 + SEG1 + SEG2 + SEG3, SEG4)
            _seg_pair_shared(embed_ptr, output_ptr, off_a, off_b, off_c,
                             off_d, out_off0, out_off1,
                             w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                             SEG0 + SEG1 + SEG2 + SEG3 + SEG4, SEG5)
        else:
            off_a0 = (hf * NUM_GRID + wf0) * HIDDEN_DIM
            off_b0 = (hf * NUM_GRID + wc0) * HIDDEN_DIM
            off_c0 = (hc * NUM_GRID + wf0) * HIDDEN_DIM
            off_d0 = (hc * NUM_GRID + wc0) * HIDDEN_DIM
            off_a1 = (hf * NUM_GRID + wf1) * HIDDEN_DIM
            off_b1 = (hf * NUM_GRID + wc1) * HIDDEN_DIM
            off_c1 = (hc * NUM_GRID + wf1) * HIDDEN_DIM
            off_d1 = (hc * NUM_GRID + wc1) * HIDDEN_DIM
            _seg_pair_diff(embed_ptr, output_ptr,
                           off_a0, off_b0, off_c0, off_d0,
                           off_a1, off_b1, off_c1, off_d1,
                           out_off0, out_off1,
                           w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                           0, SEG0)
            _seg_pair_diff(embed_ptr, output_ptr,
                           off_a0, off_b0, off_c0, off_d0,
                           off_a1, off_b1, off_c1, off_d1,
                           out_off0, out_off1,
                           w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                           SEG0, SEG1)
            _seg_pair_diff(embed_ptr, output_ptr,
                           off_a0, off_b0, off_c0, off_d0,
                           off_a1, off_b1, off_c1, off_d1,
                           out_off0, out_off1,
                           w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                           SEG0 + SEG1, SEG2)
            _seg_pair_diff(embed_ptr, output_ptr,
                           off_a0, off_b0, off_c0, off_d0,
                           off_a1, off_b1, off_c1, off_d1,
                           out_off0, out_off1,
                           w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                           SEG0 + SEG1 + SEG2, SEG3)
            _seg_pair_diff(embed_ptr, output_ptr,
                           off_a0, off_b0, off_c0, off_d0,
                           off_a1, off_b1, off_c1, off_d1,
                           out_off0, out_off1,
                           w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                           SEG0 + SEG1 + SEG2 + SEG3, SEG4)
            _seg_pair_diff(embed_ptr, output_ptr,
                           off_a0, off_b0, off_c0, off_d0,
                           off_a1, off_b1, off_c1, off_d1,
                           out_off0, out_off1,
                           w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                           SEG0 + SEG1 + SEG2 + SEG3 + SEG4, SEG5)

    @triton.jit
    def _bilinear_pos_embed_kernel_pair(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        TOTAL_ROWS,
        CHUNKS: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """Two-rows-per-program variant of ``_bilinear_pos_embed_kernel``.

        Each chunk slot covers flattened rows (2*pid, 2*pid+1); TOTAL_ROWS
        must be even, so ``slots = TOTAL_ROWS // 2`` programs suffice and
        the grid cap applies to slots. Launch contract of
        ``_bilinear_pair_body`` applies (one image, m_size even, exact
        lane split). Production default of the single-image wrapper
        since r10 (r9 NPU evidence: bitwise identical to the single-row
        kernel, BF16 3.252ms vs 3.387ms at 55% MBU, FP32 parity); the
        wrapper falls back to the single-row kernel when the contract
        fails.
        """
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            pid = tl.program_id(0) + c * grid_size
            row0 = 2 * pid
            if row0 < TOTAL_ROWS:
                _bilinear_pair_body(
                    embed_ptr, output_ptr, row0, row0, H, W, h_scale, w_scale,
                    NUM_GRID, M_SIZE, HIDDEN_DIM,
                    SEG0, SEG1, SEG2, SEG3, SEG4, SEG5)

    @triton.jit
    def _bilinear_pair_body_d6(embed_ptr, output_ptr, out_row, row, H, W,
                               h_scale, w_scale, NUM_GRID: tl.constexpr,
                               M_SIZE: tl.constexpr, HIDDEN_DIM: tl.constexpr,
                               SEG0: tl.constexpr, SEG1: tl.constexpr,
                               SEG2: tl.constexpr, SEG3: tl.constexpr,
                               SEG4: tl.constexpr, SEG5: tl.constexpr):
        """r58 diff6 variant of ``_bilinear_pair_body`` (6-load diff segs).

        Verbatim clone of ``_bilinear_pair_body`` (row mapping, weight
        chain, casts, expression order -- bitwise chain untouched) with
        ONLY the diff-branch segment swapped to ``_seg_pair_diff6``, which
        reuses the address-equal row-1 a/c loads (host window
        ``0 < w_scale <= 1.0``: wf1 == wf0+1 and wc0 == wf1 in the diff
        branch, so off_a1 == off_b0 and off_c1 == off_d0 exactly). The
        shared branch (wf1 == wf0) is unchanged -- it already serves both
        rows from 4 loads. Host gate ``_FORCE_DIFF6`` must enforce
        ``0 < w_scale <= 1.0``; w_scale > 1.0 breaks the address equality
        and is excluded. Zero divergence class: address-equal loads are
        bitwise-equal by construction (rawAB-diff6 verifies strict
        equality on-machine).
        """
        total_spatial = H * W
        spatial0 = row % total_spatial

        num_blocks_w = W // M_SIZE
        block_idx = spatial0 // (M_SIZE * M_SIZE)
        local_idx = spatial0 % (M_SIZE * M_SIZE)
        br = block_idx // num_blocks_w
        bc = block_idx % num_blocks_w
        lr = local_idx // M_SIZE
        lc = local_idx % M_SIZE
        row_i = br * M_SIZE + lr
        col0 = bc * M_SIZE + lc

        h_frac = row_i.to(tl.float32) * h_scale
        hf = h_frac.to(tl.int32)
        hc = tl.minimum(hf + 1, NUM_GRID - 1)
        dh = h_frac - hf.to(tl.float32)

        w_frac0 = col0.to(tl.float32) * w_scale
        w_frac1 = (col0 + 1).to(tl.float32) * w_scale
        wf0 = w_frac0.to(tl.int32)
        wf1 = w_frac1.to(tl.int32)
        wc0 = tl.minimum(wf0 + 1, NUM_GRID - 1)
        wc1 = tl.minimum(wf1 + 1, NUM_GRID - 1)
        dw0 = w_frac0 - wf0.to(tl.float32)
        dw1 = w_frac1 - wf1.to(tl.float32)

        w11_0 = dh * dw0
        w10_0 = dh - w11_0
        w01_0 = dw0 - w11_0
        w00_0 = 1.0 - dh - w01_0
        w11_1 = dh * dw1
        w10_1 = dh - w11_1
        w01_1 = dw1 - w11_1
        w00_1 = 1.0 - dh - w01_1

        out_dtype = output_ptr.dtype.element_ty
        w00a = w00_0.to(out_dtype)
        w01a = w01_0.to(out_dtype)
        w10a = w10_0.to(out_dtype)
        w11a = w11_0.to(out_dtype)
        w00b = w00_1.to(out_dtype)
        w01b = w01_1.to(out_dtype)
        w10b = w10_1.to(out_dtype)
        w11b = w11_1.to(out_dtype)

        out_off0 = out_row * HIDDEN_DIM
        out_off1 = out_off0 + HIDDEN_DIM

        if wf1 == wf0:
            off_a = (hf * NUM_GRID + wf0) * HIDDEN_DIM
            off_b = (hf * NUM_GRID + wc0) * HIDDEN_DIM
            off_c = (hc * NUM_GRID + wf0) * HIDDEN_DIM
            off_d = (hc * NUM_GRID + wc0) * HIDDEN_DIM
            _seg_pair_shared(embed_ptr, output_ptr, off_a, off_b, off_c,
                             off_d, out_off0, out_off1,
                             w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                             0, SEG0)
            _seg_pair_shared(embed_ptr, output_ptr, off_a, off_b, off_c,
                             off_d, out_off0, out_off1,
                             w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                             SEG0, SEG1)
            _seg_pair_shared(embed_ptr, output_ptr, off_a, off_b, off_c,
                             off_d, out_off0, out_off1,
                             w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                             SEG0 + SEG1, SEG2)
            _seg_pair_shared(embed_ptr, output_ptr, off_a, off_b, off_c,
                             off_d, out_off0, out_off1,
                             w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                             SEG0 + SEG1 + SEG2, SEG3)
            _seg_pair_shared(embed_ptr, output_ptr, off_a, off_b, off_c,
                             off_d, out_off0, out_off1,
                             w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                             SEG0 + SEG1 + SEG2 + SEG3, SEG4)
            _seg_pair_shared(embed_ptr, output_ptr, off_a, off_b, off_c,
                             off_d, out_off0, out_off1,
                             w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                             SEG0 + SEG1 + SEG2 + SEG3 + SEG4, SEG5)
        else:
            off_a0 = (hf * NUM_GRID + wf0) * HIDDEN_DIM
            off_b0 = (hf * NUM_GRID + wc0) * HIDDEN_DIM
            off_c0 = (hc * NUM_GRID + wf0) * HIDDEN_DIM
            off_d0 = (hc * NUM_GRID + wc0) * HIDDEN_DIM
            off_a1 = (hf * NUM_GRID + wf1) * HIDDEN_DIM
            off_b1 = (hf * NUM_GRID + wc1) * HIDDEN_DIM
            off_c1 = (hc * NUM_GRID + wf1) * HIDDEN_DIM
            off_d1 = (hc * NUM_GRID + wc1) * HIDDEN_DIM
            _seg_pair_diff6(embed_ptr, output_ptr,
                            off_a0, off_b0, off_c0, off_d0,
                            off_a1, off_b1, off_c1, off_d1,
                            out_off0, out_off1,
                            w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                            0, SEG0)
            _seg_pair_diff6(embed_ptr, output_ptr,
                            off_a0, off_b0, off_c0, off_d0,
                            off_a1, off_b1, off_c1, off_d1,
                            out_off0, out_off1,
                            w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                            SEG0, SEG1)
            _seg_pair_diff6(embed_ptr, output_ptr,
                            off_a0, off_b0, off_c0, off_d0,
                            off_a1, off_b1, off_c1, off_d1,
                            out_off0, out_off1,
                            w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                            SEG0 + SEG1, SEG2)
            _seg_pair_diff6(embed_ptr, output_ptr,
                            off_a0, off_b0, off_c0, off_d0,
                            off_a1, off_b1, off_c1, off_d1,
                            out_off0, out_off1,
                            w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                            SEG0 + SEG1 + SEG2, SEG3)
            _seg_pair_diff6(embed_ptr, output_ptr,
                            off_a0, off_b0, off_c0, off_d0,
                            off_a1, off_b1, off_c1, off_d1,
                            out_off0, out_off1,
                            w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                            SEG0 + SEG1 + SEG2 + SEG3, SEG4)
            _seg_pair_diff6(embed_ptr, output_ptr,
                            off_a0, off_b0, off_c0, off_d0,
                            off_a1, off_b1, off_c1, off_d1,
                            out_off0, out_off1,
                            w00a, w01a, w10a, w11a, w00b, w01b, w10b, w11b,
                            SEG0 + SEG1 + SEG2 + SEG3 + SEG4, SEG5)

    @triton.jit
    def _bilinear_pos_embed_kernel_pair_d6(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        TOTAL_ROWS,
        CHUNKS: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """r58 bench-only pair kernel with the diff6 body (load reuse).

        Structural clone of ``_bilinear_pos_embed_kernel_pair`` (same
        pair-slot mapping, CHUNKS static unroll, row0 guard) with the
        body swapped to ``_bilinear_pair_body_d6``. Bench-only: host
        gates _FORCE_DIFF6 AND the pair contract AND
        ``0 < w_scale <= 1.0`` (the diff6 address-equality window;
        w_scale > 1.0 keeps production pair). rawAB-diff6 gates it.
        """
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            pid = tl.program_id(0) + c * grid_size
            row0 = 2 * pid
            if row0 < TOTAL_ROWS:
                _bilinear_pair_body_d6(
                    embed_ptr, output_ptr, row0, row0, H, W, h_scale, w_scale,
                    NUM_GRID, M_SIZE, HIDDEN_DIM,
                    SEG0, SEG1, SEG2, SEG3, SEG4, SEG5)

    @triton.jit
    def _seg_pair_shared_2d(embed_ptr, output_ptr, off_a, off_b, off_c,
                            off_d, out_off0, HID: tl.constexpr,
                            w00a, w01a, w10a, w11a,
                            w00b, w01b, w10b, w11b,
                            START: tl.constexpr, WIDTH: tl.constexpr):
        """``_seg_pair_shared`` with the FMA/store issued as one [2, WIDTH]
        2D tile (r53).

        Loads stay 1D (both pair rows share the same 4 embed rows); the
        per-element FMA is the verbatim ``_seg_pair_shared`` expression
        with the row-0/row-1 weights stacked into [2] vectors via
        ``tl.where`` (scalar broadcast, no rounding), and the two 1D
        stores merge into one 2D store with row stride HID =
        HIDDEN_DIM. Values and addresses are identical to the scalar
        form -> bitwise identical.
        """
        if WIDTH > 0:
            cols = START + tl.arange(0, WIDTH)
            rows = tl.arange(0, 2)
            e_a = tl.load(embed_ptr + off_a + cols)
            e_b = tl.load(embed_ptr + off_b + cols)
            e_c = tl.load(embed_ptr + off_c + cols)
            e_d = tl.load(embed_ptr + off_d + cols)
            w00_2 = tl.where(rows == 0, w00a, w00b)
            w01_2 = tl.where(rows == 0, w01a, w01b)
            w10_2 = tl.where(rows == 0, w10a, w10b)
            w11_2 = tl.where(rows == 0, w11a, w11b)
            v = (w00_2[:, None] * e_a[None, :] + w01_2[:, None] * e_b[None, :]
                 + w10_2[:, None] * e_c[None, :] + w11_2[:, None] * e_d[None, :])
            tl.store(output_ptr + out_off0 + rows[:, None] * HID
                     + cols[None, :], v)

    @triton.jit
    def _seg_pair_diff_2d(embed_ptr, output_ptr,
                          off_a0, off_b0, off_c0, off_d0,
                          out_off0, HID: tl.constexpr,
                          w00a, w01a, w10a, w11a,
                          w00b, w01b, w10b, w11b,
                          START: tl.constexpr, WIDTH: tl.constexpr):
        """``_seg_pair_diff`` for the adjacent-column case, issued as 2D
        tiles (r53).

        Precondition (body-guarded): wf1 == wf0 + 1 and wc1 == wc0 + 1,
        so each of the four embed-row pairs (a/b/c/d) spans two ADJACENT
        grid columns -- in the row-major [G*G, hidden] table the second
        column's row sits exactly HID elements after the first. One
        [2, WIDTH] strided load per pair covers both columns' embed rows
        (8 scalar loads -> 4 2D loads); the per-element FMA expression
        and the cast points are verbatim ``_seg_pair_diff``; the two 1D
        stores merge into one 2D store. Element (1, c) addresses equal
        the scalar path's second-column addresses -> bitwise identical.
        """
        if WIDTH > 0:
            cols = START + tl.arange(0, WIDTH)
            rows = tl.arange(0, 2)
            e_a = tl.load(embed_ptr + off_a0
                          + rows[:, None] * HID + cols[None, :])
            e_b = tl.load(embed_ptr + off_b0
                          + rows[:, None] * HID + cols[None, :])
            e_c = tl.load(embed_ptr + off_c0
                          + rows[:, None] * HID + cols[None, :])
            e_d = tl.load(embed_ptr + off_d0
                          + rows[:, None] * HID + cols[None, :])
            w00_2 = tl.where(rows == 0, w00a, w00b)
            w01_2 = tl.where(rows == 0, w01a, w01b)
            w10_2 = tl.where(rows == 0, w10a, w10b)
            w11_2 = tl.where(rows == 0, w11a, w11b)
            v = (w00_2[:, None] * e_a + w01_2[:, None] * e_b
                 + w10_2[:, None] * e_c + w11_2[:, None] * e_d)
            tl.store(output_ptr + out_off0 + rows[:, None] * HID
                     + cols[None, :], v)

    @triton.jit
    def _bilinear_pair_body_2d(embed_ptr, output_ptr, out_row, row, H, W,
                               h_scale, w_scale,
                               NUM_GRID: tl.constexpr,
                               M_SIZE: tl.constexpr,
                               HIDDEN_DIM: tl.constexpr,
                               SEG0: tl.constexpr, SEG1: tl.constexpr,
                               SEG2: tl.constexpr, SEG3: tl.constexpr,
                               SEG4: tl.constexpr, SEG5: tl.constexpr):
        """r53 2D variant of ``_bilinear_pair_body`` (bench-only).

        Prologue (scalar math, casts, weight computation) is verbatim
        ``_bilinear_pair_body``. Branch layout:
        - ``wf1 == wf0`` (shared embed rows): ``_seg_pair_shared_2d``;
        - ``wf1 == wf0 + 1 and wc1 > wc0`` (adjacent grid columns, no
          edge clamp): ``_seg_pair_diff_2d``;
        - otherwise (e.g. wc1 clamped to wc0 at the right edge): the
          verbatim scalar ``_seg_pair_diff``.
        All three paths produce identical values at identical addresses
        -> bitwise identical to the pair/single-row kernels.
        """
        total_spatial = H * W
        spatial0 = row % total_spatial

        num_blocks_w = W // M_SIZE
        block_idx = spatial0 // (M_SIZE * M_SIZE)
        local_idx = spatial0 % (M_SIZE * M_SIZE)
        br = block_idx // num_blocks_w
        bc = block_idx % num_blocks_w
        lr = local_idx // M_SIZE
        lc = local_idx % M_SIZE
        row_i = br * M_SIZE + lr
        col0 = bc * M_SIZE + lc

        h_frac = row_i.to(tl.float32) * h_scale
        hf = h_frac.to(tl.int32)
        hc = tl.minimum(hf + 1, NUM_GRID - 1)
        dh = h_frac - hf.to(tl.float32)

        w_frac0 = col0.to(tl.float32) * w_scale
        w_frac1 = (col0 + 1).to(tl.float32) * w_scale
        wf0 = w_frac0.to(tl.int32)
        wf1 = w_frac1.to(tl.int32)
        wc0 = tl.minimum(wf0 + 1, NUM_GRID - 1)
        wc1 = tl.minimum(wf1 + 1, NUM_GRID - 1)
        dw0 = w_frac0 - wf0.to(tl.float32)
        dw1 = w_frac1 - wf1.to(tl.float32)

        w11_0 = dh * dw0
        w10_0 = dh - w11_0
        w01_0 = dw0 - w11_0
        w00_0 = 1.0 - dh - w01_0
        w11_1 = dh * dw1
        w10_1 = dh - w11_1
        w01_1 = dw1 - w11_1
        w00_1 = 1.0 - dh - w01_1

        out_dtype = output_ptr.dtype.element_ty
        w00a = w00_0.to(out_dtype)
        w01a = w01_0.to(out_dtype)
        w10a = w10_0.to(out_dtype)
        w11a = w11_0.to(out_dtype)
        w00b = w00_1.to(out_dtype)
        w01b = w01_1.to(out_dtype)
        w10b = w10_1.to(out_dtype)
        w11b = w11_1.to(out_dtype)

        out_off0 = out_row * HIDDEN_DIM
        out_off1 = out_off0 + HIDDEN_DIM

        if wf1 == wf0:
            # Adjacent columns share (hf, hc) x (wf0, wc0): 4 embed rows
            # serve both output rows (2D FMA/store, 1D loads).
            off_a = (hf * NUM_GRID + wf0) * HIDDEN_DIM
            off_b = (hf * NUM_GRID + wc0) * HIDDEN_DIM
            off_c = (hc * NUM_GRID + wf0) * HIDDEN_DIM
            off_d = (hc * NUM_GRID + wc0) * HIDDEN_DIM
            _seg_pair_shared_2d(embed_ptr, output_ptr, off_a, off_b,
                                off_c, off_d, out_off0, HIDDEN_DIM,
                                w00a, w01a, w10a, w11a,
                                w00b, w01b, w10b, w11b,
                                0, SEG0)
            _seg_pair_shared_2d(embed_ptr, output_ptr, off_a, off_b,
                                off_c, off_d, out_off0, HIDDEN_DIM,
                                w00a, w01a, w10a, w11a,
                                w00b, w01b, w10b, w11b,
                                SEG0, SEG1)
            _seg_pair_shared_2d(embed_ptr, output_ptr, off_a, off_b,
                                off_c, off_d, out_off0, HIDDEN_DIM,
                                w00a, w01a, w10a, w11a,
                                w00b, w01b, w10b, w11b,
                                SEG0 + SEG1, SEG2)
            _seg_pair_shared_2d(embed_ptr, output_ptr, off_a, off_b,
                                off_c, off_d, out_off0, HIDDEN_DIM,
                                w00a, w01a, w10a, w11a,
                                w00b, w01b, w10b, w11b,
                                SEG0 + SEG1 + SEG2, SEG3)
            _seg_pair_shared_2d(embed_ptr, output_ptr, off_a, off_b,
                                off_c, off_d, out_off0, HIDDEN_DIM,
                                w00a, w01a, w10a, w11a,
                                w00b, w01b, w10b, w11b,
                                SEG0 + SEG1 + SEG2 + SEG3, SEG4)
            _seg_pair_shared_2d(embed_ptr, output_ptr, off_a, off_b,
                                off_c, off_d, out_off0, HIDDEN_DIM,
                                w00a, w01a, w10a, w11a,
                                w00b, w01b, w10b, w11b,
                                SEG0 + SEG1 + SEG2 + SEG3 + SEG4, SEG5)
        else:
            off_a0 = (hf * NUM_GRID + wf0) * HIDDEN_DIM
            off_b0 = (hf * NUM_GRID + wc0) * HIDDEN_DIM
            off_c0 = (hc * NUM_GRID + wf0) * HIDDEN_DIM
            off_d0 = (hc * NUM_GRID + wc0) * HIDDEN_DIM
            off_a1 = (hf * NUM_GRID + wf1) * HIDDEN_DIM
            off_b1 = (hf * NUM_GRID + wc1) * HIDDEN_DIM
            off_c1 = (hc * NUM_GRID + wf1) * HIDDEN_DIM
            off_d1 = (hc * NUM_GRID + wc1) * HIDDEN_DIM
            if wf1 == wf0 + 1 and wc1 > wc0:
                # Adjacent grid columns without edge clamping: the four
                # (first-column, second-column) embed-row pairs sit
                # HIDDEN_DIM apart -> 2D strided loads + 2D store.
                _seg_pair_diff_2d(embed_ptr, output_ptr,
                                  off_a0, off_b0, off_c0, off_d0,
                                  out_off0, HIDDEN_DIM,
                                  w00a, w01a, w10a, w11a,
                                  w00b, w01b, w10b, w11b,
                                  0, SEG0)
                _seg_pair_diff_2d(embed_ptr, output_ptr,
                                  off_a0, off_b0, off_c0, off_d0,
                                  out_off0, HIDDEN_DIM,
                                  w00a, w01a, w10a, w11a,
                                  w00b, w01b, w10b, w11b,
                                  SEG0, SEG1)
                _seg_pair_diff_2d(embed_ptr, output_ptr,
                                  off_a0, off_b0, off_c0, off_d0,
                                  out_off0, HIDDEN_DIM,
                                  w00a, w01a, w10a, w11a,
                                  w00b, w01b, w10b, w11b,
                                  SEG0 + SEG1, SEG2)
                _seg_pair_diff_2d(embed_ptr, output_ptr,
                                  off_a0, off_b0, off_c0, off_d0,
                                  out_off0, HIDDEN_DIM,
                                  w00a, w01a, w10a, w11a,
                                  w00b, w01b, w10b, w11b,
                                  SEG0 + SEG1 + SEG2, SEG3)
                _seg_pair_diff_2d(embed_ptr, output_ptr,
                                  off_a0, off_b0, off_c0, off_d0,
                                  out_off0, HIDDEN_DIM,
                                  w00a, w01a, w10a, w11a,
                                  w00b, w01b, w10b, w11b,
                                  SEG0 + SEG1 + SEG2 + SEG3, SEG4)
                _seg_pair_diff_2d(embed_ptr, output_ptr,
                                  off_a0, off_b0, off_c0, off_d0,
                                  out_off0, HIDDEN_DIM,
                                  w00a, w01a, w10a, w11a,
                                  w00b, w01b, w10b, w11b,
                                  SEG0 + SEG1 + SEG2 + SEG3 + SEG4, SEG5)
            else:
                _seg_pair_diff(embed_ptr, output_ptr,
                               off_a0, off_b0, off_c0, off_d0,
                               off_a1, off_b1, off_c1, off_d1,
                               out_off0, out_off1,
                               w00a, w01a, w10a, w11a, w00b, w01b, w10b,
                               w11b,
                               0, SEG0)
                _seg_pair_diff(embed_ptr, output_ptr,
                               off_a0, off_b0, off_c0, off_d0,
                               off_a1, off_b1, off_c1, off_d1,
                               out_off0, out_off1,
                               w00a, w01a, w10a, w11a, w00b, w01b, w10b,
                               w11b,
                               SEG0, SEG1)
                _seg_pair_diff(embed_ptr, output_ptr,
                               off_a0, off_b0, off_c0, off_d0,
                               off_a1, off_b1, off_c1, off_d1,
                               out_off0, out_off1,
                               w00a, w01a, w10a, w11a, w00b, w01b, w10b,
                               w11b,
                               SEG0 + SEG1, SEG2)
                _seg_pair_diff(embed_ptr, output_ptr,
                               off_a0, off_b0, off_c0, off_d0,
                               off_a1, off_b1, off_c1, off_d1,
                               out_off0, out_off1,
                               w00a, w01a, w10a, w11a, w00b, w01b, w10b,
                               w11b,
                               SEG0 + SEG1 + SEG2, SEG3)
                _seg_pair_diff(embed_ptr, output_ptr,
                               off_a0, off_b0, off_c0, off_d0,
                               off_a1, off_b1, off_c1, off_d1,
                               out_off0, out_off1,
                               w00a, w01a, w10a, w11a, w00b, w01b, w10b,
                               w11b,
                               SEG0 + SEG1 + SEG2 + SEG3, SEG4)
                _seg_pair_diff(embed_ptr, output_ptr,
                               off_a0, off_b0, off_c0, off_d0,
                               off_a1, off_b1, off_c1, off_d1,
                               out_off0, out_off1,
                               w00a, w01a, w10a, w11a, w00b, w01b, w10b,
                               w11b,
                               SEG0 + SEG1 + SEG2 + SEG3 + SEG4, SEG5)

    @triton.jit
    def _bilinear_pos_embed_kernel_pair2d(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        TOTAL_ROWS,
        CHUNKS: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """r53: ``_bilinear_pos_embed_kernel_pair`` with the 2D pair body.

        Row mapping, grid cap handling and launch contract are verbatim
        the pair kernel; only ``_bilinear_pair_body`` ->
        ``_bilinear_pair_body_2d`` differs (2D tile issue). Bench-only:
        production dispatch is untouched until the x-pair2d rows and the
        rawAB-pair2d bitwise gate prove the case on-machine.
        """
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            pid = tl.program_id(0) + c * grid_size
            row0 = 2 * pid
            if row0 < TOTAL_ROWS:
                _bilinear_pair_body_2d(
                    embed_ptr, output_ptr, row0, row0, H, W, h_scale,
                    w_scale, NUM_GRID, M_SIZE, HIDDEN_DIM,
                    SEG0, SEG1, SEG2, SEG3, SEG4, SEG5)

    @triton.jit
    def _seg_pair_vseg(embed_ptr, output_ptr, off_a0, off_c0, off_a1,
                       off_c1, out_off0, out_off1, w00a, w10a, w00b, w10b,
                       START: tl.constexpr, WIDTH: tl.constexpr):
        """One exact (unmasked) column segment of one w_scale == 1.0 pair.

        Loads ONLY the 4 weight-bearing embed rows (a0, c0, a1, c1) of the
        production diff branch -- the wc-column rows (b0, d0, b1, d1)
        carry w01 == w11 == +0.0 exactly and b0 == a1, d0 == c1 duplicate
        a1/c1, so production pays 8 loads for 4 usable rows. Per-row FMA
        drops the exactly-zero terms: production evaluates
        ((w00*e_a0 + w01*e_b0) + w10*e_c0) + w11*e_d0 with w01 == w11 ==
        +0.0; every +/-0 addition is an identity on non-zero partial
        sums, so ``w00a * e_a0 + w10a * e_c0`` matches bitwise except on
        vanishing +/-0 corner cases (rawAB-vpair gates it on-machine).
        Segment layout identical to ``_seg_row`` (power-of-two WIDTH,
        no predication).
        """
        if WIDTH > 0:
            cols = START + tl.arange(0, WIDTH)
            e_a0 = tl.load(embed_ptr + off_a0 + cols)
            e_c0 = tl.load(embed_ptr + off_c0 + cols)
            e_a1 = tl.load(embed_ptr + off_a1 + cols)
            e_c1 = tl.load(embed_ptr + off_c1 + cols)
            val0 = w00a * e_a0 + w10a * e_c0
            val1 = w00b * e_a1 + w10b * e_c1
            tl.store(output_ptr + out_off0 + cols, val0)
            tl.store(output_ptr + out_off1 + cols, val1)

    @triton.jit
    def _bilinear_pair_vbody(embed_ptr, output_ptr, out_row, row, H, W,
                             h_scale, w_scale, NUM_GRID: tl.constexpr,
                             M_SIZE: tl.constexpr, HIDDEN_DIM: tl.constexpr,
                             SEG0: tl.constexpr, SEG1: tl.constexpr,
                             SEG2: tl.constexpr, SEG3: tl.constexpr,
                             SEG4: tl.constexpr, SEG5: tl.constexpr):
        """r54 vertical-only pair body for w_scale == 1.0 shapes.

        Row mapping, index arithmetic and the weight f32 chain are
        verbatim ``_bilinear_pair_body`` (w_frac/dw computed even though
        dw == +0.0 by the host gate -- zero bitwise risk on the weights).
        Only the embed loads differ: the four wc-column (zero-weight)
        rows of the diff branch are never loaded, and the two stores
        reuse the a/c row pairs. Launch contract identical to the pair
        kernel; the host additionally gates w_scale == 1.0 exactly, which
        implies W == G (for W > 1) so wf1 = col0 + 1 <= G - 1 stays in
        bounds and no clamp row is missed.
        """
        total_spatial = H * W
        spatial0 = row % total_spatial

        num_blocks_w = W // M_SIZE
        block_idx = spatial0 // (M_SIZE * M_SIZE)
        local_idx = spatial0 % (M_SIZE * M_SIZE)
        br = block_idx // num_blocks_w
        bc = block_idx % num_blocks_w
        lr = local_idx // M_SIZE
        lc = local_idx % M_SIZE
        row_i = br * M_SIZE + lr
        col0 = bc * M_SIZE + lc

        h_frac = row_i.to(tl.float32) * h_scale
        hf = h_frac.to(tl.int32)
        hc = tl.minimum(hf + 1, NUM_GRID - 1)
        dh = h_frac - hf.to(tl.float32)

        # Verbatim production weight chain: with w_scale == 1.0, dw0 ==
        # dw1 == +0.0 exactly, so w01 == w11 == +0.0 and the b/d column
        # terms vanish -- vpair skips their loads and FMA terms.
        w_frac0 = col0.to(tl.float32) * w_scale
        w_frac1 = (col0 + 1).to(tl.float32) * w_scale
        wf0 = w_frac0.to(tl.int32)
        wf1 = w_frac1.to(tl.int32)
        dw0 = w_frac0 - wf0.to(tl.float32)
        dw1 = w_frac1 - wf1.to(tl.float32)

        w11_0 = dh * dw0
        w10_0 = dh - w11_0
        w01_0 = dw0 - w11_0
        w00_0 = 1.0 - dh - w01_0
        w11_1 = dh * dw1
        w10_1 = dh - w11_1
        w01_1 = dw1 - w11_1
        w00_1 = 1.0 - dh - w01_1

        out_dtype = output_ptr.dtype.element_ty
        w00a = w00_0.to(out_dtype)
        w10a = w10_0.to(out_dtype)
        w00b = w00_1.to(out_dtype)
        w10b = w10_1.to(out_dtype)

        out_off0 = out_row * HIDDEN_DIM
        out_off1 = out_off0 + HIDDEN_DIM

        off_a0 = (hf * NUM_GRID + wf0) * HIDDEN_DIM
        off_c0 = (hc * NUM_GRID + wf0) * HIDDEN_DIM
        off_a1 = (hf * NUM_GRID + wf1) * HIDDEN_DIM
        off_c1 = (hc * NUM_GRID + wf1) * HIDDEN_DIM
        _seg_pair_vseg(embed_ptr, output_ptr, off_a0, off_c0, off_a1,
                       off_c1, out_off0, out_off1,
                       w00a, w10a, w00b, w10b, 0, SEG0)
        _seg_pair_vseg(embed_ptr, output_ptr, off_a0, off_c0, off_a1,
                       off_c1, out_off0, out_off1,
                       w00a, w10a, w00b, w10b, SEG0, SEG1)
        _seg_pair_vseg(embed_ptr, output_ptr, off_a0, off_c0, off_a1,
                       off_c1, out_off0, out_off1,
                       w00a, w10a, w00b, w10b, SEG0 + SEG1, SEG2)
        _seg_pair_vseg(embed_ptr, output_ptr, off_a0, off_c0, off_a1,
                       off_c1, out_off0, out_off1,
                       w00a, w10a, w00b, w10b,
                       SEG0 + SEG1 + SEG2, SEG3)
        _seg_pair_vseg(embed_ptr, output_ptr, off_a0, off_c0, off_a1,
                       off_c1, out_off0, out_off1,
                       w00a, w10a, w00b, w10b,
                       SEG0 + SEG1 + SEG2 + SEG3, SEG4)
        _seg_pair_vseg(embed_ptr, output_ptr, off_a0, off_c0, off_a1,
                       off_c1, out_off0, out_off1,
                       w00a, w10a, w00b, w10b,
                       SEG0 + SEG1 + SEG2 + SEG3 + SEG4, SEG5)

    @triton.jit
    def _bilinear_pos_embed_kernel_vpair(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        TOTAL_ROWS,
        CHUNKS: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """r54: ``_bilinear_pos_embed_kernel_pair`` with the vpair body.

        Row mapping, grid cap handling and launch contract are verbatim
        the pair kernel; only ``_bilinear_pair_body`` ->
        ``_bilinear_pair_vbody`` differs (zero-weight wc-column loads
        dropped). Bench-only: production dispatch is untouched until the
        x-vpair rows and the rawAB-vpair bitwise gate prove the case
        on-machine.
        """
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            pid = tl.program_id(0) + c * grid_size
            row0 = 2 * pid
            if row0 < TOTAL_ROWS:
                _bilinear_pair_vbody(
                    embed_ptr, output_ptr, row0, row0, H, W, h_scale,
                    w_scale, NUM_GRID, M_SIZE, HIDDEN_DIM,
                    SEG0, SEG1, SEG2, SEG3, SEG4, SEG5)

    @triton.jit
    def _bilinear_pos_embed_kernel_quad(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        TOTAL_ROWS,
        CHUNKS: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """Four-rows-per-program variant: two independent pair slots.

        Each program covers flattened rows (4*pid .. 4*pid+3) as the two
        adjacent pairs (4*pid, 4*pid+1) and (4*pid+2, 4*pid+3). Each pair
        start row (4*pid, 4*pid+2) is even, so every pair independently
        satisfies the ``_bilinear_pair_body`` contract and reuses its body
        verbatim -- per-element arithmetic is bitwise identical to the
        pair/single-row kernels. The quad only halves the program count
        versus the pair kernel and interleaves two independent
        load/FMA streams inside one program, targeting the r8 evidence
        that BF16 sits on the issue pipeline (55.2% MBU vs 124.4% FP32).
        Requires TOTAL_ROWS % 4 == 0 on top of the pair contract
        (one image, m_size even, exact lane split); the host must fall
        back otherwise. Bench-stage kernel: r9 NPU data (BF16 3.230ms /
        55.1% vs pair 3.252ms / 54.7%, FP32 3.127ms / 113.7% vs pair
        2.886ms / 123.2%) shows only a noise-level BF16 gain at a ~9%
        FP32 cost, so the pair kernel stays the production default.
        """
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            pid = tl.program_id(0) + c * grid_size
            row0 = 4 * pid
            if row0 < TOTAL_ROWS:
                _bilinear_pair_body(
                    embed_ptr, output_ptr, row0, row0, H, W, h_scale, w_scale,
                    NUM_GRID, M_SIZE, HIDDEN_DIM,
                    SEG0, SEG1, SEG2, SEG3, SEG4, SEG5)
                _bilinear_pair_body(
                    embed_ptr, output_ptr, row0 + 2, row0 + 2, H, W,
                    h_scale, w_scale,
                    NUM_GRID, M_SIZE, HIDDEN_DIM,
                    SEG0, SEG1, SEG2, SEG3, SEG4, SEG5)

    @triton.jit
    def _bilinear_pos_embed_kernel_mb(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        TOTAL_ROWS,
        CHUNKS: tl.constexpr,
        BODIES: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """r30: BODIES flat pair bodies per program (startup amortized).

        Same pair-row mapping as the pair kernel -- program g covers pair
        slots (g*BODIES .. g*BODIES+BODIES-1), i.e. flattened rows
        (2*(g*BODIES+b), 2*(g*BODIES+b)+1) -- so per-row arithmetic and
        results are bitwise identical to pair/single (rawAB-mb gates it).
        The grid cap applies to PROGRAMS (not slots): programs =
        ceil(rows/2/BODIES), CHUNKS = ceil(programs/cap) static chunks.
        r29 x-row evidence: per-program wall = S(~1.7us) + n*B(~3.4us on
        6-segment tail) -- quad/cap2048 confirmed both levers help tail
        (0.761 -> 0.612/0.654ms); BODIES amortizes S and the FLAT body
        cuts B by overlapping segment-load latencies.
        """
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            g = tl.program_id(0) + c * grid_size
            for b in tl.static_range(0, BODIES):
                row0 = 2 * (g * BODIES + b)
                if row0 < TOTAL_ROWS:
                    _bilinear_pair_body_flat(
                        embed_ptr, output_ptr, row0, row0, H, W,
                        h_scale, w_scale, NUM_GRID, M_SIZE, HIDDEN_DIM,
                        SEG0, SEG1, SEG2, SEG3, SEG4, SEG5)

    @triton.jit
    def _bilinear_pos_embed_kernel_mb_d6(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        TOTAL_ROWS,
        CHUNKS: tl.constexpr,
        BODIES: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """B16 iter: mb structure with the diff6 body (load reuse).

        Structural clone of ``_bilinear_pos_embed_kernel_mb`` with the
        body swapped to ``_bilinear_pair_body_d6`` -- mb3's program
        amortization stacked with d6's address-equal load reuse on the
        diff-branch rows. Bench-only: host gates _FORCE_MB_D6 (int
        BODIES) AND the pair contract AND 0 < w_scale <= 1.0; bitwise
        identical by construction (d6 body is address-equal reuse, zero
        divergence class)."""
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            g = tl.program_id(0) + c * grid_size
            for b in tl.static_range(0, BODIES):
                row0 = 2 * (g * BODIES + b)
                if row0 < TOTAL_ROWS:
                    _bilinear_pair_body_d6(
                        embed_ptr, output_ptr, row0, row0, H, W,
                        h_scale, w_scale, NUM_GRID, M_SIZE, HIDDEN_DIM,
                        SEG0, SEG1, SEG2, SEG3, SEG4, SEG5)

    @triton.jit
    def _seg_pair2_diff6s(embed_ptr, output_ptr,
                          off_eA0, off_eA1, off_eA2, off_eA3, off_eA4, off_eA5,
                          off_eC0, off_eC1, off_eC2,
                          SHARED_H: tl.constexpr,
                          out_off_A0, out_off_A1, out_off_B0, out_off_B1,
                          wA00, wA01, wA10, wA11,
                          wA20, wA21, wA22, wA23,
                          wB00, wB01, wB10, wB11,
                          wB20, wB21, wB22, wB23,
                          START: tl.constexpr, WIDTH: tl.constexpr):
        """B16 round 12: two VERTICALLY adjacent points (4 output rows)
        with cross-point register reuse.

        Rows (4g, 4g+1, 4g+2, 4g+3) are one m_size x m_size block: point
        A = (hA, cols c, c+1), point B = (hA+1, cols c, c+1) -- same
        columns, so dwB0 == dw0 and dwB1 == dw1 (only dh differs: dhB).
        Both points in the diff branch (host gate 0 < w_scale < 1): each
        point has 6 unique rows {hf, hc} x {wf0, wf0+1, wf0+2}. Host gate
        0 < h_scale < 1 forces hfB in {hfA, hfA+1}:
          SHARED_H=1 (hfB == hfA): B rows == A rows exactly, 0 new loads
            (12 -> 6 loads per 4 rows);
          SHARED_H=0 (hfB == hfA+1 == hcA): B hf rows == A hc rows, 3 new
            loads (12 -> 9 per 4 rows).
        Expected (1,60,80): 0.20*6 + 0.80*9 = 8.4 loads/4 rows vs 12 for
        two independent d6 pairs (-30%). FMA order per output row matches
        _seg_pair_diff6 (w00*e_a + w01*e_b + w10*e_c + w11*e_d,
        left-associative) on address-equal values: bitwise-equal class.
        """
        if WIDTH > 0:
            cols = START + tl.arange(0, WIDTH)
            eA0 = tl.load(embed_ptr + off_eA0 + cols)
            eA1 = tl.load(embed_ptr + off_eA1 + cols)
            eA2 = tl.load(embed_ptr + off_eA2 + cols)
            eA3 = tl.load(embed_ptr + off_eA3 + cols)
            eA4 = tl.load(embed_ptr + off_eA4 + cols)
            eA5 = tl.load(embed_ptr + off_eA5 + cols)
            vA0 = wA00 * eA0 + wA01 * eA1 + wA10 * eA2 + wA11 * eA3
            vA1 = wA20 * eA1 + wA21 * eA4 + wA22 * eA3 + wA23 * eA5
            tl.store(output_ptr + out_off_A0 + cols, vA0)
            tl.store(output_ptr + out_off_A1 + cols, vA1)
            if SHARED_H:
                vB0 = wB00 * eA0 + wB01 * eA1 + wB10 * eA2 + wB11 * eA3
                tl.store(output_ptr + out_off_B0 + cols, vB0)
                vB1 = wB20 * eA1 + wB21 * eA4 + wB22 * eA3 + wB23 * eA5
                tl.store(output_ptr + out_off_B1 + cols, vB1)
            else:
                eC0 = tl.load(embed_ptr + off_eC0 + cols)
                eC1 = tl.load(embed_ptr + off_eC1 + cols)
                eC2 = tl.load(embed_ptr + off_eC2 + cols)
                vB0 = wB00 * eA2 + wB01 * eA3 + wB10 * eC0 + wB11 * eC1
                tl.store(output_ptr + out_off_B0 + cols, vB0)
                vB1 = wB20 * eA3 + wB21 * eA5 + wB22 * eC1 + wB23 * eC2
                tl.store(output_ptr + out_off_B1 + cols, vB1)

    @triton.jit
    def _bilinear_pos_embed_kernel_mb_d6s(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        TOTAL_ROWS,
        CHUNKS: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
        SPLITS: tl.constexpr,
    ):
        """B16 round 12: 2 vertically adjacent points per program (4
        rows) with cross-point d6 reuse. Bench-only: _FORCE_MB_D6S;
        host gates t == 1, 0 < w_scale < 1, 0 < h_scale < 1,
        TOTAL_ROWS % 4 == 0.
        """
        grid_size = tl.num_programs(0)
        pid0 = tl.program_id(0)
        for c in tl.static_range(0, CHUNKS):
            idx = pid0 + c * grid_size
            g = idx // SPLITS
            hbase = (idx % SPLITS) * (HIDDEN_DIM // SPLITS)
            row0 = 4 * g
            if row0 < TOTAL_ROWS:
                spatial0 = row0 % (H * W)
                num_blocks_w = W // M_SIZE
                block_idx = spatial0 // (M_SIZE * M_SIZE)
                local_idx = spatial0 % (M_SIZE * M_SIZE)
                br = block_idx // num_blocks_w
                bc = block_idx % num_blocks_w
                lr = local_idx // M_SIZE
                lc = local_idx % M_SIZE
                hA = br * M_SIZE + lr
                hB = hA + 1
                col0 = bc * M_SIZE + lc

                h_fracA = hA.to(tl.float32) * h_scale
                hfA = h_fracA.to(tl.int32)
                hcA = tl.minimum(hfA + 1, NUM_GRID - 1)
                dhA = h_fracA - hfA.to(tl.float32)
                h_fracB = hB.to(tl.float32) * h_scale
                hfB = h_fracB.to(tl.int32)
                hcB = tl.minimum(hfB + 1, NUM_GRID - 1)
                dhB = h_fracB - hfB.to(tl.float32)

                w_frac0 = col0.to(tl.float32) * w_scale
                w_frac1 = (col0 + 1).to(tl.float32) * w_scale
                wfA0 = w_frac0.to(tl.int32)
                wfA1 = w_frac1.to(tl.int32)
                dw0 = w_frac0 - wfA0.to(tl.float32)
                dw1 = w_frac1 - wfA1.to(tl.float32)
                wA00 = (1.0 - dhA - (dw0 - dhA * dw0)).to(
                    output_ptr.dtype.element_ty)
                wA10 = (dhA - dhA * dw0).to(output_ptr.dtype.element_ty)
                wA01 = (dw0 - dhA * dw0).to(output_ptr.dtype.element_ty)
                wA11 = (dhA * dw0).to(output_ptr.dtype.element_ty)
                wA20 = (1.0 - dhA - (dw1 - dhA * dw1)).to(
                    output_ptr.dtype.element_ty)
                wA21 = (dw1 - dhA * dw1).to(output_ptr.dtype.element_ty)
                wA22 = (dhA - dhA * dw1).to(output_ptr.dtype.element_ty)
                wA23 = (dhA * dw1).to(output_ptr.dtype.element_ty)
                wB00 = (1.0 - dhB - (dw0 - dhB * dw0)).to(
                    output_ptr.dtype.element_ty)
                wB10 = (dhB - dhB * dw0).to(output_ptr.dtype.element_ty)
                wB01 = (dw0 - dhB * dw0).to(output_ptr.dtype.element_ty)
                wB11 = (dhB * dw0).to(output_ptr.dtype.element_ty)
                wB20 = (1.0 - dhB - (dw1 - dhB * dw1)).to(
                    output_ptr.dtype.element_ty)
                wB21 = (dw1 - dhB * dw1).to(output_ptr.dtype.element_ty)
                wB22 = (dhB - dhB * dw1).to(output_ptr.dtype.element_ty)
                wB23 = (dhB * dw1).to(output_ptr.dtype.element_ty)

                off_eA0 = hbase + (hfA * NUM_GRID + wfA0) * HIDDEN_DIM
                off_eA1 = hbase + (hfA * NUM_GRID + tl.minimum(wfA0 + 1, NUM_GRID - 1)) * HIDDEN_DIM
                off_eA2 = hbase + (hcA * NUM_GRID + wfA0) * HIDDEN_DIM
                off_eA3 = hbase + (hcA * NUM_GRID + tl.minimum(wfA0 + 1, NUM_GRID - 1)) * HIDDEN_DIM
                off_eA4 = hbase + (hfA * NUM_GRID + tl.minimum(wfA1 + 1, NUM_GRID - 1)) * HIDDEN_DIM
                off_eA5 = hbase + (hcA * NUM_GRID + tl.minimum(wfA1 + 1, NUM_GRID - 1)) * HIDDEN_DIM
                off_eC0 = hbase + (hcB * NUM_GRID + wfA0) * HIDDEN_DIM
                off_eC1 = hbase + (hcB * NUM_GRID + tl.minimum(wfA0 + 1, NUM_GRID - 1)) * HIDDEN_DIM
                off_eC2 = hbase + (hcB * NUM_GRID + tl.minimum(wfA1 + 1, NUM_GRID - 1)) * HIDDEN_DIM

                oA0 = hbase + row0 * HIDDEN_DIM
                oA1 = oA0 + HIDDEN_DIM
                oB0 = oA1 + HIDDEN_DIM
                oB1 = oB0 + HIDDEN_DIM

                shared_h = (hfB == hfA).to(tl.int32)
                if shared_h:
                    _seg_pair2_diff6s(embed_ptr, output_ptr,
                                      off_eA0, off_eA1, off_eA2, off_eA3,
                                      off_eA4, off_eA5,
                                      off_eC0, off_eC1, off_eC2,
                                      1,
                                      oA0, oA1, oB0, oB1,
                                      wA00, wA01, wA10, wA11,
                                      wA20, wA21, wA22, wA23,
                                      wB00, wB01, wB10, wB11,
                                      wB20, wB21, wB22, wB23,
                                      0, SEG0)
                    _seg_pair2_diff6s(embed_ptr, output_ptr,
                                      off_eA0, off_eA1, off_eA2, off_eA3,
                                      off_eA4, off_eA5,
                                      off_eC0, off_eC1, off_eC2,
                                      1,
                                      oA0, oA1, oB0, oB1,
                                      wA00, wA01, wA10, wA11,
                                      wA20, wA21, wA22, wA23,
                                      wB00, wB01, wB10, wB11,
                                      wB20, wB21, wB22, wB23,
                                      SEG0, SEG1)
                    _seg_pair2_diff6s(embed_ptr, output_ptr,
                                      off_eA0, off_eA1, off_eA2, off_eA3,
                                      off_eA4, off_eA5,
                                      off_eC0, off_eC1, off_eC2,
                                      1,
                                      oA0, oA1, oB0, oB1,
                                      wA00, wA01, wA10, wA11,
                                      wA20, wA21, wA22, wA23,
                                      wB00, wB01, wB10, wB11,
                                      wB20, wB21, wB22, wB23,
                                      SEG0 + SEG1, SEG2)
                    _seg_pair2_diff6s(embed_ptr, output_ptr,
                                      off_eA0, off_eA1, off_eA2, off_eA3,
                                      off_eA4, off_eA5,
                                      off_eC0, off_eC1, off_eC2,
                                      1,
                                      oA0, oA1, oB0, oB1,
                                      wA00, wA01, wA10, wA11,
                                      wA20, wA21, wA22, wA23,
                                      wB00, wB01, wB10, wB11,
                                      wB20, wB21, wB22, wB23,
                                      SEG0 + SEG1 + SEG2, SEG3)
                    _seg_pair2_diff6s(embed_ptr, output_ptr,
                                      off_eA0, off_eA1, off_eA2, off_eA3,
                                      off_eA4, off_eA5,
                                      off_eC0, off_eC1, off_eC2,
                                      1,
                                      oA0, oA1, oB0, oB1,
                                      wA00, wA01, wA10, wA11,
                                      wA20, wA21, wA22, wA23,
                                      wB00, wB01, wB10, wB11,
                                      wB20, wB21, wB22, wB23,
                                      SEG0 + SEG1 + SEG2 + SEG3, SEG4)
                    _seg_pair2_diff6s(embed_ptr, output_ptr,
                                      off_eA0, off_eA1, off_eA2, off_eA3,
                                      off_eA4, off_eA5,
                                      off_eC0, off_eC1, off_eC2,
                                      1,
                                      oA0, oA1, oB0, oB1,
                                      wA00, wA01, wA10, wA11,
                                      wA20, wA21, wA22, wA23,
                                      wB00, wB01, wB10, wB11,
                                      wB20, wB21, wB22, wB23,
                                      SEG0 + SEG1 + SEG2 + SEG3 + SEG4, SEG5)
                else:
                    _seg_pair2_diff6s(embed_ptr, output_ptr,
                                      off_eA0, off_eA1, off_eA2, off_eA3,
                                      off_eA4, off_eA5,
                                      off_eC0, off_eC1, off_eC2,
                                      0,
                                      oA0, oA1, oB0, oB1,
                                      wA00, wA01, wA10, wA11,
                                      wA20, wA21, wA22, wA23,
                                      wB00, wB01, wB10, wB11,
                                      wB20, wB21, wB22, wB23,
                                      0, SEG0)
                    _seg_pair2_diff6s(embed_ptr, output_ptr,
                                      off_eA0, off_eA1, off_eA2, off_eA3,
                                      off_eA4, off_eA5,
                                      off_eC0, off_eC1, off_eC2,
                                      0,
                                      oA0, oA1, oB0, oB1,
                                      wA00, wA01, wA10, wA11,
                                      wA20, wA21, wA22, wA23,
                                      wB00, wB01, wB10, wB11,
                                      wB20, wB21, wB22, wB23,
                                      SEG0, SEG1)
                    _seg_pair2_diff6s(embed_ptr, output_ptr,
                                      off_eA0, off_eA1, off_eA2, off_eA3,
                                      off_eA4, off_eA5,
                                      off_eC0, off_eC1, off_eC2,
                                      0,
                                      oA0, oA1, oB0, oB1,
                                      wA00, wA01, wA10, wA11,
                                      wA20, wA21, wA22, wA23,
                                      wB00, wB01, wB10, wB11,
                                      wB20, wB21, wB22, wB23,
                                      SEG0 + SEG1, SEG2)
                    _seg_pair2_diff6s(embed_ptr, output_ptr,
                                      off_eA0, off_eA1, off_eA2, off_eA3,
                                      off_eA4, off_eA5,
                                      off_eC0, off_eC1, off_eC2,
                                      0,
                                      oA0, oA1, oB0, oB1,
                                      wA00, wA01, wA10, wA11,
                                      wA20, wA21, wA22, wA23,
                                      wB00, wB01, wB10, wB11,
                                      wB20, wB21, wB22, wB23,
                                      SEG0 + SEG1 + SEG2, SEG3)
                    _seg_pair2_diff6s(embed_ptr, output_ptr,
                                      off_eA0, off_eA1, off_eA2, off_eA3,
                                      off_eA4, off_eA5,
                                      off_eC0, off_eC1, off_eC2,
                                      0,
                                      oA0, oA1, oB0, oB1,
                                      wA00, wA01, wA10, wA11,
                                      wA20, wA21, wA22, wA23,
                                      wB00, wB01, wB10, wB11,
                                      wB20, wB21, wB22, wB23,
                                      SEG0 + SEG1 + SEG2 + SEG3, SEG4)
                    _seg_pair2_diff6s(embed_ptr, output_ptr,
                                      off_eA0, off_eA1, off_eA2, off_eA3,
                                      off_eA4, off_eA5,
                                      off_eC0, off_eC1, off_eC2,
                                      0,
                                      oA0, oA1, oB0, oB1,
                                      wA00, wA01, wA10, wA11,
                                      wA20, wA21, wA22, wA23,
                                      wB00, wB01, wB10, wB11,
                                      wB20, wB21, wB22, wB23,
                                      SEG0 + SEG1 + SEG2 + SEG3 + SEG4, SEG5)

    @triton.jit
    def _bilinear_pos_embed_kernel_vmb(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        TOTAL_ROWS,
        CHUNKS: tl.constexpr,
        BODIES: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """r56: BODIES vertical-only pair bodies per program (S amortized).

        Structural clone of ``_bilinear_pos_embed_kernel_mb`` (same pair
        row mapping, grid-cap-on-programs contract, row0 < TOTAL_ROWS
        guard) with the body swapped to ``_bilinear_pair_vbody`` -- the
        w_scale == 1.0 vertical-only body whose zero-weight wc-column
        loads are dropped (rawAB-vpair gated). r30 wall model S + n*B:
        r53/r1030 healthy evidence puts mid (768 pair programs, ~19 AIV
        waves) at ~= 19*1.7us ~= 30us == observed t_dev, i.e. mid is
        PROGRAM-STARTUP bound and body compute is nearly free (vpair's
        55% instruction cut changed nothing, x-vpair-tb-mid 0.030 ==
        pair 0.030 healthy). BODIES=2 halves the program count
        (768 -> 384, ~9.6 waves) so S halves; per-element arithmetic is
        the verbatim vbody chain -> bitwise identical to vpair/pair
        modulo the documented vanishing +/-0 corners (rawAB-vmb gates
        it on-machine). Bench-only: host gates _FORCE_VMB (int BODIES)
        AND w_scale == 1.0 exact AND total_out >= _VMB_MIN_ROWS.
        """
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            g = tl.program_id(0) + c * grid_size
            for b in tl.static_range(0, BODIES):
                row0 = 2 * (g * BODIES + b)
                if row0 < TOTAL_ROWS:
                    _bilinear_pair_vbody(
                        embed_ptr, output_ptr, row0, row0, H, W,
                        h_scale, w_scale, NUM_GRID, M_SIZE, HIDDEN_DIM,
                        SEG0, SEG1, SEG2, SEG3, SEG4, SEG5)

    @triton.jit
    def _bilinear_pair_vbody_la(embed_ptr, output_ptr, out_row, row, H, W,
                                h_scale, w_scale, NUM_GRID: tl.constexpr,
                                M_SIZE: tl.constexpr, HIDDEN_DIM: tl.constexpr,
                                SEG0: tl.constexpr, SEG1: tl.constexpr,
                                SEG2: tl.constexpr, SEG3: tl.constexpr,
                                SEG4: tl.constexpr, SEG5: tl.constexpr):
        """r58 load-ahead variant of ``_bilinear_pair_vbody`` (la form).

        Index arithmetic and the weight f32 chain are verbatim
        ``_bilinear_pair_vbody``; ONLY the statement order changes: the
        six sequential ``load -> FMA -> store`` rounds become a one-deep
        software pipeline -- segment k's four embed loads are issued
        BEFORE segment k-1's FMA/store round, so each load latency
        overlaps the previous segment's store round instead of stalling
        the in-order AIV pipeline (the sequential vbody pays ~1 segment
        latency per segment; the mb flat body overlaps ALL loads but its
        hoisted live-vector set is what pays the bf16 cvt/pack penalty --
        mid x-mb4-bf16 0.042 vs f32 0.019, panel E1/E2 dtype-split
        mechanism). la is the middle form: <= 8 live vectors (two
        segments), full overlap only between adjacent segments. Zero
        arithmetic change: same values, same addresses, same store order
        -> bitwise identical to vbody/vpair/pair modulo the documented
        vanishing +/-0 corner class (rawAB-vmb-la gates it on-machine).
        All segment boundaries are constexpr, every branch resolves at
        trace time, no runtime loop (tl.static_range contract preserved
        by construction).
        """
        total_spatial = H * W
        spatial0 = row % total_spatial

        num_blocks_w = W // M_SIZE
        block_idx = spatial0 // (M_SIZE * M_SIZE)
        local_idx = spatial0 % (M_SIZE * M_SIZE)
        br = block_idx // num_blocks_w
        bc = block_idx % num_blocks_w
        lr = local_idx // M_SIZE
        lc = local_idx % M_SIZE
        row_i = br * M_SIZE + lr
        col0 = bc * M_SIZE + lc

        h_frac = row_i.to(tl.float32) * h_scale
        hf = h_frac.to(tl.int32)
        hc = tl.minimum(hf + 1, NUM_GRID - 1)
        dh = h_frac - hf.to(tl.float32)

        w_frac0 = col0.to(tl.float32) * w_scale
        w_frac1 = (col0 + 1).to(tl.float32) * w_scale
        wf0 = w_frac0.to(tl.int32)
        wf1 = w_frac1.to(tl.int32)
        dw0 = w_frac0 - wf0.to(tl.float32)
        dw1 = w_frac1 - wf1.to(tl.float32)

        w11_0 = dh * dw0
        w10_0 = dh - w11_0
        w01_0 = dw0 - w11_0
        w00_0 = 1.0 - dh - w01_0
        w11_1 = dh * dw1
        w10_1 = dh - w11_1
        w01_1 = dw1 - w11_1
        w00_1 = 1.0 - dh - w01_1

        out_dtype = output_ptr.dtype.element_ty
        w00a = w00_0.to(out_dtype)
        w10a = w10_0.to(out_dtype)
        w00b = w00_1.to(out_dtype)
        w10b = w10_1.to(out_dtype)

        out_off0 = out_row * HIDDEN_DIM
        out_off1 = out_off0 + HIDDEN_DIM

        off_a0 = (hf * NUM_GRID + wf0) * HIDDEN_DIM
        off_c0 = (hc * NUM_GRID + wf0) * HIDDEN_DIM
        off_a1 = (hf * NUM_GRID + wf1) * HIDDEN_DIM
        off_c1 = (hc * NUM_GRID + wf1) * HIDDEN_DIM

        # --- one-deep pipeline: load seg k+1, then use seg k -----------
        if SEG0 > 0:
            la_c0 = tl.arange(0, SEG0)
            la0_a0 = tl.load(embed_ptr + off_a0 + la_c0)
            la0_c0 = tl.load(embed_ptr + off_c0 + la_c0)
            la0_a1 = tl.load(embed_ptr + off_a1 + la_c0)
            la0_c1 = tl.load(embed_ptr + off_c1 + la_c0)
        if SEG1 > 0:
            la_c1 = SEG0 + tl.arange(0, SEG1)
            la1_a0 = tl.load(embed_ptr + off_a0 + la_c1)
            la1_c0 = tl.load(embed_ptr + off_c0 + la_c1)
            la1_a1 = tl.load(embed_ptr + off_a1 + la_c1)
            la1_c1 = tl.load(embed_ptr + off_c1 + la_c1)
        if SEG0 > 0:
            tl.store(output_ptr + out_off0 + la_c0,
                     w00a * la0_a0 + w10a * la0_c0)
            tl.store(output_ptr + out_off1 + la_c0,
                     w00b * la0_a1 + w10b * la0_c1)
        if SEG2 > 0:
            la_c2 = SEG0 + SEG1 + tl.arange(0, SEG2)
            la2_a0 = tl.load(embed_ptr + off_a0 + la_c2)
            la2_c0 = tl.load(embed_ptr + off_c0 + la_c2)
            la2_a1 = tl.load(embed_ptr + off_a1 + la_c2)
            la2_c1 = tl.load(embed_ptr + off_c1 + la_c2)
        if SEG1 > 0:
            tl.store(output_ptr + out_off0 + la_c1,
                     w00a * la1_a0 + w10a * la1_c0)
            tl.store(output_ptr + out_off1 + la_c1,
                     w00b * la1_a1 + w10b * la1_c1)
        if SEG3 > 0:
            la_c3 = (SEG0 + SEG1 + SEG2 + tl.arange(0, SEG3))
            la3_a0 = tl.load(embed_ptr + off_a0 + la_c3)
            la3_c0 = tl.load(embed_ptr + off_c0 + la_c3)
            la3_a1 = tl.load(embed_ptr + off_a1 + la_c3)
            la3_c1 = tl.load(embed_ptr + off_c1 + la_c3)
        if SEG2 > 0:
            tl.store(output_ptr + out_off0 + la_c2,
                     w00a * la2_a0 + w10a * la2_c0)
            tl.store(output_ptr + out_off1 + la_c2,
                     w00b * la2_a1 + w10b * la2_c1)
        if SEG4 > 0:
            la_c4 = (SEG0 + SEG1 + SEG2 + SEG3 + tl.arange(0, SEG4))
            la4_a0 = tl.load(embed_ptr + off_a0 + la_c4)
            la4_c0 = tl.load(embed_ptr + off_c0 + la_c4)
            la4_a1 = tl.load(embed_ptr + off_a1 + la_c4)
            la4_c1 = tl.load(embed_ptr + off_c1 + la_c4)
        if SEG3 > 0:
            tl.store(output_ptr + out_off0 + la_c3,
                     w00a * la3_a0 + w10a * la3_c0)
            tl.store(output_ptr + out_off1 + la_c3,
                     w00b * la3_a1 + w10b * la3_c1)
        if SEG5 > 0:
            la_c5 = (SEG0 + SEG1 + SEG2 + SEG3 + SEG4 + tl.arange(0, SEG5))
            la5_a0 = tl.load(embed_ptr + off_a0 + la_c5)
            la5_c0 = tl.load(embed_ptr + off_c0 + la_c5)
            la5_a1 = tl.load(embed_ptr + off_a1 + la_c5)
            la5_c1 = tl.load(embed_ptr + off_c1 + la_c5)
        if SEG4 > 0:
            tl.store(output_ptr + out_off0 + la_c4,
                     w00a * la4_a0 + w10a * la4_c0)
            tl.store(output_ptr + out_off1 + la_c4,
                     w00b * la4_a1 + w10b * la4_c1)
        if SEG5 > 0:
            tl.store(output_ptr + out_off0 + la_c5,
                     w00a * la5_a0 + w10a * la5_c0)
            tl.store(output_ptr + out_off1 + la_c5,
                     w00b * la5_a1 + w10b * la5_c1)

    @triton.jit
    def _bilinear_pos_embed_kernel_vmb_la(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        TOTAL_ROWS,
        CHUNKS: tl.constexpr,
        BODIES: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """r58 bench-only vmb kernel with the load-ahead body.

        Structural clone of ``_bilinear_pos_embed_kernel_vmb`` (same
        pair-slot mapping, grid-cap-on-programs contract, row0 guard)
        with the body swapped to ``_bilinear_pair_vbody_la``. Bench-only:
        host gates _FORCE_VMB_LA (int BODIES) AND w_scale == 1.0 exact
        AND total_out >= _VMB_MIN_ROWS. Panel branch C/D ammo: if the
        sequential vbody's Bv lands in (1.11, 1.45] us (x-vmb2 read-out),
        the la form overlaps segment load latencies to pull Bv toward the
        flat-form ~0.9us issue cost without the flat body's bf16 penalty.
        """
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            g = tl.program_id(0) + c * grid_size
            for b in tl.static_range(0, BODIES):
                row0 = 2 * (g * BODIES + b)
                if row0 < TOTAL_ROWS:
                    _bilinear_pair_vbody_la(
                        embed_ptr, output_ptr, row0, row0, H, W,
                        h_scale, w_scale, NUM_GRID, M_SIZE, HIDDEN_DIM,
                        SEG0, SEG1, SEG2, SEG3, SEG4, SEG5)

    @triton.jit
    def _bilinear_pos_embed_kernel_trep(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        TOTAL_UNIQUE,
        FRAME_STRIDE,
        CHUNKS: tl.constexpr,
        BODIES: tl.constexpr,
        FRAMES: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """r32: BODIES unique spatial rows per program, FRAMES copies each.

        Because h_scale/w_scale are scalar args, the t temporal frames of
        one spatial row are bitwise identical, so the kernel computes each
        unique row once (row mapping g*BODIES + b over [0, H*W), same
        body arithmetic as the single-row kernel) and stores it FRAMES
        times -- embed reads drop by FRAMES while task-book output bytes
        stay 1x. Program count = ceil(H*W/BODIES), far below the pair
        grid on t>1 shapes (tail 7203 pair rows -> 960 trep programs),
        which also amortizes startup. FRAMES is constexpr (host t value)
        because only tl.static_range compiles on triton-ascend 3.6.0.
        Bitwise gate: rawAB-trep vs single-row + native every round.
        """
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            g = tl.program_id(0) + c * grid_size
            for b in tl.static_range(0, BODIES):
                srow = g * BODIES + b
                if srow < TOTAL_UNIQUE:
                    _bilinear_trep_body(
                        embed_ptr, output_ptr, srow, H, W, h_scale, w_scale,
                        NUM_GRID, M_SIZE, HIDDEN_DIM, FRAMES, FRAME_STRIDE,
                        SEG0, SEG1, SEG2, SEG3, SEG4, SEG5)


    @triton.jit
    def _bilinear_pos_embed_kernel_trep_seg(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        TOTAL_UNIQUE,
        FRAME_STRIDE,
        N_SEGS,
        FRAMES: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """r46: segment-parallel trep -- one program per (row, segment).

        The r32 trep program walks BODIES x 6 exact segments serially
        (each segment = 4 embed row loads + FMA + FRAMES stores); the
        r45 probes isolated that chain as the tail wall (x-h512 on the
        same grid: 0.077ms vs 0.404ms). Grid = unique_rows x N_SEGS:
        ``pid % N_SEGS`` picks the segment through a runtime scalar
        branch (all six unmasked constexpr bodies are present in the
        binary, the branch only routes), ``pid // N_SEGS`` is the
        spatial row, BODIES == 1. Row mapping, weights, casts and the
        per-segment load/FMA/store are verbatim ``_bilinear_trep_body``/
        ``_seg_row_trep`` -> bitwise identical. Consecutive pids share a
        row, so the six segment programs read neighboring columns of the
        same four embed rows (L2 friendly). The scalar prologue (~30
        ALU ops) is recomputed per segment program -- cheap next to the
        memory chain it removes.
        """
        pid = tl.program_id(0)
        seg_id = pid % N_SEGS
        srow = pid // N_SEGS
        if srow < TOTAL_UNIQUE:
            total_spatial = H * W
            spatial_idx = srow % total_spatial

            num_blocks_w = W // M_SIZE
            block_idx = spatial_idx // (M_SIZE * M_SIZE)
            local_idx = spatial_idx % (M_SIZE * M_SIZE)
            br = block_idx // num_blocks_w
            bc = block_idx % num_blocks_w
            lr = local_idx // M_SIZE
            lc = local_idx % M_SIZE
            row_i = br * M_SIZE + lr
            col = bc * M_SIZE + lc

            h_frac = row_i.to(tl.float32) * h_scale
            w_frac = col.to(tl.float32) * w_scale

            hf = h_frac.to(tl.int32)
            wf = w_frac.to(tl.int32)
            hc = tl.minimum(hf + 1, NUM_GRID - 1)
            wc = tl.minimum(wf + 1, NUM_GRID - 1)

            dh = h_frac - hf.to(tl.float32)
            dw = w_frac - wf.to(tl.float32)
            w11 = dh * dw
            w10 = dh - w11
            w01 = dw - w11
            w00 = 1.0 - dh - w01

            off00 = (hf * NUM_GRID + wf) * HIDDEN_DIM
            off01 = (hf * NUM_GRID + wc) * HIDDEN_DIM
            off10 = (hc * NUM_GRID + wf) * HIDDEN_DIM
            off11 = (hc * NUM_GRID + wc) * HIDDEN_DIM
            out_off = srow * HIDDEN_DIM

            out_dtype = output_ptr.dtype.element_ty
            w00_c = w00.to(out_dtype)
            w01_c = w01.to(out_dtype)
            w10_c = w10.to(out_dtype)
            w11_c = w11.to(out_dtype)

            if seg_id == 0:
                _seg_row_trep(embed_ptr, output_ptr, off00, off01, off10,
                              off11, out_off, w00_c, w01_c, w10_c, w11_c,
                              0, SEG0, FRAMES, FRAME_STRIDE)
            if seg_id == 1:
                _seg_row_trep(embed_ptr, output_ptr, off00, off01, off10,
                              off11, out_off, w00_c, w01_c, w10_c, w11_c,
                              SEG0, SEG1, FRAMES, FRAME_STRIDE)
            if seg_id == 2:
                _seg_row_trep(embed_ptr, output_ptr, off00, off01, off10,
                              off11, out_off, w00_c, w01_c, w10_c, w11_c,
                              SEG0 + SEG1, SEG2, FRAMES, FRAME_STRIDE)
            if seg_id == 3:
                _seg_row_trep(embed_ptr, output_ptr, off00, off01, off10,
                              off11, out_off, w00_c, w01_c, w10_c, w11_c,
                              SEG0 + SEG1 + SEG2, SEG3,
                              FRAMES, FRAME_STRIDE)
            if seg_id == 4:
                _seg_row_trep(embed_ptr, output_ptr, off00, off01, off10,
                              off11, out_off, w00_c, w01_c, w10_c, w11_c,
                              SEG0 + SEG1 + SEG2 + SEG3, SEG4,
                              FRAMES, FRAME_STRIDE)
            if seg_id == 5:
                _seg_row_trep(embed_ptr, output_ptr, off00, off01, off10,
                              off11, out_off, w00_c, w01_c, w10_c, w11_c,
                              SEG0 + SEG1 + SEG2 + SEG3 + SEG4, SEG5,
                              FRAMES, FRAME_STRIDE)

    @triton.jit
    def _seg_row_trep2d(embed_ptr, output_ptr, off00, off01, off10, off11,
                        out_off, w00, w01, w10, w11,
                        START: tl.constexpr, WIDTH: tl.constexpr,
                        FRAMES: tl.constexpr, FRAME_STRIDE,
                        FP2C: tl.constexpr):
        """``_seg_row_trep`` with frame-chunked 2D stores (r45).

        Loads and the FMA are verbatim ``_seg_row_trep`` (same expression
        order, same casts) -- only the store issue changes: instead of
        FRAMES serial 1D stores per segment, frames are chunked FP2C at a
        time into one 2D ``tl.store`` of shape [FP2C, WIDTH] (row f lands
        at ``out_off + f * FRAME_STRIDE``). r44 isolated the store side as
        the trep wall (fr1 FRAMES=1 0.154ms vs production FRAMES=3
        0.404ms on the same load chain); fewer, wider store instructions
        target that wall. The frame mask folds to constant-true on full
        chunks (``f0`` is a ``tl.static_range`` constant, so the backend
        sees ``f0 + FP2C <= FRAMES`` at trace time). Values and addresses
        are identical to the 1D store loop -> bitwise safe.
        """
        if WIDTH > 0:
            cols = START + tl.arange(0, WIDTH)
            e00 = tl.load(embed_ptr + off00 + cols)
            e01 = tl.load(embed_ptr + off01 + cols)
            e10 = tl.load(embed_ptr + off10 + cols)
            e11 = tl.load(embed_ptr + off11 + cols)
            val = w00 * e00 + w01 * e01 + w10 * e10 + w11 * e11
            v2 = tl.broadcast_to(val[None, :], (FP2C, WIDTH))
            for f0 in tl.static_range(0, FRAMES, FP2C):
                fs = f0 + tl.arange(0, FP2C)
                fmask = fs < FRAMES
                ptrs = (output_ptr + out_off + fs[:, None] * FRAME_STRIDE
                        + cols[None, :])
                tl.store(ptrs, v2, mask=fmask[:, None])

    @triton.jit
    def _bilinear_trep_body_2d(embed_ptr, output_ptr, spatial_row, H, W,
                               h_scale, w_scale, NUM_GRID: tl.constexpr,
                               M_SIZE: tl.constexpr,
                               HIDDEN_DIM: tl.constexpr,
                               FRAMES: tl.constexpr, FRAME_STRIDE,
                               FP2C: tl.constexpr,
                               SEG0: tl.constexpr, SEG1: tl.constexpr,
                               SEG2: tl.constexpr, SEG3: tl.constexpr,
                               SEG4: tl.constexpr, SEG5: tl.constexpr):
        """``_bilinear_trep_body`` with the r45 2D frame-chunked stores.

        Scalar math, address arithmetic and cast points are verbatim
        ``_bilinear_trep_body``; the six exact segments call
        ``_seg_row_trep2d`` instead of ``_seg_row_trep``. Same values to
        the same addresses -> output bitwise identical to the production
        trep kernel (rawAB-trep2d gates it every round).
        """
        total_spatial = H * W
        spatial_idx = spatial_row % total_spatial

        num_blocks_w = W // M_SIZE
        block_idx = spatial_idx // (M_SIZE * M_SIZE)
        local_idx = spatial_idx % (M_SIZE * M_SIZE)
        br = block_idx // num_blocks_w
        bc = block_idx % num_blocks_w
        lr = local_idx // M_SIZE
        lc = local_idx % M_SIZE
        row_i = br * M_SIZE + lr
        col = bc * M_SIZE + lc

        h_frac = row_i.to(tl.float32) * h_scale
        w_frac = col.to(tl.float32) * w_scale

        hf = h_frac.to(tl.int32)
        wf = w_frac.to(tl.int32)
        hc = tl.minimum(hf + 1, NUM_GRID - 1)
        wc = tl.minimum(wf + 1, NUM_GRID - 1)

        dh = h_frac - hf.to(tl.float32)
        dw = w_frac - wf.to(tl.float32)
        w11 = dh * dw
        w10 = dh - w11
        w01 = dw - w11
        w00 = 1.0 - dh - w01

        off00 = (hf * NUM_GRID + wf) * HIDDEN_DIM
        off01 = (hf * NUM_GRID + wc) * HIDDEN_DIM
        off10 = (hc * NUM_GRID + wf) * HIDDEN_DIM
        off11 = (hc * NUM_GRID + wc) * HIDDEN_DIM
        out_off = spatial_row * HIDDEN_DIM

        out_dtype = output_ptr.dtype.element_ty
        w00_c = w00.to(out_dtype)
        w01_c = w01.to(out_dtype)
        w10_c = w10.to(out_dtype)
        w11_c = w11.to(out_dtype)

        _seg_row_trep2d(embed_ptr, output_ptr, off00, off01, off10, off11,
                        out_off, w00_c, w01_c, w10_c, w11_c, 0, SEG0,
                        FRAMES, FRAME_STRIDE, FP2C)
        _seg_row_trep2d(embed_ptr, output_ptr, off00, off01, off10, off11,
                        out_off, w00_c, w01_c, w10_c, w11_c, SEG0, SEG1,
                        FRAMES, FRAME_STRIDE, FP2C)
        _seg_row_trep2d(embed_ptr, output_ptr, off00, off01, off10, off11,
                        out_off, w00_c, w01_c, w10_c, w11_c,
                        SEG0 + SEG1, SEG2, FRAMES, FRAME_STRIDE, FP2C)
        _seg_row_trep2d(embed_ptr, output_ptr, off00, off01, off10, off11,
                        out_off, w00_c, w01_c, w10_c, w11_c,
                        SEG0 + SEG1 + SEG2, SEG3, FRAMES, FRAME_STRIDE, FP2C)
        _seg_row_trep2d(embed_ptr, output_ptr, off00, off01, off10, off11,
                        out_off, w00_c, w01_c, w10_c, w11_c,
                        SEG0 + SEG1 + SEG2 + SEG3, SEG4,
                        FRAMES, FRAME_STRIDE, FP2C)
        _seg_row_trep2d(embed_ptr, output_ptr, off00, off01, off10, off11,
                        out_off, w00_c, w01_c, w10_c, w11_c,
                        SEG0 + SEG1 + SEG2 + SEG3 + SEG4, SEG5,
                        FRAMES, FRAME_STRIDE, FP2C)

    @triton.jit
    def _bilinear_pos_embed_kernel_trep2d(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        TOTAL_UNIQUE,
        FRAME_STRIDE,
        CHUNKS: tl.constexpr,
        BODIES: tl.constexpr,
        FRAMES: tl.constexpr,
        FP2C: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """r45: ``_bilinear_pos_embed_kernel_trep`` with 2D frame-chunked
        stores. Row mapping (g*BODIES + b over [0, H*W)), bodies formula,
        chunk loop and segment layout are verbatim the production trep
        kernel; only ``_bilinear_trep_body`` -> ``_bilinear_trep_body_2d``
        differs (store issue). Bench-only diagnostic: production dispatch
        is untouched until the x-trep2d rows prove the store wall moves.
        """
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            g = tl.program_id(0) + c * grid_size
            for b in tl.static_range(0, BODIES):
                srow = g * BODIES + b
                if srow < TOTAL_UNIQUE:
                    _bilinear_trep_body_2d(
                        embed_ptr, output_ptr, srow, H, W, h_scale, w_scale,
                        NUM_GRID, M_SIZE, HIDDEN_DIM, FRAMES, FRAME_STRIDE,
                        FP2C, SEG0, SEG1, SEG2, SEG3, SEG4, SEG5)

    @triton.jit
    def _seg_row_trep2d_m1(embed_ptr, output_ptr, off00, off01, off10,
                           off11, out_off, w00, w01, w10, w11,
                           HIDDEN: tl.constexpr, BLOCK_D: tl.constexpr,
                           FRAMES: tl.constexpr, FRAME_STRIDE,
                           FP2C: tl.constexpr):
        """r50: masked single-segment ``_seg_row_trep2d`` (one serial
        round instead of six).

        The exact pow2 segmentation (h=1000 -> 512+256+128+64+32+8) makes
        every program walk six serial load->FMA->store rounds; r45/r49
        isolated that chain as the tail latency wall (x-h512 single-segment
        probe 0.073ms vs 0.216ms for the six-segment h=1000 trep2d row in
        r49 -- 1.5x per-byte). This variant issues ONE [BLOCK_D] round with
        ``cols < HIDDEN`` masks on the embed loads (other=0.0) and on the
        frame-chunked 2D store. In-range lanes run the identical FMA on
        the identical addresses (masked lanes load 0.0 and are never
        stored; the store column mask is mandatory because a [1024] block
        overhangs the 1000-wide row into the next row's linear range)
        -> output bitwise identical to trep2d/trep; the rawAB-trep2dm1
        gate verifies it every round.
        """
        cols = tl.arange(0, BLOCK_D)
        cmask = cols < HIDDEN
        e00 = tl.load(embed_ptr + off00 + cols, mask=cmask, other=0.0)
        e01 = tl.load(embed_ptr + off01 + cols, mask=cmask, other=0.0)
        e10 = tl.load(embed_ptr + off10 + cols, mask=cmask, other=0.0)
        e11 = tl.load(embed_ptr + off11 + cols, mask=cmask, other=0.0)
        val = w00 * e00 + w01 * e01 + w10 * e10 + w11 * e11
        v2 = tl.broadcast_to(val[None, :], (FP2C, BLOCK_D))
        for f0 in tl.static_range(0, FRAMES, FP2C):
            fs = f0 + tl.arange(0, FP2C)
            fmask = fs < FRAMES
            ptrs = (output_ptr + out_off + fs[:, None] * FRAME_STRIDE
                    + cols[None, :])
            tl.store(ptrs, v2, mask=fmask[:, None] & cmask[None, :])

    @triton.jit
    def _bilinear_trep_body_2d_m1(embed_ptr, output_ptr, spatial_row, H, W,
                                  h_scale, w_scale, NUM_GRID: tl.constexpr,
                                  M_SIZE: tl.constexpr,
                                  HIDDEN_DIM: tl.constexpr,
                                  FRAMES: tl.constexpr, FRAME_STRIDE,
                                  FP2C: tl.constexpr, BLOCK_D: tl.constexpr):
        """``_bilinear_trep_body_2d`` with the r50 single masked segment.

        Scalar math, address arithmetic and cast points are verbatim
        ``_bilinear_trep_body_2d``; the six exact segments are replaced by
        ONE ``_seg_row_trep2d_m1`` call over [BLOCK_D]. Same values to the
        same addresses -> output bitwise identical to the production
        trep/trep2d kernels (rawAB-trep2dm1 gates it).
        """
        total_spatial = H * W
        spatial_idx = spatial_row % total_spatial

        num_blocks_w = W // M_SIZE
        block_idx = spatial_idx // (M_SIZE * M_SIZE)
        local_idx = spatial_idx % (M_SIZE * M_SIZE)
        br = block_idx // num_blocks_w
        bc = block_idx % num_blocks_w
        lr = local_idx // M_SIZE
        lc = local_idx % M_SIZE
        row_i = br * M_SIZE + lr
        col = bc * M_SIZE + lc

        h_frac = row_i.to(tl.float32) * h_scale
        w_frac = col.to(tl.float32) * w_scale

        hf = h_frac.to(tl.int32)
        wf = w_frac.to(tl.int32)
        hc = tl.minimum(hf + 1, NUM_GRID - 1)
        wc = tl.minimum(wf + 1, NUM_GRID - 1)

        dh = h_frac - hf.to(tl.float32)
        dw = w_frac - wf.to(tl.float32)
        w11 = dh * dw
        w10 = dh - w11
        w01 = dw - w11
        w00 = 1.0 - dh - w01

        off00 = (hf * NUM_GRID + wf) * HIDDEN_DIM
        off01 = (hf * NUM_GRID + wc) * HIDDEN_DIM
        off10 = (hc * NUM_GRID + wf) * HIDDEN_DIM
        off11 = (hc * NUM_GRID + wc) * HIDDEN_DIM
        out_off = spatial_row * HIDDEN_DIM

        out_dtype = output_ptr.dtype.element_ty
        w00_c = w00.to(out_dtype)
        w01_c = w01.to(out_dtype)
        w10_c = w10.to(out_dtype)
        w11_c = w11.to(out_dtype)

        _seg_row_trep2d_m1(embed_ptr, output_ptr, off00, off01, off10,
                           off11, out_off, w00_c, w01_c, w10_c, w11_c,
                           HIDDEN_DIM, BLOCK_D, FRAMES, FRAME_STRIDE, FP2C)

    @triton.jit
    def _bilinear_pos_embed_kernel_trep2d_m1(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        TOTAL_UNIQUE,
        FRAME_STRIDE,
        CHUNKS: tl.constexpr,
        BODIES: tl.constexpr,
        FRAMES: tl.constexpr,
        FP2C: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """r50 bench-only: ``_bilinear_pos_embed_kernel_trep2d`` with the
        six exact segments collapsed into ONE masked segment (see
        ``_seg_row_trep2d_m1``). Row mapping, bodies formula, chunk loop
        and launch contract verbatim trep2d; BLOCK_D = next_pow2(HIDDEN).
        Production dispatch is untouched until the x-trep2dm1 rows prove
        the serial-chain wall moves (pre-registered: tail-f32 probe
        <=0.191ms -> promote the tail route next round).
        """
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            g = tl.program_id(0) + c * grid_size
            for b in tl.static_range(0, BODIES):
                srow = g * BODIES + b
                if srow < TOTAL_UNIQUE:
                    _bilinear_trep_body_2d_m1(
                        embed_ptr, output_ptr, srow, H, W, h_scale, w_scale,
                        NUM_GRID, M_SIZE, HIDDEN_DIM, FRAMES, FRAME_STRIDE,
                        FP2C, BLOCK_D)

    @triton.jit
    def _bilinear_trep_body_flat(embed_ptr, output_ptr, spatial_row, H, W,
                                 h_scale, w_scale, NUM_GRID: tl.constexpr,
                                 M_SIZE: tl.constexpr,
                                 HIDDEN_DIM: tl.constexpr,
                                 FRAMES: tl.constexpr, FRAME_STRIDE,
                                 SEG0: tl.constexpr, SEG1: tl.constexpr,
                                 SEG2: tl.constexpr, SEG3: tl.constexpr,
                                 SEG4: tl.constexpr, SEG5: tl.constexpr):
        """Flat multi-segment variant of ``_bilinear_trep_body`` (r38).

        r37 (job 1011) closed the machine-state diagnosis: the degraded
        small-kernel rows are a machine-level slowdown (EVPROBE warm ==
        cold, x-inwarm no recovery, flat per-iter durations), so the
        remaining lever on the latency-bound tail rows is kernel
        structure. The r32 trep body issues six sequential
        load->FMA->store rounds (each stalls the in-order AIV pipeline on
        its own memory latency -- the r29 S+B model B ~= 3.4us/segment on
        the 6-segment tail shape). This variant applies the r30 pair-flat
        pattern: each WAVE issues ALL its segment loads back-to-back
        first (four embed rows per segment), then does the FMAs and the
        FRAMES stores. Wave = 3 segments keeps the live register set
        bounded (4 vectors x (512+256+128) lanes fp32 ~= 14KB, inside the
        256KB UB). Scalar math, FMA order and cast points are verbatim
        ``_bilinear_trep_body`` -> bitwise identical output (same argument
        as the r30 pair-flat; rawAB-trepflat gates it every round).
        """
        total_spatial = H * W
        spatial_idx = spatial_row % total_spatial

        num_blocks_w = W // M_SIZE
        block_idx = spatial_idx // (M_SIZE * M_SIZE)
        local_idx = spatial_idx % (M_SIZE * M_SIZE)
        br = block_idx // num_blocks_w
        bc = block_idx % num_blocks_w
        lr = local_idx // M_SIZE
        lc = local_idx % M_SIZE
        row_i = br * M_SIZE + lr
        col = bc * M_SIZE + lc

        h_frac = row_i.to(tl.float32) * h_scale
        w_frac = col.to(tl.float32) * w_scale

        hf = h_frac.to(tl.int32)
        wf = w_frac.to(tl.int32)
        hc = tl.minimum(hf + 1, NUM_GRID - 1)
        wc = tl.minimum(wf + 1, NUM_GRID - 1)

        dh = h_frac - hf.to(tl.float32)
        dw = w_frac - wf.to(tl.float32)
        w11 = dh * dw
        w10 = dh - w11
        w01 = dw - w11
        w00 = 1.0 - dh - w01

        off00 = (hf * NUM_GRID + wf) * HIDDEN_DIM
        off01 = (hf * NUM_GRID + wc) * HIDDEN_DIM
        off10 = (hc * NUM_GRID + wf) * HIDDEN_DIM
        off11 = (hc * NUM_GRID + wc) * HIDDEN_DIM
        out_off = spatial_row * HIDDEN_DIM

        out_dtype = output_ptr.dtype.element_ty
        w00_c = w00.to(out_dtype)
        w01_c = w01.to(out_dtype)
        w10_c = w10.to(out_dtype)
        w11_c = w11.to(out_dtype)

        # wave 1: segs 0-2, all loads issued before any FMA
        if SEG0 > 0:
            s0c = tl.arange(0, SEG0)
            s0a = tl.load(embed_ptr + off00 + s0c)
            s0b = tl.load(embed_ptr + off01 + s0c)
            s0cv = tl.load(embed_ptr + off10 + s0c)
            s0d = tl.load(embed_ptr + off11 + s0c)
        if SEG1 > 0:
            s1c = SEG0 + tl.arange(0, SEG1)
            s1a = tl.load(embed_ptr + off00 + s1c)
            s1b = tl.load(embed_ptr + off01 + s1c)
            s1cv = tl.load(embed_ptr + off10 + s1c)
            s1d = tl.load(embed_ptr + off11 + s1c)
        if SEG2 > 0:
            s2c = SEG0 + SEG1 + tl.arange(0, SEG2)
            s2a = tl.load(embed_ptr + off00 + s2c)
            s2b = tl.load(embed_ptr + off01 + s2c)
            s2cv = tl.load(embed_ptr + off10 + s2c)
            s2d = tl.load(embed_ptr + off11 + s2c)
        if SEG0 > 0:
            val0 = w00_c * s0a + w01_c * s0b + w10_c * s0cv + w11_c * s0d
            for f in tl.static_range(0, FRAMES):
                tl.store(output_ptr + out_off + f * FRAME_STRIDE + s0c, val0)
        if SEG1 > 0:
            val1 = w00_c * s1a + w01_c * s1b + w10_c * s1cv + w11_c * s1d
            for f in tl.static_range(0, FRAMES):
                tl.store(output_ptr + out_off + f * FRAME_STRIDE + s1c, val1)
        if SEG2 > 0:
            val2 = w00_c * s2a + w01_c * s2b + w10_c * s2cv + w11_c * s2d
            for f in tl.static_range(0, FRAMES):
                tl.store(output_ptr + out_off + f * FRAME_STRIDE + s2c, val2)
        # wave 2: segs 3-5
        if SEG3 > 0:
            s3c = SEG0 + SEG1 + SEG2 + tl.arange(0, SEG3)
            s3a = tl.load(embed_ptr + off00 + s3c)
            s3b = tl.load(embed_ptr + off01 + s3c)
            s3cv = tl.load(embed_ptr + off10 + s3c)
            s3d = tl.load(embed_ptr + off11 + s3c)
        if SEG4 > 0:
            s4c = SEG0 + SEG1 + SEG2 + SEG3 + tl.arange(0, SEG4)
            s4a = tl.load(embed_ptr + off00 + s4c)
            s4b = tl.load(embed_ptr + off01 + s4c)
            s4cv = tl.load(embed_ptr + off10 + s4c)
            s4d = tl.load(embed_ptr + off11 + s4c)
        if SEG5 > 0:
            s5c = (SEG0 + SEG1 + SEG2 + SEG3 + SEG4
                   + tl.arange(0, SEG5))
            s5a = tl.load(embed_ptr + off00 + s5c)
            s5b = tl.load(embed_ptr + off01 + s5c)
            s5cv = tl.load(embed_ptr + off10 + s5c)
            s5d = tl.load(embed_ptr + off11 + s5c)
        if SEG3 > 0:
            val3 = w00_c * s3a + w01_c * s3b + w10_c * s3cv + w11_c * s3d
            for f in tl.static_range(0, FRAMES):
                tl.store(output_ptr + out_off + f * FRAME_STRIDE + s3c, val3)
        if SEG4 > 0:
            val4 = w00_c * s4a + w01_c * s4b + w10_c * s4cv + w11_c * s4d
            for f in tl.static_range(0, FRAMES):
                tl.store(output_ptr + out_off + f * FRAME_STRIDE + s4c, val4)
        if SEG5 > 0:
            val5 = w00_c * s5a + w01_c * s5b + w10_c * s5cv + w11_c * s5d
            for f in tl.static_range(0, FRAMES):
                tl.store(output_ptr + out_off + f * FRAME_STRIDE + s5c, val5)

    @triton.jit
    def _bilinear_pos_embed_kernel_trep_flat(
        embed_ptr,
        output_ptr,
        H,
        W,
        h_scale,
        w_scale,
        TOTAL_UNIQUE,
        FRAME_STRIDE,
        CHUNKS: tl.constexpr,
        BODIES: tl.constexpr,
        FRAMES: tl.constexpr,
        NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        SEG0: tl.constexpr,
        SEG1: tl.constexpr,
        SEG2: tl.constexpr,
        SEG3: tl.constexpr,
        SEG4: tl.constexpr,
        SEG5: tl.constexpr,
    ):
        """r38: launch wrapper for the flat trep body.

        Same grid mapping and launch contract as
        ``_bilinear_pos_embed_kernel_trep`` (BODIES unique spatial rows
        per program, row mapping g*BODIES + b, FRAMES constexpr stores);
        only the body is the flat-load variant. Bitwise gate:
        rawAB-trepflat every round.
        """
        grid_size = tl.num_programs(0)
        for c in tl.static_range(0, CHUNKS):
            g = tl.program_id(0) + c * grid_size
            for b in tl.static_range(0, BODIES):
                srow = g * BODIES + b
                if srow < TOTAL_UNIQUE:
                    _bilinear_trep_body_flat(
                        embed_ptr, output_ptr, srow, H, W, h_scale, w_scale,
                        NUM_GRID, M_SIZE, HIDDEN_DIM, FRAMES, FRAME_STRIDE,
                        SEG0, SEG1, SEG2, SEG3, SEG4, SEG5)

    # ---- r42: corner kernel (logical-row w-group load sharing) --------------
    #
    # r24/r38 pre-approved tail direction. One program per LOGICAL output
    # row (fixed hf/hc/dh); the kernel walks the distinct w-source groups
    # in-kernel: consecutive columns with the same floor(col*w_scale) form
    # one contiguous interval (the floor is monotone in col), so each
    # (group, segment) loads its 4 embed rows ONCE and the FMA/store
    # re-issues per column from registers. Every processed column
    # recomputes its own weights verbatim from its own index and the chain
    # advances >= 1 column per slot, so coverage and values are bitwise
    # safe for any MAX_RUN/GROUPS the host passes (rawAB-corner gates it).

    @triton.jit
    def _corner_seg_store(output_ptr, out_off, e00, e01, e10, e11,
                          w00, w01, w10, w11,
                          START: tl.constexpr, WIDTH: tl.constexpr,
                          FRAMES: tl.constexpr, FRAME_STRIDE):
        """FMA + FRAMES stores for one column/segment, tiles pre-loaded.

        ``val`` and the frame-store loop are verbatim ``_seg_row_trep``
        (weights arrive pre-cast to the output dtype); the embed tiles are
        loaded once per (group, segment) by the caller and shared across
        every column of the group. WIDTH == 0 compiles to nothing.
        """
        if WIDTH > 0:
            cols = START + tl.arange(0, WIDTH)
            val = w00 * e00 + w01 * e01 + w10 * e10 + w11 * e11
            for f in tl.static_range(0, FRAMES):
                tl.store(output_ptr + out_off + f * FRAME_STRIDE + cols, val)

    @triton.jit
    def _bilinear_pos_embed_kernel_corner(
        embed_ptr, output_ptr, H, W, h_scale, w_scale, FRAME_STRIDE,
        CHUNKS: tl.constexpr, ROWS_PER_PROG: tl.constexpr,
        FRAMES: tl.constexpr, NUM_GRID: tl.constexpr,
        M_SIZE: tl.constexpr, HIDDEN_DIM: tl.constexpr,
        MAX_RUN: tl.constexpr, GROUPS: tl.constexpr,
        SEG0: tl.constexpr, SEG1: tl.constexpr, SEG2: tl.constexpr,
        SEG3: tl.constexpr, SEG4: tl.constexpr, SEG5: tl.constexpr,
    ):
        """r42 bench kernel: one program per logical output row, in-kernel
        w-group load sharing (the "corner" direction, r24 floor analysis).

        Program = logical row ``r`` in [0, H). Columns sharing
        floor(col*w_scale) form one contiguous interval, loaded once per
        segment; every processed column recomputes its own weights verbatim
        from its own index, so per-element results are bitwise identical to
        pair/single/trep. Output addressing uses the exact inverse of the
        production spatial-merge block remap:
        spatial(r, c) = (r//M*nbw + c//M)*M^2 + (r%M)*M + c%M.
        """
        grid_size = tl.num_programs(0)
        for cc in tl.static_range(0, CHUNKS):
            g = tl.program_id(0) + cc * grid_size
            for b in tl.static_range(0, ROWS_PER_PROG):
                r = g * ROWS_PER_PROG + b
                if r < H:
                    h_frac = r.to(tl.float32) * h_scale
                    hf = h_frac.to(tl.int32)
                    hc = tl.minimum(hf + 1, NUM_GRID - 1)
                    dh = h_frac - hf.to(tl.float32)
                    out_dtype = output_ptr.dtype.element_ty
                    nbw = W // M_SIZE
                    br = r // M_SIZE
                    lr = r % M_SIZE
                    c_lo = g * 0  # int32 scalar zero
                    for _gi in tl.static_range(0, GROUPS):
                        if c_lo < W:
                            w_frac_lo = c_lo.to(tl.float32) * w_scale
                            wf_g = w_frac_lo.to(tl.int32)
                            wc_g = tl.minimum(wf_g + 1, NUM_GRID - 1)
                            off00 = (hf * NUM_GRID + wf_g) * HIDDEN_DIM
                            off01 = (hf * NUM_GRID + wc_g) * HIDDEN_DIM
                            off10 = (hc * NUM_GRID + wf_g) * HIDDEN_DIM
                            off11 = (hc * NUM_GRID + wc_g) * HIDDEN_DIM
                            # equal-floor run length (floor monotone in col)
                            cnt = 1
                            for j in tl.static_range(1, MAX_RUN):
                                c_j = c_lo + j
                                ok_j = (c_j < W) & (
                                    (c_j.to(tl.float32) * w_scale).to(tl.int32)
                                    == wf_g)
                                cnt = tl.where(ok_j, cnt + 1, cnt)
                            c_hi = c_lo + cnt
                            if SEG0 > 0:
                                e00 = tl.load(embed_ptr + off00
                                              + tl.arange(0, SEG0))
                                e01 = tl.load(embed_ptr + off01
                                              + tl.arange(0, SEG0))
                                e10 = tl.load(embed_ptr + off10
                                              + tl.arange(0, SEG0))
                                e11 = tl.load(embed_ptr + off11
                                              + tl.arange(0, SEG0))
                                for j in tl.static_range(0, MAX_RUN):
                                    c_j = c_lo + j
                                    w_frac_j = c_j.to(tl.float32) * w_scale
                                    in_g = (c_j < c_hi) & (
                                        w_frac_j.to(tl.int32) == wf_g)
                                    if in_g:
                                        bc = c_j // M_SIZE
                                        lc = c_j % M_SIZE
                                        spatial = (br * nbw + bc) * (
                                            M_SIZE * M_SIZE) + lr * M_SIZE + lc
                                        out_off = spatial * HIDDEN_DIM
                                        dw = w_frac_j - w_frac_j.to(
                                            tl.int32).to(tl.float32)
                                        w11 = dh * dw
                                        w10 = dh - w11
                                        w01 = dw - w11
                                        w00 = 1.0 - dh - w01
                                        _corner_seg_store(
                                            output_ptr, out_off, e00, e01,
                                            e10, e11, w00.to(out_dtype),
                                            w01.to(out_dtype),
                                            w10.to(out_dtype),
                                            w11.to(out_dtype),
                                            0, SEG0, FRAMES,
                                            FRAME_STRIDE)
                            if SEG1 > 0:
                                e00 = tl.load(embed_ptr + off00
                                              + SEG0
                                              + tl.arange(0, SEG1))
                                e01 = tl.load(embed_ptr + off01
                                              + SEG0
                                              + tl.arange(0, SEG1))
                                e10 = tl.load(embed_ptr + off10
                                              + SEG0
                                              + tl.arange(0, SEG1))
                                e11 = tl.load(embed_ptr + off11
                                              + SEG0
                                              + tl.arange(0, SEG1))
                                for j in tl.static_range(0, MAX_RUN):
                                    c_j = c_lo + j
                                    w_frac_j = c_j.to(tl.float32) * w_scale
                                    in_g = (c_j < c_hi) & (
                                        w_frac_j.to(tl.int32) == wf_g)
                                    if in_g:
                                        bc = c_j // M_SIZE
                                        lc = c_j % M_SIZE
                                        spatial = (br * nbw + bc) * (
                                            M_SIZE * M_SIZE) + lr * M_SIZE + lc
                                        out_off = spatial * HIDDEN_DIM
                                        dw = w_frac_j - w_frac_j.to(
                                            tl.int32).to(tl.float32)
                                        w11 = dh * dw
                                        w10 = dh - w11
                                        w01 = dw - w11
                                        w00 = 1.0 - dh - w01
                                        _corner_seg_store(
                                            output_ptr, out_off, e00, e01,
                                            e10, e11, w00.to(out_dtype),
                                            w01.to(out_dtype),
                                            w10.to(out_dtype),
                                            w11.to(out_dtype),
                                            SEG0, SEG1, FRAMES,
                                            FRAME_STRIDE)
                            if SEG2 > 0:
                                e00 = tl.load(embed_ptr + off00
                                              + SEG0 + SEG1
                                              + tl.arange(0, SEG2))
                                e01 = tl.load(embed_ptr + off01
                                              + SEG0 + SEG1
                                              + tl.arange(0, SEG2))
                                e10 = tl.load(embed_ptr + off10
                                              + SEG0 + SEG1
                                              + tl.arange(0, SEG2))
                                e11 = tl.load(embed_ptr + off11
                                              + SEG0 + SEG1
                                              + tl.arange(0, SEG2))
                                for j in tl.static_range(0, MAX_RUN):
                                    c_j = c_lo + j
                                    w_frac_j = c_j.to(tl.float32) * w_scale
                                    in_g = (c_j < c_hi) & (
                                        w_frac_j.to(tl.int32) == wf_g)
                                    if in_g:
                                        bc = c_j // M_SIZE
                                        lc = c_j % M_SIZE
                                        spatial = (br * nbw + bc) * (
                                            M_SIZE * M_SIZE) + lr * M_SIZE + lc
                                        out_off = spatial * HIDDEN_DIM
                                        dw = w_frac_j - w_frac_j.to(
                                            tl.int32).to(tl.float32)
                                        w11 = dh * dw
                                        w10 = dh - w11
                                        w01 = dw - w11
                                        w00 = 1.0 - dh - w01
                                        _corner_seg_store(
                                            output_ptr, out_off, e00, e01,
                                            e10, e11, w00.to(out_dtype),
                                            w01.to(out_dtype),
                                            w10.to(out_dtype),
                                            w11.to(out_dtype),
                                            SEG0 + SEG1, SEG2, FRAMES,
                                            FRAME_STRIDE)
                            if SEG3 > 0:
                                e00 = tl.load(embed_ptr + off00
                                              + SEG0 + SEG1 + SEG2
                                              + tl.arange(0, SEG3))
                                e01 = tl.load(embed_ptr + off01
                                              + SEG0 + SEG1 + SEG2
                                              + tl.arange(0, SEG3))
                                e10 = tl.load(embed_ptr + off10
                                              + SEG0 + SEG1 + SEG2
                                              + tl.arange(0, SEG3))
                                e11 = tl.load(embed_ptr + off11
                                              + SEG0 + SEG1 + SEG2
                                              + tl.arange(0, SEG3))
                                for j in tl.static_range(0, MAX_RUN):
                                    c_j = c_lo + j
                                    w_frac_j = c_j.to(tl.float32) * w_scale
                                    in_g = (c_j < c_hi) & (
                                        w_frac_j.to(tl.int32) == wf_g)
                                    if in_g:
                                        bc = c_j // M_SIZE
                                        lc = c_j % M_SIZE
                                        spatial = (br * nbw + bc) * (
                                            M_SIZE * M_SIZE) + lr * M_SIZE + lc
                                        out_off = spatial * HIDDEN_DIM
                                        dw = w_frac_j - w_frac_j.to(
                                            tl.int32).to(tl.float32)
                                        w11 = dh * dw
                                        w10 = dh - w11
                                        w01 = dw - w11
                                        w00 = 1.0 - dh - w01
                                        _corner_seg_store(
                                            output_ptr, out_off, e00, e01,
                                            e10, e11, w00.to(out_dtype),
                                            w01.to(out_dtype),
                                            w10.to(out_dtype),
                                            w11.to(out_dtype),
                                            SEG0 + SEG1 + SEG2, SEG3, FRAMES,
                                            FRAME_STRIDE)
                            if SEG4 > 0:
                                e00 = tl.load(embed_ptr + off00
                                              + SEG0 + SEG1 + SEG2 + SEG3
                                              + tl.arange(0, SEG4))
                                e01 = tl.load(embed_ptr + off01
                                              + SEG0 + SEG1 + SEG2 + SEG3
                                              + tl.arange(0, SEG4))
                                e10 = tl.load(embed_ptr + off10
                                              + SEG0 + SEG1 + SEG2 + SEG3
                                              + tl.arange(0, SEG4))
                                e11 = tl.load(embed_ptr + off11
                                              + SEG0 + SEG1 + SEG2 + SEG3
                                              + tl.arange(0, SEG4))
                                for j in tl.static_range(0, MAX_RUN):
                                    c_j = c_lo + j
                                    w_frac_j = c_j.to(tl.float32) * w_scale
                                    in_g = (c_j < c_hi) & (
                                        w_frac_j.to(tl.int32) == wf_g)
                                    if in_g:
                                        bc = c_j // M_SIZE
                                        lc = c_j % M_SIZE
                                        spatial = (br * nbw + bc) * (
                                            M_SIZE * M_SIZE) + lr * M_SIZE + lc
                                        out_off = spatial * HIDDEN_DIM
                                        dw = w_frac_j - w_frac_j.to(
                                            tl.int32).to(tl.float32)
                                        w11 = dh * dw
                                        w10 = dh - w11
                                        w01 = dw - w11
                                        w00 = 1.0 - dh - w01
                                        _corner_seg_store(
                                            output_ptr, out_off, e00, e01,
                                            e10, e11, w00.to(out_dtype),
                                            w01.to(out_dtype),
                                            w10.to(out_dtype),
                                            w11.to(out_dtype),
                                            SEG0 + SEG1 + SEG2 + SEG3, SEG4, FRAMES,
                                            FRAME_STRIDE)
                            if SEG5 > 0:
                                e00 = tl.load(embed_ptr + off00
                                              + SEG0 + SEG1 + SEG2 + SEG3 + SEG4
                                              + tl.arange(0, SEG5))
                                e01 = tl.load(embed_ptr + off01
                                              + SEG0 + SEG1 + SEG2 + SEG3 + SEG4
                                              + tl.arange(0, SEG5))
                                e10 = tl.load(embed_ptr + off10
                                              + SEG0 + SEG1 + SEG2 + SEG3 + SEG4
                                              + tl.arange(0, SEG5))
                                e11 = tl.load(embed_ptr + off11
                                              + SEG0 + SEG1 + SEG2 + SEG3 + SEG4
                                              + tl.arange(0, SEG5))
                                for j in tl.static_range(0, MAX_RUN):
                                    c_j = c_lo + j
                                    w_frac_j = c_j.to(tl.float32) * w_scale
                                    in_g = (c_j < c_hi) & (
                                        w_frac_j.to(tl.int32) == wf_g)
                                    if in_g:
                                        bc = c_j // M_SIZE
                                        lc = c_j % M_SIZE
                                        spatial = (br * nbw + bc) * (
                                            M_SIZE * M_SIZE) + lr * M_SIZE + lc
                                        out_off = spatial * HIDDEN_DIM
                                        dw = w_frac_j - w_frac_j.to(
                                            tl.int32).to(tl.float32)
                                        w11 = dh * dw
                                        w10 = dh - w11
                                        w01 = dw - w11
                                        w00 = 1.0 - dh - w01
                                        _corner_seg_store(
                                            output_ptr, out_off, e00, e01,
                                            e10, e11, w00.to(out_dtype),
                                            w01.to(out_dtype),
                                            w10.to(out_dtype),
                                            w11.to(out_dtype),
                                            SEG0 + SEG1 + SEG2 + SEG3 + SEG4, SEG5, FRAMES,
                                            FRAME_STRIDE)
                            c_lo = c_hi

    def triton_pos_embed_interpolate(
        embed_weight: torch.Tensor,
        t: int,
        h: int,
        w: int,
        num_grid_per_side: int,
        m_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Launch the fused Triton kernel for one (t,h,w) grid.

        Returns a tensor of shape ``(t * h * w, hidden_dim)`` with the
        bilinearly-interpolated position embeddings in spatial-merge order.
        Signature identical to upstream ``triton_pos_embed_interpolate``.
        """
        _validate_inputs(embed_weight, t, h, w, num_grid_per_side, m_size, dtype)
        hidden_dim = embed_weight.shape[1]
        total_out = t * h * w
        output = torch.empty(
            total_out,
            hidden_dim,
            device=embed_weight.device,
            dtype=dtype,
        )

        h_scale = float(num_grid_per_side - 1) / float(h - 1) if h > 1 else 0.0
        w_scale = float(num_grid_per_side - 1) / float(w - 1) if w > 1 else 0.0

        block_d, tail_block, tail_start, exact_main = _hidden_split(hidden_dim)
        segs = _pow2_segments(hidden_dim)
        s0, s1, s2, s3, s4, s5 = segs if segs else (0, 0, 0, 0, 0, 0)

        # r10 production default: the pair kernel (2 rows/program) whenever
        # its launch contract holds -- even m_size (hence even W, so pairs
        # never straddle a t-boundary), even total row count and exact
        # power-of-two column segments (the pair kernel is fully
        # unmasked). r9 NPU evidence (16x100x168, 268800 rows): bitwise
        # identical to the single-row kernel, BF16 3.252ms vs 3.387ms
        # single-row (55% MBU, issue-bound) with FP32 parity (2.886ms vs
        # 2.875ms). Falls back to the single-row kernel otherwise (odd
        # m_size, over-deep hidden split).
        use_pair = segs is not None and m_size % 2 == 0 and total_out % 2 == 0
        # r24 bench-only overrides (bitwise-neutral; never set in production)
        cap = _GRID_CAP_OVERRIDE if _GRID_CAP_OVERRIDE else _GRID_CAP
        if (_FORCE_QUAD and segs is not None and m_size % 2 == 0
                and total_out % 4 == 0):
            # Quad kernel: 4 rows per program as two independent pair
            # slots -- r9-proven bitwise identical to pair/single.
            slots_q = total_out // 4
            chunks_q = (slots_q + cap - 1) // cap
            grid_q = (min(slots_q, cap),)
            _bilinear_pos_embed_kernel_quad[grid_q](
                embed_weight,
                output,
                h,
                w,
                h_scale,
                w_scale,
                total_out,
                chunks_q,
                num_grid_per_side,
                m_size,
                hidden_dim,
                s0,
                s1,
                s2,
                s3,
                s4,
                s5,
                **_launch_opts(),
            )
            return output
        # r28 production default: the 2D-vectorised band kernel
        # (source-row-group load sharing) whenever the host grouping yields
        # enough rows per group (>= _BAND_MIN_GROUP_ROWS); r26 op_summary
        # evidence showed the per-row independent embed loads cost 4-5x the
        # task-book bytes on the failing shapes (tail 20.9%, big-video
        # 53.4%, raw 53-55% FP32 MBU), and r27 showed the v1 per-row-loop
        # band form is 3x slower than pair (no CSE of the repeated embed
        # loads). Falls back to pair/single when groups are too small
        # (mid/short-video/small-A: 1-2 rows per group, lines already
        # green) or numpy is unavailable. Bitwise-identical output (per-row
        # weight arithmetic unchanged; only the embed offsets are shared).
        band_tbl = None
        if (segs is not None and not _BAND_DISABLE):
            band_tbl = _build_band_table(
                t, h, w, num_grid_per_side, m_size, h_scale, w_scale,
                hidden_dim, embed_weight.device)
        if band_tbl is not None:
            band_rows_t, band_off_t, num_bands = band_tbl
            chunks_b = (num_bands + cap - 1) // cap
            grid_b = (min(num_bands, cap),)
            _bilinear_pos_embed_kernel_band2[grid_b](
                embed_weight,
                output,
                band_rows_t,
                band_off_t,
                h,
                w,
                h_scale,
                w_scale,
                num_bands,
                total_out,
                chunks_b,
                _BAND_R,
                num_grid_per_side,
                m_size,
                hidden_dim,
                _BAND_TILE_D,
                s0,
                s1,
                s2,
                s3,
                s4,
                s5,
                **_launch_opts(),
            )
            return output
        # r42 bench-only: corner kernel (logical-row w-group load
        # sharing). _FORCE_CORNER is flipped only by run.py diagnostics;
        # production dispatch is unchanged. Requires t > 1 (frame
        # replication) and the exact-segment contract; grid = (h,) with
        # one logical row per program, CHUNKS=1.
        if _FORCE_CORNER and segs is not None and t > 1:
            max_run_c = _corner_shape_params(w, w_scale)
            _bilinear_pos_embed_kernel_corner[(h,)](
                embed_weight,
                output,
                h,
                w,
                h_scale,
                w_scale,
                h * w * hidden_dim,
                1,
                1,
                t,
                num_grid_per_side,
                m_size,
                hidden_dim,
                max_run_c,
                num_grid_per_side,
                s0,
                s1,
                s2,
                s3,
                s4,
                s5,
                **_launch_opts(),
            )
            return output
        # r32 production default: temporal-frame replication when t > 1.
        # h_scale/w_scale are scalar args, so all t frames of a spatial row
        # are bitwise identical -> compute each unique spatial row once,
        # store FRAMES copies (embed reads drop by t; writes unchanged).
        # Gated on the exact-segment contract (segs is not None) and a
        # minimum total row count so tiny t>1 shapes (short-video lines
        # already green) keep pair. BODIES reuses the mb tuning knobs --
        # r55: via the dedicated _TREP_TARGET_PROGRAMS knob (1024, the
        # pre-r55 _MB_TARGET_PROGRAMS value) so the r55 mb4-large
        # promotion (_MB_TARGET_PROGRAMS 512) cannot silently change the
        # line-checked trep paths (adversarial-review finding: tb-tail
        # trep BODIES would have moved 4->8 with no same-run A/B data).
        use_trep = (segs is not None and t > 1 and not _TREP_DISABLE
                    and total_out >= _TREP_MIN_ROWS)
        if use_trep:
            unique_rows = h * w
            bodies_t = max(1, min(_MB_MAX_BODIES,
                                  (unique_rows + _TREP_TARGET_PROGRAMS - 1)
                                  // _TREP_TARGET_PROGRAMS))
            programs_t = (unique_rows + bodies_t - 1) // bodies_t
            chunks_t = (programs_t + cap - 1) // cap
            grid_t = (min(programs_t, cap),)
            if _TREP_SEG:
                # r46 bench-only: segment-parallel trep (rawAB-trepseg
                # gated). Grid expands to unique_rows x N_SEGS -- both
                # trep shapes stay under the 65535 program cap (tail
                # 28800, big-video 33600); anything larger keeps the
                # r32 path below.
                n_segs = sum(1 for _s in (s0, s1, s2, s3, s4, s5) if _s > 0)
                seg_grid = unique_rows * n_segs
                if seg_grid <= cap:
                    _bilinear_pos_embed_kernel_trep_seg[(seg_grid,)](
                        embed_weight,
                        output,
                        h,
                        w,
                        h_scale,
                        w_scale,
                        unique_rows,
                        h * w * hidden_dim,
                        n_segs,
                        t,
                        num_grid_per_side,
                        m_size,
                        hidden_dim,
                        s0,
                        s1,
                        s2,
                        s3,
                        s4,
                        s5,
                        **_launch_opts(),
                    )
                    return output
            if _TREP_FLAT:
                # r38 bench-only: flat-load trep body (rawAB-trepflat
                # gated); production keeps the r32 body below.
                _bilinear_pos_embed_kernel_trep_flat[grid_t](
                    embed_weight,
                    output,
                    h,
                    w,
                    h_scale,
                    w_scale,
                    unique_rows,
                    h * w * hidden_dim,
                    chunks_t,
                    bodies_t,
                    t,
                    num_grid_per_side,
                    m_size,
                    hidden_dim,
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    **_launch_opts(),
                )
                return output
            # r51 production: masked single-segment trep2d (m1) for the
            # tail route (hidden whose exact pow2 decomposition is long,
            # e.g. 1000 -> 512+256+128+64+32+8). r50 pre-registered: probe
            # 0.079/0.085ms <=0.191ms, rawAB-trep2dm1 bitwise gate passed
            # both dtypes; collapses the six-segment serial chain to ONE
            # masked [BLOCK_D=next_pow2(hidden)] round. Gated to low mask
            # waste (<=1/8 of BLOCK_D) so hidden=1152 big-video (2 segs,
            # 43.8% waste) keeps the r48 exact-segment trep2d below.
            # FP2C=t (full frame chunks), row mapping/bodies/chunks verbatim.
            block_d_m1 = 1 << (hidden_dim - 1).bit_length()
            n_segs_m1 = sum(1 for _s in (s0, s1, s2, s3, s4, s5) if _s > 0)
            if (_TREP_M1 and n_segs_m1 >= 4
                    and (block_d_m1 - hidden_dim) * 8 <= block_d_m1):
                _bilinear_pos_embed_kernel_trep2d_m1[grid_t](
                    embed_weight,
                    output,
                    h,
                    w,
                    h_scale,
                    w_scale,
                    unique_rows,
                    h * w * hidden_dim,
                    chunks_t,
                    bodies_t,
                    t,
                    t,
                    num_grid_per_side,
                    m_size,
                    hidden_dim,
                    block_d_m1,
                    **_launch_opts(),
                )
                return output
            # r48 production: 2D frame-chunked store trep (trep2d). Same
            # row mapping / bodies / segments as the r32 trep body above --
            # only the FRAMES 1D stores are issued as one [FP2C, WIDTH] 2D
            # store per exact segment (values/addresses identical, rawAB-
            # trep2d bitwise gate passed r45b/r46/r47). FP2C=FRAMES=t makes
            # every store chunk full (zero masked lanes): tail t=3 -> fp3
            # (0.175ms vs 0.404ms trep at r47), big-video t=16 -> fp16.
            # r46 pre-registered: fp3 tail-f32 <=0.191ms -> promote to
            # production; r47 measured 0.175ms. Dtype/args unchanged.
            _bilinear_pos_embed_kernel_trep2d[grid_t](
                embed_weight,
                output,
                h,
                w,
                h_scale,
                w_scale,
                unique_rows,
                h * w * hidden_dim,
                chunks_t,
                bodies_t,
                t,
                t,
                num_grid_per_side,
                m_size,
                hidden_dim,
                s0,
                s1,
                s2,
                s3,
                s4,
                s5,
                **_launch_opts(),
            )
            return output
        # r31 production default: flat multi-body pair kernel for
        # mid-size single-image shapes (shallow pair grids), plain pair
        # kernel elsewhere. Same pair-row mapping as the pair kernel
        # (bitwise identical by construction; rawAB-mb gates every
        # round). Small shapes (< _MB_MIN_ROWS) keep the plain pair
        # kernel: their lines are green and a shrunk grid must keep
        # enough programs to fill the AIVs. Very large shapes
        # (>= _MB_MAX_ROWS) keep pair too: r30 proved mb loses there
        # (big-video-f32 3.606ms vs pair 2.770ms, both op_summary).
        use_mb = (segs is not None and m_size % 2 == 0 and total_out % 2 == 0
                  and not _MB_DISABLE and total_out >= _MB_MIN_ROWS
                  and total_out < _MB_MAX_ROWS)
        if use_mb:
            slots_mb = total_out // 2
            if (_MB_D6_PROMOTE and t == 1 and 0.0 < w_scale < 1.0):
                # r59 production (was r59 bench hook _FORCE_MB_D6=1): mb1d6
                # is the fastest measured large-bf16 structure on the local
                # SoC (9.0 ns/row vs mb2 16.75, -46%, bitwise identical).
                # Gate t==1 and 0<w_scale<1 confines the blast radius to
                # the tb-large line: small-A/mid have w_scale >= 1.0, and
                # short-video/big-video/tail/multi all have t > 1.
                bodies_d6 = 1
                chunks_d6 = (slots_mb + cap - 1) // cap
                grid_d6 = (min(slots_mb, cap),)
                _bilinear_pos_embed_kernel_mb_d6[grid_d6](
                    embed_weight,
                    output,
                    h,
                    w,
                    h_scale,
                    w_scale,
                    total_out,
                    chunks_d6,
                    bodies_d6,
                    num_grid_per_side,
                    m_size,
                    hidden_dim,
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    **_launch_opts(),
                )
                return output
            if _FORCE_MB_D6 and 0.0 < w_scale <= 1.0:
                bodies_d6 = max(1, min(_MB_MAX_BODIES, int(_FORCE_MB_D6)))
                programs_d6 = (slots_mb + bodies_d6 - 1) // bodies_d6
                chunks_d6 = (programs_d6 + cap - 1) // cap
                grid_d6 = (min(programs_d6, cap),)
                _bilinear_pos_embed_kernel_mb_d6[grid_d6](
                    embed_weight,
                    output,
                    h,
                    w,
                    h_scale,
                    w_scale,
                    total_out,
                    chunks_d6,
                    bodies_d6,
                    num_grid_per_side,
                    m_size,
                    hidden_dim,
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    **_launch_opts(),
                )
                return output
            if (_FORCE_MB_D6S and t == 1 and 0.0 < w_scale < 1.0
                    and 0.0 < h_scale < 1.0 and total_out % 4 == 0):
                # B16 round 12 bench hook: 2 vertically adjacent points
                # per program (4 rows), cross-point d6 register reuse.
                # Gated so hfB in {hfA, hfA+1} (0 < h_scale < 1) and both
                # points stay in the diff branch (0 < w_scale < 1).
                slots_s = total_out // 4
                chunks_s = (slots_s + cap - 1) // cap
                grid_s = (min(slots_s, cap),)
                _bilinear_pos_embed_kernel_mb_d6s[grid_s](
                    embed_weight,
                    output,
                    h,
                    w,
                    h_scale,
                    w_scale,
                    total_out,
                    chunks_s,
                    num_grid_per_side,
                    m_size,
                    hidden_dim,
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    SPLITS=1,
                    **_launch_opts(),
                )
                return output
            if (_FORCE_MB_D6S2 and t == 1 and 0.0 < w_scale < 1.0
                    and 0.0 < h_scale < 1.0 and total_out % 4 == 0
                    and hidden_dim % 2 == 0):
                # B16 round 18: cross-point load sharing of d6s WITH the
                # program count restored -- 2 vertically adjacent points
                # per program on a hidden HALF (576 of 1152). Same traffic
                # saving as d6s (-30% loads) at mb1d6's program count.
                slots_s = total_out // 4
                programs_s = slots_s * 2
                chunks_s = (programs_s + cap - 1) // cap
                grid_s = (min(programs_s, cap),)
                hs = _pow2_segments(hidden_dim // 2)
                _bilinear_pos_embed_kernel_mb_d6s[grid_s](
                    embed_weight,
                    output,
                    h,
                    w,
                    h_scale,
                    w_scale,
                    total_out,
                    chunks_s,
                    num_grid_per_side,
                    m_size,
                    hidden_dim,
                    hs[0],
                    hs[1],
                    hs[2],
                    hs[3],
                    hs[4],
                    hs[5],
                    SPLITS=2,
                    **_launch_opts(),
                )
                return output
            if _MB_BODY_OVERRIDE:
                bodies = int(_MB_BODY_OVERRIDE)
            else:
                bodies = max(1, min(_MB_MAX_BODIES,
                                    slots_mb // _MB_TARGET_PROGRAMS))
            if bodies >= 2:
                programs_mb = (slots_mb + bodies - 1) // bodies
                chunks_mb = (programs_mb + cap - 1) // cap
                grid_mb = (min(programs_mb, cap),)
                _bilinear_pos_embed_kernel_mb[grid_mb](
                    embed_weight,
                    output,
                    h,
                    w,
                    h_scale,
                    w_scale,
                    total_out,
                    chunks_mb,
                    bodies,
                    num_grid_per_side,
                    m_size,
                    hidden_dim,
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    **_launch_opts(),
                )
                return output
        if use_pair:
            # r58 bench-only hook (00:40 spec R1): diff6 pair kernel with
            # the diff-branch load reuse. On 0 < w_scale <= 1.0 the diff
            # branch (wf1 != wf0) forces wf1 == wf0 + 1 with wf0 + 1 <=
            # G-1, so wc0 == wf1 and off_a1 == off_b0, off_c1 == off_d0
            # exactly (algebra in _bilinear_pair_body_d6): the two dropped
            # loads are pure address-equal reuse -- ZERO divergence class,
            # stronger than the vpair +/-0 class; rawAB-diff6 gates it
            # strictly. Placed BEFORE the vmb production dispatch so
            # _FORCE_DIFF6 can also bypass vmb on w_scale == 1.0 shapes
            # (bench-hook precedence). w_scale > 1.0 breaks the window
            # and keeps production. Production unchanged (None default).
            if _FORCE_DIFF6 and 0.0 < w_scale <= 1.0:
                slots_d6 = total_out // 2
                chunks_d6 = (slots_d6 + cap - 1) // cap
                grid_d6 = (min(slots_d6, cap),)
                _bilinear_pos_embed_kernel_pair_d6[grid_d6](
                    embed_weight,
                    output,
                    h,
                    w,
                    h_scale,
                    w_scale,
                    total_out,
                    chunks_d6,
                    num_grid_per_side,
                    m_size,
                    hidden_dim,
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    **_launch_opts(),
                )
                return output
            # r58 bench-only hook (00:40 spec R2): vmb with the load-ahead
            # vbody (_bilinear_pair_vbody_la) -- statement-order overlap
            # of each segment's load latency with the previous segment's
            # store round; zero arithmetic change -> bitwise identical to
            # the sequential vbody modulo the same vanishing +/-0 corner
            # class (rawAB-vmb-la gates it, same caliber as rawAB-vmb).
            # _FORCE_VMB_LA carries the BODIES int (same meaning as
            # _FORCE_VMB); _VMB_DISABLE keeps the legacy-pair A/B leg
            # available. Placed before the vmb production dispatch.
            if (_FORCE_VMB_LA and not _VMB_DISABLE and w_scale == 1.0
                    and total_out >= _VMB_MIN_ROWS):
                slots_la = total_out // 2
                bodies_la = max(1, min(_MB_MAX_BODIES, int(_FORCE_VMB_LA)))
                programs_la = (slots_la + bodies_la - 1) // bodies_la
                chunks_la = (programs_la + cap - 1) // cap
                grid_la = (min(programs_la, cap),)
                _bilinear_pos_embed_kernel_vmb_la[grid_la](
                    embed_weight,
                    output,
                    h,
                    w,
                    h_scale,
                    w_scale,
                    total_out,
                    chunks_la,
                    bodies_la,
                    num_grid_per_side,
                    m_size,
                    hidden_dim,
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    **_launch_opts(),
                )
                return output
            # r53 bench-only hook: _FORCE_PAIR2D forces the 2D pair kernel
            # (same pair-row mapping, [2, WIDTH] 2D strided embed loads /
            # FMA / store). Values and addresses are identical to the
            # scalar pair kernel -> bitwise identical (rawAB-pair2d gates
            # it every round). Production dispatch is unchanged (None);
            # callers that force this on mb-dispatched shapes must also
            # set _MB_DISABLE=True so the mb block above does not return
            # first.
            if _FORCE_PAIR2D:
                slots = total_out // 2
                chunks = (slots + cap - 1) // cap
                grid = (min(slots, cap),)
                _bilinear_pos_embed_kernel_pair2d[grid](
                    embed_weight,
                    output,
                    h,
                    w,
                    h_scale,
                    w_scale,
                    total_out,
                    chunks,
                    num_grid_per_side,
                    m_size,
                    hidden_dim,
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    **_launch_opts(),
                )
                return output
            # r55 PRODUCTION: vpair dispatches on w_scale == 1.0 shapes
            # (_VPAIR_PROMOTE). _FORCE_VPAIR stays as a bench override and
            # _VPAIR_DISABLE lets run.py bench the legacy pair kernel in
            # the same run (within-run A/B). Same pair-row mapping and
            # weight chain, zero-weight wc-column embed loads dropped ->
            # bitwise identical except vanishing +/-0 corners (rawAB-vpair
            # gates it every round; r54 both dtypes PASSED on-machine).
            if ((_FORCE_VPAIR or _VPAIR_PROMOTE)
                    and not _VPAIR_DISABLE and w_scale == 1.0):
                slots = total_out // 2
                chunks = (slots + cap - 1) // cap
                grid = (min(slots, cap),)
                _bilinear_pos_embed_kernel_vpair[grid](
                    embed_weight,
                    output,
                    h,
                    w,
                    h_scale,
                    w_scale,
                    total_out,
                    chunks,
                    num_grid_per_side,
                    m_size,
                    hidden_dim,
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    **_launch_opts(),
                )
                return output
            # r57 PRODUCTION dispatch (pre-registered 1033 verdict): vmb
            # BODIES=2 promotes on w_scale == 1.0 shapes with enough rows
            # to amortize (mid 1536 qualifies, short-video 800 stays plain
            # pair via _VMB_MIN_ROWS; its line is already green). Evidence:
            # rawAB-vmb bitwise PASS both dtypes on-machine; x-vmb2 device
            # p50 17.8us/17.3us vs pair 30.6/30.2us -> S halved BOTH
            # dtypes, no dtype-split. _VMB_DISABLE lets run.py bench the
            # legacy pair kernel in the same run (within-run A/B, keeps
            # rawAB gates non-tautological). _FORCE_VMB (int BODIES)
            # remains the bench override for the 3/4 curve points.
            if ((_FORCE_VMB or _VMB_PROMOTE)
                    and not _VMB_DISABLE and w_scale == 1.0
                    and total_out >= _VMB_MIN_ROWS):
                slots_vmb = total_out // 2
                bodies_vmb = max(1, min(_MB_MAX_BODIES,
                                        int(_FORCE_VMB or _VMB_PROMOTE)))
                programs_vmb = (slots_vmb + bodies_vmb - 1) // bodies_vmb
                chunks_vmb = (programs_vmb + cap - 1) // cap
                grid_vmb = (min(programs_vmb, cap),)
                _bilinear_pos_embed_kernel_vmb[grid_vmb](
                    embed_weight,
                    output,
                    h,
                    w,
                    h_scale,
                    w_scale,
                    total_out,
                    chunks_vmb,
                    bodies_vmb,
                    num_grid_per_side,
                    m_size,
                    hidden_dim,
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    **_launch_opts(),
                )
                return output
            # Ascend caps TOTAL grid programs (EE1003); the cap applies to
            # pair slots and CHUNKS is a static unroll over the cap.
            slots = total_out // 2
            chunks = (slots + cap - 1) // cap
            grid = (min(slots, cap),)
            _bilinear_pos_embed_kernel_pair[grid](
                embed_weight,
                output,
                h,
                w,
                h_scale,
                w_scale,
                total_out,
                chunks,
                num_grid_per_side,
                m_size,
                hidden_dim,
                s0,
                s1,
                s2,
                s3,
                s4,
                s5,
                **_launch_opts(),
            )
            return output

        # Single-row fallback: Ascend caps TOTAL grid programs (EE1003);
        # r6 sweep: cap 32768 + per-program CHUNKS static unroll is
        # fastest on large grids.
        chunks = (total_out + cap - 1) // cap
        grid = (min(total_out, cap),)

        _bilinear_pos_embed_kernel[grid](
            embed_weight,
            output,
            h,
            w,
            h_scale,
            w_scale,
            total_out,
            chunks,
            num_grid_per_side,
            m_size,
            hidden_dim,
            block_d,
            tail_start,
            tail_block,
            exact_main,
            s0,
            s1,
            s2,
            s3,
            s4,
            s5,
            **_launch_opts(),
        )
        return output

    def triton_pos_embed_interpolate_merged(
        embed_weight: torch.Tensor,
        grid_thw_list: list,
        num_grid_per_side: int,
        m_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Merged single-launch multi-image variant.

        Equivalent to ``torch.cat([triton_pos_embed_interpolate(embed, t, h,
        w, ...) for t, h, w in grid_thw_list], dim=0)`` but with one kernel
        launch and no cat copy. Requires all images to share the same
        num_grid_per_side / m_size / hidden_dim (single-model constraint).

        Args:
            embed_weight: [G*G, hidden_dim] learnable position table.
            grid_thw_list: list of (t, h, w) tuples, at least one image.
            num_grid_per_side: position-table edge length G.
            m_size: spatial-merge edge length.
            dtype: output dtype (FP32 or BF16).

        Returns:
            Tensor of shape ``(sum(t*h*w), hidden_dim)`` in spatial-merge
            order per image, images concatenated in list order.
        """
        assert len(grid_thw_list) >= 1, "grid_thw_list must contain at least one image"
        hidden_dim = embed_weight.shape[1]

        cu_rows = [0]
        img_hw = []
        img_scale = []
        for t, h, w in grid_thw_list:
            _validate_inputs(embed_weight, t, h, w, num_grid_per_side, m_size, dtype)
            cu_rows.append(cu_rows[-1] + t * h * w)
            img_hw.extend([h, w])
            img_scale.extend(
                [
                    float(num_grid_per_side - 1) / float(h - 1) if h > 1 else 0.0,
                    float(num_grid_per_side - 1) / float(w - 1) if w > 1 else 0.0,
                ]
            )

        total_out = cu_rows[-1]
        device = embed_weight.device
        output = torch.empty(total_out, hidden_dim, device=device, dtype=dtype)

        block_d, tail_block, tail_start, exact_main = _hidden_split(hidden_dim)
        segs = _pow2_segments(hidden_dim)
        s0, s1, s2, s3, s4, s5 = segs if segs else (0, 0, 0, 0, 0, 0)
        num_images = len(grid_thw_list)

        # Ascend caps TOTAL grid programs (EE1003); r6 sweep: cap 32768 +
        # per-program CHUNKS static unroll is fastest on large grids.
        chunks = (total_out + _GRID_CAP - 1) // _GRID_CAP
        grid = (min(total_out, _GRID_CAP),)

        if num_images <= 4:
            # r33: amortise the image-location chain over BODIES rows. The
            # gcd keeps every body block inside one image (see knob docs).
            bodies = 1
            if not _MERGED_MB_DISABLE and total_out >= _MERGED_MB_MIN_ROWS:
                g = 0
                for t, h, w in grid_thw_list:
                    n = t * h * w
                    while n:
                        g, n = n, g % n
                bodies = max(1, min(g, _MERGED_MB_MAX_BODIES))
            if bodies > 1:
                n_bodies = -(-total_out // bodies)
                chunks = -(-n_bodies // _GRID_CAP)
                grid = (min(n_bodies, _GRID_CAP),)
            # Host-known scalars: no per-program dependent global-load
            # chain (r18 evidence: that chain dominated the device-tensor
            # variant, 14.8 Gelem/s vs 58+ single-image). Ends beyond
            # NUM_IMAGES are inert (trace-time guards) and use INT_MAX /
            # dummy sizes.
            ends = cu_rows[1:] + [2147483647] * (4 - num_images)
            hws = img_hw + [1, 1] * (4 - num_images)
            scales = img_scale + [0.0, 0.0] * (4 - num_images)
            _bilinear_pos_embed_merged_scalar_kernel[grid](
                embed_weight,
                output,
                ends[0],
                ends[1],
                ends[2],
                ends[3],
                hws[0],
                hws[1],
                hws[2],
                hws[3],
                hws[4],
                hws[5],
                hws[6],
                hws[7],
                scales[0],
                scales[1],
                scales[2],
                scales[3],
                scales[4],
                scales[5],
                scales[6],
                scales[7],
                total_out,
                chunks,
                bodies,
                num_images,
                num_grid_per_side,
                m_size,
                hidden_dim,
                block_d,
                tail_start,
                tail_block,
                exact_main,
                s0,
                s1,
                s2,
                s3,
                s4,
                s5,
                **_launch_opts(),
            )
            return output

        # > 4 images: device-tensor variant (padded prefix sums).
        cu_rows_t = torch.tensor(cu_rows, dtype=torch.int32, device=device)
        img_hw_t = torch.tensor(img_hw, dtype=torch.int32, device=device)
        img_scale_t = torch.tensor(img_scale, dtype=torch.float32, device=device)
        max_images = triton.next_power_of_2(num_images)

        _bilinear_pos_embed_merged_kernel[grid](
            embed_weight,
            output,
            cu_rows_t,
            img_hw_t,
            img_scale_t,
            num_images,
            total_out,
            chunks,
            max_images,
            num_grid_per_side,
            m_size,
            hidden_dim,
            block_d,
            tail_start,
            tail_block,
            exact_main,
            s0,
            s1,
            s2,
            s3,
            s4,
            s5,
            **_launch_opts(),
        )
        return output

else:  # pragma: no cover - environment without triton (import must not fail)

    def triton_pos_embed_interpolate(*_args, **_kwargs):
        raise RuntimeError("triton is not available in this environment")

    def triton_pos_embed_interpolate_merged(*_args, **_kwargs):
        raise RuntimeError("triton is not available in this environment")


def pos_embed_interpolate_native(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Eager PyTorch bilinear position-embedding interpolation (benchmark).

    Faithful port of upstream ``pos_embed_interpolate_native``
    (qwen3_vl.py:301-368); serves as the precision benchmark and the
    vllm-ascend native fallback path.
    """
    assert h % m_size == 0 and w % m_size == 0, (
        f"h={h} and w={w} must be divisible by m_size={m_size}"
    )
    hidden_dim = embed_weight.shape[1]
    device = embed_weight.device

    h_idxs = torch.linspace(
        0,
        num_grid_per_side - 1,
        h,
        dtype=torch.float32,
        device=device,
    )
    w_idxs = torch.linspace(
        0,
        num_grid_per_side - 1,
        w,
        dtype=torch.float32,
        device=device,
    )

    h_floor = h_idxs.to(torch.long)
    w_floor = w_idxs.to(torch.long)
    h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
    w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

    dh = h_idxs - h_floor
    dw = w_idxs - w_floor

    dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
    h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
    h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = 1 - dh_grid - w01

    h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
    w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
    h_grid_idx = h_grid * num_grid_per_side

    indices = (h_grid_idx + w_grid).reshape(4, -1)
    weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
    weights = weights.to(dtype=dtype)

    embeds = embed_weight[indices]
    embeds *= weights
    combined = embeds.sum(dim=0)

    combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)
    combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
    repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
    return repeated.to(dtype=dtype)


# ---- mojo_opset dispatch bindings (functions/_dispatch.load_impl contract:
# ---- each registered op_id must resolve to an <op_id>_fwd callable here.)
# The triton entry points are the module's public functions (stubs that
# raise when triton is absent); the native reference is device-agnostic.
vision_pos_embed_interp_fwd = triton_pos_embed_interpolate
vision_pos_embed_interp_merged_fwd = triton_pos_embed_interpolate_merged
vision_pos_embed_interp_native_fwd = pos_embed_interpolate_native
