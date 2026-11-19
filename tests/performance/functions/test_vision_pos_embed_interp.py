# SPDX-License-Identifier: Apache-2.0
"""vision_pos_embed_interp performance suite (task-book caliber).

Standalone extraction of the measurement protocol used by the iteration
harness (job_staging/vpe-tb-bench/run.py + wrapper.py, r52+ baseline):

- MBU = bytes / t / bw_fill with
  bytes = (G^2*hidden + t*h*w*hidden) * esize  (== 2 x (G^2*hidden +
  thw*hidden) for bf16 -- the task-book formula, per image for merged);
- primary device metric: torch_npu.profiler op_summary "Task Duration(us)"
  summed over all NPU kernels of one profiled session / iters. Sessions
  run 1 pre-warm call outside the window + 30 timed calls inside
  (task-book 5 warmup + 30 iters; the first in-window call is warm).
  The op_summary CSVs are written by a CANN parse child at PROCESS EXIT
  (in-process polling is architecturally blind -- r24/r25 evidence), so
  the official numbers are harvested in a parent process after the bench
  child exits and mapped 1:1 to rows by mtime order == session order.
- provisional fallbacks inside the bench child (flagged via timing_src):
  chrome-trace kernel durations, then npu Event device-span median
  (includes idle-stream launch gap on small shapes -- conservative).
- fill_ pure-write calibration: 512 MB bf16 block, host-caliber median,
  used as the MBU denominator (this operator is write-dominant);
- clock_warm(120 ms) sustained fill load before every session (DVFS ramp:
  idle AICore down-clocks halve latency-bound kernels -- r33+ evidence).

Task-book lines (each run in float32 and bfloat16):
  tb-small-A       (1,8,16)     hidden=1152  not worse than native (host)
  tb-mid           (1,32,48)    hidden=1152  MBU >= 60%
  tb-large         (1,60,80)    hidden=1152  MBU >= 60%
  tb-short-video   (3,16,16)    hidden=1152  MBU >= 60%
  tb-big-video     (16,100,168) hidden=1152  MBU >= 60%
  tb-tail          (3,60,80)    hidden=1000  MBU >= 50%  (tail block)
  tb-multi-merged  [(1,8,16),(1,32,48),(3,16,16)] hidden=1152  >= 60%
  raw_prod_mb      16x100x168   production dispatch, 3 warmup + 10 iters,
                                MBU >= 60%

Run:
  pytest tests/performance/functions/test_bench_vit_pos_embed.py -v
  python3 tests/performance/functions/test_bench_vit_pos_embed.py [--out DIR]
Exit code 0 iff every line-checked row passes and the op_summary harvest
succeeded (otherwise the provisional timings are reported and the verdict
is marked unverifiable, exit 1).

Local-smoke switches (degraded SoCs where builtin Fill/elementwise ops and
the CANN parse child are unavailable -- C-level hard exits, not catchable):
  VPE_BENCH_NATIVE_DEV=cpu          A-line vs CPU native reference
  VPE_BENCH_BW_GBPS=<gbps>          external pure-write BW denominator
  VPE_BENCH_NO_CLOCK_WARM=1         skip the fill-based DVFS ramp
  VPE_BENCH_ALLOW_FALLBACK_TIMING=1 accept npu_event device-span timings
                                     (harvest not required)
"""
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time

# ---- import bootstrap (works outside pytest where no conftest runs) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))  # repo root
for _p in (_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

NPU_AVAILABLE = False
try:
    import torch_npu  # noqa: F401

    NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    pass

import pytest  # noqa: E402

from mojo_opset.functions.vision_pos_embed_interp import (  # noqa: E402
    pos_embed_interpolate_native,
    triton_pos_embed_interpolate,
    triton_pos_embed_interpolate_merged,
)

# ---- task-book constants (identical to the iteration harness) ----
DEV = "npu"
G = 48
HIDDEN = 1152
M = 2
DTYPES = [torch.float32, torch.bfloat16]
TAIL_HIDDEN = 1000
LARGE_GRID = (16, 100, 168)
MULTI = [(1, 8, 16), (1, 32, 48), (3, 16, 16)]
LARGE_ROWS = LARGE_GRID[0] * LARGE_GRID[1] * LARGE_GRID[2]

# (tag, grid_or_multi, hidden, line_pct, mode, warmup, iters)
LINE_SPECS = [
    ("tb-small-A", (1, 8, 16), HIDDEN, 0, "single", 5, 30),
    ("tb-mid", (1, 32, 48), HIDDEN, 60, "single", 5, 30),
    ("tb-large", (1, 60, 80), HIDDEN, 60, "single", 5, 30),
    ("tb-short-video", (3, 16, 16), HIDDEN, 60, "single", 5, 30),
    ("tb-big-video", (16, 100, 168), HIDDEN, 60, "single", 5, 30),
    ("tb-tail", (3, 60, 80), TAIL_HIDDEN, 50, "single", 5, 30),
    ("tb-multi-merged", MULTI, HIDDEN, 60, "merged", 5, 30),
    ("raw_prod_mb", LARGE_GRID, HIDDEN, 60, "single", 3, 10),
]

_SESSION_SEQ = [0]
_WARM_TENSOR = None

# A-line baseline device. Official caliber: NPU-native. On hosts where
# the NPU native op package is unavailable (op JSON gaps on some
# 910B-class vnpu stacks), set VPE_BENCH_NATIVE_DEV=cpu to baseline the
# A-line against the CPU native reference -- a documented deviation that
# only affects the A-line rows (all MBU lines are unaffected).
NATIVE_DEV = os.environ.get("VPE_BENCH_NATIVE_DEV", DEV)

# Optional MBU denominator override (GB/s). Official path: the fill_
# pure-write calibration below. On stacks where the built-in Fill op is
# unavailable, pass an externally measured pure-write bandwidth
# (e.g. a triton write probe) to keep the suite runnable; the deviation
# is flagged in the summary (bw_fill_src).
_BW_OVERRIDE_GBPS = float(os.environ.get("VPE_BENCH_BW_GBPS", "0") or 0)


def clock_warm(ms=120):
    """Sustained fill load to raise the AICore DVFS clock before a timed
    session. r33/r34 evidence: after an idle gap the AICore down-clocks
    (idle ~800 MHz vs rated 1650); a short-kernel session never triggers
    the ramp on its own. A sustained fill_ load ramps curFreq to 1650
    MHz within ~100 ms. Bandwidth-bound kernels and the fill calibration
    self-warm and are unaffected.
    """
    if os.environ.get("VPE_BENCH_NO_CLOCK_WARM"):
        # fill_ (builtin) hard-exits the process on degraded SoCs (C-level
        # acl error, not catchable); set this to skip the ramp entirely.
        return
    global _WARM_TENSOR
    try:
        if _WARM_TENSOR is None:
            _WARM_TENSOR = torch.empty(512 * 1024 * 1024 // 2,
                                       dtype=torch.bfloat16, device=DEV)
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) * 1000.0 < ms:
            _WARM_TENSOR.fill_(1.0)
        torch.npu.synchronize()
    except Exception as e:
        # built-in Fill op unavailable on this stack (degraded host);
        # DVFS ramp skipped -- documented deviation, local smoke only.
        if not getattr(clock_warm, "_warned", False):
            clock_warm._warned = True
            print(f"clock_warm skipped (Fill op unavailable): "
                  f"{repr(e)[:100]}", flush=True)


def bench_host(fn, warmup=5, iters=30):
    """Host-caliber wall-clock median in ms (5 warmup + 30 iters).

    Includes launch overhead -- used for the A-line native comparison,
    the fill calibration, and as the t_dev mapping guard.
    """
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000.0  # ms


def _op_summary_us(pre_files, iters):
    """Parse the newest CANN-exported op_summary CSV (export-only mode).

    torch_npu.profiler auto-exports PROF_<...>/mindstudio_profiler_output/
    op_summary_*.csv under the cwd at profiler exit; its
    "Task Duration(us)" column is the per-kernel device Duration -- the
    task-book metric. r20-r21 evidence: chrome_trace key filtering yields
    0us on this stack while op_summary holds the true Duration sums.
    Returns (total_us, label) or (None, None).
    """
    if os.environ.get("VPE_SKIP_OP_SUMMARY_POLL"):
        # The CANN parse child writes the CSVs after THIS process exits;
        # in-process polling is architecturally blind (r24 proof). The
        # parent process harvests post-exit instead.
        return None, None
    for _ in range(20):
        files = [f for f in glob.glob(
            "export_only_prof_dir/**/mindstudio_profiler_output/"
            "op_summary_*.csv", recursive=True) if f not in pre_files]
        if files:
            f = max(files, key=os.path.getmtime)
            age = time.time() - os.path.getmtime(f)
            if age < 1.5:
                time.sleep(1.5 - age)
            try:
                tot = 0.0
                n = 0
                with open(f, newline="") as fh:
                    for row in csv.DictReader(fh):
                        try:
                            tot += float(row.get("Task Duration(us)") or 0)
                            n += 1
                        except (TypeError, ValueError):
                            pass
                if tot > 0:
                    return tot, f"op_summary n={n}"
            except Exception:
                pass
        time.sleep(1.0)
    return None, None


def measure_session(fn, warmup=5, iters=30, tag="x"):
    """Task-book primary metric: sum of NPU kernel durations / iters (ms).

    Layer 1: op_summary CSV (official truth; harvested post-exit by the
    parent -- in-process only if VPE_SKIP_OP_SUMMARY_POLL is unset).
    Layer 2: chrome-trace kernel durations from the same profile context.
    Layer 3: npu Event device-span median (includes the idle-stream
    launch gap on small shapes -- conservative, flagged via src).
    Returns (ms, src, sess_idx) or (None, "unavailable", sess_idx).
    """
    sess = None
    errs = []
    pre_prof_files = set(glob.glob(
        "export_only_prof_dir/**/mindstudio_profiler_output/"
        "op_summary_*.csv", recursive=True))

    def _log_layers():
        if errs:
            print(f"TIMING_LAYERS {tag}: " + " | ".join(errs[:8]),
                  flush=True)

    clock_warm()
    fn()
    torch.npu.synchronize()
    try:
        from torch_npu.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.NPU]) as prof:
            sess = _SESSION_SEQ[0]
            _SESSION_SEQ[0] += 1
            for _ in range(iters):
                fn()
            torch.npu.synchronize()
        try:
            tot_us, label = _op_summary_us(pre_prof_files, iters)
            if tot_us:
                _log_layers()
                return tot_us / 1000.0 / iters, label, sess
            errs.append(f"op_summary: {label or 'no new file'}")
        except Exception as e:
            errs.append(f"op_summary: {repr(e)[:120]}")
        try:
            tp = f"/tmp/bench_vpe_trace_{tag}.json"
            prof.export_chrome_trace(tp)
            data = json.load(open(tp))
            evs = data if isinstance(data, list) else data.get(
                "traceEvents", [])
            tot = 0.0
            for e in evs:
                if not isinstance(e, dict) or e.get("ph") != "X":
                    continue
                name = str(e.get("name", ""))
                if name in ("Computing", "Free", "Others", "Epilogue"):
                    continue
                tot += float(e.get("dur", 0) or 0)
            if tot > 0:
                _log_layers()
                return tot / 1000.0 / iters, "chrome_trace", sess
            errs.append(f"chrome_trace: 0us over {len(evs)} events")
        except Exception as e:
            errs.append(f"chrome_trace: {repr(e)[:120]}")
    except Exception as e:
        errs.append(f"profile: {repr(e)[:120]}")
    # fallback: npu Event device-span median
    try:
        ev0 = torch.npu.Event(enable_timing=True)
        ev1 = torch.npu.Event(enable_timing=True)
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        ts = []
        for _ in range(iters):
            ev0.record()
            fn()
            ev1.record()
            torch.npu.synchronize()
            ts.append(ev0.elapsed_time(ev1))
        ts.sort()
        _log_layers()
        return ts[len(ts) // 2], "npu_event", sess
    except Exception as e:
        errs.append(f"npu_event: {repr(e)[:120]}")
    _log_layers()
    return None, "unavailable", sess


def esize(dtype):
    return 4 if dtype == torch.float32 else 2


def bench_row(tag, dtype, grid_or_multi, hidden, line_pct, mode="single",
              warmup=5, iters=30):
    """One task-book shape row: bytes/MBU per task-book formula + line.

    grid_or_multi: (t, h, w) tuple for single-image, or list of tuples
    for the merged multi-image path.
    """
    # CPU-side scale: torch element-wise builtins (muls) HARD-EXIT the
    # process on degraded SoCs (C-level acl error, not catchable); the
    # copy to DEV is the only NPU builtin this suite relies on.
    emb = (torch.randn(G * G, hidden, dtype=dtype) * 0.25).to(DEV)
    emb_nat = emb.cpu() if NATIVE_DEV == "cpu" else emb
    if mode == "merged":
        assert isinstance(grid_or_multi, list)
        imgs = grid_or_multi
        tri_fn = lambda: triton_pos_embed_interpolate_merged(
            emb, imgs, G, M, dtype)
        nat_fn = lambda: torch.cat(
            [pos_embed_interpolate_native(emb_nat, t, h, w, G, M, dtype)
             for t, h, w in imgs], dim=0)
        rows_n = sum(t * h * w for t, h, w in imgs)
        bytes_total = sum(G * G * hidden + t * h * w * hidden
                          for t, h, w in imgs) * esize(dtype)
        grid_s = "+".join(f"{t}x{h}x{w}" for t, h, w in imgs)
    else:
        t, h, w = grid_or_multi
        tri_fn = lambda: triton_pos_embed_interpolate(emb, t, h, w, G, M, dtype)
        nat_fn = lambda: pos_embed_interpolate_native(
            emb_nat, t, h, w, G, M, dtype)
        rows_n = t * h * w
        bytes_total = (G * G * hidden + rows_n * hidden) * esize(dtype)
        grid_s = f"{t}x{h}x{w}"

    try:
        t_dev, src, sess = measure_session(tri_fn, warmup=warmup,
                                           iters=iters, tag=tag)
    except Exception as e:
        t_dev, src, sess = None, f"error:{repr(e)[:80]}", None
        print(f"device timing unavailable for {tag}: {repr(e)[:150]}",
              flush=True)
    t_host = bench_host(tri_fn, warmup=warmup, iters=iters)
    if t_dev:
        # t_dev is in ms -> convert to seconds for the bytes/s ratio
        mbu = bytes_total / (t_dev / 1000.0) / BW_FILL[0] * 100
    else:
        # profiler unavailable: report host-caliber MBU (contains launch
        # overhead -- conservative), flagged via timing_src
        mbu = bytes_total / (t_host / 1000.0) / BW_FILL[0] * 100
    # A-line (line_pct == 0): end-to-end not worse than native (host
    # caliber); only measured for shapes small enough to stay cheap.
    t_nat = (bench_host(nat_fn, warmup=3, iters=10) if rows_n <= 15000
             else None)
    if line_pct == 0:
        line_pass = (t_host <= t_nat) if t_nat is not None else None
    elif t_dev:
        line_pass = bool(mbu >= line_pct)
    else:
        line_pass = f"host-fallback:{mbu:.1f}%"
    row = {"config": tag, "dtype": str(dtype), "grid": grid_s,
           "hidden": hidden, "mode": mode, "out_rows": rows_n,
           "bytes": bytes_total,
           "iters": iters, "sess_idx": sess,
           "t_dev_ms": round(t_dev, 3) if t_dev else "",
           "timing_src": src,
           "t_host_ms": round(t_host, 3),
           "t_native_host_ms": round(t_nat, 3) if t_nat is not None else "",
           "mbu_pct": round(mbu, 1), "line_pct": line_pct,
           "line_pass": line_pass if line_pass is not None else ""}
    print("BENCH " + json.dumps(row), flush=True)
    return row


# calibrated fill_ write bandwidth (bytes/s), set by calibrate_fill()
BW_FILL = [0.0]
# denominator source: "fill_" (official) or "env-override" (degraded host)
BW_SRC = ["fill_"]


def calibrate_fill():
    """fill_ pure-write bandwidth calibration (task-book denominator).

    512 MB bf16 block, host-caliber median of 5 warmup + 30 iters. This
    operator is write-dominant (output writes == input reads in bytes),
    so the pure-write fill rate is the correct denominator.
    """
    if _BW_OVERRIDE_GBPS > 0:
        BW_FILL[0] = _BW_OVERRIDE_GBPS * 1e9
        BW_SRC[0] = "env-override"
        print(f"using VPE_BENCH_BW_GBPS = {_BW_OVERRIDE_GBPS} GB/s "
              "(override, Fill op unavailable)", flush=True)
        return
    print("calibrating fill_ write bandwidth (bf16 block, 5 warmup + 30) "
          "...", flush=True)
    cal = torch.empty(512 * 1024 * 1024 // 2, dtype=torch.bfloat16,
                      device=DEV)
    BW_FILL[0] = cal.numel() * 2 / (bench_host(lambda: cal.fill_(1.0))
                                    / 1000.0)
    BW_SRC[0] = "fill_"
    print(f"fill_ peak write BW = {BW_FILL[0] / 1e9:.1f} GB/s", flush=True)
    del cal


def run_bench():
    """Bench phase (child process): calibrate + run every line row.

    Writes bench_raw.json (rows + bw_fill + session count); exits. The
    parent process harvests the op_summary CSVs after this process dies.
    """
    global _SESSION_SEQ
    calibrate_fill()
    rows = []
    print("running task-book line matrix ...", flush=True)
    for dtype in DTYPES:
        for tag, grid, hidden, line, mode, warmup, iters in LINE_SPECS:
            rows.append(bench_row(tag, dtype, grid, hidden, line, mode,
                                  warmup, iters))
    json.dump({"bw_fill_gbps": round(BW_FILL[0] / 1e9, 1),
               "rows": rows,
               "sessions": _SESSION_SEQ[0]},
              open("bench_raw.json", "w"), indent=1)
    print(f"DONE bench rows={len(rows)} sessions={_SESSION_SEQ[0]}",
          flush=True)
    return 0


def _find_csvs(out_dir):
    return glob.glob(os.path.join(
        out_dir, "export_only_prof_dir", "**", "op_summary_*.csv"),
        recursive=True)


def harvest(out_dir):
    """Post-exit op_summary harvest (parent process).

    The CANN parse child writes one op_summary CSV per profiler session
    AFTER the bench process exits (r24/r25 evidence: in-process polling
    is blind). Poll up to 90 s (3 s stability), map CSVs 1:1 to the
    profiled rows by mtime order == session order (sessions run
    sequentially), recompute t_dev / MBU / line verdicts, and copy the
    evidence CSVs. Returns the final summary dict.
    """
    raw = json.load(open(os.path.join(out_dir, "bench_raw.json")))
    rows = raw.get("rows", [])
    bw_fill = raw.get("bw_fill_gbps", 0) * 1e9
    profiled = [r for r in rows if r.get("sess_idx") is not None]
    print(f"bench rows={len(rows)} sessions={raw.get('sessions', 0)} "
          f"profiled={len(profiled)} bw_fill={bw_fill / 1e9:.1f} GB/s",
          flush=True)

    new = []
    deadline = time.time() + 90
    while time.time() < deadline:
        new = _find_csvs(out_dir)
        if profiled and len(new) >= len(profiled):
            time.sleep(3.0)  # stability: no growth for 3 s
            newer = _find_csvs(out_dir)
            if len(newer) == len(new):
                new = newer
                break
            new = newer
        time.sleep(1.0)

    harvest_ok = bool(profiled) and len(new) >= len(profiled)
    if not harvest_ok:
        print(f"HARVEST FAIL: {len(new)} CSVs vs {len(profiled)} sessions "
              f"after 90 s; rows keep provisional timings", flush=True)
    else:
        # one CSV per session; (mtime, path) order == session order.
        # extra CSVs (stray sessions) -> keep the newest N.
        new.sort(key=lambda f: (os.path.getmtime(f), f))
        if len(new) > len(profiled):
            print(f"harvest: {len(new)} CSVs > {len(profiled)} sessions; "
                  f"using newest {len(profiled)}", flush=True)
            new = new[-len(profiled):]
        print(f"harvested {len(new)} op_summary CSVs", flush=True)
        ev_dir = os.path.join(out_dir, "op_summary_evidence")
        os.makedirs(ev_dir, exist_ok=True)
        for i, (r, f) in enumerate(zip(profiled, new)):
            try:
                tot = 0.0
                n = 0
                with open(f, newline="") as fh:
                    for crow in csv.DictReader(fh):
                        try:
                            tot += float(crow.get("Task Duration(us)") or 0)
                            n += 1
                        except (TypeError, ValueError):
                            pass
                if tot <= 0:
                    r["timing_src"] += "|csv-empty"
                    continue
                t_dev = tot / 1000.0 / (r.get("iters") or 30)
                t_host = r.get("t_host_ms")
                if t_host and t_dev > float(t_host) * 1.2:
                    # mapping guard: device time cannot exceed host time
                    r["timing_src"] += f"|csv-mismatch(n={n})"
                    continue
                r["t_dev_ms"] = round(t_dev, 3)
                r["timing_src"] = f"op_summary n={n}"
                if r.get("bytes") and r.get("line_pct"):
                    r["mbu_pct"] = round(
                        r["bytes"] / (t_dev / 1000.0) / bw_fill * 100, 1)
                    r["line_pass"] = bool(r["mbu_pct"] >= r["line_pct"])
                shutil.copy(f, os.path.join(
                    ev_dir, f"{i:02d}_{r.get('config', 'row')}"
                    f"_{str(r.get('dtype', '')).split('.')[-1]}.csv"))
            except Exception as e:
                r["timing_src"] += "|csv-parse-error"
                print(f"csv parse failed for {r.get('config')}: "
                      f"{repr(e)[:120]}", flush=True)

    # ---- line verdicts (task-book rules) ----
    line_fail = [r for r in rows if r.get("line_pass") is False]
    line_pass = [r for r in rows if r.get("line_pass") is True]
    line_fb = [r for r in rows if isinstance(r.get("line_pass"), str)
               and r["line_pass"].startswith("host-fallback")]
    for r in line_fail:
        print(f"FAIL line-{r['config']}-{str(r['dtype']).split('.')[-1]} "
              f"| MBU {r['mbu_pct']} < {r['line_pct']}%", flush=True)
    for r in line_pass:
        print(f"PASS line-{r['config']}-{str(r['dtype']).split('.')[-1]} "
              f"| MBU {r['mbu_pct']} >= {r['line_pct']}%"
              if r["line_pct"] else
              f"PASS line-{r['config']}-{str(r['dtype']).split('.')[-1]}"
              f" | A-line: not worse than native", flush=True)
    for r in line_fb:
        print(f"FAIL linefb-{r['config']}-{str(r['dtype']).split('.')[-1]}"
              f" | profiler unavailable, host-caliber MBU {r['mbu_pct']} "
              f"(line {r['line_pct']}%)", flush=True)

    allow_fallback = bool(os.environ.get(
        "VPE_BENCH_ALLOW_FALLBACK_TIMING"))
    summary = {"bw_fill_gbps": round(bw_fill / 1e9, 1),
               "bw_fill_src": BW_SRC[0],
               "harvest_ok": harvest_ok,
               "allow_fallback_timing": allow_fallback,
               "line_fail": [f"line-{r['config']}-{str(r['dtype']).split('.')[-1]}"
                             for r in line_fail],
               "line_fallback": [
                   f"linefb-{r['config']}-{str(r['dtype']).split('.')[-1]}"
                   for r in line_fb],
               "pass": len(line_pass),
               "total": len(rows),
               "rows": rows}
    with open(os.path.join(out_dir, "bench_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    if rows:
        with open(os.path.join(out_dir, "bench_sweep.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    ok = (harvest_ok or allow_fallback) and not line_fail and not line_fb
    print(f"DONE precision-style verdict: pass={len(line_pass)}/{len(rows)}; "
          f"line_fail={len(line_fail)}; fallback={len(line_fb)}; "
          f"harvest_ok={harvest_ok}", flush=True)
    summary["verdict_ok"] = ok
    return summary


def _run_suite(out_dir):
    """Parent flow: spawn the bench child, then harvest after exit."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"spawning bench child (out={out_dir}) ...", flush=True)
    env = {**os.environ, "VPE_SKIP_OP_SUMMARY_POLL": "1"}
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "_bench"],
        cwd=out_dir, env=env)
    rc = proc.wait()
    print(f"bench child exit={rc} after {time.time() - t0:.0f}s", flush=True)
    raw_path = os.path.join(out_dir, "bench_raw.json")
    if rc != 0 or not os.path.isfile(raw_path):
        return {"harvest_ok": False, "line_fail": ["bench-child"],
                "line_fallback": [], "pass": 0, "total": 0, "rows": [],
                "verdict_ok": False,
                "bw_fill_gbps": 0,
                "detail": f"bench child rc={rc}, no usable bench_raw.json"}
    return harvest(out_dir)


def main(argv):
    if argv and argv[0] == "_bench":
        return run_bench()
    out = "./bench_out"
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    return 0 if _run_suite(out).get("verdict_ok") else 1


@pytest.mark.skipif(not NPU_AVAILABLE, reason="requires an Ascend NPU")
def test_performance_lines(tmp_path):
    """Task-book performance lines (both dtypes).

    Full matrix on a 910B-class A2 machine takes ~20-30 min (16 line
    rows x 2 dtypes, each a 5+30 profiler session, plus the post-exit
    op_summary harvest window). Run explicitly:
        pytest tests/performance/functions/test_bench_vit_pos_embed.py -v
    """
    out = tmp_path / "bench_out"
    out.mkdir()
    summary = _run_suite(str(out))
    assert summary["harvest_ok"], (
        "op_summary harvest failed; device timings unverifiable "
        f"(detail: {summary.get('detail', '')})")
    assert not summary["line_fail"], (
        f"line checks failed: {summary['line_fail']} "
        f"(MBU detail in {out / 'bench_sweep.csv'})")
    assert not summary["line_fallback"], (
        f"profiler-unavailable fallback rows: {summary['line_fallback']}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
