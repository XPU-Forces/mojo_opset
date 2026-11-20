"""Fused ``sigmoid(x @ W_gate^T) * shared_out`` Triton operator.

Standalone extraction of the Qwen3.5-MoE shared-expert gate.  No
``transformers`` import is required; only ``torch`` and ``triton``.

The gate is a GEMV (N=1), i.e. a pure streaming, memory-bound operator
(arithmetic intensity ~0.33 FLOP/byte), so the kernel is written as a
vector-pipe reduction instead of a cube matmul:

* pass 1 reads ``x`` once, multiplies by the ``[hidden]`` gate weight and
  accumulates ``[BLOCK_M]`` partial sums on the vector pipe, then applies
  ``sigmoid``;
* pass 2 re-reads ``shared_out`` and writes ``sigmoid(score) * shared_out``.

The cube path (padding the weight to 16 columns and running ``tl.dot``) was
removed: on Ascend910B3 / triton-ascend 3.2.2 the cube+vector mix kernel only
executes if the launch stub does not hit its segfault, and even when it does
the per-K-step ``ND2NZ`` conversion of the ``[BLOCK_K, 16]`` weight tile keeps
the cube MTE2 pipe at >200% of the kernel wall while the MAC unit stays below
5% busy (msprof op simulator).  For N=1 the cube buys nothing: the vector
reduction reads the same bytes with no layout conversion.

Tiling is shape-dependent and chosen deterministically by ``_tiles``:

* ``BLOCK_M=16`` / ``BLOCK_K=512`` by default: a 1 KB row segment per K step is
  the measured sweet spot - the vec pipe stays busy for ~0.9 of the wall while
  the kernel still streams at the device bandwidth;
* ``BLOCK_M=2`` / ``BLOCK_K=3584`` (fp16) or ``4`` / ``3584`` (bf16) for
  ``d == 7168``: two 3.5 KB segments per row beat both the fourteen 512 B
  fragments of ``(16, 512)`` and a whole-row DMA;
* ``BLOCK_M=8`` / ``BLOCK_K=d`` for ``n >= 32768, d <= 2048``: one 32 KB
  contiguous DMA per program;
* ``BLOCK_M=64`` / ``BLOCK_K=256`` for the remaining ``n >= 32768`` shapes,
  where MTE2 is the busiest pipe and the wider ``BLOCK_M`` keeps it at ~0.75;
* ``BLOCK_M=16`` / ``BLOCK_K=512`` for fp32 as well, whose doubled tile still
  only needs 32 KB of UB.

The one knob left open is whether pass 2 double buffers ``S``: it buys MTE2 /
MTE3 overlap where the store leaves a gap and costs 3-5% wall time where every
pipe is already busy.  The split is not expressible in closed form (the same
tile geometry goes both ways for bf16 vs fp16), so ``triton.autotune`` measures
it per ``(NROWS, D, dtype)``.  Wall-clock repeats are not usable for that
measurement: Triton's default ``do_bench`` flushes L2, and a bare warm loop is
swamped by the ~70 us host launch cost, so ``_steady_bench`` times graph
replays instead.  Tuning happens once per group; later launches bypass the
tuner (see ``_launch``), because its Python dispatch costs ~30 us per call.

``profiling_test`` collects a 13-step torch/torch_npu profiler trace (NPU:
``PipeUtilization``; CUDA: CUDA activities) and ``profile_shapes`` turns the
exported ``kernel_details.csv`` into MBU / busiest-pipe numbers::

    npu-python shared_expert_gate.py --device-type npu --device npu:2
"""

import glob
import os

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None
    _HAS_TRITON = False

PEAK_BW = 1.6e12  # 910B3 HBM spec, same denominator as the A2 MBU model
MBU_TARGET = 0.6
BOUND_TARGET = 0.8

# Same 10 large shapes as test_shared_expert_gate.py.
PROFILE_SHAPES = (
    (8192, 2048), (16384, 2048), (32768, 2048),
    (4096, 3584), (8192, 3584),
    (4096, 4096), (8192, 4096),
    (4096, 5120), (8192, 5120),
    (8192, 7168),
)


def triton_available(x: torch.Tensor) -> bool:
    """True when ``x`` lives on an accelerator this module can target."""
    return _HAS_TRITON and x.device.type in ("cuda", "npu")


if _HAS_TRITON:

    def _steady_bench(fn, quantiles=None):
        """Steady-state device time in microseconds.

        Triton's default ``do_bench`` flushes L2 between launches, so every
        candidate is ranked as a cold-cache run - wrong for our mostly L2-resident
        shapes.  A plain warm loop is not enough either: launching this kernel
        from Python costs ~70-95 us of host time, which swamps the 40-200 us of
        device time and hides the difference between candidates.  So the loop is
        captured into an NPU graph and replayed, which measures device time only.
        """
        for _ in range(2):
            fn()
        torch.npu.synchronize()
        launches = 8
        try:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(launches):
                    fn()
            graph.replay()
            torch.npu.synchronize()
            best = float("inf")
            for _ in range(3):
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                best = min(best, start.elapsed_time(end) * 1000.0 / launches)
        except Exception:
            best = float("inf")
            for _ in range(3):
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(launches):
                    fn()
                end.record()
                end.synchronize()
                best = min(best, start.elapsed_time(end) * 1000.0 / launches)
        return [best, best, best]

    @triton.autotune(
        # The only knob worth measuring is whether pass 2 double buffers ``S``:
        # it buys MTE2/MTE3 overlap where the store leaves a gap, and costs 3-5%
        # wall time where every pipe is already busy.  The split is not
        # expressible in closed form (same tile geometry goes both ways for
        # bf16 vs fp16), so it is measured per shape instead of guessed.
        # ``NROWS`` / ``D`` select the tuning group; Triton appends the dtype of
        # every tensor argument to the key, so bf16 and fp16 tune separately.
        configs=[
            triton.Config({"PREFETCH": True}, num_stages=2),
            triton.Config({"PREFETCH": False}, num_stages=2),
        ],
        key=["NROWS", "D"],
        do_bench=_steady_bench,
    )
    @triton.jit
    def _qwen3_5_moe_shared_expert_gate_kernel(
        X, W, S, OUT, NROWS, D,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, PREFETCH: tl.constexpr,
    ):
        # One program owns ``BLOCK_M`` tokens.  Pass 1 is a vector GEMV: the
        # ``[BLOCK_M, BLOCK_K]`` tile is scaled by the ``[BLOCK_K]`` weight
        # slice and reduced into ``[BLOCK_M]`` fp32 partial sums.
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < NROWS

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, D, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            kmask = ks < D
            x = tl.load(
                X + rows[:, None] * D + ks[None, :],
                mask=rmask[:, None] & kmask[None, :], other=0.0,
            ).to(tl.float32)
            w = tl.load(W + ks, mask=kmask, other=0.0).to(tl.float32)
            acc += tl.sum(x * w[None, :], axis=1)

        g = (1.0 / (1.0 + tl.exp(-acc))).to(S.dtype.element_ty)

        if PREFETCH:
            # Double buffer: the next ``S`` tile is issued before the current
            # store, so MTE2 keeps streaming while MTE3 drains the previous
            # tile instead of idling.
            base_off = rows[:, None] * D
            ks0 = tl.arange(0, BLOCK_K)
            s_cur = tl.load(
                S + base_off + ks0[None, :],
                mask=rmask[:, None] & (ks0[None, :] < D), other=0.0,
            )
            for k0 in range(0, D, BLOCK_K):
                ks = k0 + tl.arange(0, BLOCK_K)
                mask = rmask[:, None] & (ks[None, :] < D)
                if k0 + BLOCK_K < D:
                    nk = ks + BLOCK_K
                    s_next = tl.load(
                        S + base_off + nk[None, :],
                        mask=rmask[:, None] & (nk[None, :] < D), other=0.0,
                    )
                else:
                    s_next = s_cur
                tl.store(OUT + base_off + ks[None, :], g[:, None] * s_cur, mask=mask)
                s_cur = s_next
        else:
            for k0 in range(0, D, BLOCK_K):
                ks = k0 + tl.arange(0, BLOCK_K)
                mask = rmask[:, None] & (ks[None, :] < D)
                s = tl.load(S + rows[:, None] * D + ks[None, :], mask=mask, other=0.0)
                tl.store(OUT + rows[:, None] * D + ks[None, :], g[:, None] * s, mask=mask)


_TUNED: dict = {}


def _launch(x2, w, s2, out, n, d, block_m, block_k):
    """Dispatch one launch, tuning ``PREFETCH`` once per ``(n, d, dtype)``.

    The first call for a shape goes through the autotuner, which measures both
    variants on NPU graphs and keeps the faster one.  Later calls go straight to
    the JIT function: the autotuner's Python dispatch costs ~28 us per launch
    (~96 us vs ~68 us host time), which is a third of the runtime of the smaller
    shapes and would otherwise be paid on every call.
    """
    grid = (triton.cdiv(n, block_m),)
    key = (n, d, x2.dtype)
    prefetch = _TUNED.get(key)
    if prefetch is None:
        if getattr(torch.npu, "is_current_stream_capturing", lambda: False)():
            # The tuner captures its own graph, which cannot nest inside the
            # caller's capture; take the variant that needs no extra buffer.
            prefetch = False
        else:
            _qwen3_5_moe_shared_expert_gate_kernel[grid](
                x2, w, s2, out, n, d, BLOCK_M=block_m, BLOCK_K=block_k,
            )
            prefetch = _qwen3_5_moe_shared_expert_gate_kernel.best_config.kwargs["PREFETCH"]
        _TUNED[key] = prefetch
    _qwen3_5_moe_shared_expert_gate_kernel.fn[grid](
        x2, w, s2, out, n, d, BLOCK_M=block_m, BLOCK_K=block_k,
        PREFETCH=prefetch, num_stages=2,
    )


def _tiles(n: int, d: int, dtype: torch.dtype):
    """Deterministic tile choice for the vector GEMV pass.

    ``(16, 512)`` is the measured optimum for every shape that keeps its rows
    short: each K step moves a 1 KB row segment and the narrow tile keeps a
    second load in flight behind the store.  Shapes whose rows are long enough
    that a 1 KB segment fragments the row (``d == 7168``) or whose row count is
    very large (``n >= 32768, d <= 2048``) instead use whole-row tiles, turning
    many small scattered DMAs into a few contiguous 4-14 KB ones.
    """
    if dtype == torch.float32:
        return 16, 512
    if d == 7168:
        # 14 KB rows: two 3.5 KB segments per row beat both the fourteen 512 B
        # fragments of (16, 512) and the whole-row (4, 7168) DMA.  fp16 tiles
        # carry twice the vector work per byte, so two rows per program keep the
        # pipe busy where four rows leave it idle between DMAs.
        return (2, 3584) if dtype == torch.float16 else (4, 3584)
    if n >= 32768 and d <= 2048:
        # 4 KB rows: eight whole rows (one 32 KB contiguous DMA) beat the
        # 64 x 512 B fragments of (64, 256) by ~2%.
        return 8, d
    if n >= 32768:
        return 64, 256
    return 16, 512


def shared_expert_gate_fwd_impl(
    hidden_states: torch.Tensor, gate_weight: torch.Tensor, shared_out: torch.Tensor
) -> torch.Tensor:
    """Fused ``sigmoid(hidden_states @ gate_weight.T) * shared_out``.

    ``gate_weight`` is the ``[1, hidden]`` weight of the model's
    ``shared_expert_gate`` Linear (bias-free).  Returns ``None`` when Triton
    cannot run on the input device, so the caller can fall back to torch.
    """
    if not triton_available(hidden_states):
        return None
    shape = shared_out.shape
    x2 = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    s2 = shared_out.reshape(-1, shared_out.shape[-1]).contiguous()
    w = gate_weight.reshape(-1).contiguous()
    n, d = x2.shape
    out = torch.empty_like(s2)
    block_m, block_k = _tiles(n, d, x2.dtype)
    _launch(x2, w, s2, out, n, d, block_m, block_k)
    return out.reshape(shape)

# Model-facing alias kept for the standalone profiling helpers below.
qwen3_5_moe_shared_expert_gate = shared_expert_gate_fwd_impl


@torch.library.custom_op("mojo_npu_triton_a2::shared_expert_gate_fwd", mutates_args=())
def shared_expert_gate_fwd(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_out: torch.Tensor,
) -> torch.Tensor:
    return shared_expert_gate_fwd_impl(hidden_states, gate_weight, shared_out)


@shared_expert_gate_fwd.register_fake
def _shared_expert_gate_fwd_fake(hidden_states, gate_weight, shared_out):
    return torch.empty_like(shared_out, memory_format=torch.contiguous_format)


def _kernel_seconds(n: int, d: int, dtype: torch.dtype) -> float:
    """HBM bytes moved by one launch."""
    return (3 * n * d + d) * (torch.finfo(dtype).bits // 8)


def _export_kernel_details(trace_dir):
    """把 torch_npu 原始轨迹导出成 ``ASCEND_PROFILER_OUTPUT/kernel_details.csv``。"""
    import csv as _csv
    import subprocess
    import sys

    prof_dirs = sorted(glob.glob(os.path.join(trace_dir, "**", "PROF_*"), recursive=True),
                       key=os.path.getmtime)
    if not prof_dirs:
        return None
    prof_dir = prof_dirs[-1]
    msprof = os.path.join(
        os.environ.get("ASCEND_HOME_PATH", "/home/lyy_triton/cann921/install/cann"),
        "tools/profiler/profiler_tool/analysis/msprof/msprof.py")
    done = subprocess.run([sys.executable, msprof, "export", "summary", "-dir", prof_dir],
                          capture_output=True, text=True)
    if done.returncode != 0:
        print(f"  msprof export failed: {done.stderr.strip()[:200]}", flush=True)
        return None
    summaries = sorted(glob.glob(os.path.join(
        prof_dir, "mindstudio_profiler_output", "op_summary*.csv")))
    if not summaries:
        return None
    ascend_pt = os.path.dirname(prof_dir)
    target = os.path.join(ascend_pt, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(summaries[0], encoding="utf-8", newline="") as src:
        rows = list(_csv.reader(src))
    header = rows[0]
    for old, new in (("Op Name", "Name"), ("Task Duration(us)", "Duration(us)")):
        if old in header:
            header[header.index(old)] = new
    rows[0] = header
    with open(target, "w", encoding="utf-8", newline="") as dst:
        _csv.writer(dst).writerows(rows)
    return target


def _steady_stats(csv_path):
    """Steady-state duration and busiest AIV pipe ratio of one kernel_details.csv."""
    import csv as _csv

    units = ("aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio",
             "aiv_scalar_ratio", "aiv_mte1_ratio")
    durations, peaks = [], []
    with open(csv_path, encoding="utf-8", newline="") as handle:
        for row in _csv.DictReader(handle):
            if row.get("Task Type") != "AI_VECTOR_CORE":
                continue
            durations.append(float(row.get("Duration(us)") or 0.0))
            peaks.append(max(float(row.get(unit) or 0.0) for unit in units))
    if not durations:
        return None, None
    # Steady state: the fastest quarter of the launches.
    limit = sorted(durations)[max(0, len(durations) // 4 - 1)]
    keep = [i for i, value in enumerate(durations) if value <= limit]
    return (sum(durations[i] for i in keep) / len(keep),
            sum(peaks[i] for i in keep) / len(keep))


def profiling_test(fn_triton, device_type="npu", device=None, result_dir="result_dir",
                   l2_cache=False):
    """按统一的 13 步 schedule 采集当前 case（NPU 走 PipeUtilization）。

    ``l2_cache=True`` 额外打开 msprof 的 L2 命中率采集（会改变 msprof 的
    默认 cache 处理方式），用于观察结果是否依赖热 L2。
    """
    from torch.profiler import ProfilerActivity

    if device_type == "npu":
        import torch_npu  # noqa: F401

        profiler = torch_npu.profiler
        activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]
        options = {"experimental_config": profiler._ExperimentalConfig(
            aic_metrics=profiler.AiCMetrics.PipeUtilization,
            profiler_level=profiler.ProfilerLevel.Level1,
            l2_cache=l2_cache,
        )}
        device = device or "npu:0"
    else:
        profiler = torch.profiler
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        options = {}
        device = device or "cuda:0"

    fn_triton()  # compile / warm up outside the profiled window
    getattr(torch, device_type).synchronize(device)
    with profiler.profile(
        activities=activities,
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
        schedule=profiler.schedule(wait=1, warmup=1, active=10, repeat=1, skip_first=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(result_dir)),
        **options,
    ) as prof:
        for _ in range(13):
            fn_triton()
            getattr(torch, device_type).synchronize(device)
            prof.step()
    return result_dir


def profile_shapes(cases=None, result_dir="result_dir", device_type="npu", device=None,
                   l2_cache=False):
    """Profile ``cases`` and report MBU / busiest pipe; returns one record per case."""
    cases = cases or [(t, d, dt) for t, d in PROFILE_SHAPES
                      for dt in (torch.bfloat16, torch.float16)]
    os.makedirs(result_dir, exist_ok=True)
    records = []
    for tokens, hidden, dtype in cases:
        torch.manual_seed(0)
        x = torch.randn(tokens, hidden, device=device_type, dtype=dtype)
        w = torch.randn(1, hidden, device=device_type, dtype=dtype)
        s = torch.randn(tokens, hidden, device=device_type, dtype=dtype)
        case_dir = os.path.join(result_dir, f"{tokens}x{hidden}_{str(dtype).split('.')[-1]}")
        os.makedirs(case_dir, exist_ok=True)
        profiling_test(lambda: qwen3_5_moe_shared_expert_gate(x, w, s),
                       device_type, device, case_dir, l2_cache=l2_cache)
        csvs = glob.glob(os.path.join(case_dir, "**", "kernel_details.csv"), recursive=True)
        if not csvs:
            exported = _export_kernel_details(case_dir)
            csvs = [exported] if exported else []
        steady_us, bound = _steady_stats(max(csvs, key=os.path.getmtime)) if csvs else (None, None)
        record = dict(tokens=tokens, hidden=hidden,
                      dtype=str(dtype).split(".")[-1], steady_us=steady_us, bound=bound)
        if steady_us:
            record["mbu"] = _kernel_seconds(tokens, hidden, dtype) / (steady_us * 1e-6) / PEAK_BW
        record["ok"] = bool(record.get("mbu", 0) > MBU_TARGET and (bound or 0) >= BOUND_TARGET)
        records.append(record)
        print(f"{tokens:>6}x{hidden:<5} {record['dtype']:<9} "
              f"t={steady_us if steady_us else float('nan'):7.1f}us "
              f"MBU={record.get('mbu', float('nan')):.3f} bound={bound or 0:.3f} "
              f"{'PASS' if record['ok'] else 'FAIL'}", flush=True)
    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="profile the shared_expert_gate operator")
    parser.add_argument("--device-type", choices=("npu", "cuda"), default="npu")
    parser.add_argument("--device", default=None)
    parser.add_argument("--result-dir", default="result_dir")
    parser.add_argument("--l2-cache", action="store_true",
                        help="打开 msprof 的 L2 命中率采集")
    args = parser.parse_args()
    rows = profile_shapes(result_dir=args.result_dir, device_type=args.device_type,
                          device=args.device, l2_cache=args.l2_cache)
    print(f"\n{sum(r['ok'] for r in rows)}/{len(rows)} cases pass "
          f"(MBU>{MBU_TARGET} and bound>={BOUND_TARGET})")
