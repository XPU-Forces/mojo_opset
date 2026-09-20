"""Compare two pytest performance JSON artifacts; standard library only."""

import argparse
import json
import math
import statistics
import sys

from pathlib import Path

METRICS = ("profiler_span_us", "profiler_sum_us", "event_us", "e2e_us")
MEMORY_FIELDS = ("baseline_allocated_bytes", "peak_allocated_bytes", "incremental_peak_bytes")
MEMORY_METRICS = ("peak_allocated_bytes", "incremental_peak_bytes")


def validate(report):
    if report.get("schema_version") != 2 or report.get("exitstatus") != 0:
        raise ValueError("Only successful schema-version-2 runs can be compared; regenerate old baselines")
    if report.get("skipped"):
        raise ValueError("Run contains skipped performance cases; do not promote an incomplete baseline")
    for field in ("environment", "measurement"):
        if not isinstance(report.get(field), dict) or not report[field]:
            raise ValueError(f"Report is missing {field}")
    repeats = report["measurement"].get("repeats", 0)
    memory_contract = report["measurement"].get("memory")
    if memory_contract not in (None, "allocator_peak_after_timing_v1"):
        raise ValueError("Unknown memory measurement contract")
    timers = report["measurement"].get("timers", [])
    reduction = report["measurement"].get("reduction", "span")
    if not timers or len(set(timers)) != len(timers) or set(timers) - {"profiler", "event", "e2e"}:
        raise ValueError("Invalid timer selection")
    if reduction not in ("span", "sum"):
        raise ValueError("Invalid profiler reduction")
    expected = {f"profiler_{reduction}_us" if name == "profiler" else f"{name}_us" for name in timers}
    if not isinstance(repeats, int) or repeats < 3:
        raise ValueError("Reports require at least 3 samples per metric")
    rows = {}
    for row in report.get("cases", []):
        name = row.get("id")
        if not isinstance(name, str) or not name or name in rows:
            raise ValueError(f"Missing or duplicate case id: {name!r}")
        if not isinstance(row.get("parameters"), dict):
            raise ValueError(f"Missing workload parameters for {name}")
        if set(row.get("metrics", {})) != expected:
            raise ValueError(f"Missing or unexpected metrics for {name}")
        memory = row.get("memory")
        if memory_contract:
            if not isinstance(memory, dict) or set(memory) != set(MEMORY_FIELDS):
                raise ValueError(f"Missing or incomplete memory report for {name}")
            if any(type(value) is not int or value < 0 for value in memory.values()):
                raise ValueError(f"Invalid memory bytes for {name}")
            if (memory["peak_allocated_bytes"] < memory["baseline_allocated_bytes"]
                    or memory["incremental_peak_bytes"] !=
                    memory["peak_allocated_bytes"] - memory["baseline_allocated_bytes"]):
                raise ValueError(f"Inconsistent memory peaks for {name}")
        elif memory is not None:
            raise ValueError("Memory report requires a measurement contract")
        for metric in expected:
            samples = row.get("metrics", {}).get(metric, {}).get("samples", [])
            if len(samples) != repeats or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
                for x in samples
            ):
                raise ValueError(f"Invalid or incomplete samples: {name}/{metric}")
        rows[name] = row
    if not rows:
        raise ValueError("Report contains no performance cases")
    return rows


def compare_reports(baseline, current, threshold_percent=5.0):
    if not math.isfinite(threshold_percent) or threshold_percent <= 0:
        raise ValueError("threshold must be finite and positive")
    old, new = validate(baseline), validate(current)
    for field in ("environment", "measurement"):
        if baseline[field] != current[field]:
            keys = sorted(set(baseline[field]) | set(current[field]))
            changed = [key for key in keys if baseline[field].get(key) != current[field].get(key)]
            raise ValueError(f"Incompatible {field}: {', '.join(changed)}")
    rows = []
    for name in sorted(old.keys() | new.keys()):
        if name not in old or name not in new:
            rows.append({"id": name, "metric": "all", "status": "new" if name not in old else "missing"})
            continue
        before, after = old[name], new[name]
        # Changing the selected implementation is a legitimate optimization.
        workload = lambda row: {key: value for key, value in row["parameters"].items() if key != "implementation"}
        if workload(before) != workload(after):
            raise ValueError(f"Workload changed under the same case id: {name}")
        for metric in sorted(before["metrics"]):
            a, b = (row["metrics"][metric]["samples"] for row in (before, after))
            old_median, new_median = statistics.median(a), statistics.median(b)
            noise = 3 * max(statistics.median(abs(x - old_median) for x in a),
                            statistics.median(abs(x - new_median) for x in b))
            delta = 100 * (new_median / old_median - 1)
            status = "stable"
            if noise > old_median * threshold_percent / 100:
                status = "noisy"
            elif new_median > old_median * (1 + threshold_percent / 100):
                status = "regression"
            elif new_median < old_median * (1 - threshold_percent / 100):
                status = "improvement"
            rows.append({
                "id": name, "metric": metric, "status": status,
                "baseline_us": old_median, "current_us": new_median, "delta_percent": delta,
                "speedup": old_median / new_median, "noise_us": noise,
                "implementations": [before["parameters"].get("implementation"),
                                    after["parameters"].get("implementation")],
            })
        if baseline["measurement"].get("memory"):
            for metric in MEMORY_METRICS:
                a, b = before["memory"][metric], after["memory"][metric]
                delta = 100 * (b / a - 1) if a else (0.0 if b == 0 else None)
                status = ("regression" if b > a * (1 + threshold_percent / 100) else
                          "improvement" if b < a * (1 - threshold_percent / 100) else "stable")
                rows.append({"id": name, "metric": metric, "status": status,
                             "baseline_bytes": a, "current_bytes": b,
                             "delta_bytes": b - a, "delta_percent": delta,
                             "implementations": [before["parameters"].get("implementation"),
                                                 after["parameters"].get("implementation")]})
    return rows


def markdown(rows, baseline, current):
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    old = baseline.get("revision", {}).get("commit") or "unknown"
    new = current.get("revision", {}).get("commit") or "unknown"
    lines = ["# Mojo performance comparison", "", f"Baseline: `{old}`; current: `{new}`.", "",
             "Positive latency delta means slower. Profiler span/sum, stream event and synchronized e2e "
             "are separate metrics.",
             "Noise uses a 3×MAD heuristic, not a statistical confidence interval.", "",
             "| Case | Metric | Impl (old → new) | Baseline µs | Current µs | Latency Δ | Speedup | Status |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        if row["metric"] in MEMORY_METRICS:
            continue
        if row["status"] in ("new", "missing"):
            lines.append(f"| {escape(row['id'])} | all | — | — | — | — | — | {row['status']} |")
        else:
            impl = " → ".join(str(x) for x in row["implementations"])
            lines.append(f"| {escape(row['id'])} | {row['metric']} | {escape(impl)} | "
                         f"{row['baseline_us']:.3f} | {row['current_us']:.3f} | "
                         f"{row['delta_percent']:+.2f}% | {row['speedup']:.3f}× | {row['status']} |")
    memory_rows = [row for row in rows if row["metric"] in MEMORY_METRICS]
    if memory_rows:
        lines += ["", "## Device memory", "",
                  "Allocator-tracked allocated memory; positive delta means more memory. "
                  "Absolute peak includes the input/state pool; incremental peak subtracts its live baseline. "
                  "The live baseline is retained only in JSON. A zero baseline has no percentage increase.", "",
                  "| Case | Metric | Baseline MiB | Current MiB | Δ MiB | Δ % | Status |",
                  "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
        for row in memory_rows:
            delta = "N/A" if row["delta_percent"] is None else f"{row['delta_percent']:+.2f}%"
            lines.append(f"| {escape(row['id'])} | {row['metric']} | "
                         f"{row['baseline_bytes'] / 1024**2:.3f} | {row['current_bytes'] / 1024**2:.3f} | "
                         f"{row['delta_bytes'] / 1024**2:+.3f} | {delta} | {row['status']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--threshold-percent", type=float, default=5.0)
    parser.add_argument("--output", type=Path, help="Write Markdown report (also printed to stdout)")
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() in (args.baseline.resolve(), args.current.resolve()):
            raise ValueError("Markdown output must not overwrite an input report")
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        current = json.loads(args.current.read_text(encoding="utf-8"))
        rows = compare_reports(baseline, current, args.threshold_percent)
        text = markdown(rows, baseline, current)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        print(text, end="")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Cannot compare performance reports: {error}", file=sys.stderr)
        return 2
    statuses = {row["status"] for row in rows}
    if "missing" in statuses or not any(row["metric"] in METRICS for row in rows):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
