#!/usr/bin/env python3
"""Parse perf log and extract operator avg time from NPU profiling data.

Usage:
    python scripts/parse_perf_op_stats.py perf.log [-o output.csv]
"""

import argparse
import csv
import os
import re
import sys

# regex: pytest node IDs — mojo_opset/tests/.../test_kv_cache.py::test_store_paged_kv[2-2-128-128-...]
CASE_NODE_RE = re.compile(r"^(mojo_opset/\S+?::test[\w\[\]-]*)")
# regex: Profile dir = ./npu_profiling/a92518f4fe3b_..._ascend_pt
PROFILE_DIR_RE = re.compile(r"Profile dir = (\S+)")

TARGET_OP_NAMES = ["_store_paged_kv_cache_kernel", "ScatterPaKvCache"]


def parse_log(log_path: str) -> list[tuple[str, str]]:
    case_dirs: list[tuple[str, str]] = []
    current_case: str | None = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            case_match = CASE_NODE_RE.search(line)
            if case_match:
                current_case = case_match.group(1)
                continue
            if current_case is not None:
                profile_match = PROFILE_DIR_RE.search(line)
                if profile_match:
                    case_dirs.append((current_case, profile_match.group(1)))
                    current_case = None

    return case_dirs


def resolve_profile_dir(profile_dir: str, log_dir: str) -> str:
    if os.path.isabs(profile_dir):
        return profile_dir
    return os.path.normpath(os.path.join(log_dir, profile_dir))


def get_op_avg_time(profile_dir: str, log_dir: str) -> tuple[str | None, float | None]:
    resolved = resolve_profile_dir(profile_dir, log_dir)
    csv_path = os.path.join(resolved, "ASCEND_PROFILER_OUTPUT", "op_statistic.csv")

    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found", file=sys.stderr)
        return None, None

    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            op_type = row.get("OP Type", "").strip()
            if op_type in TARGET_OP_NAMES:
                raw = row.get("Avg Time(us)", "").strip()
                try:
                    return op_type, float(raw)
                except ValueError:
                    print(f"  WARNING: bad Avg Time '{raw}' for '{op_type}' in {csv_path}", file=sys.stderr)
                    return op_type, None

    print(f"  WARNING: no target operator in {csv_path}", file=sys.stderr)
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Parse perf log and extract operator performance.")
    parser.add_argument("log_file", help="Path to perf log file")
    parser.add_argument("-o", "--output", default="perf_op_summary.csv", help="Output CSV path")
    args = parser.parse_args()

    log_path = os.path.abspath(args.log_file)
    log_dir = os.path.dirname(log_path)

    if not os.path.exists(log_path):
        print(f"ERROR: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    case_dirs = parse_log(log_path)
    if not case_dirs:
        print("No (case_node, profile_dir) pairs found in log.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(case_dirs)} test case(s).")

    results = []
    for case_node, profile_dir in case_dirs:
        op_name, avg_time = get_op_avg_time(profile_dir, log_dir)
        results.append({
            "case_node": case_node,
            "op_name": op_name or "N/A",
            "avg_time_us": avg_time if avg_time is not None else "N/A",
        })

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_node", "op_name", "avg_time_us"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Written {len(results)} entries to {args.output}")


if __name__ == "__main__":
    main()
