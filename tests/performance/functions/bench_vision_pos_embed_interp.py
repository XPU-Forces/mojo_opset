"""Task-specific Duration-sum benchmark, separate from Mojo's span timer.

Run from the repository root with:
    python -m tests.performance.functions.bench_vision_pos_embed_interp --output /path/to/new/run
This implements the published measurement description; the official attachment
named in the taskbook was not included in its download.
"""

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time

from datetime import datetime
from datetime import timezone
from pathlib import Path

import torch
import torch_npu

from mojo_opset import functions as F
from mojo_opset.utils.target import detect_target

CASES = [
    ("small", [(1, 8, 16)], 1152, None),
    ("medium", [(1, 32, 48)], 1152, 0.60),
    ("large", [(1, 60, 80)], 1152, 0.60),
    ("short_video", [(3, 16, 16)], 1152, 0.60),
    ("large_video", [(16, 100, 168)], 1152, 0.60),
    ("multi_image", [(1, 8, 16), (1, 32, 48), (3, 16, 16)], 1152, 0.60),
    ("tail", [(3, 60, 80)], 1000, 0.50),
]


def profile(fn, directory, warmup=5, repeats=30):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        data_simplification=False,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
        experimental_config=config,
    ):
        for _ in range(repeats):
            fn()
            torch.npu.synchronize()
    files = list(directory.rglob("kernel_details.csv"))
    if len(files) != 1:
        raise RuntimeError(f"expected one device kernel_details.csv, found {files}")
    with files[0].open() as stream:
        rows = list(csv.DictReader(stream))
    durations = [float(row["Duration(us)"]) for row in rows]
    if not durations or min(durations) <= 0:
        raise RuntimeError("profiler produced no valid positive device durations")
    return {
        "total_device_us": sum(durations) / repeats,
        "kernel_count_per_call": len(rows) / repeats,
        "kernel_durations_us": durations,
        "csv": str(files[0]),
    }


def calibrate(directory, elements=256 * 1024 * 1024):
    source = torch.empty(elements, device="npu", dtype=torch.bfloat16)
    target = torch.empty_like(source)
    source.fill_(0.25)
    fill = profile(lambda: target.fill_(0.5), directory / "fill")
    copy = profile(lambda: target.copy_(source), directory / "copy")
    # Each contiguous BF16 probe is one kernel. Fail instead of inventing
    # per-call samples if a future runtime changes its execution structure.
    if fill["kernel_count_per_call"] != 1 or copy["kernel_count_per_call"] != 1:
        raise RuntimeError("calibration requires one fill/copy kernel per call to compute the 30-sample median")
    write_bytes = elements * 2
    fill_us = statistics.median(fill["kernel_durations_us"])
    copy_us = statistics.median(copy["kernel_durations_us"])
    return {
        "elements": elements,
        "dtype": "bfloat16",
        "fill_bytes": write_bytes,
        "copy_bytes": write_bytes * 2,
        "fill_GBs": write_bytes / fill_us / 1000,
        "copy_GBs": write_bytes * 2 / copy_us / 1000,
        "fill_median_us": fill_us,
        "copy_median_us": copy_us,
        "fill": fill,
        "copy": copy,
        "denominator": "fill_ pure write",
    }


def synchronized_wall_us(fn, repeats=30):
    samples = []
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1000)
    return {"mean": statistics.mean(samples), "median": statistics.median(samples), "samples": samples}


def benchmark(output, selected=None, device=0):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(device)
    smi = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True).stdout
    (output / "npu-smi.txt").write_text(smi)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "task-description development benchmark; official suite unavailable",
        "measurement": {
            "warmup": 5,
            "repeats": 30,
            "primary": "sum of all NPU kernel Duration(us) / 30",
            "cache": "warm; no explicit cache flushing",
            "e2e": "mean of 30 synchronized wall samples",
        },
        "environment": {
            "target": str(detect_target()),
            "device": torch.npu.get_device_name(device),
            "python": platform.python_version(),
            "ascend_home": os.environ.get("ASCEND_HOME_PATH"),
            "packages": {n: importlib.metadata.version(n) for n in ("torch", "torch-npu", "triton-ascend")},
        },
        "cases": [],
        "expected_cases": [name for name, _, _, _ in CASES if not selected or name in selected],
    }
    report["calibration"] = calibrate(output / "calibration")
    try:
        for name, grids, d, threshold in CASES:
            if selected and name not in selected:
                continue
            torch.manual_seed(42)
            weight = torch.randn(48 * 48, d, device="npu", dtype=torch.bfloat16) * 0.25
            implementations = {}
            for impl in ("torch", "triton"):
                if len(grids) == 1:
                    call = lambda: F.vision_pos_embed_interp(
                        weight, *grids[0], 48, 2, weight.dtype, implementation=impl
                    )
                else:
                    call = lambda: F.vision_pos_embed_interp_batch(
                        weight, grids, 48, 2, weight.dtype, implementation=impl
                    )
                measured = profile(call, output / name / impl)
                measured["e2e_us"] = synchronized_wall_us(call)
                implementations[impl] = measured
            native, fused = implementations["torch"], implementations["triton"]
            byte_count = sum(2 * (48 * 48 * d + t * h * w * d) for t, h, w in grids)
            mbu = byte_count / fused["total_device_us"] / report["calibration"]["fill_GBs"] / 1000
            passed = (
                (mbu >= threshold)
                if threshold is not None
                else (
                    fused["total_device_us"] <= native["total_device_us"]
                    and fused["e2e_us"]["mean"] <= native["e2e_us"]["mean"]
                )
            )
            row = {
                "name": name,
                "grids": grids,
                "hidden": d,
                "dtype": "bfloat16",
                "read_write_bytes": byte_count,
                "MBU": mbu,
                "required_MBU": threshold,
                "passed": passed,
                "single_launch": fused["kernel_count_per_call"] == 1,
                "native": native,
                "triton": fused,
                "speedup": native["total_device_us"] / fused["total_device_us"],
            }
            report["cases"].append(row)
            (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: row[k] for k in ("name", "MBU", "passed", "single_launch", "speedup")}), flush=True)
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        report["complete"] = len(report["cases"]) == len(report["expected_cases"])
        report["passed"] = (
            bool(report["cases"])
            and report["complete"]
            and not report.get("error")
            and all(row["passed"] and row["single_launch"] for row in report["cases"])
        )
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", help="comma-separated case names; default all seven")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    benchmark(args.output, set(args.cases.split(",")) if args.cases else None, args.device)


if __name__ == "__main__":
    main()
