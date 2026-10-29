"""Performance acceptance for fused gated normalization on Ascend."""

import csv
import glob
import json
import os
import subprocess
import time

from functools import partial
from pathlib import Path

import pytest
import torch
import torch_npu

from mojo_opset import functions as F
from triton.backends.ascend.testing import do_bench_npu_profiler

DTYPE = torch.bfloat16
ELEMENT_SIZE = DTYPE.itemsize


CASES = [
    ("gdn-256-n128-swish", 256, 128, "gdn", "swish", None, 0, None),
    ("gdn-2560-n128-swish", 2560, 128, "gdn", "swish", None, 0, None),
    ("gdn-32768-n128-swish", 32768, 128, "gdn", "swish", None, 0, 60.0),
    ("gdn-32768-n128-sigmoid", 32768, 128, "gdn", "sigmoid", None, 0, 60.0),
    ("gdn-131072-n128-swish", 131072, 128, "gdn", "swish", None, 0, 60.0),
    ("qk-65536-n128", 65536, 128, "none", "swish", None, 0, 60.0),
    ("group-1024-n2048", 1024, 2048, "group", "swish", 128, 0, None),
    ("group-8192-n2048", 8192, 2048, "group", "swish", 128, 0, 60.0),
    ("quant-32768-n512", 32768, 512, "none", "swish", None, 8, 60.0),
    ("gdn-524288-n128-swish", 524288, 128, "gdn", "swish", None, 0, 60.0),
    ("tail-32771-n192-swish", 32771, 192, "2d", "swish", None, 0, 50.0),
]


def _make_inputs(case, device):
    _, M, N, gate_kind, _, _, quant_heads, _ = case
    torch.manual_seed(20261030)
    x = torch.randn((M, N), device=device, dtype=DTYPE)
    weight = torch.randn((N,), device=device, dtype=DTYPE)
    z = None
    if gate_kind == "gdn":
        heads = 8
        padded = torch.empty((M // heads, heads, N + 32), device=device, dtype=DTYPE)
        padded.normal_()
        z = padded[..., :N]
    elif gate_kind == "2d":
        z = torch.randn((M, N), device=device, dtype=DTYPE)
    args = (x, weight, None, 1e-6)
    kwargs = {
        "z": z,
        "is_rms_norm": True,
        "activation": case[4],
        "quant_heads": quant_heads,
    }
    if gate_kind == "group":
        kwargs["group_size"] = case[5]
        kwargs["norm_before_gate"] = False
    return args, kwargs


def _traffic_bytes(case):
    _, M, N, gate_kind, _, _, quant_heads, _ = case
    if quant_heads:
        return 2 * M * N * ELEMENT_SIZE + M * N + 4 * M // quant_heads + 2 * N * ELEMENT_SIZE
    rows = 3 if gate_kind != "none" else 2
    return (rows * M * N + 2 * N) * ELEMENT_SIZE


def _find_kernel_csvs(root):
    return glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)


def _sum_kernel_us(root):
    paths = _find_kernel_csvs(root)
    if not paths:
        raise RuntimeError(f"kernel_details.csv not found under {root}")
    total = 0.0
    rows = 0
    for path in paths:
        with open(path, newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                duration = row.get("Duration(us)") or row.get("Duration(μs)")
                if duration is None or not duration.strip():
                    continue
                try:
                    total += float(duration)
                    rows += 1
                except ValueError:
                    continue
    if rows == 0:
        raise RuntimeError(f"no valid kernel durations in {paths}")
    return total, paths


def _profile_call(fn, output_dir, warmup=5, active=30):
    if os.path.exists(output_dir):
        raise FileExistsError(output_dir)
    elapsed_ms = do_bench_npu_profiler(
        fn,
        warmup=warmup,
        active=active,
        prof_dir=output_dir,
        keep_res=True,
    )
    if elapsed_ms == float("inf"):
        raise RuntimeError(f"profiler collection failed under {output_dir}")
    return elapsed_ms * 1e3, _find_kernel_csvs(output_dir)


def _e2e_us(fn, warmup=5, active=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(active):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1e6 / active


def _calibrate(device, output_root):
    n_elements = 128 * 1024 * 1024
    source = torch.empty(n_elements, device=device, dtype=DTYPE)
    target = torch.empty_like(source)
    fill_us, _ = _profile_call(
        lambda: source.fill_(1.0),
        str(output_root / "calibration-fill"),
    )
    copy_us, _ = _profile_call(
        lambda: target.copy_(source),
        str(output_root / "calibration-copy"),
    )
    tensor_bytes = n_elements * ELEMENT_SIZE
    return {
        "fill_us": fill_us,
        "fill_bandwidth_GBps": tensor_bytes / (fill_us * 1e-6) / 1e9,
        "copy_us": copy_us,
        "copy_bandwidth_GBps": 2 * tensor_bytes / (copy_us * 1e-6) / 1e9,
    }


def test_layernorm_gated_performance(pytestconfig):
    implementation = pytestconfig.getoption("--mojo-implementation") or "triton"
    device = f"npu:{pytestconfig.getoption('--perf-device')}"
    output_root = Path(pytestconfig.getoption("--perf-output")).parent / "layernorm_gated_raw"
    output_root.mkdir(parents=True, exist_ok=True)

    calibration = _calibrate(device, output_root)
    bandwidth_bytes_s = calibration["copy_bandwidth_GBps"] * 1e9
    records = []
    failures = []
    for index, case in enumerate(CASES):
        if case[6] and "950" not in torch.npu.get_device_name(0):
            records.append(
                {
                    "case": case[0],
                    "status": "skipped",
                    "reason": "FP8 quant capability is formally required only on Ascend 950",
                }
            )
            continue
        args, kwargs = _make_inputs(case, device)
        fused = partial(
            F.layernorm_gated,
            *args,
            **kwargs,
            implementation=implementation,
        )
        fused_us, csv_paths = _profile_call(fused, str(output_root / f"case-{index:02d}"))
        traffic = _traffic_bytes(case)
        mbu = traffic / (fused_us * 1e-6) / bandwidth_bytes_s
        record = {
            "case": case[0],
            "M": case[1],
            "N": case[2],
            "traffic_bytes": traffic,
            "traffic_MB": traffic / 1e6,
            "fused_us": fused_us,
            "MBU": mbu,
            "threshold": case[7],
            "profiler_csv": [os.path.relpath(path, output_root) for path in csv_paths],
        }
        if case[7] is not None and mbu < case[7] / 100:
            failures.append(f"{case[0]}: MBU={mbu:.2%} < {case[7]:g}%")
        if traffic < 16 * 1024 * 1024:
            fused_e2e = _e2e_us(fused)
            eager = partial(
                F.layernorm_gated,
                *args,
                **kwargs,
                implementation="torch",
            )
            eager_us, _ = _profile_call(
                eager,
                str(output_root / f"case-{index:02d}-torch"),
            )
            eager_e2e = _e2e_us(eager)
            record.update(
                e2e_fused_us=fused_e2e,
                eager_us=eager_us,
                eager_e2e_us=eager_e2e,
            )
            if fused_us > eager_us:
                failures.append(f"{case[0]}: fused {fused_us:.2f}us > eager {eager_us:.2f}us")
            if fused_e2e > eager_e2e:
                failures.append(
                    f"{case[0]}: fused e2e {fused_e2e:.2f}us > eager e2e {eager_e2e:.2f}us"
                )
        records.append(record)
        del args, kwargs, fused
        torch.npu.empty_cache()

    report_path = Path(pytestconfig.getoption("--perf-output"))
    output_path = report_path.with_name(
        f"{report_path.stem}.layernorm_gated{report_path.suffix or '.json'}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "implementation": implementation,
        "device": torch.npu.get_device_name(0),
        "calibration": calibration,
        "records": records,
        "failures": failures,
        "npu_smi": subprocess.run(
            ["npu-smi", "info"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"layernorm_gated performance JSON: {output_path}")
    assert not failures, "\n".join(failures)
