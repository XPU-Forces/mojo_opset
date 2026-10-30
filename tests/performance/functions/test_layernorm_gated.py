"""Acceptance benchmark for fused gated normalization on Ascend NPUs.

Measurement follows the task definition: one call is 5 warmup launches followed by
30 profiled launches, and the reported time is the sum of every device kernel the
profiler records divided by 30 (the default ``do_bench_npu_profiler`` window). The
memory-bandwidth denominator is calibrated in-process with contiguous BF16 probes.
"""

import glob
import inspect
import json
import os
import shutil
import subprocess
import time

from pathlib import Path
from typing import NamedTuple
from typing import Optional

import pytest
import torch

pytest.importorskip("torch_npu")

from mojo_opset import functions as F

do_bench_npu_profiler = pytest.importorskip("triton.backends.ascend.testing").do_bench_npu_profiler

DTYPE = torch.bfloat16
ELEMENT_SIZE = DTYPE.itemsize
CALIBRATION_ELEMENTS = 128 * 1024 * 1024


class Case(NamedTuple):
    name: str
    M: int
    N: int
    gate: str  # "gdn" (3D strided), "2d", "none"
    activation: str
    group_size: Optional[int]
    quant_heads: int
    threshold: Optional[float]


CASES = [
    Case("gdn-256-n128-swish", 256, 128, "gdn", "swish", None, 0, None),
    Case("gdn-2560-n128-swish", 2560, 128, "gdn", "swish", None, 0, None),
    Case("gdn-32768-n128-swish", 32768, 128, "gdn", "swish", None, 0, 60.0),
    Case("gdn-32768-n128-sigmoid", 32768, 128, "gdn", "sigmoid", None, 0, 60.0),
    Case("gdn-131072-n128-swish", 131072, 128, "gdn", "swish", None, 0, 60.0),
    Case("qk-65536-n128", 65536, 128, "none", "swish", None, 0, 60.0),
    Case("group-1024-n2048", 1024, 2048, "2d", "swish", 128, 0, None),
    Case("group-8192-n2048", 8192, 2048, "2d", "swish", 128, 0, 60.0),
    Case("quant-32768-n512", 32768, 512, "none", "swish", None, 8, 60.0),
    Case("gdn-524288-n128-swish", 524288, 128, "gdn", "swish", None, 0, 60.0),
    Case("tail-32771-n192-swish", 32771, 192, "2d", "swish", None, 0, 50.0),
]


def _make_inputs(case: Case, device):
    torch.manual_seed(20261030)
    x = torch.randn((case.M, case.N), device=device, dtype=DTYPE)
    weight = torch.randn((case.N,), device=device, dtype=DTYPE)
    z = None
    if case.gate == "gdn":
        heads = 8
        padded = torch.empty((case.M // heads, heads, case.N + 32), device=device, dtype=DTYPE)
        padded.normal_()
        z = padded[..., :case.N]
    elif case.gate == "2d":
        z = torch.randn((case.M, case.N), device=device, dtype=DTYPE)
    kwargs = {
        "z": z,
        "is_rms_norm": True,
        "activation": case.activation,
        "quant_heads": case.quant_heads,
    }
    if case.group_size is not None:
        kwargs["group_size"] = case.group_size
        kwargs["norm_before_gate"] = False
    return (x, weight, None, 1e-6), kwargs


def _traffic_bytes(case: Case) -> int:
    if case.quant_heads:
        return (
            2 * case.M * case.N * ELEMENT_SIZE
            + case.M * case.N
            + 4 * case.M // case.quant_heads
            + 2 * case.N * ELEMENT_SIZE
        )
    reads = 3 if case.gate != "none" else 2
    return (reads * case.M * case.N + 2 * case.N) * ELEMENT_SIZE


def _find_kernel_csvs(root: Path):
    return glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)


_DO_BENCH_KWARGS = set(inspect.signature(do_bench_npu_profiler).parameters)


def _profile_call(fn, output_dir, warmup=5, active=30):
    output_dir = Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    requested = {"warmup": warmup, "active": active, "prof_dir": str(output_dir), "keep_res": True}
    result = do_bench_npu_profiler(fn, **{k: v for k, v in requested.items() if k in _DO_BENCH_KWARGS})
    # The default window (no target_kernel_name) sums every recorded kernel per call.
    if isinstance(result, dict):
        result = result.get("mean", result.get("total", float("inf")))
    elapsed_ms = float(result)
    if not elapsed_ms < float("inf"):
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


def _calibrate(device, output_root: Path):
    source = torch.empty(CALIBRATION_ELEMENTS, device=device, dtype=DTYPE)
    target = torch.empty_like(source)
    fill_us, _ = _profile_call(lambda: source.fill_(1.0), output_root / "calibration-fill")
    copy_us, _ = _profile_call(lambda: target.copy_(source), output_root / "calibration-copy")
    tensor_bytes = CALIBRATION_ELEMENTS * ELEMENT_SIZE
    return {
        "elements": CALIBRATION_ELEMENTS,
        "dtype": str(DTYPE),
        "fill_us": fill_us,
        "fill_bandwidth_GBps": tensor_bytes / (fill_us * 1e-6) / 1e9,
        "copy_us": copy_us,
        "copy_bandwidth_GBps": 2 * tensor_bytes / (copy_us * 1e-6) / 1e9,
    }


@pytest.mark.api("functions.layernorm_gated")
def test_layernorm_gated_performance(pytestconfig):
    implementation = pytestconfig.getoption("--mojo-implementation") or "triton"
    device_index = pytestconfig.getoption("--perf-device")
    device = f"npu:{device_index}"
    device_name = torch.npu.get_device_name(device_index)
    report_path = Path(pytestconfig.getoption("--perf-output"))
    output_root = report_path.parent / "layernorm_gated_raw"
    output_root.mkdir(parents=True, exist_ok=True)

    # The denominator is the measured copy probe, per the task definition.
    calibration = _calibrate(device, output_root)
    bandwidth_bytes_s = calibration["copy_bandwidth_GBps"] * 1e9
    records = []
    failures = []
    for index, case in enumerate(CASES):
        if case.quant_heads and "950" not in device_name:
            records.append(
                {
                    "case": case.name,
                    "status": "skipped",
                    "reason": "FP8 quant is formally required only on Ascend 950",
                }
            )
            continue
        args, kwargs = _make_inputs(case, device)
        fused = lambda: F.layernorm_gated(*args, **kwargs, implementation=implementation)  # noqa: E731
        fused_us, csv_paths = _profile_call(fused, output_root / f"case-{index:02d}")
        traffic = _traffic_bytes(case)
        mbu = traffic / (fused_us * 1e-6) / bandwidth_bytes_s
        record = {
            "case": case.name,
            "M": case.M,
            "N": case.N,
            "gate": case.gate,
            "quant_heads": case.quant_heads,
            "traffic_bytes": traffic,
            "traffic_MB": round(traffic / 1e6, 3),
            "fused_us": round(fused_us, 3),
            "MBU": round(mbu, 5),
            "threshold": case.threshold,
            "profiler_csv": [os.path.relpath(path, output_root) for path in csv_paths],
        }
        if case.threshold is not None and mbu < case.threshold / 100:
            failures.append(f"{case.name}: MBU={mbu:.2%} < {case.threshold:g}%")
        if traffic < 16 * 1024 * 1024:
            # Small shapes are judged against the eager composition instead of MBU.
            fused_e2e = _e2e_us(fused)
            eager = lambda: F.layernorm_gated(*args, **kwargs, implementation="torch")  # noqa: E731
            eager_us, _ = _profile_call(eager, output_root / f"case-{index:02d}-torch")
            eager_e2e = _e2e_us(eager)
            record.update(
                e2e_fused_us=round(fused_e2e, 3),
                eager_us=round(eager_us, 3),
                eager_e2e_us=round(eager_e2e, 3),
            )
            if fused_us > eager_us:
                failures.append(f"{case.name}: fused {fused_us:.2f}us > eager {eager_us:.2f}us")
            if fused_e2e > eager_e2e:
                failures.append(
                    f"{case.name}: fused e2e {fused_e2e:.2f}us > eager e2e {eager_e2e:.2f}us"
                )
        records.append(record)
        del args, kwargs, fused
        torch.npu.empty_cache()

    output_path = report_path.parent / f"{report_path.stem}.layernorm_gated.json"
    result = {
        "implementation": implementation,
        "device": device_name,
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
