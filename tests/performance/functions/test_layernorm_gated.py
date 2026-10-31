"""Performance acceptance matrix for fused gated normalization on Ascend.

One measurement is the profiler window the task prescribes: five warmup launches
followed by thirty profiled ones, reported as the summed device-kernel time per
launch.  Memory bandwidth is calibrated in the same process with a contiguous
BF16 fill and copy, and MBU is the case traffic divided by that measured copy
bandwidth.  Small cases are additionally held against the eager torch path,
because a single short kernel cannot be judged on bandwidth alone.

Shape matrix, thresholds and the notion of "traffic" come from the task
statement.
"""

import glob
import inspect
import json
import os
import shutil
import subprocess
import time

from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu")

from mojo_opset import functions as F

_bench = pytest.importorskip("triton.backends.ascend.testing")
_profile = getattr(_bench, "do_bench_npu_profiler", None) or getattr(_bench, "do_bench_npu", None)
if _profile is None:
    pytest.skip("triton-ascend ships no NPU profiler benchmark helper", allow_module_level=True)

BF16 = torch.bfloat16
ROW_BYTES = BF16.itemsize
PROBE_ELEMENTS = 128 * 1024 * 1024
WARMUP = 5
ACTIVE = 30
SMALL_TRAFFIC = 16 * 1024 * 1024
_PROFILE_KWARGS = frozenset(inspect.signature(_profile).parameters)


def _case(name, rows, cols, gate, activation, group, quant_heads, floor):
    return {
        "name": name,
        "M": rows,
        "N": cols,
        "gate": gate,
        "activation": activation,
        "group_size": group,
        "quant_heads": quant_heads,
        "threshold": floor,
    }


CASES = [
    _case("gdn-256-n128-swish", 256, 128, "gdn", "swish", None, 0, None),
    _case("gdn-2560-n128-swish", 2560, 128, "gdn", "swish", None, 0, None),
    _case("gdn-32768-n128-swish", 32768, 128, "gdn", "swish", None, 0, 60.0),
    _case("gdn-32768-n128-sigmoid", 32768, 128, "gdn", "sigmoid", None, 0, 60.0),
    _case("gdn-131072-n128-swish", 131072, 128, "gdn", "swish", None, 0, 60.0),
    _case("qk-65536-n128", 65536, 128, "none", "swish", None, 0, 60.0),
    _case("group-1024-n2048", 1024, 2048, "2d", "swish", 128, 0, None),
    _case("group-8192-n2048", 8192, 2048, "2d", "swish", 128, 0, 60.0),
    _case("quant-32768-n512", 32768, 512, "none", "swish", None, 8, 60.0),
    _case("gdn-524288-n128-swish", 524288, 128, "gdn", "swish", None, 0, 60.0),
    _case("tail-32771-n192-swish", 32771, 192, "2d", "swish", None, 0, 50.0),
]


def _inputs(case, device):
    """Fix the seed so every run of the matrix profiles the same data."""

    torch.manual_seed(20261030)
    x = torch.randn((case["M"], case["N"]), device=device, dtype=BF16)
    weight = torch.randn((case["N"],), device=device, dtype=BF16)
    gate = None
    if case["gate"] == "gdn":
        heads = 8
        padded = torch.empty((case["M"] // heads, heads, case["N"] + 32), device=device, dtype=BF16)
        padded.normal_()
        gate = padded[..., : case["N"]]
    elif case["gate"] == "2d":
        gate = torch.randn((case["M"], case["N"]), device=device, dtype=BF16)
    kwargs = {
        "z": gate,
        "is_rms_norm": True,
        "activation": case["activation"],
        "quant_heads": case["quant_heads"],
    }
    if case["group_size"] is not None:
        kwargs["group_size"] = case["group_size"]
        kwargs["norm_before_gate"] = False
    return (x, weight, None, 1e-6), kwargs


def _traffic(case):
    """Bytes the fused kernel is expected to move for this case.

    A quantised call writes the FP8 tile and the per-block scale instead of the
    BF16 output; every other call reads x, optionally z, and writes y.
    """

    rows, cols = case["M"], case["N"]
    if case["quant_heads"]:
        return 2 * rows * cols * ROW_BYTES + rows * cols + 4 * rows // case["quant_heads"] + 2 * cols * ROW_BYTES
    streams = 3 if case["gate"] != "none" else 2
    return (streams * rows * cols + 2 * cols) * ROW_BYTES


def _csv_traces(directory):
    return glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)


def _kernel_us(fn, directory, warmup=WARMUP, active=ACTIVE):
    """Profiled microseconds per call, plus the profiler's kernel traces."""

    directory = Path(directory)
    shutil.rmtree(directory, ignore_errors=True)
    request = {"warmup": warmup, "active": active, "prof_dir": str(directory), "keep_res": True}
    measured = _profile(fn, **{key: value for key, value in request.items() if key in _PROFILE_KWARGS})
    if isinstance(measured, dict):
        measured = measured.get("mean", measured.get("total", float("inf")))
    measured = float(measured)
    if not measured < float("inf"):
        raise RuntimeError(f"the profiler returned nothing for {directory}")
    return measured * 1e3, _csv_traces(directory)


def _wall_us(fn, warmup=WARMUP, active=ACTIVE):
    """End-to-end microseconds per call, launch overhead included."""

    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(active):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1e6 / active


def _calibrate(device, root):
    """Fill and copy probes on BF16; the copy one is the MBU denominator."""

    source = torch.empty(PROBE_ELEMENTS, device=device, dtype=BF16)
    target = torch.empty_like(source)
    moved = PROBE_ELEMENTS * ROW_BYTES
    fill_us, _ = _kernel_us(lambda: source.fill_(1.0), root / "calibration-fill")
    copy_us, _ = _kernel_us(lambda: target.copy_(source), root / "calibration-copy")
    return {
        "elements": PROBE_ELEMENTS,
        "dtype": str(BF16),
        "fill_us": fill_us,
        "fill_bandwidth_GBps": moved / (fill_us * 1e-6) / 1e9,
        "copy_us": copy_us,
        "copy_bandwidth_GBps": 2 * moved / (copy_us * 1e-6) / 1e9,
    }


def _timed(args, kwargs, implementation):
    return lambda: F.layernorm_gated(*args, **kwargs, implementation=implementation)


def _judge(record, mbu, case, floor):
    """Record every way this case can fail the acceptance matrix."""

    problems = []
    if floor is not None and mbu < floor / 100:
        problems.append(f"{case['name']}: MBU={record['MBU']:.2%} < {floor:g}%")
    if record["traffic_bytes"] < SMALL_TRAFFIC:
        if record["fused_us"] > record["eager_us"]:
            problems.append(f"{case['name']}: fused {record['fused_us']:.2f}us > eager {record['eager_us']:.2f}us")
        if record["e2e_fused_us"] > record["eager_e2e_us"]:
            problems.append(
                f"{case['name']}: fused e2e {record['e2e_fused_us']:.2f}us > eager e2e {record['eager_e2e_us']:.2f}us"
            )
    return problems


@pytest.mark.api("functions.layernorm_gated")
def test_layernorm_gated_performance(pytestconfig):
    implementation = pytestconfig.getoption("--mojo-implementation") or "triton"
    device_index = pytestconfig.getoption("--perf-device")
    device = f"npu:{device_index}"
    device_name = torch.npu.get_device_name(device_index)
    report = Path(pytestconfig.getoption("--perf-output"))
    root = report.parent / "layernorm_gated_raw"
    root.mkdir(parents=True, exist_ok=True)

    calibration = _calibrate(device, root)
    copy_bytes_per_s = calibration["copy_bandwidth_GBps"] * 1e9
    records, failures = [], []

    for index, case in enumerate(CASES):
        if case["quant_heads"] and "950" not in device_name:
            records.append(
                {
                    "case": case["name"],
                    "status": "skipped",
                    "reason": "FP8 quant is formally required only on Ascend 950",
                }
            )
            continue

        args, kwargs = _inputs(case, device)
        fused = _timed(args, kwargs, implementation)
        fused_us, traces = _kernel_us(fused, root / f"case-{index:02d}")
        traffic = _traffic(case)
        mbu = traffic / (fused_us * 1e-6) / copy_bytes_per_s
        record = {
            "case": case["name"],
            "M": case["M"],
            "N": case["N"],
            "gate": case["gate"],
            "quant_heads": case["quant_heads"],
            "traffic_bytes": traffic,
            "traffic_MB": round(traffic / 1e6, 3),
            "fused_us": round(fused_us, 3),
            "MBU": round(mbu, 5),
            "threshold": case["threshold"],
            "profiler_csv": [os.path.relpath(path, root) for path in traces],
        }
        if traffic < SMALL_TRAFFIC:
            eager = _timed(args, kwargs, "torch")
            eager_us, _ = _kernel_us(eager, root / f"case-{index:02d}-torch")
            record.update(
                e2e_fused_us=round(_wall_us(fused), 3),
                eager_us=round(eager_us, 3),
                eager_e2e_us=round(_wall_us(eager), 3),
            )
        failures.extend(_judge(record, mbu, case, case["threshold"]))
        records.append(record)
        del args, kwargs, fused
        torch.npu.empty_cache()

    payload = {
        "implementation": implementation,
        "device": device_name,
        "calibration": calibration,
        "records": records,
        "failures": failures,
        "npu_smi": subprocess.run(["npu-smi", "info"], check=False, capture_output=True, text=True).stdout,
    }
    output = report.parent / f"{report.stem}.layernorm_gated.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"layernorm_gated performance JSON: {output}")
    assert not failures, "\n".join(failures)
