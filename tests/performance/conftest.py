import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import traceback

from datetime import datetime
from datetime import timezone
from pathlib import Path

import pytest
import torch

from mojo_opset import Target
from mojo_opset import config
from mojo_opset.functions._dispatch import resolve_implementation
from mojo_opset.utils import target as target_utils

from ._adapters import load_platform
from .memory import measure_memory
from .timing import measure


def _git(*args):
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def pytest_configure(config):
    if config.option.collectonly:
        return
    if getattr(config.option, "numprocesses", 0) or hasattr(config, "workerinput"):
        raise pytest.UsageError("Performance tests require serial execution; disable pytest-xdist")
    if config.getoption("--perf-warmup") < 1 or config.getoption("--perf-repeats") < 3:
        raise pytest.UsageError("--perf-warmup must be >= 1 and --perf-repeats >= 3")
    timers = config.getoption("--perf-timers").split(",")
    if not timers or len(set(timers)) != len(timers) or set(timers) - {"profiler", "event", "e2e"}:
        raise pytest.UsageError("--perf-timers must contain distinct profiler,event,e2e values")
    if config.getoption("--perf-instances") < 2:
        raise pytest.UsageError("--perf-instances must be >= 2")
    output = Path(config.getoption("--perf-output")).resolve()
    if output.suffix != ".json":
        raise pytest.UsageError("--perf-output must be a .json path")
    config._mojo_perf_report = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "revision": {"commit": _git("rev-parse", "HEAD"), "dirty": _git("status", "--porcelain")},
        "cases": [],
        "skipped": [],
    }


@pytest.fixture(scope="session")
def perf_environment(pytestconfig, request):
    implementation, target, device_type = request.getfixturevalue("accuracy_backend")
    adapter = load_platform(Target.parse(target).platform)
    index = pytestconfig.getoption("--perf-device")
    adapter.runtime.set_device(index)
    properties = adapter.runtime.get_device_properties(index)
    l2 = adapter.l2_bytes(properties)
    report = pytestconfig._mojo_perf_report
    report["host"] = socket.gethostname()
    report["environment"] = {
        "target": target,
        "device_name": str(properties.name),
        "device_index": index,
        "total_memory": int(properties.total_memory),
        "l2_bytes": int(l2) if l2 is not None else None,
        "runner_id": os.getenv("MOJO_PERF_RUNNER_ID", socket.gethostname()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": _version("torch-npu"),
        "torch_mlu": _version("torch-mlu"),
        "triton": _version("triton"),
        "triton_x": _version("byted-triton-x"),
    }
    report["measurement"] = {
        "timers": sorted(pytestconfig.getoption("--perf-timers").split(",")),
        "profiler_timer": "per_call_kernel_v4",
        "event_timer": "current_stream_batch_v1",
        "e2e_timer": "synchronized_wall_v2",
        "memory": "allocator_peak_after_timing_v1",
        "batch": 1,
        "instances": pytestconfig.getoption("--perf-instances"),
        "reduction": "span",
        "cache": "rotate",
        "warmup": pytestconfig.getoption("--perf-warmup"),
        "repeats": pytestconfig.getoption("--perf-repeats"),
        "harness_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
            + Path(__file__).with_name("timing.py").read_bytes()
            + Path(__file__).with_name("memory.py").read_bytes()
            + Path(__file__).with_name("_workload.py").read_bytes()
            + Path(adapter.__file__).read_bytes()
        ).hexdigest(),
    }
    report["config"] = config.get_config() if implementation is None else None
    return adapter, f"{device_type}:{index}", target, implementation


@pytest.fixture
def benchmark(request, perf_environment, monkeypatch):
    adapter, device, target, implementation = perf_environment
    monkeypatch.setattr(target_utils, "_AUTO_TARGET", Target.parse(target))
    torch.manual_seed(42)
    report = request.config._mojo_perf_report

    def run(*, factory, op, **parameters):
        declared = {
            name
            for marker in request.node.iter_markers("api")
            for name in marker.kwargs.get("ops", (api.removeprefix("functions.") for api in marker.args))
        }
        if op not in declared:
            raise ValueError(f"{request.node.nodeid}: benchmark op {op!r} is missing from its api mark")
        if any(row["id"] == request.node.nodeid for row in report["cases"]):
            raise ValueError("Call benchmark once per test; parametrize separate workloads")
        selected = implementation or resolve_implementation(op, Target.parse(target))
        count = report["measurement"]["instances"]
        workload = [factory() for _ in range(count)]
        metrics = measure(
            workload,
            adapter,
            warmup=request.config.getoption("--perf-warmup"),
            repeats=request.config.getoption("--perf-repeats"),
            timers=report["measurement"]["timers"],
        )
        row = {
            "id": request.node.nodeid,
            "parameters": {
                **parameters,
                "op": op,
                "implementation": selected,
                "workload_sha256": hashlib.sha256(Path(request.node.path).read_bytes()).hexdigest(),
            },
            "metrics": metrics,
            "memory": measure_memory(workload, adapter, device),
        }
        report["cases"].append(row)
        return row

    return run


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()
    report = getattr(item.config, "_mojo_perf_report", None)
    if report is not None and "performance" in Path(item.path).parts:
        if result.skipped:
            report["skipped"].append(item.nodeid)
        elif result.failed and call.excinfo is not None:
            # The report is already formatted. Drop finished frames' tensor
            # references so a failed benchmark cannot retain its rotate inputs.
            traceback.clear_frames(call.excinfo.value.__traceback__)
            gc.collect()


def pytest_sessionfinish(session, exitstatus):
    report = getattr(session.config, "_mojo_perf_report", None)
    if report is None:
        return
    report["exitstatus"] = int(exitstatus)
    output = Path(session.config.getoption("--perf-output"))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    terminal = session.config.pluginmanager.getplugin("terminalreporter")
    if terminal:
        terminal.write_line(f"Performance JSON: {output}")
